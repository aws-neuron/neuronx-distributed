"""
Core functions for model tracing, compilation, and distributed checkpoint sharding operations.
"""
import os
import re
import time
import inspect
import pathlib
import logging
import warnings
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set

import torch
from safetensors.torch import save_file
import torch_neuronx
from libneuronxla import neuron_xla_compile # type: ignore
from torch_neuronx.proto import metaneff_pb2
from torch_neuronx.pyhlo import hlo_pb2
from torch_neuronx.utils.utils import get_platform_target, SUPPORTED_TYPES
try:
    from libneuronxla import neuron_xla_wlo_compile # type: ignore
except ImportError:
    # This is a temporary check to allow users to upgrade LibNeuronXla and utilize
    # neuron persistent cache feature as part of neuron_xla_wlo_compile().
    warnings.warn("neuron_xla_wlo_compile() API requires a later version of libneuronxla, "
    "upgrade to enable Neuron persistent cache for HLO compilation.", category=ImportWarning)
    neuron_xla_wlo_compile = None

from neuronx_distributed.parallel_layers import parallel_state
import neuronx_distributed.trace.hlo_utils as hlo_utils
from neuronx_distributed.trace.model_builder_utils import (
    ModelBuilderConstants,
    TraceArtifacts,
    CompilationArtifacts,
    WLOArtifacts,
    LayoutTransformerArtifacts,
    generate_key,
    ModelParamInfo,
    ProvidedArgInfo,
)
from neuronx_distributed.trace.trace import get_sharded_checkpoint, preprocess_checkpoint

logger = logging.getLogger("Neuron")


def append_default_compiler_flags(compiler_args: Optional[str] = "") -> str:
    if not compiler_args:
        compiler_args = "--enable-saturate-infinity --auto-cast=none --model-type=transformer -O1"

    # ensure --auto-cast value is part of compiler arg
    if "--auto-cast" not in compiler_args:
        compiler_args += " --auto-cast=none "
    elif "--auto-cast=none" not in compiler_args:
        # user provided other value, emit a warning
        logger.warning(
            "Compiler argument ``--auto-cast=none`` not detected. Other values might result in lower accuracy"
        )

    if not re.search(r'-O\d+', compiler_args):
        compiler_args += " -O1 "

    # append `--verbose=35`
    if "--verbose" not in compiler_args:
        compiler_args += " --verbose=35"

    return compiler_args


def _get_weight_names_to_skip(
    hlo_module: hlo_pb2.HloModuleProto,
    weight_name_to_idx: Dict[str, int],
    weights_to_skip_layout_optimization: Set
) -> Set:
    """
    Determines which weight names should be skipped based on environment variables and HLO analysis.
    
    Args:
        hlo_module: The HLO module to analyze
        weight_name_to_idx: Mapping of weight names to their indices
        
    Returns:
        Set of weight names that should be skipped, or None
    """
    weight_names_to_skip = set(weights_to_skip_layout_optimization)
    
    # Get computation map for easy lookup
    id_to_computation = {cpt.id: cpt for cpt in hlo_module.computations}
    entry_computation = id_to_computation[hlo_module.entry_computation_id]

    # Create idx to weight name mapping
    idx_to_weight_name = {
        idx: weight_name
        for weight_name, idx in weight_name_to_idx.items()
    }

    # Get NKI kernel weight names to skip
    weight_names_to_skip = hlo_utils.get_nki_kernel_weight_names(
        entry_cpt=entry_computation,
        idx_to_weight_name=idx_to_weight_name,
        nki_kernel_weight_names=weight_names_to_skip,
        id_to_computation=id_to_computation
    )
        
    return weight_names_to_skip


def _update_metaneff_with_user_input_key(
    metaneff: metaneff_pb2.MetaNeff,
    flattener: Any,
    provided_args: List[ProvidedArgInfo]
) -> None:
    """
    Updates metaneff with user input keys for all positional and keyword arguments.

    Args:
        metaneff: The metaneff to update.
        flattener: Object containing layout and exclude information.
            - layout: Tuple of indices representing the original parameter ordering (e.g., (0,1,2,3))
            - exclude: List of indices to exclude from the inputs (e.g., [1,3]), or None if no exclusions
        provided_args: List of ProvidedArgInfo for all provided args.
    """
    layout = flattener.layout
    exclude = set(flattener.exclude) if flattener.exclude is not None else set()

    metaneff_to_original_idx = {}
    metaneff_idx = 0

    for orig_idx in layout:
        # Skip excluded indices
        if orig_idx in exclude:
            continue

        # Map the metaneff index to the original parameter index
        metaneff_to_original_idx[metaneff_idx] = orig_idx
        metaneff_idx += 1

    # Update the user_input_key for each input tensor
    for idx, input_tensor in enumerate(metaneff.input_tensors):
        if input_tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT:
            # Get the original parameter index for this metaneff input
            if idx in metaneff_to_original_idx:
                orig_idx = metaneff_to_original_idx[idx]
                # Set the user_input_key using the corresponding provided_args name
                input_tensor.user_input_key = provided_args[orig_idx].param_name.encode("utf-8")
        else:
            # Break out of the loop when we encounter the first non-user-input metatensor
            break


def _validate_model(model: Union[Callable, torch.nn.Module]) -> inspect.Signature:
    """
    Validates the model and returns its function signature.
    
    Args:
        model: The model to validate
        
    Returns:
        The model's function signature
        
    Raises:
        ValueError: If model is None
        NotImplementedError: If model contains unsupported signature patterns
    """
    if model is None:
        raise ValueError("Model cannot be None")
        
    # Get the model's function signature
    if isinstance(model, torch.nn.Module):
        model_sig = inspect.signature(model.forward)
    else:
        model_sig = inspect.signature(model)

    # Check if signature is empty
    if not model_sig.parameters:
        raise ValueError("Model must have at least one parameter")

    # Check for *args and **kwargs
    params = model_sig.parameters
    var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())
    var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    
    if var_positional or var_keyword:
        raise NotImplementedError(
            "Methods with *args or **kwargs are not supported. "
            "Please explicitly specify all parameters."
        )
        
    return model_sig


def _validate_model_params(params: Any) -> List[ModelParamInfo]:
    """
    Validates and processes the model's function signature parameters.
    
    Args:
        params: The signature parameters to validate
        
    Returns:
        List of ModelParamInfo objects
        
    Raises:
        ValueError: If parameters have invalid default values
    """
    model_params = []
    for name, param in params.items():
        if name != 'self':
            is_positional = param.default == inspect.Parameter.empty
            
            # Check that parameters with default values only have None as default
            if not is_positional and param.default is not None:
                raise ValueError(
                    f"Parameter '{name}' has a non-None default value: {param.default}. "
                    f"Only None is allowed as a default value for parameters."
                )
                
            model_params.append(ModelParamInfo(
                param_name=name, 
                is_positional=is_positional
            ))
            
    return model_params


def _validate_args(
    args: Union[None, torch.Tensor, Tuple[torch.Tensor, ...]],
    model_params: List[ModelParamInfo]
) -> Tuple[torch.Tensor, ...]:
    """
    Validates positional arguments.
    
    Args:
        args: The positional arguments to validate
        model_params: List of model's parameter information
        
    Returns:
        Validated tuple of tensors
        
    Raises:
        ValueError: If args are invalid
    """
    if args is None:
        return tuple()
        
    if isinstance(args, torch.Tensor):
        args = (args,)
        
    if not isinstance(args, tuple) or not all(isinstance(t, torch.Tensor) for t in args):
        raise ValueError("args must be either None, a single tensor, or a tuple of tensors")

    # Get total number of parameters (both required and optional)
    total_params = len(model_params)
    
    # Check if we have too many arguments
    if len(args) > total_params:
        raise ValueError(
            f"Too many positional arguments. Model accepts {total_params} "
            f"but received {len(args)}"
        )

    return args


def _validate_kwargs(
    kwargs: Optional[Dict[str, torch.Tensor]],
    model_params: List[ModelParamInfo],
    provided_args: List[ProvidedArgInfo]
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Validates keyword arguments.
    
    Args:
        kwargs: The keyword arguments to validate
        model_params: List of model's parameter information
        provided_args: List of provided args information
        
    Returns:
        Validated kwargs dictionary
        
    Raises:
        ValueError: If kwargs are invalid
    """
    if kwargs is None:
        return None
        
    if not isinstance(kwargs, dict):
        raise ValueError("kwargs must be a dictionary")

    # Validate kwargs against function signature
    param_names = {p.param_name for p in model_params}
    invalid_keys = set(kwargs.keys()) - param_names
    if invalid_keys:
        raise ValueError(
            f"Found unexpected keys in kwargs: {invalid_keys}. "
            f"Valid keys are: {list(param_names)}"
        )
    
    # Check for parameters that were already provided as positional arguments
    positional_param_names = {p.param_name for p in provided_args}
    duplicate_keys = set(kwargs.keys()) & positional_param_names
    if duplicate_keys:
        raise ValueError(
            f"Parameters {duplicate_keys} were already provided as positional arguments "
            f"and cannot be overridden by keyword arguments"
        )
        
    # Validate tensor types
    for key, value in kwargs.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Value for key '{key}' must be a tensor")
            
    return kwargs


def _process_example_inputs(
    model: Union[Callable, torch.nn.Module],
    args: Union[None, torch.Tensor, Tuple[torch.Tensor, ...]],
    kwargs: Optional[Dict[str, torch.Tensor]]
) -> Tuple[List[ProvidedArgInfo], List[ModelParamInfo]]:
    """
    Process and validate input tensors from both args and kwargs.

    Args:
        model: The model whose inputs are being processed.
        args: Positional arguments to the model.
        kwargs: Keyword arguments to the model.

    Returns:
        Tuple containing:
            - List of argument information for all the provided arguments
            - List of parameter information for all parameters defined in 
              the model's function signature

    Raises:
        ValueError: If inputs are invalid or missing required tensors.
    """
    # Validate inputs
    model_sig = _validate_model(model)
    model_params = _validate_model_params(model_sig.parameters)
    validated_args = _validate_args(args, model_params)

    # Create map of param_name to its ModelParamInfo for easy lookup
    param_info_map = {p.param_name: p for p in model_params}
    
    # Process positional args
    provided_args = []
    for idx, tensor in enumerate(validated_args):
        param_name = model_params[idx].param_name
        provided_args.append(ProvidedArgInfo(
            param_name=param_name,
            is_positional=param_info_map[param_name].is_positional,
            tensor=tensor
        ))
    
    # Validate and process kwargs
    validated_kwargs = _validate_kwargs(kwargs, model_params, provided_args)
    
    # Track which required parameters have been provided
    provided_param_names = {arg.param_name for arg in provided_args}
    
    if validated_kwargs:
        for param_info in model_params:
            if param_info.param_name in validated_kwargs:
                provided_args.append(ProvidedArgInfo(
                    param_name=param_info.param_name,
                    is_positional=param_info_map[param_info.param_name].is_positional,
                    tensor=validated_kwargs[param_info.param_name]
                ))
                provided_param_names.add(param_info.param_name)

    # Check if all required parameters are provided
    missing_required = [
        p.param_name for p in model_params 
        if p.is_positional and p.param_name not in provided_param_names
    ]
    
    if missing_required:
        raise ValueError(
            f"Missing required parameters: {missing_required}. "
            f"These must be provided either as positional arguments or keyword arguments."
        )

    if not provided_args:
        raise ValueError("At least one input tensor must be provided via args or kwargs")

    return provided_args, model_params


def trace(
    model: Union[Callable, torch.nn.Module],
    args: Union[None, torch.Tensor, Tuple[torch.Tensor, ...]] = None,
    kwargs: Optional[Dict[str, torch.Tensor]] = None,
    spmd: bool = True,
    preserve_parameters: bool = True,
    weights_to_skip_layout_optimization: Optional[Set] = None
) -> TraceArtifacts:
    """
    Traces a model with the given example inputs.

    Args:
        model: The model to be traced.
        args: 
            The example inputs to be used for tracing in the form of positional arguments.
        kwargs: 
            The example inputs to be used for tracing in the form of keyword arguments.
        spmd: Whether to use SPMD for tracing. Currently only SPMD=True
            is supported. Defaults to True.
        preserve_parameters: Whether to preserve parameters. Recommended to be
            False if tracing only one bucket for a particular module. Should be set to True
            when tracing multiple buckets of the same module.
        weights_to_skip_layout_optimization: A set of weight names to skip during layout optimization.

    Returns:
        TraceArtifacts: An instance of the TraceArtifacts class
    """
    if not spmd:
        raise NotImplementedError("MPMD tracing is not currently supported")

    if model is None:
        raise ValueError("Model cannot be None")

    # Process the example_inputs and get argument info
    provided_args, model_params = _process_example_inputs(model, args, kwargs)
    example_inputs = OrderedDict((arg.param_name, arg.tensor) for arg in provided_args)

    try:
        # Generate HLO with processed inputs
        hlo_artifacts = torch_neuronx.xla_impl.trace.generate_hlo(
            model,
            example_inputs,
            inline_weights_to_neff=False,
            return_weights=False,
            output_aliased_tensor=False,
            cpu_backend=True,
            preserve_parameters=preserve_parameters,
            enable_aliasing=True,
            treat_inputs_as_kwargs=True,
        )
    except Exception as e:
        logger.error(f"HLO generation failed: {str(e)}")
        raise RuntimeError(f"HLO generation failed: {str(e)}") from e
    
    # Get weight names to skip
    weight_names_to_skip = _get_weight_names_to_skip(
        hlo_module=hlo_artifacts.hlo_module,
        weight_name_to_idx=hlo_artifacts.weight_name_to_idx,
        weights_to_skip_layout_optimization=set() if weights_to_skip_layout_optimization is None else weights_to_skip_layout_optimization,
    )

    # Update metaneff with user input keys
    _update_metaneff_with_user_input_key(
        hlo_artifacts.metaneff,
        hlo_artifacts.flattener,
        provided_args,
    )

    return TraceArtifacts(
        hlo=hlo_artifacts.hlo_module,
        metaneff=hlo_artifacts.metaneff,
        flattener=hlo_artifacts.flattener,
        packer=hlo_artifacts.packer,
        weight_name_to_idx=hlo_artifacts.weight_name_to_idx,
        weight_names_to_skip=weight_names_to_skip,
        provided_args=provided_args,
        model_params=model_params,
    )


def compile(
    hlo_module: hlo_pb2.HloModuleProto,
    metaneff: Any,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[str] = None,
    key: Optional[str] = None
) -> CompilationArtifacts:
    """
    Compiles the traced model with the Neuron Compiler to a Neuron Executable File Format (NEFF).

    Args:
        hlo_module: The HLO module representing the computational graph to be compiled.
        metaneff: The meta information for the Neuron Executable File Format (NEFF).
        compiler_workdir: Path to store compiler artifacts. If None, uses a default path.
        compiler_args: Compiler flags for neuronx-cc.
        key: Key to tag the bucket with a meaningful name. If None, a hash of the HLO will be used.

    Returns:
        An object containing the path to the compiled NEFF.
    """
    # Input validation
    missing_args = {
        'hlo_module': hlo_module,
        'metaneff': metaneff
    }
    missing = [arg_name for arg_name, arg_value in missing_args.items() if not arg_value]

    if missing:
        raise ValueError(f"Required compile arguments missing: {', '.join(missing)}")

    try:
        compile_start_time = time.time()

        # Generate key to tag the bucket with a meaningful name
        key = generate_key(hlo_module, key)
        logger.info(f"Started compilation for {key}")

        # Set-up compiler workdir
        timestamp = datetime.now().isoformat().replace(':', '-')
        output_dir = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR if compiler_workdir is None else compiler_workdir, key, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Compiler workdir for {key} is: {output_dir}")

        # Set compiler flags
        compiler_args = append_default_compiler_flags(compiler_args=compiler_args)
        if '--logfile' not in compiler_args:
            compiler_args += f" --logfile={os.path.join(output_dir, ModelBuilderConstants.LOG_FILE_DEFAULT_NAME)}"

        logger.info(f"Neuron compiler flags: {compiler_args}")

        module_bytes = hlo_module.SerializeToString()
        platform_target = get_platform_target(compiler_args)
        # Save HLO
        os.makedirs(os.path.join(output_dir, "model"), exist_ok=True)
        hlo_path = os.path.join(output_dir, "model", ModelBuilderConstants.GRAPH_HLO_FILE)
        hlo_utils.write_hlo(hlo_path, hlo_module)
        # Compile HLO
        neff_bytes = neuron_xla_compile(module_bytes=module_bytes,
                                        compiler_flags=compiler_args,
                                        input_format="hlo",
                                        platform_target=platform_target,
                                        cache_key=None,
                                        retry_failed_compilation=False,
                                        lazy=True,
                                        use_cache=False,
                                        cache_dir=None,
                                        work_dir=pathlib.Path(output_dir).absolute())
        # Save NEFF
        neff_path = os.path.join(output_dir, ModelBuilderConstants.NEFF_FILE)
        with open(neff_path, "wb") as f:
            f.write(neff_bytes)

        # Save metaneff
        metaneff_path = os.path.join(output_dir, ModelBuilderConstants.METANEFF_FILE)
        with open(metaneff_path, "wb") as f:
            f.write(metaneff.SerializeToString())

    except Exception as e:
        logger.error(f"Compilation failed for {key}: {str(e)}")
        raise RuntimeError(f"Compilation failed for {key}") from e

    logger.info(f"Finished compilation for {key} in {time.time() - compile_start_time} seconds")

    return CompilationArtifacts(neff_filepath=neff_path)


def _update_compiler_args_for_compile_wlo(
    compiler_args: Union[str, List[str], None],
    output_dir: str
) -> Union[str, List[str]]:
    """
    Updates the compiler args for the priority model.

    Args:
        compiler_args: Compiler flags for neuronx-cc.
        output_dir: Priority model compiler workdir.

    Returns:
        Updated compiler args. If input was a string, returns
        a space-separated string of args. If input was a list, returns a list of args.
        If input was None, returns a space-separated string of required args.

    Note:
        "--model-type=transformer" and "--enable-internal-neff-wrapper" are mandatory
        compiler args required for compile_wlo() and will be added if not present.
    """
    required_args = ["--model-type=transformer", "--enable-internal-neff-wrapper", "--auto-cast=none", "--verbose=35"]
    logfile_arg = f"--logfile={os.path.join(output_dir, ModelBuilderConstants.LOG_FILE_DEFAULT_NAME)}"

    if compiler_args is None:
        return " ".join(required_args + [logfile_arg])

    if isinstance(compiler_args, str):
        args_list = compiler_args.split()
    elif isinstance(compiler_args, list):
        args_list = compiler_args.copy()
    else:
        raise TypeError(f"compiler_args must be either a string, a list of strings, or None. Got {type(compiler_args).__name__}")

    # Add required arguments if they're not present
    args_list.extend(arg for arg in required_args if arg not in args_list)
    if not any(arg.startswith("--logfile") for arg in args_list):
        args_list.append(logfile_arg)

    return " ".join(args_list) if isinstance(compiler_args, str) else args_list


def compile_wlo(
    hlo_module: hlo_pb2.HloModuleProto,
    metaneff: Any,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[List[str], str]] = None,
    key: Optional[str] = None
) -> WLOArtifacts:
    """
    Compiles the priority model with the Neuron Compiler.

    Args:
        hlo_module: The HLO module representing the computational graph to be compiled.
        metaneff: The meta information for the Neuron Executable File Format (NEFF).
        compiler_workdir: Path to store compiler artifacts. If None, uses a default path.
        compiler_args: Additional compiler flags for neuronx-cc.
        key: Key to tag the bucket with a meaningful name. If None, a hash of the HLO will be used.

    Returns:
        An object containing the path to the compiled NEFF and wrapped NEFF HLO.
    """
    # Input validation
    missing_args = {
        'hlo_module': hlo_module,
        'metaneff': metaneff
    }
    missing = [arg_name for arg_name, arg_value in missing_args.items() if not arg_value]

    if missing:
        raise ValueError(f"Required compile_wlo arguments missing: {', '.join(missing)}")

    try:
        priority_model_compile_start_time = time.time()

        # Generate key to tag the bucket with a meaningful name
        key = generate_key(hlo_module, key)
        logger.info(f"Started compilation for the priority model {key}")

        # Set-up compiler workdir
        timestamp = datetime.now().isoformat().replace(':', '-')
        output_dir = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR if compiler_workdir is None else compiler_workdir, key, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Compiler workdir for the priority model {key} is: {output_dir}")

        # Set compiler flags for WLO
        compiler_args = _update_compiler_args_for_compile_wlo(compiler_args=compiler_args, output_dir=output_dir)

        module_bytes = hlo_module.SerializeToString()
        platform_target = get_platform_target(compiler_args)
        # Save HLO
        os.makedirs(os.path.join(output_dir, "model"), exist_ok=True)
        hlo_path = os.path.join(output_dir, "model", ModelBuilderConstants.GRAPH_HLO_FILE)
        hlo_utils.write_hlo(hlo_path, hlo_module)
        # Generate NEFF
        neff_bytes, wrapped_neff_bytes = neuron_xla_wlo_compile(module_bytes, compiler_args,
                                                                input_format="hlo",
                                                                platform_target=platform_target,
                                                                cache_key=None,
                                                                retry_failed_compilation=False,
                                                                lazy=True,
                                                                use_cache=False,
                                                                cache_dir=None,
                                                                work_dir=pathlib.Path(output_dir).absolute(),
                                                                create_subdir=False)
        # Save NEFF
        neff_path = os.path.join(output_dir, ModelBuilderConstants.NEFF_FILE)
        with open(neff_path, "wb") as f:
            f.write(neff_bytes)

        # Save wrapped NEFF
        wrapped_neff_path = None
        if wrapped_neff_bytes:
            wrapped_neff_path = os.path.join(output_dir, ModelBuilderConstants.WRAPPED_NEFF_FILE)
            with open(wrapped_neff_path, "wb") as f:
                f.write(wrapped_neff_bytes)

        # Save metaneff
        metaneff_path = os.path.join(output_dir, ModelBuilderConstants.METANEFF_FILE)
        with open(metaneff_path, "wb") as f:
            f.write(metaneff.SerializeToString())

    except Exception as e:
        logger.error(f"Compilation failed for priority model {key}: {str(e)}")
        raise RuntimeError(f"Compilation failed for priority model {key}") from e

    logger.info(f"Finished compilation for priority model {key} in {time.time() - priority_model_compile_start_time} seconds")

    return WLOArtifacts(
        neff_filepath=neff_path,
        wrapped_neff_hlo_filepath=wrapped_neff_path
    )


def compile_layout_transformer(
    wlo_artifacts: WLOArtifacts,
    priority_model_weight_name_to_idx: Dict[str, int],
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    logical_nc_config: int = 1,
) -> Optional[LayoutTransformerArtifacts]:
    """
    Compiles a layout transformer for weight layout optimization.

    This function takes Weight Layout Optimization (WLO) artifacts and compiles them into
    a layout transformer, generating necessary HLO, NEFF, and metaneff files.

    Args:
        wlo_artifacts: Weight Layout Optimization artifacts containing
            wrapped NEFF HLO filepath and other related information.
        priority_model_weight_name_to_idx: Mapping of weight names
            to their corresponding indices in the priority model.
        compiler_workdir: Path to store compiler
            artifacts. If None, uses a default path.
        logical_nc_config: Logical NC configuration parameter.
            Defaults to 1.

    Returns:
        An object containing paths to generated HLO, NEFF,
        and metaneff files, or None if no layout transformation is needed.
    """
    # Input validation
    missing_args = {
        'wlo_artifacts': wlo_artifacts,
        'priority_model_weight_name_to_idx': priority_model_weight_name_to_idx
    }
    missing = [arg_name for arg_name, arg_value in missing_args.items() if not arg_value]

    if missing:
        raise ValueError(f"Required compile_layout_transformer arguments missing: {', '.join(missing)}")

    try:
        layout_transform_compile_start_time = time.time()
        logger.info("Started compilation for layout transfomer")

        # Check if hlo_stub / wrapped_neff exists
        if wlo_artifacts.wrapped_neff_hlo_filepath is None:
            logger.info("No changes on weight layout, falling back to the existing weight layout")
            return None

        # Read the WLO stub
        wlo_stub = hlo_utils.read_hlo(wlo_artifacts.wrapped_neff_hlo_filepath)
        if not wlo_stub:
            raise ValueError("Failed to read WLO stub")

        # Set-up compiler workdir
        timestamp = datetime.now().isoformat().replace(':', '-')
        output_dir = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR if compiler_workdir is None else compiler_workdir, "layout_opt", timestamp)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Compiler workdir for the layout transformer is: {output_dir}")

        # Prepare HLO
        wlt_hlo = hlo_utils.extract_weight_layout_transform_hlo(
            hlo_stub=wlo_stub,
            weight_name_to_idx=priority_model_weight_name_to_idx,
        )

        # Set compiler flags
        compiler_args = (
            "--model-type=transformer" +
            f" --lnc={logical_nc_config}" +
            " --internal-hlo2tensorizer-options='--experimental-unsafe-fp8e4m3fn-as-fp8e4m3 --verify-hlo=true'" +
            f" --logfile={os.path.join(output_dir, 'log-neuron-cc.txt')}" +
            " --verbose=35"
        )

        module_bytes = wlt_hlo.SerializeToString()
        platform_target = get_platform_target(compiler_args)
        metaneff = hlo_utils.prepare_metaneff_for_wlt_hlo(
            wlt_hlo=wlt_hlo,
            weight_name_to_idx=priority_model_weight_name_to_idx,
        )
        metaneff_str = metaneff.SerializeToString()

        # Generate NEFF
        wlt_neff_bytes = neuron_xla_compile(module_bytes=module_bytes,
                                            compiler_flags=compiler_args,
                                            input_format="hlo",
                                            platform_target=platform_target,
                                            cache_key=None,
                                            retry_failed_compilation=False,
                                            lazy=True,
                                            use_cache=False,
                                            cache_dir=None,
                                            work_dir=pathlib.Path(output_dir).absolute())

        # Save HLO
        os.makedirs(os.path.join(output_dir, "model"), exist_ok=True)
        hlo_path = os.path.join(output_dir, "model", ModelBuilderConstants.GRAPH_HLO_FILE)
        hlo_utils.write_hlo(hlo_path, wlt_hlo)

        # Save NEFF
        neff_path = os.path.join(output_dir, ModelBuilderConstants.NEFF_FILE)
        with open(neff_path, "wb") as f:
            f.write(wlt_neff_bytes)

        # Save metaneff
        metaneff_path = os.path.join(output_dir, ModelBuilderConstants.METANEFF_FILE)
        with open(metaneff_path, "wb") as f:
            f.write(metaneff_str)

    except Exception as e:
        logger.error(f"Compilation failed for layout transformer: {str(e)}")
        raise RuntimeError("Compilation failed for layout transformer") from e

    logger.info(f"Done compilation for layout transformer in {time.time() - layout_transform_compile_start_time} seconds")

    return LayoutTransformerArtifacts(
        hlo_filepath=hlo_path,
        neff_filepath=neff_path,
        metaneff_filepath=metaneff_path,
    )


def shard_checkpoint(
    checkpoint: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    start_rank: Optional[int] = None,
    end_rank: Optional[int] = None,
    load_on_device: bool = False,
    serialize_path: Optional[str] = None
) -> List[Dict[str, torch.Tensor]]:
    """
    Shard a model checkpoint across tensor parallel ranks for distributed execution.

    This function splits a model checkpoint dictionary into multiple sharded dictionaries,
    each corresponding to a specific tensor parallel rank. The sharded checkpoints can be
    optionally serialized to disk and/or loaded directly onto Neuron devices.

    Parameters
    ----------
    checkpoint:
        The model checkpoint dictionary mapping parameter names to tensor values.
    model:
        The PyTorch model to be sharded.
    start_rank:
        The starting rank for sharding. Must be in the range [0, tp_degree).
        If None, start_rank will be set to 0.
    end_rank:
        The ending rank for sharding. Must be in the range [start_rank, tp_degree)
        If None, end_rank will be set to tp_degree - 1.
        If end rank == start rank, this means only the shard for that rank is generated.
    load_on_device:
        If True, loads sharded tensors onto corresponding Neuron devices.
        Requires running on a supported Neuron instance (trn1, inf2, trn2, etc.).
    serialize_path:
        If provided, saves each sharded checkpoint to this directory path.

    Returns
    -------
    A list of sharded checkpoint dictionaries, one for each tensor parallel rank.

    Raises
    ------
    AssertionError
        If NxD parallel state has not been initialized.
    SystemError
        If attempting to load tensors on device when not running on a supported Neuron instance.

    Notes
    -----
    - Requires NxD parallel state to be initialized before calling
    - The function preprocesses the checkpoint for the model before sharding
        **It is highly recommended to send a copy of the sharded checkpoint
        if used later in your application to avoid unintended side effects.**
    - When serializing to disk, checkpoints are saved as safetensors files
    """
    assert parallel_state.model_parallel_is_initialized(), "NxD parallel state has not been initialized"
    preprocess_checkpoint(model, checkpoint)
    tp_degree: int = parallel_state.get_tensor_model_parallel_size()
    if start_rank is None:
        start_rank = 0
    assert 0 <= start_rank < tp_degree, f"start_rank must be in range [0, {tp_degree}), but found {start_rank=}."
    if end_rank is None:
        end_rank = tp_degree - 1
    assert start_rank <= end_rank < tp_degree, f"end_rank must be in range [{start_rank}, {tp_degree}) but found {end_rank=}."

    sharded_checkpoint = []
    for tp_rank in range(start_rank, end_rank + 1, 1): # end_rank + 1 as range interval is [start,end)
        sharded_sd = checkpoint.copy()
        get_sharded_checkpoint(
            sharded_sd,
            model,
            tp_rank,
            tp_degree,
            is_cached=True
        )
        if serialize_path is not None:
            if not os.path.exists(serialize_path):
                os.makedirs(serialize_path, exist_ok=True)
            save_file(
                {k:v.contiguous() for k,v in sharded_sd.items()},
                os.path.join(
                    serialize_path,
                    f"tp{tp_rank}_sharded_checkpoint.safetensors"
                )
            )
        if load_on_device:
            # check if on a Neuron instance
            try:
                platform = get_platform_target()
            except (RuntimeError, IOError):
                platform = "UNSUPPORTED"
            finally:
                if platform not in SUPPORTED_TYPES:
                    raise SystemError("Attempted to load sharded weights onto Neuron on a non-Neuron/unsupported instance. Please set load_on_device=False.")

            for key in sharded_sd:
                sharded_sd[key] = sharded_sd[key].to(
                    f"privateuseone:{tp_rank}"
                ) # non contiguous tensors are made contiguous in our runtime integration

        sharded_checkpoint.append(sharded_sd)

    return sharded_checkpoint
