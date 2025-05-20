import concurrent.futures
import os
import shutil
import time
import logging
import hashlib
from pathlib import Path
import contextlib
import pathlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set
import warnings

import torch
import torch_xla
import torch.distributed
import torch_neuronx
import torch_neuronx.xla_impl
import torch_neuronx.xla_impl.trace
from libneuronxla import neuron_xla_compile # type: ignore
from torch_neuronx import BucketModelConfig
from torch_neuronx.proto import metaneff_pb2
from torch_neuronx.pyhlo import hlo_pb2
from torch_neuronx.xla_impl.trace import get_torch_dtype, HloArtifacts, generate_neff, NeffArtifacts
from torch_neuronx.utils import get_platform_target
from packaging import version
if version.parse(torch.__version__) >= version.parse("2.1"):
    from torch_neuronx.experimental.profiler.v2_x.custom_op_name import hlo_debug
else:
    hlo_debug = contextlib.nullcontext()

from safetensors.torch import save_file, load_file

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.spmd import (
    NxDModel,
    NxDModelExecutor,
    SPMDBucketModelScript,
    default_bucket_kernel,
    StateInitializer,
)
from neuronx_distributed.trace.trace import _mock_parallel_state, get_sharded_checkpoint, preprocess_checkpoint
from neuronx_distributed.trace.mock_torchdist import mock_distributed
from neuronx_distributed.trace.model_builder_utils import (
    ModelBuilderConstants,
    TraceArtifacts,
    CompilationArtifacts,
    generate_key,
)
import neuronx_distributed.trace.hlo_utils as hlo_utils
from neuronx_distributed.utils.model_utils import init_on_device
from neuronx_distributed.utils.safetensors_utils import remove_duplicate_tensors

ModelInputType = List[Union[Tuple[Union[torch.Tensor, List[torch.Tensor]]], torch.Tensor]]
logger = logging.getLogger("Neuron")

try:
    from libneuronxla import neuron_xla_wlo_compile # type: ignore
except ImportError:
    # This is a temporary check to allow users to upgrade LibNeuronXla and utilize
    # neuron persistent cache feature as part of neuron_xla_wlo_compile().
    warnings.warn("neuron_xla_wlo_compile() API requires a later version of libneuronxla, "
    "upgrade to enable Neuron persistent cache for HLO compilation.", category=ImportWarning)
    neuron_xla_wlo_compile = None


# Constants
NEFF_FILE = "graph.neff"
WRAPPED_NEFF_FILE = "wrapped_neff.hlo"
GRAPH_HLO_FILE = "graph.hlo"
LOG_FILE_DEFAULT_NAME = "log-neuron-cc.txt"


def get_hash_module(hlo_module, flags):
    # Hashing is pretty fast and negligible compared to compilation time
    hash_gen = hashlib.sha256()
    text = str(hlo_module)
    if flags is not None:
        text += flags.replace(" ", "")
    hash_gen.update(text.encode('utf-8'))
    hash = str(hash_gen.hexdigest())[:20]
    return hash


def trace(
    model: Union[Callable, torch.nn.Module],
    example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    spmd: bool = True,
    preserve_parameters: bool = True
) -> TraceArtifacts:
    """
    Traces a model with the given example inputs.

    Args:
        model (Union[Callable, torch.nn.Module]): The model to be traced.
        example_inputs (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): 
            The example inputs to be used for tracing. Can be either a single tensor
            or a tuple of tensors.
        spmd (bool, optional): Whether to use SPMD for tracing. Currently only SPMD=True
            is supported. Defaults to True.
        preserve_parameters (bool, Optional): Whether to preserve parameters. Recommended to be
            False if tracing only one bucket for a particular module. Should be set to True
            when tracing multiple buckets of the same module.

    Returns:
        TraceArtifacts: An instance of the TraceArtifacts class containing the traced HLO artifacts,
        metaneff, flattener, packer and weight name to idx mapping components.
    """
    if not spmd:
        raise NotImplementedError("MPMD tracing is not currently supported.")

    # Validate and preprocess the inputs
    if isinstance(example_inputs, torch.Tensor):
        processed_inputs = (example_inputs,)
    elif isinstance(example_inputs, tuple) and all(isinstance(t, torch.Tensor) for t in example_inputs):
        processed_inputs = example_inputs
    else:
        raise ValueError("example_inputs must be either a single tensor or a tuple of tensors")

    # Generate HLO
    hlo_artifacts = torch_neuronx.xla_impl.trace.generate_hlo(
        model,
        processed_inputs,
        inline_weights_to_neff=False,
        return_weights=False,
        output_aliased_tensor=False,
        cpu_backend=True,
        preserve_parameters=preserve_parameters,
        enable_aliasing=True,
    )

    return TraceArtifacts(
        hlo=hlo_artifacts.hlo_module,
        metaneff=hlo_artifacts.metaneff,
        flattener=hlo_artifacts.flattener,
        packer=hlo_artifacts.packer,
        weight_name_to_idx=hlo_artifacts.weight_name_to_idx
    )

def compile(
    hlo_module: hlo_pb2.HloModuleProto,
    flattener: Any,
    packer: Any,
    metaneff: Any,
    weight_name_to_idx: Optional[Dict[str, int]] = None,
    compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
    compiler_args: Optional[Union[str, List[str]]] = None,
    key: Optional[str] = None
) -> CompilationArtifacts:
    """
    Compiles the traced model with the Neuron Compiler to a Neuron Executable File Format (NEFF).

    Args:
        hlo_module (hlo_pb2.HloModuleProto): The HLO module representing the computational graph to be compiled.
        flattener (Any): Function to flatten inputs.
        packer (Any): Function to pack outputs.
        metaneff (Any): The meta information for the Neuron Executable File Format (NEFF).
        weight_name_to_idx (Optional[Dict[str, int]]): Dictionary mapping weight names to their indices.
        compiler_workdir (Optional[Union[str, pathlib.Path]]): Path to store compiler artifacts. If None, uses a default path.
        compiler_args (Optional[Union[str, List[str]]]): Compiler flags for neuronx-cc.
        key (Optional[str]): Key to tag the bucket with a meaningful name. If None, a hash of the HLO will be used.

    Returns:
        CompilationArtifacts: An object containing the path to the compiled NEFF.
    """
    # Input validation
    missing_args = {
        'hlo_module': hlo_module,
        'flattener': flattener,
        'packer': packer,
        'metaneff': metaneff
    }
    missing = [arg_name for arg_name, arg_value in missing_args.items() if not arg_value]
    
    if missing:
        raise ValueError(f"Required trace arguments missing: {', '.join(missing)}")

    try:
        logger.info(f"Started compilation for {key}")
        compile_start_time = time.time()

        # Generate key to tag the bucket with a meaningful name
        key = generate_key(hlo_module, key)

        # Set-up compiler workdir
        timestamp = datetime.now().isoformat().replace(':', '-')
        output_dir = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR if compiler_workdir is None else compiler_workdir, key, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Compiler workdir for {key} is: {output_dir}")

        # Set compiler flags
        if compiler_args is None:
            compiler_args = "--enable-saturate-infinity --auto-cast=none --model-type=transformer -O1"
        if isinstance(compiler_args, str):
            if '--logfile' not in compiler_args:
                compiler_args += f" --logfile={os.path.join(output_dir, ModelBuilderConstants.LOG_FILE_DEFAULT_NAME)}"
        elif isinstance(compiler_args, list):
            if not any('--logfile' in arg for arg in compiler_args):
                compiler_args.append(f"--logfile={os.path.join(output_dir, ModelBuilderConstants.LOG_FILE_DEFAULT_NAME)}")
        else:
            raise TypeError(f"compiler_args must be either a string or a list of strings, got {type(compiler_args).__name__}")

        # Prepare HloArtifacts
        hlo_artifacts = HloArtifacts(
            hlo_module=hlo_module,
            flattener=flattener,
            packer=packer,
            metaneff=metaneff,
            weights=None,
            constant_parameter_tensors=None,
            weight_name_to_idx=weight_name_to_idx
        )

        # Generate NEFF
        neff_artifacts = generate_neff(
            hlo_artifacts=hlo_artifacts,
            compiler_workdir=output_dir,
            compiler_args=compiler_args,
            inline_weights_to_neff=False
        )

        # Save metaneff
        metaneff_path = os.path.join(output_dir, ModelBuilderConstants.METANEFF_FILE)
        with open(metaneff_path, "wb") as f:
            f.write(hlo_artifacts.metaneff.SerializeToString())

        logger.info(f"Finished compilation for {key} in {time.time() - compile_start_time} seconds")

        return CompilationArtifacts(neff_filename=neff_artifacts.neff_filename)

    except Exception as e:
        logger.error(f"Compilation failed for {key}: {str(e)}")
        raise RuntimeError(f"Compilation failed for {key}") from e




class ModelBuilderV2:
    """
    A class for tracing and compiling models for Neuron devices.

    This class provides functionality to trace and compile models for efficient
    execution on Neuron hardware. It supports SPMD (Single Program Multiple Data)
    tracing and compilation.
    """
    def __init__(
        self,
        model: Union[Callable, torch.nn.Module],
        world_size: int,
        weights_to_skip_layout_optimization: Optional[Set] = None
    ):
        """
        Initialize the ModelBuilderV2.

        Args:
            model (Union[Callable, torch.nn.Module]): The PyTorch model to be traced and compiled.
            world_size (int): The number of parallel processes for SPMD execution.
            weights_to_skip_layout_optimization (Optional[Set]): A set of weight names to skip during layout optimization.

        Raises:
            AssertionError: If the torch-neuronx version is not compatible.
        """
        if not torch_neuronx.__version__.startswith("2"):
            raise AssertionError(
                f"ModelBuilderV2 requires torch-neuronx>=2.* but found torch-neuronx=={torch_neuronx.__version__}."
            )

        self.model = model
        self.world_size = world_size
        self.traces: Dict[str, TraceArtifacts] = {}
        self.weights_to_skip_layout_optimization = weights_to_skip_layout_optimization
    
    def trace(
        self,
        example_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        tag: Optional[str] = None,
        spmd: bool = True,
    ):
        """
        Traces a model with given example inputs and stores the resulting trace artifacts.

        Args:
            example_inputs (Union[torch.Tensor, Tuple[torch.Tensor, ...]]):
                The example inputs to be used for tracing. Can be either a single tensor 
                or a tuple of tensors.
            tag (Optional[str]): A unique identifier for this trace. If None, a default tag will be generated.
            spmd (bool): Whether to use SPMD for tracing. Currently only SPMD=True
                is supported. Defaults to True.

        Returns:
            self: Returns the instance to allow method chaining.
        """
        raise NotImplementedError("The trace method has not been implemented yet")
    
    def compile(
        self,
        priority_model_key: Optional[str] = None,
        compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
        compiler_args: Optional[Union[str, List[str]]] = None,
    ):
        """
        Compiles the traced model using the Neuron compiler, generating
        a Neuron Executable File Format (NEFF) for each trace.

        Args:
            priority_model_key (Optional[str]): Key of the model to prioritize during compilation.
            compiler_workdir (Optional[Union[str, pathlib.Path]]):
                Path to store compiler artifacts. If None, uses a default path.
            compiler_args (Optional[Union[str, List[str]]]): Compiler flags for neuronx-cc.

        Returns:
            The compiled model, ready for execution on Neuron hardware.
        """
        raise NotImplementedError("The compile method has not been implemented yet")


# TODO write a generic class which can accept a function as well
class BaseModelInstance:
    def __init__(self, module_cls, input_output_aliases):
        self.module_cls = module_cls
        self.module = None
        self.input_output_aliases = [input_output_aliases]

    def load_module(self):
        self.module = self.module_cls()

    def get(self, bucket_rank, **kwargs):
        return self.module, self.input_output_aliases[0]


def init_process_wrapper(args):
    def init_process(rank, world_size, backend, tp_degree, pp_degree, ep_degree):
        """Initialize the distributed environment and return results."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)

        parallel_state.initialize_model_parallel(tp_degree, pp_degree, ep_degree, skip_collective_init=True)

    init_process(*args)


class ModelContainer:
    def __init__(self, model_instance, example_inputs, compiler_args, bucket_config, priority_model_idx):
        self.model_instance: BaseModelInstance = model_instance
        self.example_inputs = example_inputs
        self.compiler_args = compiler_args
        self.bucket_config: Optional[BucketModelConfig] = bucket_config
        self.priority_model_idx = priority_model_idx
        self.hlo_artifact_collection = None
        self.neff_artifact_collection = None

        # these are determined later through the trace function
        self.num_params = None
        self.num_user_inputs = None  # accounts for excluded inputs
        self.num_states = None
        self.num_weights = None


class JITWrapper(torch.nn.Module):
    """
    Makes a python object like Flattener and Packer JIT traceable.
    """

    def __init__(self, func, is_flattener):
        super().__init__()
        self.func = func
        self.is_flattener = is_flattener

    def forward(self, inputs: List[torch.Tensor]):
        # flattener expects a tuple while packer expects a list
        if self.is_flattener:
            return self.func(tuple(inputs))
        else:
            return self.func(inputs)


class ModelBuilder:
    def __init__(
            self,
            router,
            tp_degree,
            checkpoint_loader,
            start_rank_id=0,
            pp_degree=1,
            ep_degree=1,
            local_ranks_size=None,
            world_size=None,
            compiler_workdir=None,
            master_proc_env_vars=None,
            debug=False,
            num_cores_per_group=1,
            init_custom_process_group_fn=None,
            logical_nc_config=1,
            weights_to_skip_layout_optimization=set()
    ):
        if not torch_neuronx.__version__.startswith("2"):
            raise AssertionError(
                f"ModelBuilder requires torch-neuronx>=2.* but found torch-neuronx=={torch_neuronx.__version__}."
            )

        if world_size is None:
            world_size = tp_degree * pp_degree * ep_degree
        elif local_ranks_size is None:
            raise ValueError(
                "`world_size` is specified, but `local_ranks_size` is None. Cannot determine local_ranks_size.")

        # only active if world size was none
        if local_ranks_size is None:
            local_ranks_size = world_size

        self.router = router
        # TODO: These parameters will be wrapped into NeuronDistributedConfig
        self.tp_degree = tp_degree
        self.pp_degree = pp_degree
        self.ep_degree = ep_degree
        self.world_size = world_size
        self.start_rank_id = start_rank_id
        self.local_ranks_size = local_ranks_size
        self.checkpoint_loader = checkpoint_loader
        self.compiler_workdir = compiler_workdir if compiler_workdir else "/tmp/nxd_model/"
        self.model_collection: Dict[str, ModelContainer] = {}
        self.master_proc_env_vars: Optional[Dict[str, str]] = master_proc_env_vars
        self.debug = debug
        self.num_cores_per_group = num_cores_per_group
        self.init_custom_process_group_fn = init_custom_process_group_fn
        self.logical_nc_config = logical_nc_config
        self.weights_to_skip_layout_optimization = weights_to_skip_layout_optimization

    def add(
            self,
            key: str,
            model_instance: BaseModelInstance,
            example_inputs: ModelInputType,
            compiler_args: Optional[Union[str, List[str]]] = None,
            bucket_config: Optional[BucketModelConfig] = None,
            priority_model_idx: Optional[int] = None,
    ) -> "ModelBuilder":
        """
        Adds a model to the model collection to be traced.
        """
        if compiler_args is None:
            compiler_args = "--enable-saturate-infinity --auto-cast=none --model-type=transformer -O1"

        # This does not validate if the HLOs are same across all ranks.
        # _validate_traceable(model_instance.module, self.tp_degree, force_custom_init_on_device=True)

        warnings.warn(
            "'bucket_config' will be deprecated in a future release."
            "Bucket routing will be automatically determined by inputs.",
            category=DeprecationWarning
        )
        if bucket_config:
            bucket_config.store_example_inputs(example_inputs)

        self.model_collection[key] = ModelContainer(
            model_instance, example_inputs, compiler_args, bucket_config, priority_model_idx
        )
        return self

    def trace(self, initialize_model_weights=True, dry_run=False):
        """
        Trace and compile a NxD model into a NEFF that can be excuted on neuron
        devices.

        Currently we need to put this function into another process to
        explictly release runtime at the end, so that it can clean memory
        garbage on devices.
        """
        start_time = time.time()
        prev_sharing_strategy = torch.multiprocessing.get_sharing_strategy()
        torch.multiprocessing.set_sharing_strategy("file_system")

        if self.master_proc_env_vars:
            for env_var, val in self.master_proc_env_vars.items():
                os.environ[env_var] = val

        # Clean compiler working dir
        if os.path.exists(self.compiler_workdir):
            shutil.rmtree(self.compiler_workdir)

        weight_names_to_skip = set(self.weights_to_skip_layout_optimization)
        num_hlos = 0
        logger.info(f"Generating HLOs for the following models: {list(self.model_collection.keys())}")
        trace_start_time = time.time()
        with mock_distributed(world_size=self.world_size):
            backend = 'xla'
            torch.distributed.init_process_group(backend, rank=0, world_size=self.world_size)
            parallel_state.initialize_model_parallel(self.tp_degree, self.pp_degree, self.ep_degree,
                                                     skip_collective_init=True, lnc_size=self.logical_nc_config)
            parallel_state.set_aot_mode(True)

            flash_decoding_enabled = self.num_cores_per_group > 1
            if flash_decoding_enabled:
                logger.info(f"Init kv group in model builder with multiplier: {self.num_cores_per_group}")
                parallel_state.initialize_kv_group(self.num_cores_per_group, sequential_ranks_in_group=True)

            if self.init_custom_process_group_fn:
                self.init_custom_process_group_fn()

            for key in self.model_collection:
                model_artifacts = self.model_collection[key]
                bucket_degree = len(model_artifacts.example_inputs)
                num_hlos += bucket_degree

                logger.info(f"Generating {bucket_degree} hlos for key: {key}")
                hlo_artifact_collection = self._generate_hlo(key)
                model_artifacts.hlo_artifact_collection = hlo_artifact_collection
                hm = hlo_artifact_collection[0].hlo_module
                id_to_computation = {cpt.id: cpt for cpt in hm.computations}
                entry_computation = id_to_computation[hm.entry_computation_id]
                model_artifacts.num_params = len([i for i in entry_computation.instructions if i.opcode == "parameter"])
                idx_to_weight_name = {idx: weight_name for weight_name, idx in hlo_artifact_collection[0].weight_name_to_idx.items()}

                # Check if kernel weights should be marked as transposable or not
                if not self.transpose_kernel_weights():
                    weight_names_to_skip = hlo_utils.get_nki_kernel_weight_names(
                        entry_computation, idx_to_weight_name, weight_names_to_skip, id_to_computation
                    )
                else:
                    logger.info("TRANSPOSE_KERNEL_WEIGHTS is set")

            # clear up the parallel state and distributed group for next batch of tracing
            parallel_state.set_aot_mode(False)
            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()

        logger.info(f"Generated all HLOs in {time.time() - trace_start_time} seconds")

        self._mark_weight_in_priority_hlo(weight_names_to_skip)

        def submit_compilation_job_with_cache(key, bucket_rank, **kwargs):
            neuron_xla_compile_kwargs = {
                "module_bytes": kwargs["module_bytes"],
                "compiler_flags": kwargs["compiler_flags"],
                "input_format": "hlo",
                "platform_target": get_platform_target(kwargs["compiler_flags"]),
                "cache_key": kwargs["cache_key"],
                "retry_failed_compilation": False,
                "lazy": True,
                "use_cache": True,
                "cache_dir": None,
                "work_dir": kwargs["work_dir"],
                "create_subdir": False,
            }
            try:
                compile_result = neuron_xla_compile(**neuron_xla_compile_kwargs)
            except TypeError:
                # Backward compatibility with earlier versions that lack create_subdir arg.
                del neuron_xla_compile_kwargs["create_subdir"]
                compile_result = neuron_xla_compile(**neuron_xla_compile_kwargs)

            return key, bucket_rank, compile_result

        for key, model_artifacts in self.model_collection.items():
            # init placeholder for all hlo
            model_artifacts.neff_artifact_collection = [None] * len(model_artifacts.hlo_artifact_collection)

            if model_artifacts.priority_model_idx is not None:
                priority_model_start_time = time.time()
                logger.info("Starting compilation for the priority HLO")
                bucket_rank = model_artifacts.priority_model_idx
                hlo_artifacts = model_artifacts.hlo_artifact_collection[bucket_rank]
                logger.info(f"'{key}' is the priority model with bucket rank {bucket_rank}")
                hlo_module = hlo_artifacts.hlo_module
                module_bytes = hlo_module.SerializeToString()
                output_dir = os.path.join(self.compiler_workdir, key, f"_tp0_bk{bucket_rank}")
                self._create_output_dir(output_dir)

                # TODO: improve these compiler flags if possible
                compiler_args = model_artifacts.compiler_args
                if '--logfile' not in compiler_args:
                    compiler_args += f" --logfile={Path(os.path.join(output_dir, LOG_FILE_DEFAULT_NAME)).absolute()}"
                compiler_args += " --enable-internal-neff-wrapper"
                platform_target = get_platform_target(compiler_args)
                module_hash = get_hash_module(hlo_module, compiler_args)
                if neuron_xla_wlo_compile:
                    neff_bytes, wrapped_neff_bytes = neuron_xla_wlo_compile(module_bytes, compiler_args,
                                                                            input_format="hlo",
                                                                            platform_target=platform_target,
                                                                            cache_key=module_hash,
                                                                            retry_failed_compilation=False,
                                                                            lazy=True,
                                                                            use_cache=True,
                                                                            cache_dir=None,
                                                                            work_dir=Path(output_dir).absolute(),
                                                                            create_subdir=False)
                    self._create_output_dir(os.path.join(output_dir, "model"))
                    hlo_utils.write_hlo(os.path.join(output_dir, "model", GRAPH_HLO_FILE), hlo_module)
                    neff_artifacts = self.write_neff_to_file(neff_bytes, os.path.join(output_dir, NEFF_FILE))
                    if wrapped_neff_bytes:
                        # This file is only generated when the weights need to be optimized
                        wrapped_neff_path = os.path.join(output_dir, WRAPPED_NEFF_FILE)
                        with open(wrapped_neff_path, 'wb') as f:
                            f.write(wrapped_neff_bytes)
                else:
                    logger.debug("Falling back to generate_neff() as libneuronxla is outdated")
                    neff_artifacts = torch_neuronx.xla_impl.trace.generate_neff(
                        hlo_artifacts,
                        os.path.join(self.compiler_workdir, key, f"_tp0_bk{bucket_rank}"),
                        compiler_args,
                        False,
                    )

                # The neff is still valid for this SPMD model
                self.model_collection[key].neff_artifact_collection[bucket_rank] = neff_artifacts
                logger.info(f"Done compilation for the priority HLO in {time.time() - priority_model_start_time} seconds")

        self._add_layout_optimization_to_remaining_hlo()

        if dry_run:
            logger.info(f"Saving HLOs and commands to {self.compiler_workdir} for dry run")
            for key, model_artifacts in self.model_collection.items():
                for bucket_rank, hlo_artifacts in enumerate(model_artifacts.hlo_artifact_collection):
                    if bucket_rank == model_artifacts.priority_model_idx:
                        # Priority model HLO is already saved.
                        continue

                    compiler_workdir = os.path.join(self.compiler_workdir, key, f"_tp0_bk{bucket_rank}")
                    compiler_target = torch_neuronx.xla_impl.trace.setup_compiler_dirs(
                        hlo_artifacts.hlo_module,
                        compiler_workdir,
                        hlo_artifacts.constant_parameter_tensors,
                        False,
                    )

                    # Write the command that produces the NEFF
                    command = torch_neuronx.xla_impl.trace.get_compile_command(compiler_target, compiler_workdir, compiler_args)
                    torch_neuronx.xla_impl.trace.save_compile_command(command, compiler_workdir)

            # Prepare weight layout transformation model
            self._prepare_weight_layout_transform_model(dry_run=True)

            logger.info(f"Saved HLOs and commands to {self.compiler_workdir} for dry run in {time.time() - start_time} seconds")
            return None

        logger.info("Starting compilation for all HLOs")
        compile_start_time = time.time()
        executor = concurrent.futures.ThreadPoolExecutor()
        jobs = []
        for key, model_artifacts in self.model_collection.items():
            for bucket_rank, hlo_artifacts in enumerate(model_artifacts.hlo_artifact_collection):
                if bucket_rank == model_artifacts.priority_model_idx:
                    # no need to compile the priority model again
                    continue
                output_dir = os.path.join(self.compiler_workdir, key, f"_tp0_bk{bucket_rank}")
                self._create_output_dir(output_dir)

                compiler_args = model_artifacts.compiler_args
                if '--logfile' not in compiler_args:
                    compiler_args += f" --logfile={Path(output_dir, LOG_FILE_DEFAULT_NAME).absolute()}"

                hlo_module = hlo_artifacts.hlo_module
                module_bytes = hlo_module.SerializeToString()
                module_hash = get_hash_module(hlo_module, compiler_args)
                self._create_output_dir(os.path.join(output_dir, "model"))
                hlo_utils.write_hlo(os.path.join(output_dir, "model", GRAPH_HLO_FILE), hlo_module)

                jobs.append(
                    executor.submit(
                        submit_compilation_job_with_cache,
                        key,
                        bucket_rank,
                        module_bytes=module_bytes,
                        compiler_flags=compiler_args,
                        cache_key=module_hash,
                        work_dir=Path(output_dir).absolute(),
                    )
                )

        for future in concurrent.futures.as_completed(jobs):
            key, bucket_rank, neff_bytes = future.result()
            output_dir = os.path.join(self.compiler_workdir, key, f"_tp0_bk{bucket_rank}")
            neff_artifacts = self.write_neff_to_file(neff_bytes, os.path.join(output_dir, NEFF_FILE))
            self.model_collection[key].neff_artifact_collection[bucket_rank] = neff_artifacts

        # Save metaneff
        for key, model_artifacts in self.model_collection.items():
            for bucket_rank, hlo_artifacts in enumerate(model_artifacts.hlo_artifact_collection):
                path = os.path.join(self.compiler_workdir, key, f"_tp0_bk{bucket_rank}", "metaneff.pb")
                with open(path, "wb") as f:
                    f.write(hlo_artifacts.metaneff)

        logger.info(f"Finished Compilation for all HLOs in {time.time() - compile_start_time} seconds")

        nxd_model_executor = self.build_nxd_model()

        if initialize_model_weights:
            if self.local_ranks_size < self.world_size:
                raise RuntimeError("initialize_model_weights argument must be False when tracing for multi-node")

            self.shard_checkpoint(self.compiler_workdir)

            weights = []
            for rank in range(self.start_rank_id, self.start_rank_id + self.local_ranks_size):
                ckpt = load_file(os.path.join(self.compiler_workdir, f"tp{rank}_sharded_checkpoint.safetensors"))
                weights.append(ckpt)

            start_rank_tensor = torch.tensor([self.start_rank_id], dtype=torch.int32, device="cpu")
            nxd_model_executor.nxd_model.initialize(weights, start_rank_tensor)
            logger.info("NxD Model Initialized")

        torch.multiprocessing.set_sharing_strategy(prev_sharing_strategy)
        if initialize_model_weights:
            nxd_model_executor.nxd_model.initialize_with_saved_weights(torch.tensor(0))

        logger.info(f"Finished building model in {time.time() - start_time} seconds")
        return nxd_model_executor

    def shard_checkpoint(self, serialize_path=None):
        if serialize_path is not None and not os.path.exists(serialize_path):
            os.makedirs(serialize_path)

        source_model_key = list(self.model_collection.keys())[0]
        model_container = self.model_collection[source_model_key]
        logger.info(
            f"Sharding Weights for ranks: {self.start_rank_id}...{self.start_rank_id + self.local_ranks_size - 1}")
        start_time_shard = time.monotonic()

        sharded_checkpoints = []
        with mock_distributed(world_size=self.world_size), init_on_device(torch.device("meta"),
                                                                          force_custom_init_on_device=True):
            torch.distributed.init_process_group(backend="xla", rank=0, world_size=self.world_size)
            parallel_state.initialize_model_parallel(self.tp_degree,
                                                     self.pp_degree,
                                                     self.ep_degree,
                                                     skip_collective_init=True,
                                                     lnc_size=self.logical_nc_config)
            if self.init_custom_process_group_fn:
                self.init_custom_process_group_fn()

            model_container.model_instance.load_module()
            func_kwargs = (
                {}
                if model_container.bucket_config is None
                else model_container.bucket_config.get_func_kwargs_for_bucket_rank(0)
            )
            if "bucket_rank" in func_kwargs:
                func_kwargs.pop("bucket_rank")  # to avoid multiple definition of bucket_rank
            model, io_aliases = model_container.model_instance.get(0, **func_kwargs)
            checkpoint = self.checkpoint_loader()
            preprocess_checkpoint(model, checkpoint)

            for rank in range(self.start_rank_id, self.start_rank_id + self.local_ranks_size):
                sharded_checkpoints.append(self.shard_weights_with_cache(rank, model, checkpoint, serialize_path))

            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()
        logger.info(f"Done Sharding weights in {time.monotonic() - start_time_shard}")
        return sharded_checkpoints

    def _generate_hlo(
            self,
            key,
    ):
        run_context = contextlib.nullcontext()

        if self.debug:
            # Adds metadata into the HLO
            torch_xla._XLAC._set_ir_debug(True)
            os.environ["XLA_IR_DEBUG"] = "1"
            os.environ["XLA_HLO_DEBUG"] = "1"
            # add_var_names=False avoids traversing the complete HLO graph,
            # instead only minimal information to metadata.
            # For e.g., op_name="NeuronLlamaForCausalLM[.1]/ModelBuilder[.1]/NeuronFusedSpecModel[.1]/
            #                       NeuronLlamaModel[.2]/ParallelEmbedding[.2]/aten__index_select"
            run_context = hlo_debug(add_var_names=False)

        with run_context:
            model_input_container = self.model_collection[key]
            logger.info(f"Started loading module {key}")
            start_time = time.time()
            model_input_container.model_instance.load_module()
            logger.info(f"Finished loading module {key} in {time.time() - start_time} seconds")
            example_input_collection = model_input_container.example_inputs
            bucket_config = model_input_container.bucket_config
            bucket_degree = len(example_input_collection)

            hlo_artifact_collection = []

            # Memory optimization: Throw away weights if only one bucket is used. When using more
            # than one model, we must preserve the weights or the aliasing equality check will
            # fail. This logic should be removed when automatic aliasing is implemented. Automatic
            # aliasing will derive aliases from the model forward call instead of explicitly
            # requiring a static dictionary. This allows the backing allocation to change between
            # forward calls.
            preserve_weights = False
            if bucket_degree > 1:
                preserve_weights = True

            for bucket_rank in range(bucket_degree):
                example_inputs = example_input_collection[bucket_rank]
                func_kwargs = {} if bucket_config is None else bucket_config.get_func_kwargs_for_bucket_rank(
                    bucket_rank)
                if "bucket_rank" in func_kwargs:
                    func_kwargs.pop("bucket_rank")  # to avoid multiple definition of bucket_rank
                func, input_output_aliases = model_input_container.model_instance.get(bucket_rank, **func_kwargs)

                logger.info(f"generating HLO: {key}, input example shape = {example_inputs[0].shape}")
                start_time = time.time()

                hlo_artifacts = torch_neuronx.xla_impl.trace.generate_hlo(
                    func, example_inputs, input_output_aliases,
                    inline_weights_to_neff=False,
                    return_weights=False,
                    output_aliased_tensor=False,
                    cpu_backend=True,
                    preserve_parameters=preserve_weights,
                )
                hlo_artifacts.metaneff = hlo_artifacts.metaneff.SerializeToString()
                hlo_artifact_collection.append(hlo_artifacts)

                logger.info(
                    f"Finished generating HLO for {key} in {time.time() - start_time} seconds, "
                    f"input example shape = {example_inputs[0].shape}"
                )

        return hlo_artifact_collection

    def shard_weights(self, rank, model_container: ModelContainer, serialize_path: Optional[str] = None) -> None:
        checkpoint = self.checkpoint_loader()
        with mock_distributed(world_size=self.world_size), init_on_device(torch.device("meta"),
                                                                          force_custom_init_on_device=True):
            torch.distributed.init_process_group(backend="xla", rank=0, world_size=self.world_size)
            parallel_state.initialize_model_parallel(self.tp_degree,
                                                     self.pp_degree,
                                                     self.ep_degree,
                                                     skip_collective_init=True,
                                                     lnc_size=self.logical_nc_config)
            if self.init_custom_process_group_fn:
                self.init_custom_process_group_fn()

            model_container.model_instance.load_module()
            func_kwargs = (
                {}
                if model_container.bucket_config is None
                else model_container.bucket_config.get_func_kwargs_for_bucket_rank(0)
            )
            if "bucket_rank" in func_kwargs:
                func_kwargs.pop("bucket_rank")  # to avoid multiple definition of bucket_rank
            model, io_aliases = model_container.model_instance.get(0, **func_kwargs)

            get_sharded_checkpoint(checkpoint, model, rank, self.tp_degree)

            if serialize_path is not None:
                save_file({ k: v.contiguous() for k, v in checkpoint.items() }, os.path.join(serialize_path, f"tp{rank}_sharded_checkpoint.safetensors"))

            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()
        return checkpoint

    def shard_weights_with_cache(self, rank, model, checkpoint, serialize_path: Optional[str] = None) -> None:
        sharded_checkpoint = checkpoint.copy()
        get_sharded_checkpoint(sharded_checkpoint, model, rank, self.tp_degree, is_cached=True)

        if serialize_path is not None:
            save_file({ k: v.contiguous() for k, v in sharded_checkpoint.items() }, os.path.join(serialize_path, f"tp{rank}_sharded_checkpoint.safetensors"))
        return sharded_checkpoint

    def build_state_initializer(self):
        shapes = {}
        dtypes = {}

        # Take any metaneff
        source_model_key = list(self.model_collection.keys())[0]
        metaneff = metaneff_pb2.MetaNeff()
        metaneff_str = self.model_collection[source_model_key].hlo_artifact_collection[0].metaneff
        metaneff.ParseFromString(metaneff_str)
        for tensor in metaneff.input_tensors:
            if tensor.type is metaneff_pb2.MetaTensor.Type.INPUT_STATE:
                # proto keys are bytes not strings, and casting as a string causes it to be "b'key'"
                checkpoint_key = str(tensor.checkpoint_key).replace("b'", "").replace("'", "")
                shapes[checkpoint_key] = list(tensor.shape)
                dtypes[checkpoint_key] = get_torch_dtype(tensor.data_type)
        if len(shapes):
            return torch.jit.script(
                StateInitializer(shapes=shapes, dtypes=dtypes, local_ranks_size=self.local_ranks_size))
        else:
            return None

    def build_flattener_map(self):
        flattener_map = []
        for key, model_container in self.model_collection.items():
            flattener = JITWrapper(func=model_container.hlo_artifact_collection[0].flattener, is_flattener=True)
            example_inputs = model_container.example_inputs
            flattener_script = torch.jit.trace(flattener, ([*example_inputs[0]],), strict=False)
            flattener_map.append((key, flattener_script))
        return torch.nn.ModuleDict(flattener_map)

    def build_packer(self, packer):
        # Take any metaneff
        source_model_key = list(self.model_collection.keys())[0]
        metaneff = metaneff_pb2.MetaNeff()
        metaneff_str = self.model_collection[source_model_key].hlo_artifact_collection[0].metaneff
        metaneff.ParseFromString(metaneff_str)

        # create example outputs from metaneff
        example_outputs = []
        for i, meta_tensor in enumerate(metaneff.output_tensors):
            if i not in metaneff.output_aliases_to:
                example_outputs.append(
                    torch.zeros(list(meta_tensor.shape), dtype=get_torch_dtype(meta_tensor.data_type))
                )

        # return jit traced packer
        jit_wrapped_packer = JITWrapper(packer, False)
        return torch.jit.trace(jit_wrapped_packer, (example_outputs,), strict=False)

    def build_nxd_model(self):
        model_map_input = []
        for key, model_container in self.model_collection.items():

            models = [
                (self._read_neff_from_path(neff_artifacts.neff_filename), hlo_artifacts.metaneff)
                for hlo_artifacts, neff_artifacts in zip(
                    model_container.hlo_artifact_collection, model_container.neff_artifact_collection
                )
            ]

            buckets = [torch.classes.neuron.SPMDModel(neff, metaneff, self.local_ranks_size, self.world_size) for
                       neff, metaneff in models]

            spmd_bucket_model_executor = SPMDBucketModelScript(compiled_models=buckets)
            with torch_neuronx.contexts.disable_nrt_load():
                spmd_bucket_model_executor = torch.jit.script(spmd_bucket_model_executor)
            model_map_input.append((key, spmd_bucket_model_executor))

        state_initializer = self.build_state_initializer()
        model_map = torch.nn.ModuleDict(model_map_input)
        flattener_map = self.build_flattener_map()

        input_shape_map = {}
        # use to jit trace NxDModelExecutor
        example_inputs = None
        for key, model_container in self.model_collection.items():
            # example_inputs is of type List[Tuple[Tensor, Tensor, ...]]
            example_inputs = model_container.example_inputs
            for i, example_input in enumerate(example_inputs):
                # torch.Size type is not a concept in a jit model, it's just List[int]
                input_shape_map[str([list(tensor.shape) for tensor in example_input])] = (key, i)

        packer = next(iter(self.model_collection.values())).hlo_artifact_collection[0].packer
        traced_packer = self.build_packer(packer)

        # Get weight layout transformation model
        wlt_model = self._prepare_weight_layout_transform_model()

        with torch_neuronx.contexts.disable_nrt_load():
            nxd_model = NxDModel(
                models=model_map,
                flattener_map=flattener_map,
                input_shape_map=input_shape_map,
                packer=traced_packer,
                state_initializer=state_initializer,
                weight_loader=wlt_model,
                start_rank_id=self.start_rank_id
            )
            nxd_model = torch.jit.script(nxd_model)

            # mock model as initialized so jit trace doesn't fail
            nxd_model.mock_initialization(True)
            nxd_model_executor = torch.jit.trace(NxDModelExecutor(nxd_model), example_inputs[0], strict=False)
            nxd_model_executor.nxd_model.mock_initialization(False)

        return nxd_model_executor

    @staticmethod
    def _create_output_dir(output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    def _read_neff_from_path(self, neff_path: str):
        with open(neff_path, "rb") as f:
            return f.read()

    @staticmethod
    def write_neff_to_file(neff_bytes, neff_path) -> NeffArtifacts:
        with open(neff_path, 'wb') as f:
            f.write(neff_bytes)
        return NeffArtifacts(neff_path)

    def _get_priority_hlo_artifact(self) -> HloArtifacts:
        for model_artifacts in self.model_collection.values():
            if model_artifacts.priority_model_idx is not None:
                return model_artifacts.hlo_artifact_collection[model_artifacts.priority_model_idx]
        return None

    def _should_optimize_layout(self):
        return self._get_priority_hlo_artifact() is not None

    def _mark_weight_in_priority_hlo(self, weight_names_to_skip: set):
        """
        Mark weights in the priority HLO, so compiler will suggest optimal
        layout for the weights.
        """
        if not self._should_optimize_layout():
            logger.info("Can't find a priority model, skip marking weights")
            return
        priority_hlo_artifacts = self._get_priority_hlo_artifact()

        hlo_utils.add_weight_idx_attr_to_hlo(
            hlo=priority_hlo_artifacts.hlo_module,
            weight_name_to_idx=priority_hlo_artifacts.weight_name_to_idx,
            weight_names_to_skip=weight_names_to_skip
        )

    def _get_hlo_stub(self):
        """
        Read the HLO stub if it is there, otherwise return None
        """
        neff_artifacts = None
        for model_artifacts in self.model_collection.values():
            if model_artifacts.priority_model_idx is not None:
                neff_artifacts = model_artifacts.neff_artifact_collection[model_artifacts.priority_model_idx]
        assert neff_artifacts.neff_filename is not None, "Can't find the path for the NEFF from the priority model"
        hlo_stub_filepath = neff_artifacts.neff_filename.replace(NEFF_FILE, WRAPPED_NEFF_FILE)

        if os.path.exists(hlo_stub_filepath):
            return hlo_utils.read_hlo(hlo_stub_filepath)
        else:
            return None

    def _add_layout_optimization_to_remaining_hlo(self):
        """
        Apply the layout transformation suggestion from the priority HLO to
        other HLOs, so they all can benefit.

        This is a no-op if there is no suggestion on weight layout.
        """
        if not self._should_optimize_layout():
            logger.info("Can't find a priority model, skip optimizing weight layout for other HLOs")
            return

        hlo_stub = self._get_hlo_stub()
        if hlo_stub is None:
            logger.info("No changes on weight layout, skip updating weight layout for other HLOs")
            return

        start_time = time.time()
        priority_hlo_artifacts = self._get_priority_hlo_artifact()
        weight_name_to_transform_cpt = hlo_utils.get_layout_transform_map(
            hlo_stub=hlo_stub, weight_name_to_idx=priority_hlo_artifacts.weight_name_to_idx
        )

        for model_collection_key, model_artifacts in self.model_collection.items():
            for bucket_rank, hlo_artifacts in enumerate(model_artifacts.hlo_artifact_collection):
                if bucket_rank == model_artifacts.priority_model_idx:
                    continue
                hlo_utils.append_layout_computation_to_hlo(hlo_artifacts, weight_name_to_transform_cpt)

                original_hlo_file_name = f"{model_collection_key}_{bucket_rank}.hlo"
                optimized_hlo_file_name = f"{model_collection_key}_{bucket_rank}_optimized.hlo"

                # Write the current HLO into a temporary file to convert it into optimal layout
                hlo_utils.write_hlo(original_hlo_file_name, hlo_artifacts.hlo_module)

                # Convert the inputs in HLO to optimal shape
                hlo_utils.convert_inputs_to_optimal_shape(original_hlo_file_name, optimized_hlo_file_name)

                # Overwrite current HLO with new optimized layout version
                logger.info("Updating the hlo module with optimized layout")
                hlo_artifacts.hlo_module = hlo_utils.cleanup_after_layout_transformation(
                    hlo_utils.read_hlo(optimized_hlo_file_name), hlo_artifacts.hlo_module.frontend_attributes.map
                )

                # Cleanup intermediate files
                os.remove(original_hlo_file_name)
                os.remove(optimized_hlo_file_name)

        logger.info(f"Done optimizing weight layout for all HLOs in {time.time() - start_time} seconds")

    def _prepare_weight_layout_transform_model(self, dry_run=False):
        """
        Generate a NEFF for weight layout transformation, which will be run on
        device before actual inference.

        This will return None if there is no changes on weight layout.
        """
        if not self._should_optimize_layout():
            logger.info("Can't find a priority model, falling back to the existing weight layout")
            return

        hlo_stub = self._get_hlo_stub()
        if hlo_stub is None:
            logger.info("No changes on weight layout, falling back to the existing weight layout")
            return

        # Clear existing dir
        layout_dir = os.path.join(self.compiler_workdir, "layout_opt")
        if os.path.exists(layout_dir):
            shutil.rmtree(layout_dir)
        os.makedirs(layout_dir)

        # Prepare HLO
        weight_name_to_idx = self._get_priority_hlo_artifact().weight_name_to_idx
        wlt_hlo = hlo_utils.extract_weight_layout_transform_hlo(
            hlo_stub=hlo_stub,
            weight_name_to_idx=weight_name_to_idx,
        )

        compiler_args = (
            "--model-type=transformer -O1" +
            f" --lnc={self.logical_nc_config}" +
            " --internal-hlo2tensorizer-options=--experimental-unsafe-fp8e4m3fn-as-fp8e4m3" +
            f" --logfile={os.path.join(layout_dir, 'log-neuron-cc.txt')}"
        )

        if dry_run:
            compiler_target = torch_neuronx.xla_impl.trace.setup_compiler_dirs(
                wlt_hlo,
                layout_dir,
                None,
                False,
            )

            # Write the command that produces the NEFF
            command = torch_neuronx.xla_impl.trace.get_compile_command(compiler_target, layout_dir, compiler_args)
            torch_neuronx.xla_impl.trace.save_compile_command(command, layout_dir)
            return None

        metaneff = hlo_utils.prepare_metaneff_for_wlt_hlo(
            wlt_hlo=wlt_hlo,
            weight_name_to_idx=weight_name_to_idx,
        )
        metaneff_str = metaneff.SerializeToString()
        metaneff_path = os.path.join(layout_dir, "metaneff")
        with open(metaneff_path, "wb") as f:
            f.write(metaneff_str)

        wlt_hlo_artifact = HloArtifacts(
            hlo_module=wlt_hlo,
            flattener=None,
            packer=None,
            metaneff=metaneff,
            weights=None,
            constant_parameter_tensors=None,
            weight_name_to_idx=weight_name_to_idx,
        )

        # Generate NEFF
        # Why do we add unsafe cast option?
        # Its a HILO pass so should not affect any latency or layout. Also makes it work for
        # fp8 models
        wlt_neff_artifact = torch_neuronx.xla_impl.trace.generate_neff(
            wlt_hlo_artifact,
            compiler_workdir=layout_dir,
            compiler_args=compiler_args,
            inline_weights_to_neff=False,
        )
        wlt_neff = self._read_neff_from_path(wlt_neff_artifact.neff_filename)

        # Experimental options to transform weight layout
        # TODO:
        # 1. update the metaneff on input weight shape for model inference neff
        #    and update the weight shape in runtime after weight layout transformation
        # 2. make this an argument in model builder once the feature is stable
        if hlo_utils.NXD_LAYOUT_TRANSFORMATION_OPTIONS in os.environ:
            overriden_option = os.environ[hlo_utils.NXD_LAYOUT_TRANSFORMATION_OPTIONS]
            logger.info(f"Overriden option for weight layout: {overriden_option}")

            if overriden_option == hlo_utils.NXD_LAYOUT_ON_CPU_AND_SERIALIZE:
                layout_dir = wlt_neff_artifact.neff_filename.strip(f"/{NEFF_FILE}")
                hlo_path = os.path.join(layout_dir, f"model/{GRAPH_HLO_FILE}")
                self.args_for_cpu_transformation = [
                    hlo_path,
                    metaneff_path,
                    self.start_rank_id,
                    self.local_ranks_size,
                ]
            elif overriden_option == hlo_utils.NXD_LAYOUT_ON_DEVICE_AND_SERIALIZE:
                self.args_for_on_device_and_serialization = [
                    metaneff_path,
                    self.start_rank_id,
                    self.local_ranks_size,
                    wlt_neff_artifact.neff_filename,
                ]
            else:
                raise ValueError(f"Unknown layout option: {overriden_option}")

            logger.info("Skipping the default layout flow to transform on load")
            return

        # Build the model on runtime
        wlt_model = torch.classes.neuron.LayoutTransformation(wlt_neff, metaneff_str, self.local_ranks_size)
        logger.info("Done preparing weight layout transformation")
        return wlt_model

    def transform_weight_layout_with_overriden_option(self, sharded_checkpoint_dir):
        """
        Transform the weight layout in an alternative option.

        TODO:
        1. pass NXD_LAYOUT_TRANSFORMATION_OPTIONS as an argument to this function
           once the feature is stable.
        """

        if hlo_utils.NXD_LAYOUT_TRANSFORMATION_OPTIONS not in os.environ:
            return

        overriden_option = os.environ[hlo_utils.NXD_LAYOUT_TRANSFORMATION_OPTIONS]

        if overriden_option == hlo_utils.NXD_LAYOUT_ON_CPU_AND_SERIALIZE:
            hlo_path, metaneff_path, start_rank_id, local_ranks_size = self.args_for_cpu_transformation

            hlo_utils.transform_weight_layout_on_cpu(
                hlo_filename=hlo_path,
                metaneff_filename=metaneff_path,
                start_rank_id=start_rank_id,
                local_ranks_size=local_ranks_size,
                sharded_checkpoint_dir=sharded_checkpoint_dir,
            )

        elif overriden_option == hlo_utils.NXD_LAYOUT_ON_DEVICE_AND_SERIALIZE:
            metaneff_path, start_rank_id, local_ranks_size, wlt_neff_path = self.args_for_on_device_and_serialization

            hlo_utils.transform_weight_layout_on_device_and_save_to_disk(
                metaneff_filename=metaneff_path,
                start_rank_id=start_rank_id,
                local_ranks_size=local_ranks_size,
                wlt_neff_path=wlt_neff_path,
                sharded_checkpoint_dir=sharded_checkpoint_dir,
            )

        else:
            raise ValueError(f"Unknown layout option: {overriden_option}")

    @staticmethod
    def transpose_kernel_weights() -> bool:
        """ Check if TRANSPOSE_KERNEL_WEIGHTS env variable is set. It defaults to 0 (i.e., False). """
        return (os.getenv("TRANSPOSE_KERNEL_WEIGHTS", '0') == '1')
