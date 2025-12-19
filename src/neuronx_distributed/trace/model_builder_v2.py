"""
ModelBuilderV2 -- Simplified version of ModelBuilder API that is more flexible and extensible.

This API is approaching stabilization.
"""
import time
import pathlib
import logging
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set

import torch
import torch_neuronx

from neuronx_distributed.trace.functions import (
    trace,
    compile,
    compile_wlo,
    compile_layout_transformer
)
import neuronx_distributed.trace.hlo_utils as hlo_utils
from neuronx_distributed.trace.model_builder_utils import (
    ModelBuilderConstants,
    TraceArtifacts,
    generate_key,
)
from neuronx_distributed.trace.nxd_model import NxDModel


logger = logging.getLogger("Neuron")


class ModelBuilder:
    """
    A class for tracing and compiling models for Neuron devices.

    This class provides functionality to trace and compile models for efficient
    execution on Neuron hardware. It supports SPMD (Single Program Multiple Data)
    tracing and compilation.
    """
    def __init__(
        self,
        model: Union[Callable, torch.nn.Module],
        weights_to_skip_layout_optimization: Optional[Set] = None
    ):
        """
        Initialize the ModelBuilder.

        Args:
            model: The PyTorch model to be traced and compiled.
            weights_to_skip_layout_optimization: A set of weight names to skip during layout optimization.

        Raises:
            AssertionError: If the torch-neuronx version is not compatible.
        """
        if not torch_neuronx.__version__.startswith("2"):
            raise AssertionError(
                f"ModelBuilder requires torch-neuronx>=2.* but found torch-neuronx=={torch_neuronx.__version__}."
            )

        self.model = model
        self.weights_to_skip_layout_optimization = weights_to_skip_layout_optimization

        self.trace_artifacts_collection: Dict[str, TraceArtifacts] = {}
        self.world_size = ModelBuilderConstants.DEFAULT_WORLD_SIZE

        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()

    def trace(
        self,
        args: Union[None, torch.Tensor, Tuple[torch.Tensor, ...]] = None,
        kwargs: Optional[Dict[str, torch.Tensor]] = None,
        tag: Optional[str] = None,
        spmd: bool = True,
    ):
        """
        Traces a model with given example inputs and stores the resulting trace artifacts.

        Args:
            args: 
                The example inputs to be used for tracing in the form of positional arguments.
            kwargs: 
                The example inputs to be used for tracing in the form of keyword arguments.
            tag: A unique identifier for this trace. If None, a default tag will be generated.
            spmd: Whether to use SPMD for tracing. Currently only SPMD=True
                is supported. Defaults to True.

        Returns:
            self: Returns the instance to allow method chaining.
        """
        trace_start_time = time.time()
        if tag is not None:
            logger.info(f"Started tracing {tag}")

        # Trace the model by leveraging trace() fundamental unit
        trace_artifacts = trace(
            model=self.model,
            args=args,
            kwargs=kwargs,
            spmd=spmd,
            preserve_parameters=True,
            weights_to_skip_layout_optimization=self.weights_to_skip_layout_optimization,
        )

        # Generate a tag if not provided
        tag = generate_key(trace_artifacts.hlo, tag)
        
        self.trace_artifacts_collection[tag] = trace_artifacts

        logger.info(f"Finished tracing {tag} in {time.time() - trace_start_time} seconds")

        return self


    def compile(
        self,
        priority_model_key: Optional[str] = None,
        compiler_workdir: Optional[Union[str, pathlib.Path]] = None,
        compiler_args: Optional[Union[str, Dict[str, str]]] = None,
        max_workers: Optional[int] = None,
    ) -> NxDModel:
        """
        Compiles the traced model using the Neuron compiler, generating
        a Neuron Executable File Format (NEFF) for each trace.

        Args:
            priority_model_key: Key of the model to prioritize during compilation.
                If provided, weight layout optimization will be suggested based on this model,
                and then it will be applied to all the other models.
            compiler_workdir: Path to store compiler artifacts.
            compiler_args: Either compiler flags for neuronx-cc as a string
                to use for all buckets, or a dictionary mapping bucket tags to their specific compiler flags. 
                When using a dictionary, all bucket tags must have corresponding compiler flags.
                If None, default compiler flags will be used from the compile()/compile_wlo() fundamental unit.
            max_workers: Maximum number of worker threads for parallel compilation.
                If None, uses the default value from ThreadPoolExecutor.

        Returns:
            neuronx_distributed.trace.nxd_model.NxDModel: Constructed NxDModel.

        Raises:
            ValueError: If no traces are available for compilation or if the priority_model_key
                is invalid.
        """
        if not self.trace_artifacts_collection:
            raise ValueError("No traces available for compilation. Call trace() first.")

        if priority_model_key and priority_model_key not in self.trace_artifacts_collection:
            raise ValueError(f"Invalid priority_model_key: {priority_model_key}")

        # Handle compiler args
        if isinstance(compiler_args, dict):
            # When dict is provided, ensure all buckets have compiler args
            missing_tags = set(self.trace_artifacts_collection.keys()) - set(compiler_args.keys())
            if missing_tags:
                raise ValueError(f"Missing compiler args for buckets: {missing_tags}")
        elif isinstance(compiler_args, str):
            # When string is provided, use same args for all buckets
            compiler_args = {tag: compiler_args for tag in self.trace_artifacts_collection.keys()}
        elif compiler_args is None:
            # When None is provided, let compile() fundamental unit handle default args
            compiler_args = None

        compilation_results: Dict[str, Any] = {}
        compile_start_time = time.time()
        logger.info("Starting compilation process")

        try:
            # Handle priority model compilation
            self._compile_priority_model(
                priority_model_key, 
                compiler_workdir,
                compiler_args,
                compilation_results
            )

            # Parallel compilation of remaining models
            self._compile_non_priority_models(
                priority_model_key,
                compiler_workdir,
                compiler_args,
                max_workers,
                compilation_results
            )

        except Exception as e:
            raise RuntimeError("Compilation process failed") from e

        logger.info(
            f"Finished compilation for all the models in "
            f"{time.time() - compile_start_time} seconds"
        )

        try:
            return self._build_nxd_model(compilation_results=compilation_results)
        except Exception as e:
            raise RuntimeError(f"Failed to build NxDModel: {str(e)}") from e


    def _compile_priority_model(
        self,
        priority_model_key: Optional[str],
        compiler_workdir: Optional[Union[str, pathlib.Path]],
        compiler_args: Optional[Dict[str, str]],
        compilation_results: Dict[str, Any]
    ) -> None:
        """Handles compilation of the priority model if specified."""
        if not priority_model_key:
            logger.info("Skipping weight layout optimization")
            return None

        priority_model_trace_artifacts = self.trace_artifacts_collection[priority_model_key]
        priority_model_compiler_args = compiler_args[priority_model_key] if compiler_args else None

        # Mark weights for WLO
        hlo_utils.mark_weights_for_wlo(
            trace_artifacts=priority_model_trace_artifacts,
            weights_to_skip_layout_optimization=priority_model_trace_artifacts.weight_names_to_skip,
        )

        # Compile priority model with WLO
        wlo_artifacts = compile_wlo(
            hlo_module=priority_model_trace_artifacts.hlo,
            metaneff=priority_model_trace_artifacts.metaneff,
            compiler_workdir=compiler_workdir,
            compiler_args=priority_model_compiler_args,
            key=priority_model_key
        )
        compilation_results[priority_model_key] = wlo_artifacts

        # Compile layout transformer
        layout_transformer_artifacts = compile_layout_transformer(
            wlo_artifacts=wlo_artifacts,
            priority_model_weight_name_to_idx=priority_model_trace_artifacts.weight_name_to_idx,
            compiler_workdir=compiler_workdir
        )
        compilation_results[ModelBuilderConstants.LAYOUT_TRANSFORMER_KEY] = layout_transformer_artifacts


    def _compile_non_priority_models(
        self,
        priority_model_key: Optional[str],
        compiler_workdir: Optional[Union[str, pathlib.Path]],
        compiler_args: Optional[Dict[str, str]],
        max_workers: Optional[int],
        compilation_results: Dict[str, Any]
    ) -> None:
        """Handles parallel compilation of remaining models."""
        models_to_compile = {
            k: v for k, v in self.trace_artifacts_collection.items()
            if k != priority_model_key
        }

        if not models_to_compile:
            return

        logger.info(f"Starting parallel compilation of {len(models_to_compile)} models")

        def compile_with_layout_transformation(trace_artifacts, compiler_workdir, compiler_args, key):
            # Apply layout transformation
            if priority_model_key and \
                priority_model_key in self.trace_artifacts_collection and \
                priority_model_key in compilation_results:
                priority_model_trace_artifacts = self.trace_artifacts_collection[priority_model_key]
                wlo_artifacts = compilation_results[priority_model_key]
                hlo_utils.apply_layout_transformation(
                    trace_artifacts=trace_artifacts,
                    priority_model_trace_artifacts=priority_model_trace_artifacts,
                    wlo_artifacts=wlo_artifacts,
                    key=key
                )
            
            # Compile
            return compile(
                trace_artifacts.hlo,
                trace_artifacts.metaneff,
                compiler_workdir,
                compiler_args,
                key,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {}

            for key, trace_artifacts in models_to_compile.items():
                future = executor.submit(
                    compile_with_layout_transformation,
                    trace_artifacts,
                    compiler_workdir,
                    compiler_args[key] if compiler_args else None,
                    key,
                )
                future_to_key[future] = key

            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                compilation_results[key] = future.result()


    def _build_nxd_model(
        self,
        compilation_results: Dict[str, Any]
    ) -> NxDModel:
        """Builds and configures the NxDModel."""
        nxd_model = NxDModel(
            world_size=self.world_size,
            layout_transformer=compilation_results.get(ModelBuilderConstants.LAYOUT_TRANSFORMER_KEY)
        )

        for key in self.trace_artifacts_collection.keys():
            nxd_model.add(
                key=key,
                trace_artifacts=self.trace_artifacts_collection[key],
                compilation_artifacts=compilation_results[key],
            )
        
        logger.info("NxD Model Built")

        return nxd_model