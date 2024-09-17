import concurrent.futures
import multiprocessing
import os
import shutil
import time
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch_neuronx
import torch_neuronx.xla_impl
import torch_neuronx.xla_impl.trace
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_neuronx import BucketModelConfig
from torch_neuronx.proto import metaneff_pb2
from torch_neuronx.xla_impl.trace import get_torch_dtype, HloArtifacts

from safetensors.torch import save_file, load_file

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.spmd import (
    NxDModel,
    NxDModelExecutor,
    SPMDBucketModel,
    SPMDBucketModelScript,
    default_bucket_kernel,
    StateInitializer)
from neuronx_distributed.trace.trace import _mock_parallel_state, get_sharded_checkpoint
from neuronx_distributed.utils.model_utils import init_on_device
import neuronx_distributed.trace.hlo_utils as hlo_utils

ModelInputType = List[Union[Tuple[Union[torch.Tensor, List[torch.Tensor]]], torch.Tensor]]
logger = logging.getLogger("Neuron")


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


class ModelContainer:
    def __init__(self, model_instance, example_inputs, compiler_args, bucket_config, priority_model_idx):
        self.model_instance: BaseModelInstance = model_instance
        self.example_inputs = example_inputs
        self.compiler_args = compiler_args
        self.bucket_config: BucketModelConfig = bucket_config
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
        if (self.is_flattener):
            return self.func(tuple(inputs))
        else:
            return self.func(inputs)

class ModelBuilder:
    def __init__(
        self,
        router,
        tp_degree,
        checkpoint_loader,
        compiler_workdir=None,
        master_proc_env_vars=None,
    ):
        if not torch_neuronx.__version__.startswith("2"):
            raise AssertionError(
                f"ModelBuilder requires torch-neuronx>=2.* but found torch-neuronx=={torch_neuronx.__version__}."
            )

        self.router = router
        self.tp_degree = tp_degree
        self.checkpoint_loader = checkpoint_loader
        self.compiler_workdir = compiler_workdir if compiler_workdir else "/tmp/nxd_model/"

        self.model_collection: Dict[str, ModelContainer] = {}
        self.master_proc_env_vars: Optional[Dict[str, str]] = master_proc_env_vars

    def add(
        self,
        key: str,
        model_instance: BaseModelInstance,
        example_inputs: ModelInputType,
        compiler_args: Union[str, List[str]] = None,
        bucket_config: BucketModelConfig = None,
        priority_model_idx: int = None,
    ) -> None:
        """
        Adds a model to the model collection to be traced.
        """
        if compiler_args is None:
            compiler_args = "--enable-saturate-infinity --auto-cast=none --model-type=transformer -O1"

        # This does not validate if the HLOs are same across all ranks.
        # _validate_traceable(model_instance.module, self.tp_degree, force_custom_init_on_device=True)

        if bucket_config:
            bucket_config.store_example_inputs(example_inputs)

        self.model_collection[key] = ModelContainer(
            model_instance, example_inputs, compiler_args, bucket_config, priority_model_idx
        )
        return self

    def trace(
        self,
        tp_degree=None,
        initialize_model_weights=True
    ):
        if tp_degree is None:
            tp_degree = self.tp_degree
        else:
            self.tp_degree = tp_degree

        ctx = multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        mp_q = manager.Queue()

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "2022"
        os.environ["NEURONCORE_NUM_DEVICES"] = str(tp_degree)  # for pjrt
        os.environ["WORLD_SIZE"] = str(tp_degree)
        prev_sharing_strategy = torch.multiprocessing.get_sharing_strategy()
        torch.multiprocessing.set_sharing_strategy("file_system")

        if self.master_proc_env_vars:
            for env_var, val in self.master_proc_env_vars.items():
                os.environ[env_var] = val

        # Clean compiler working dir
        if os.path.exists(self.compiler_workdir):
            shutil.rmtree(self.compiler_workdir)

        num_hlos = 0
        logger.info(f"Generating HLOs for the following models: {list(self.model_collection.keys())}")
        for key in self.model_collection:
            model_artifacts = self.model_collection[key]
            bucket_degree = 1 if not model_artifacts.bucket_config else model_artifacts.bucket_config.bucket_degree
            num_hlos += bucket_degree
            logger.info(f"Generating {bucket_degree} hlos for key: {key}")
            xmp.spawn(
                self._generate_hlo,
                args=(
                    key,
                    mp_q,
                ),
                start_method="spawn",
                nprocs=self.tp_degree,
            )

            hlo_artifact_collection = mp_q.get()
            model_artifacts.hlo_artifact_collection = hlo_artifact_collection
            hm = hlo_artifact_collection[0].hlo_module
            id_to_computation = {cpt.id: cpt for cpt in hm.computations}
            entry_computation = id_to_computation[hm.entry_computation_id]
            model_artifacts.num_params = len([i for i in entry_computation.instructions if i.opcode == "parameter"])

        self._mark_weight_in_priority_hlo()

        def submit_compilation_job(key, bucket_rank, args):
            return key, bucket_rank, torch_neuronx.xla_impl.trace.generate_neff(*args)

        logger.info("Started compilation for all HLOs")
        for key, model_artifacts in self.model_collection.items():
            # init placeholder for all hlo
            model_artifacts.neff_artifact_collection = [None] * len(model_artifacts.hlo_artifact_collection)

            if model_artifacts.priority_model_idx is not None:
                bucket_rank = model_artifacts.priority_model_idx
                hlo_artifacts = model_artifacts.hlo_artifact_collection[bucket_rank]

                neff_artifacts = torch_neuronx.xla_impl.trace.generate_neff(
                    hlo_artifacts,
                    os.path.join(self.compiler_workdir, key, f"_tp0_bk{bucket_rank}"),
                    # TODO: improve these compiler flags if possiable
                    model_artifacts.compiler_args + " --enable-internal-neff-wrapper",
                    False,
                )
                # The neff is still valid for this SPMD model
                self.model_collection[key].neff_artifact_collection[bucket_rank] = neff_artifacts
        logger.info("Done compilation for the priority HLO")

        self._add_layout_optimization_to_remaining_hlo()

        executor = concurrent.futures.ThreadPoolExecutor()
        jobs = []
        for key, model_artifacts in self.model_collection.items():
            for bucket_rank, hlo_artifacts in enumerate(model_artifacts.hlo_artifact_collection):
                if bucket_rank == model_artifacts.priority_model_idx:
                    # no need to compile the priority model again
                    continue
                jobs.append(
                    executor.submit(
                        submit_compilation_job,
                        key,
                        bucket_rank,
                        (
                            hlo_artifacts,
                            os.path.join(self.compiler_workdir, key, f"_tp0_bk{bucket_rank}"),
                            model_artifacts.compiler_args,
                            False,
                        ),
                    )
                )

        for future in concurrent.futures.as_completed(jobs):
            key, bucket_rank, neff_artifacts = future.result()
            self.model_collection[key].neff_artifact_collection[bucket_rank] = neff_artifacts

        # Save metaneff
        for key, model_artifacts in self.model_collection.items():
            for bucket_rank, hlo_artifacts in enumerate(model_artifacts.hlo_artifact_collection):
                path = os.path.join(self.compiler_workdir, key, f"_tp0_bk{bucket_rank}", "metaneff.pb")
                with open(path, 'wb') as f:
                    f.write(hlo_artifacts.metaneff)

        logger.info("Finished Compilation for all HLOs")

        logger.info("Finished Compilation for all HLOs")

        nxd_model_executor = self.build_nxd_model()

        if (initialize_model_weights):
            self.shard_checkpoint(self.compiler_workdir)

            weights = []
            for rank in range(self.tp_degree):
                ckpt = load_file(os.path.join(self.compiler_workdir, f"tp{rank}_sharded_checkpoint.safetensors"))
                weights.append(ckpt)

            nxd_model_executor.nxd_model.initialize(weights)
            logger.info("NxD Model Initialized")

        torch.multiprocessing.set_sharing_strategy(prev_sharing_strategy)

        return nxd_model_executor

    def shard_checkpoint(self, serialize_path):
        if not os.path.exists(serialize_path):
            os.makedirs(serialize_path)

        source_model_key = list(self.model_collection.keys())[0]
        logger.info("Sharding Weights")
        for rank in range(self.tp_degree):
            self.shard_weights(rank, self.model_collection[source_model_key], serialize_path)
        logger.info("Done Sharding weights")


    def _generate_hlo(
        self,
        rank,
        key,
        mp_q,
    ):
        os.environ["RANK"] = str(rank)
        torch.distributed.init_process_group("xla", init_method="pjrt://")
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=self.tp_degree)

        if rank == 0:
            model_input_container = self.model_collection[key]
            logger.info(f"Started loading module {key}")
            start_time = time.time()
            model_input_container.model_instance.load_module()
            logger.info(f"Finished loading module {key} in {time.time() - start_time} seconds")
            example_input_collection = model_input_container.example_inputs
            bucket_config = model_input_container.bucket_config

            bucket_degree = 1
            if bucket_config is not None:
                bucket_degree = bucket_config.bucket_degree

            hlo_artifact_collection = []
            for bucket_rank in range(bucket_degree):
                example_inputs = example_input_collection[bucket_rank]
                func_kwargs = (
                    {} if bucket_config is None else bucket_config.get_func_kwargs_for_bucket_rank(bucket_rank)
                )
                if "bucket_rank" in func_kwargs:
                    func_kwargs.pop("bucket_rank")  # to avoid multiple definition of bucket_rank
                func, input_output_aliases = model_input_container.model_instance.get(bucket_rank, **func_kwargs)

                hlo_artifacts = torch_neuronx.xla_impl.trace.generate_hlo(
                    func, example_inputs, input_output_aliases, False, False, False
                )
                hlo_artifacts.metaneff = hlo_artifacts.metaneff.SerializeToString()
                hlo_artifact_collection.append(hlo_artifacts)

            mp_q.put(hlo_artifact_collection)

    def shard_weights(self, rank, model_container: ModelContainer, serialize_path: str):
        checkpoint = self.checkpoint_loader()
        _mock_parallel_state(self.tp_degree, rank)
        with init_on_device(torch.device("meta"), force_custom_init_on_device=True):
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

        save_file(checkpoint, os.path.join(serialize_path, f"tp{rank}_sharded_checkpoint.safetensors"))

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
                checkpoint_key = str(tensor.checkpoint_key).replace("b'","").replace("'","")
                shapes[checkpoint_key] = list(tensor.shape)
                dtypes[checkpoint_key] = get_torch_dtype(tensor.data_type)
        if len(shapes):
            return torch.jit.script(StateInitializer(shapes=shapes, dtypes=dtypes, tp_degree=self.tp_degree))
        else:
            return None

    def build_flattener_map(self):
        flattener_map = []
        for key, model_container in self.model_collection.items():
            flattener = JITWrapper(func=model_container.hlo_artifact_collection[0].flattener,is_flattener=True)
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
        for i,meta_tensor in enumerate(metaneff.output_tensors):
            if i not in metaneff.output_aliases_to:
                example_outputs.append(torch.zeros(list(meta_tensor.shape),dtype=get_torch_dtype(meta_tensor.data_type)))

        # return jit traced packer
        jit_wrapped_packer = JITWrapper(packer,False)
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

            buckets = [torch.classes.neuron.SPMDModel(neff, metaneff, self.tp_degree) for neff, metaneff in models]

            spmd_bucket_model_executor = SPMDBucketModelScript(compiled_models=buckets)
            with torch_neuronx.contexts.disable_nrt_load():
                spmd_bucket_model_executor = torch.jit.script(spmd_bucket_model_executor)
            if model_container.bucket_config is None:
                bucket_kernel = torch.jit.script(default_bucket_kernel)
                bucket_kernel_constant_args = ()
            else:
                bucket_kernel = model_container.bucket_config.bucket_kernel()
                bucket_kernel_constant_args = model_container.bucket_config.bucket_kernel_constant_args
            spmd_bucket_model = SPMDBucketModel(
                bucket_kernel,
                bucket_kernel_constant_args,
                spmd_bucket_model_executor
            )
            with torch_neuronx.contexts.disable_nrt_load():
                spmd_bucket_model = torch.jit.script(spmd_bucket_model)
            model_map_input.append((key, spmd_bucket_model))

        state_initializer = self.build_state_initializer()

        model_map = torch.nn.ModuleDict(model_map_input)

        flattener_map = self.build_flattener_map()

        input_shape_map = {}
        # use to jit trace NxDModelExecutor
        example_inputs = None
        for key, model_container in self.model_collection.items():
            # example_inputs is of type List[Tuple[Tensor, Tensor, ...]]
            example_inputs = model_container.example_inputs
            for example_input in example_inputs:
                # torch.Size type is not a concept in a jit model, it's just List[int]
                input_shape_map[str([list(tensor.shape) for tensor in example_input])] = key

        packer = next(iter(self.model_collection.values())).hlo_artifact_collection[0].packer
        traced_packer = self.build_packer(packer)

        # Get weight layout transformation model
        wlt_model = self._prepare_weight_layout_transform_model()

        with torch_neuronx.contexts.disable_nrt_load():
            nxd_model = NxDModel(
                models=model_map,
                tp_degree=self.tp_degree,
                flattener_map=flattener_map,
                input_shape_map=input_shape_map,
                packer=traced_packer,
                state_initializer=state_initializer,
                weight_loader=wlt_model,
            )
            with torch_neuronx.contexts.disable_nrt_load():
                nxd_model = torch.jit.script(nxd_model)

                # mock model as initialized so jit trace doesn't fail
                nxd_model.mock_initialization(True)
                nxd_model_executor = torch.jit.trace(NxDModelExecutor(nxd_model),example_inputs[0],strict=False)
                nxd_model_executor.nxd_model.mock_initialization(False)

        return nxd_model_executor

    def _read_neff_from_path(self, neff_path: str):
        with open(neff_path, "rb") as f:
            return f.read()

    def _get_priority_hlo_artifact(self) -> HloArtifacts:
        for model_artifacts in self.model_collection.values():
            if model_artifacts.priority_model_idx is not None:
                return model_artifacts.hlo_artifact_collection[model_artifacts.priority_model_idx]
        return None

    def _should_optimize_layout(self):
        return self._get_priority_hlo_artifact() is not None

    def _mark_weight_in_priority_hlo(self):
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
        hlo_stub_filepath = neff_artifacts.neff_filename.replace("graph.neff", "wrapped_neff.hlo")

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

        priority_hlo_artifacts = self._get_priority_hlo_artifact()
        weight_name_to_transform_cpt = hlo_utils.get_layout_transform_map(
            hlo_stub=hlo_stub,
            weight_name_to_idx=priority_hlo_artifacts.weight_name_to_idx
        )

        for model_artifacts in self.model_collection.values():
            for bucket_rank, hlo_artifacts in enumerate(model_artifacts.hlo_artifact_collection):
                if bucket_rank == model_artifacts.priority_model_idx:
                    continue
                hlo_utils.append_layout_computation_to_hlo(hlo_artifacts, weight_name_to_transform_cpt)
        logger.info("Done optimizing weight layout for all HLOs")

    def _prepare_weight_layout_transform_model(self):
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
        wlt_neff_artifact = torch_neuronx.xla_impl.trace.generate_neff(
            wlt_hlo_artifact,
            compiler_workdir=layout_dir,
            compiler_args="--model-type=transformer -O1",
            inline_weights_to_neff=False,
        )
        wlt_neff = self._read_neff_from_path(wlt_neff_artifact.neff_filename)

        # Build the model on runtime
        wlt_model = torch.classes.neuron.LayoutTransformation(wlt_neff, metaneff_str, self.tp_degree)
        logger.info("Done preparing weight layout transformation")
        return wlt_model
