import copy
import logging
import time
from types import MethodType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch_xla.core.xla_model as xm
from torch import nn
from torch_xla.distributed.parallel_loader import MpDeviceLoader

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.grads import bucket_allreduce_gradients
from neuronx_distributed.pipeline import scheduler
from neuronx_distributed.pipeline.comm import (
    recv_from,
    recv_python_object,
    rmsg,
    send,
    send_python_object,
)
from neuronx_distributed.pipeline.partition import (
    analyze_pipeline_module,
    analyze_shared_weights_across_stages,
    create_partitions,
    partition_traced_model,
)
from neuronx_distributed.pipeline.scheduler import InferenceSchedule, TrainSchedule
from neuronx_distributed.pipeline.timeline import PPTimeline
from neuronx_distributed.pipeline.trace import trace_model
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.utils.model_utils import (
    maybe_materalize_model,
    move_model_to_device,
    reinit_model,
)
from neuronx_distributed.utils.serialization import (
    SerializationManager,
    TensorMeta,
    find_loss_from_output_and_spec,
)

logger = get_logger()


class NxDPPModel(nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        transformer_layer_cls: Optional[Any] = None,
        num_microbatches: int = 1,
        output_loss_value_spec: Optional[Union[Dict, Tuple]] = None,
        return_mb_loss: bool = False,
        broadcast_and_average_loss: bool = False,
        use_zero1_optimizer: bool = False,
        use_optimizer_wrapper: bool = False,
        pipeline_cuts: Optional[List[str]] = None,
        input_names: Optional[List[str]] = None,
        leaf_module_cls: Optional[List[Any]] = None,
        autowrap_functions: Optional[Tuple[ModuleType]] = None,
        autowrap_modules: Optional[Tuple[Callable, ...]] = None,
        autowrap_obj_methods: Optional[Dict[Any, List[Callable]]] = None,
        tracer_cls: Optional[Union[str, Any]] = None,
        param_init_fn: Optional[Any] = None,
        trace_file_path: Optional[str] = None,
        return_loss_on_cpu: Optional[bool] = True,
        auto_partition: Optional[bool] = False,
        _use_gloo_for_metadata_comm: bool = False,
        _debug_mode: bool = False,
        _debug_pp_size: int = 8,
        _debug_pp_rank: int = 0,
    ):
        """
        Model wrapper to run pipeline parallelism
        Inputs:
            module:
                Module to be distributed with pipeline parallelism

            transformer_layer_cls:
                The module class of transformer layers

            num_microbatches:
                Number of pipeline microbatchs

            output_loss_value_spec:
                The ``output_loss_value_spec`` value can be specified to disambiguate
                which value in the output of `forward` is the loss value on which NxDPPModel should apply
                backpropagation. For example, if your ``forward`` returns a tuple ``(loss, model_out)``,
                you can specify ``output_loss_value_spec=(True, False)``. Or, if your ``forward`` returns
                a dict ``{'loss': loss_value, 'model_out': model_out}``, you can specify
                ``output_loss_value_spec={'loss': True, 'model_out': False}``
                referred from https://github.com/pytorch/PiPPy/blob/main/pippy/IR.py#L697

            return_mb_loss:
                Whether return a list of loss for all microbatchs

            broadcast_and_average_loss:
                Whether to broadcast loss to all PP ranks and average across dp ranks, when set to True
                return_mb_loss must be False

            use_zero1_optimizer:
                Whether ZeRO-1 optimizer is used.

            use_optimizer_wrapper
                Whether optimizer wrapper is used.

            pipeline_cuts:
                A list of layer names that will be used to annotate pipeline stage boundaries

            input_names:
                The input names that will be used for tracing,
                which will be the same as the model inputs during runtime.

            leaf_module_cls:
                A list of module classes that should be treated as leaf nodes during tracing.
                Note transformer layer class will be by default treat as leaf nodes.

            autowrap_modules: (symbolic tracing only)
                Python modules whose functions should be wrapped automatically
                without needing to use fx.wrap().
                reference https://github.com/pytorch/pytorch/blob/main/torch/fx/_symbolic_trace.py#L241

            autowrap_functions: (symbolic tracing only)
                Python functions that should be wrapped automatically without
                needing to use fx.wrap().
                reference https://github.com/pytorch/pytorch/blob/main/torch/fx/_symbolic_trace.py#L241

            autowrap_obj_methods: (symbolic tracing only)
                Wrapping the methods in certain object that you want to avoid tracing. Most common use case
                is to wrap some methods in some modules. For example assume the top module is `model` and the method
                that should skip tracing is `model.some_module.some_method`, one can pass autowrap_obj_methods={model.some_module: ["some_method"]}

            tracer_cls:
                User provided tracer class for symbolic tracing. It can be "hf", "torch" or any tracer class
                user created.

            param_init_fn:
                Function used to initialize parameters. This is useful if user wants to use meta device to do
                delayed parameter initialization. param_init_fn should take a module as input and initialize the
                parameters that belongs to this module only (not for submodules).

            trace_file_path:
                The file location to save the timeline file. Setting to None will not create timeline

            auto_partition:
                Boolean to indicate whether to use auto_partition. If auto_partition is True, pipeline_cuts should not be provided by the user as they will be selected during initialization.

        Usage:
            User can feed the partition config into the NxDPPModel, then tracing and partition will
            be done during model initialization. Example usage:

                model = NxDPPModel(...pipeline_cuts=["layer1"]...)

            User can also call trace/cut_pipeline_stage/partition manually. Example usage:

                model = NxDPPModel(...)
                model.trace(input_names=...)
                model.cut_pipeline_stage("layer1")
                model.partition()

        """
        super(NxDPPModel, self).__init__()
        if not parallel_state.model_parallel_is_initialized() and not _debug_mode:
            raise RuntimeError(
                "Model parallelism needs to be initialzed before applying NxDPPModel wrapper. Please call neuronx_distributed.parallel_layers.initialize_model_parallel(pipeline_model_parallel_size, tensor_model_parallel_size)"  # noqa: E501
            )
        if transformer_layer_cls is None:
            raise ValueError("NxDPPModel requires transformer_layer_cls as input")
        self.original_torch_module = module
        self.traced_model = None
        self.paritioned_model = None
        self.pipeline_cuts = []
        self.partitioned = False
        self.model_moved_to_device = False
        self.shape_traced = False
        self.return_mb_loss = return_mb_loss
        self.broadcast_and_average_loss = broadcast_and_average_loss
        self.use_zero1_optimizer = use_zero1_optimizer
        self.use_optimizer_wrapper = use_optimizer_wrapper
        self.return_loss_on_cpu = return_loss_on_cpu
        if self.broadcast_and_average_loss:
            assert not self.return_mb_loss, "When broadcast_and_average_loss is True return_mb_loss must be False"
        self.output_loss_value_spec = output_loss_value_spec
        self.transformer_layer_cls = transformer_layer_cls
        self.num_microbatches = num_microbatches
        self.pipeline_parallel_size = (
            parallel_state.get_pipeline_model_parallel_size() if not _debug_mode else _debug_pp_size
        )
        self.pipeline_parallel_rank = (
            parallel_state.get_pipeline_model_parallel_rank() if not _debug_mode else _debug_pp_rank
        )
        self.serialization_manager = SerializationManager()
        self.param_init_fn = param_init_fn
        self.input_names = None
        self._metadata_comm_type = "gloo" if _use_gloo_for_metadata_comm else "tcp"
        if not parallel_state.is_tcp_store_available() and self._metadata_comm_type == "tcp":
            logger.warning(f"Can not get default tcp_store, fall back to use gloo for metadata communication")
            self._metadata_comm_type = "gloo"

        # Internal attributes for amp
        self._autocast_enabled = False
        self._autocast_dtype = None

        # Outputs from analyze_pipeline_module
        self.stage_id_to_IO_input_names = None
        self.stage_id_to_model_input_names = None
        self.stage_id_to_input_count = None
        self.stage_id_to_output_count = None

        self.shared_weights_name_to_pg = {}
        self.local_name_to_original_name = {}
        self.original_name_to_local_name = {}
        self.clear_minibatch_state()
        if not _debug_mode:
            self._set_distributed()
            # timeline needs to be created after init dist
            self.timeline = PPTimeline(trace_file_path, self.pipeline_parallel_rank)
        if auto_partition and pipeline_cuts is not None and len(pipeline_cuts) > 0:
            raise RuntimeError(
                "auto_partition is True and pipeline_cuts are provided. If auto_partition is set to True, pipeline_cuts must not be set as the cuts will automatically be computed. If pipeline cuts are provided then auto_partition should be False."
            )
        elif auto_partition and self.pipeline_parallel_size > 1:
            # Auto partition the model layers
            model_layers = self.get_model_layers(self.original_torch_module, self.transformer_layer_cls)
            if len(model_layers) == 0:
                raise ValueError(f"No modules of type {self.transformer_layer_cls} found in the model.")
            if torch.distributed.get_rank() == 0:
                logger.info("Model transformer layers are: \n{}".format(model_layers))
            pipeline_cuts = create_partitions(self.pipeline_parallel_size, model_layers)
            if torch.distributed.get_rank() == 0:
                logger.info("Pipeline cuts are: \n{}".format(pipeline_cuts))

        # If pipeline cuts are set, directly run tracing and partition
        if pipeline_cuts is not None and len(pipeline_cuts) > 0:
            self.trace(
                input_names=input_names,
                leaf_modules=leaf_module_cls,
                autowrap_functions=autowrap_functions,
                autowrap_modules=autowrap_modules,
                autowrap_obj_methods=autowrap_obj_methods,
                tracer_cls=tracer_cls,
            )
            for pp_cut in pipeline_cuts:
                self.cut_pipeline_stage(pp_cut)
            self.partition()

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.original_torch_module, name)

    def clear_minibatch_state(self):
        """
        Initialize/clear the runtime placeholders
        """
        self.training = None
        self.losses = []
        self.model_inputs_iter = None
        self.input_name_to_iter_idx = None
        self.current_mb = -1
        # Name map to object that will pass along the stage for current microbatch
        self.current_mb_pass_along_io = {}
        # Inputs that requires grads for each mb
        self.mb_to_inputs_for_grads = {mb: [] for mb in range(self.num_microbatches)}
        # Outputs that requires grads for each mb
        self.mb_to_outputs_for_grads = {mb: [] for mb in range(self.num_microbatches)}
        # Used to pass the stage input from _fwd_preprocess_task to _fwd_step_task
        self.current_mb_stage_input = None
        # Used to pass the stage output from _fwd_step_task to _fwd_postprocess_task
        self.current_mb_stage_output = None
        # Used to pass the grads recvd from _bwd_preprocess_task to _bwd_step_task
        self.current_mb_grads = None

    def trace(
        self,
        input_names: Optional[List[str]] = None,
        leaf_modules: Optional[List[Any]] = None,
        autowrap_functions: Optional[List[Callable]] = None,
        autowrap_modules: Optional[List[ModuleType]] = None,
        autowrap_obj_methods: Optional[Dict[Any, List[Callable]]] = None,
        tracer_cls: Optional[Union[str, Any]] = None,
    ):
        """
        Trace the module and create the fx.GraphModule
        Inputs:
            input_names:
                The input arg/kward names that will for tracing

            leaf_modules:
                torch.nn.Module classes that should be treated as leaf module.
                The transformer layer will be treated as leaf by default

            autowrap_modules:
                Python modules whose functions should be wrapped automatically
                without needing to use fx.wrap().

            autowrap_function:
                Python functions that should be wrapped automatically without
                needing to use fx.wrap().
        """
        if self.traced_model is not None:
            logging.warning("NxDPPModel.trace() is called while model is already partitioned, skipping...")
            return
        # [TODO] Handle input names better
        self.input_names = input_names
        if leaf_modules is None:
            leaf_modules = []
        leaf_modules.append(self.transformer_layer_cls.__name__)
        self.traced_model = trace_model(
            self.original_torch_module,
            input_names=input_names,
            leaf_modules=leaf_modules,
            autowrap_functions=autowrap_functions,
            autowrap_modules=autowrap_modules,
            autowrap_obj_methods=autowrap_obj_methods,
            tracer_cls=tracer_cls,
        )
        if torch.distributed.get_rank() == 0:
            logger.debug(rmsg(f"traced_model: {self.traced_model}"))

    def partition(self):
        if self.partitioned:
            logging.warning("NxDPPModel.partition() is called while model is already partitioned, skipping...")
            return
        num_stages = len(self.pipeline_cuts) + 1

        if self.pipeline_parallel_size != num_stages:
            raise ValueError(
                f"User cut stages {num_stages} mismatch the initialized pipeline parallel size {self.pipeline_parallel_size}"  # noqa: E501
            )
        assert (
            self.traced_model is not None
        ), "NxDPPModel.trace() should be called before calling NxDPPModel.partition()"

        # Partition
        qualname_map = {}
        self.paritioned_model = partition_traced_model(self.traced_model, qualname_map)
        if torch.distributed.get_rank() == 0:
            logger.debug(rmsg(f"paritioned_model: {self.paritioned_model}"))

        # Extract the local module
        self.partitions = []
        for name, module in self.paritioned_model.named_children():
            if torch.distributed.get_rank() == 0:
                logger.debug(rmsg(f"partition {name}: {module}"))
            self.partitions.append(module)

        (
            self.stage_id_to_IO_input_names,
            self.stage_id_to_model_input_names,
            self.stage_id_to_input_count,
            self.stage_id_to_output_count,
        ) = analyze_pipeline_module(self.paritioned_model)

        assert self.pipeline_parallel_size == len(
            self.partitions
        ), f"pipeline_parallel_size {self.pipeline_parallel_size} != number submodules {len(self.paritioned_model)}"

        self.local_module = self.partitions[self.pipeline_parallel_rank]
        if parallel_state.get_tensor_model_parallel_rank() == 0 and parallel_state.get_data_parallel_rank() == 0:
            logger.debug(
                rmsg(f"pipeline_parallel_rank {self.pipeline_parallel_rank} local_module {self.local_module }")
            )

        self._post_partition(qualname_map)

    def _post_partition(self, qualname_map):
        # Create name mapping between original parameter to partitioned parameter
        self._build_parameter_buffer_name_mapping(qualname_map)
        self.partitioned = True

        # Locate shared weights between stages
        self.register_shared_weights()

        # Create pipeline schedule
        self.create_schedule()

        # Materialize local module to CPU then move to XLA device
        self._maybe_materialize_local_module()

        # grad is only enabled for local parameters
        self._disable_grad_for_nonlocal()

    def create_schedule(self):
        self.eval_scheduler = InferenceSchedule(
            self.num_microbatches, self.pipeline_parallel_size, self.pipeline_parallel_rank
        )
        self.train_scheduler = TrainSchedule(
            self.num_microbatches, self.pipeline_parallel_size, self.pipeline_parallel_rank
        )
        logger.debug(rmsg(f"eval_schedule {[task for task in self.eval_scheduler.steps()]}"))
        logger.debug(rmsg(f"train_schedule {[task for task in self.train_scheduler.steps()]}"))

    def _build_parameter_buffer_name_mapping(self, qualname_map):
        """
        After partition, FX will change the parameter name. Here we build a mapping between local parameter name
        and the original parameter name. The general rule is:
        - `split_module` call for FX will create each partition as a child module
        - Each child module will be named with submod_ prefix
        - `split_module` call will return qualname_map as the mapping of original name to changed name
        For example:
            Original name: transformer.h.2.attn.c_proj.bias
            Changed name from split call: submod_0.transformer_h_2.attn.c_proj.bias
            We create transformer.h.2.attn.c_proj.bias : transformer_h_2.attn.c_proj.bias mapping to translate names
        """
        local_state_dict = self.local_module.state_dict(keep_vars=True)
        orginal_state_dict = self.original_torch_module.state_dict(keep_vars=True)
        qualname_map_local = {}
        for local_name, origin_name in qualname_map.items():
            assert local_name.startswith("submod_"), f"Found local module name with out submod_ prefix {local_name}"
            if local_name.startswith(f"submod_{self.pipeline_parallel_rank}"):
                # remove the submod_ prefix
                local_name = ".".join(local_name.split(".")[1:])
                qualname_map_local[local_name] = origin_name
        for local_name, _ in local_state_dict.items():
            origin_name = local_name
            for local, orgin in qualname_map_local.items():
                if local_name.startswith(local):
                    origin_name = local_name.replace(local, orgin)
            assert (
                origin_name in orginal_state_dict
            ), f"parameter/buffer name {origin_name} is missing from original torch model"
            self.local_name_to_original_name[local_name] = origin_name
            self.original_name_to_local_name[origin_name] = local_name
        logger.debug(rmsg(f"local_name_to_original_name {self.local_name_to_original_name}"))
        logger.debug(rmsg(f"original_name_to_local_name {self.original_name_to_local_name}"))

    def register_shared_weights(self):
        """
        Analyze the shared weights. Each shared weight will be assigned with a process group for grads sync.
        """
        # [TODO] Handle the case that the shared weights only requires grads on partial PP rank
        shared_weights = analyze_shared_weights_across_stages(self.traced_model, self.partitions)
        for weights in shared_weights:
            ranks_ = []
            current_rank_name = None
            for name, stage in weights:
                ranks_.append(stage)
                if stage == self.pipeline_parallel_rank:
                    current_rank_name = name
            pg = self._create_pg_with_ranks(ranks_)
            logger.debug(
                rmsg(f"Found shared weights with (name, stage): {(current_rank_name, self.pipeline_parallel_rank)}")
            )
            if current_rank_name is not None:
                self.shared_weights_name_to_pg[current_rank_name] = pg
                logger.info(
                    rmsg(
                        f"Register shared weight with (name, stage): {(current_rank_name, self.pipeline_parallel_rank)}"
                    )
                )

    def _reduce_shared_weights(self):
        for shared_name, pg in self.shared_weights_name_to_pg.items():
            for n, p in self.local_module.named_parameters():
                if shared_name == n and p.requires_grad:
                    assert p.grad is not None, f"Found shared weight {n} has None grad"
                    logger.debug(rmsg(f"Reduce shared weight {shared_name}"))
                    torch.distributed.all_reduce(p.grad, group=pg)
                    break

    def _sync_shared_weights(self):
        for shared_name, pg in self.shared_weights_name_to_pg.items():
            for n, p in self.local_module.named_parameters():
                if shared_name == n:
                    logger.info(rmsg(f"Sync shared weight {shared_name}"))
                    with torch.no_grad():
                        # Set parameter data to zeros except for the first rank in the shared group
                        if torch.distributed.get_rank(group=pg) != 0:
                            p.data.zero_()
                        torch.distributed.all_reduce(p.data, group=pg)
                    break

    def _mark_pipeline_cuts(self, cut_point):
        # Internal API to mark the cut in the graph
        for node in self.traced_model.graph.nodes:
            if node.op == "call_module" and node.target == cut_point:
                node.meta["partition"] = True
                return
        raise RuntimeError(f"cut_point {cut_point} does not exist in the graph")

    def cut_pipeline_stage(self, cut_point):
        # [TODO] make sure cut point is only at transformer layers
        assert (
            cut_point not in self.pipeline_cuts
        ), f"each cutpoint can be only marked once, but {cut_point} is marked twice"
        assert (
            not self.partitioned
        ), f"cut_pipeline_stage({cut_point}) is called after model is partitioned, which is not allowed."
        assert self.traced_model is not None, "cut_pipeline_stage must be called after trace"
        self.pipeline_cuts.append(cut_point)
        self._mark_pipeline_cuts(cut_point)

    def _validate_partitioned(self):
        if not self.partitioned:
            raise RuntimeError(
                "Model has not been partitioned yet. Call this method after first step when using autopartitioning."
            )

    def _verify_inputs(self, kwargs):
        if self.input_names is not None:
            if set(kwargs.keys()) != set(self.input_names):
                raise RuntimeError(
                    f"train/eval inputs ({set(kwargs.keys())}) must be same as the tracing input names {set(self.input_names)}"  # noqa: E501
                )

    def _disable_grad_for_nonlocal(self):
        # disable grads for non-local parameters
        local_p = set([p for p in self.local_parameters()])
        for n, p in self.named_parameters():
            if p not in local_p:
                p.requires_grad = False
                p.grad = None

    def _prepare_run(self, kwargs, train=True):
        self._validate_partitioned()
        self.move_model_to_device()
        self._prepare_inputs_and_infer_shape(kwargs, train=train)

    def _prepare_inputs_and_infer_shape(self, kwargs, train=True):
        """
        Tracing the tensor shapes so that no need to launch blocking send/recv python object calls during runtime.
        This is only required for Neuron since Neuron requires all neffs that contain the same CC be loaded
        at the same time, the blocking send/recv will prevent all ranks to load the CC neff, which will cause a hang.
        XLA will track live tensors and build graph based on them, we garbage collect all live tensor during tracing
        so that XLA will not capture any graph, but the tensor shapes will be sent/recvd during the tracing.
        The only graphs of this tracing call should be the input graphs which come from parallel loader
        and graphs that create the parameters.
        """
        if not self.shape_traced:
            logger.info(rmsg("Running tracing to infer tensor shapes..."))
            start = time.time()

            self.tracing = True
            with torch.set_grad_enabled(train):
                self._create_model_inputs_iter(kwargs)
                self._fwd_preprocess_task()
                self._fwd_step_task()
                self._fwd_postprocess_task()
                self.clear_minibatch_state()
                self.shape_traced = True
            self.tracing = False
            self._mark_step()
            end = time.time()
            logger.info(rmsg(f"Tensor shapes inference finished, total consumed time {end-start}s"))
            logger.debug(
                rmsg(
                    f"After tracing current stage's stage_id_to_IO_input_names {self.stage_id_to_IO_input_names[self.pipeline_parallel_rank]}"  # noqa: E501
                )
            )

        # Need to create input iters again since the old one is garbage collected
        self._create_model_inputs_iter(kwargs)

    def run_train(self, **kwargs):
        self._autocast_enabled = torch.is_autocast_enabled()
        self._autocast_dtype = torch.get_autocast_gpu_dtype()
        with torch.cuda.amp.autocast(enabled=False):
            loss = self._run_train(**kwargs)
        self._autocast_enabled = False
        self._autocast_dtype = None
        return loss

    def run_eval(self, **kwargs):
        self._autocast_enabled = torch.is_autocast_enabled()
        self._autocast_dtype = torch.get_autocast_gpu_dtype()
        with torch.cuda.amp.autocast(enabled=False):
            loss = self._run_eval(**kwargs)
        self._autocast_enabled = False
        self._autocast_dtype = None
        return loss

    def _run_train(self, **kwargs):
        self._prepare_run(kwargs)

        self.training = True
        self.local_module.train()
        self._exec_schedule(self.train_scheduler)
        loss = self._process_loss()
        self.clear_minibatch_state()
        self.timeline.mark_step_end()
        return loss

    def _run_eval(self, **kwargs):
        self._prepare_run(kwargs, train=False)

        self.training = False
        self.local_module.eval()
        with torch.no_grad():
            self._exec_schedule(self.eval_scheduler)
            loss = self._process_loss()
            self.clear_minibatch_state()
            self.timeline.mark_step_end()
            return loss

    def get_batch_iterator(self, batch: List[torch.Tensor]):
        """
        Reference from Apex https://github.com/NVIDIA/apex/blob/master/apex/transformer/pipeline_parallel/utils.py#L122
        Create a list of microbatches from a list of local minibatches.

        This function creates a list of `k`th microbatches from a list of local minibatches.
        `a local minibatch` consists of `global_batch_size / data_parallel_size` samples.
        """
        batch_size = None
        for t in batch:
            if batch_size is None:
                batch_size = t.size()[0]
            else:
                assert batch_size == t.size()[0], f"batch dimension does not match, {batch_size} and {t.size()[0]}"
        if batch_size % self.num_microbatches != 0:
            raise RuntimeError(
                f"Input batch size {batch_size} must be divisible with the num_microbatches {self.num_microbatches}"
            )
        micro_batch_size = batch_size // self.num_microbatches

        all_batches = []
        for k in range(self.num_microbatches):
            start = k * micro_batch_size
            end = start + micro_batch_size
            microbatch = list()
            for x in batch:
                size = len(x)
                assert size > start and size >= end, "size issue microbatch"
                microbatch.append(x[start:end])
            assert len(microbatch) > 0, "Microbatch length less than 0"
            all_batches.append(microbatch)
        return self._get_microbatch_dataloader(all_batches)

    def _create_model_inputs_iter(self, input_kwargs):
        """
        Create a data iterator for microbatches if current PP rank requires any model input
        """
        self._verify_inputs(input_kwargs)
        inputs_ = []
        input_name_to_iter_idx = {}
        for inp_name in self.stage_id_to_model_input_names[self.pipeline_parallel_rank].keys():
            assert (
                inp_name in input_kwargs
            ), f"stage {self.pipeline_parallel_rank} requires model input {inp_name}, which is not provided in the input list {input_kwargs.keys()}"  # noqa: E501
            input_name_to_iter_idx[inp_name] = len(inputs_)
            t = input_kwargs[inp_name]
            # Model inputs should be torch CPU tensors
            if not isinstance(t, torch.Tensor):
                raise ValueError(f"batch input must be a torch tensor, but getting {type(t)}")
            if t.device != torch.device("cpu"):
                raise ValueError(
                    f"model inputs should be all on cpu device, but getting {inp_name} on device {t.device}"
                )
            inputs_.append(t)
        if len(inputs_) > 0:
            self.model_inputs_iter = iter(self.get_batch_iterator(inputs_))
            self.input_name_to_iter_idx = input_name_to_iter_idx

    def _handle_stage_outputs(self, outputs, stage):
        # If there is only a single output from graph
        # make output a list so the indexing will be right
        if self.stage_id_to_output_count[stage] == 1:
            outputs = [outputs]
        else:
            if len(outputs) != self.stage_id_to_output_count[stage]:
                raise RuntimeError(
                    f"Stage {stage} number outputs ({len(outputs)}) mismatches with compiled result ({self.stage_id_to_output_count[stage]})"  # noqa: E501
                )
        return outputs

    def _fwd_step_task(self):
        """
        Major duties of for this task:
        - Run current stage forward step for current mb
        - Collect loss for the last pipeline stage
        - Collect the outputs that require grad for bwd
        """
        self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardStepTask")
        if not self.tracing:
            if self.current_mb_stage_input is None:
                raise RuntimeError(rmsg("Running ForwardStepTask but current_mb_stage_input is None"))
            if self.current_mb_stage_input[1] != self.current_mb:
                raise RuntimeError(
                    rmsg(
                        f"Running ForwardStepTask for mb {self.current_mb} but current_mb_stage_input contains mb {self.current_mb_stage_input[1]}"  # noqa: E501
                    )
                )
            if self.current_mb_stage_output is not None:
                raise RuntimeError(rmsg("Running ForwardStepTask but current_mb_stage_output is not None"))
            if len(self.mb_to_outputs_for_grads[self.current_mb]) != 0:
                raise RuntimeError(
                    "Running ForwardStepTask but mb_to_outputs_for_grads already contains outputs for current mb"
                )
        inputs = self.current_mb_stage_input[0]
        self.current_mb_stage_input = None

        # Run local pipeline stage
        self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardStep")

        # [TODO]: From Torch 2.1 we can specify device_type='xla' or use torch_xla's autocast, need to revisit this once we have PT2.1 support
        with torch.autocast(enabled=self._autocast_enabled, dtype=self._autocast_dtype, device_type="cuda"):
            outputs = self.local_module(*inputs)

        if not self.tracing:
            self._mark_step()
        self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardStep")

        if self.pipeline_parallel_rank == self.pipeline_parallel_size - 1:
            # [TODO] Add inference support
            current_mb_loss = find_loss_from_output_and_spec(outputs, self.output_loss_value_spec)
            if current_mb_loss is None:
                raise RuntimeError(
                    f"User provided output_loss_value_spec {self.output_loss_value_spec} failed to fetch the loss"
                )
            self.losses.append(current_mb_loss)
        else:
            outputs = self._handle_stage_outputs(outputs, self.pipeline_parallel_rank)
            self.current_mb_stage_output = (outputs, self.current_mb)
            if self.training and not self.tracing:
                # Need to collect the mb_to_outputs_for_grads here since _fwd_postprocess_task sometimes run after _bwd_preprocess_task   # noqa: E501
                current_stage_output_IO = self.stage_id_to_IO_input_names[self.pipeline_parallel_rank + 1]
                # Iterate through current_stage_output_IO to enforce tensor order between PP ranks
                for name, out in current_stage_output_IO.items():
                    pass_along_io = False
                    if out.output_idx is not None:
                        # Current stage outputs
                        current_output = outputs[out.output_idx]
                    else:
                        current_output = self.current_mb_pass_along_io[name]
                        pass_along_io = True
                    _, tx_list, tensor_meta = self.serialization_manager.serialize(current_output)
                    for idx, t in enumerate(tx_list):
                        # [TODO] Add support, requires for cross attention
                        if pass_along_io and t.requires_grad:
                            raise RuntimeError(
                                f"Does not support tensors that require grads to pass along! IO name {name} current stage {self.pipeline_parallel_rank}"  # noqa: E501
                            )
                        logger.debug(
                            rmsg(
                                f"fwd mb {self.current_mb} collect {name}'s {idx}th tensor meta {tensor_meta[idx]} for bwd"  # noqa: E501
                            )
                        )
                        # Collect the outputs that require grad for bwd
                        # Current stage output and next stage input should match exactly
                        if self.training and t.requires_grad:
                            self.mb_to_outputs_for_grads[self.current_mb].append(t)
        self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardStepTask")

    def _bwd_step_task(self):
        """
        Major duties of for this task:
        - Run current stage forward step for current mb
        """
        self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardStepTask")
        # Last pipeline stage will directly do backprop from loss
        if self.pipeline_parallel_rank == self.pipeline_parallel_size - 1:
            self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardStep")
            scaled_loss = self.losses[self.current_mb] / self.num_microbatches
            scaled_loss.backward()
            self._mark_step()
            self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardStep")
            self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardStepTask")
            return

        if self.current_mb_grads is None:
            raise RuntimeError("Running BackwardStepTask but current_mb_grads is None")
        if self.current_mb_grads[1] != self.current_mb:
            raise RuntimeError(
                f"Running BackwardStepTask for mb {self.current_mb} but current_mb_grads contains mb {self.current_mb_grads[1]}"  # noqa: E501
            )
        if len(self.mb_to_outputs_for_grads[self.current_mb]) == 0:
            raise RuntimeError(
                "Running BackwardStepTask but mb_to_outputs_for_grads is does not contain outputs for current mb"
            )
        grads = self.current_mb_grads[0]
        self.current_mb_grads = None
        outputs = self.mb_to_outputs_for_grads.pop(self.current_mb)

        self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardStep")
        # Run local pipeline stage
        torch.autograd.backward(outputs, grad_tensors=grads)
        self._mark_step()
        self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardStep")
        self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardStepTask")

    def _fwd_preprocess_task(self):
        """
        Major duties of for this task:
        - Receive the tensor shapes during tracing
        - Get the model inputs from model_inputs_iter for current mb
        - Receive the current stage IO from previous stage
        - Collect the inputs that are required by current stage
        - Collect the inputs that require grad for bwd
        - Collect the inputs that will pass along this stage
        """
        self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardPreprocessTask")
        if not self.tracing:
            if self.current_mb_stage_input is not None:
                raise RuntimeError(rmsg("Running ForwardPreprocessTask but current_mb_stage_input is not None"))
            if len(self.mb_to_inputs_for_grads[self.current_mb]) != 0:
                raise RuntimeError(
                    rmsg(
                        "Running ForwardPreprocessTask but mb_to_inputs_for_grads already contains inputs for current mb"  # noqa: E501
                    )
                )

        inputs = [None] * self.stage_id_to_input_count[self.pipeline_parallel_rank]

        # Get the model inputs from model_inputs_iter
        if self.model_inputs_iter is not None:
            self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardFetchInput")
            model_inputs = next(self.model_inputs_iter)
            for name, idx in self.stage_id_to_model_input_names[self.pipeline_parallel_rank].items():
                inputs[idx] = model_inputs[self.input_name_to_iter_idx[name]]
            self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardFetchInput")

        for name, inp in self.stage_id_to_IO_input_names[self.pipeline_parallel_rank].items():
            self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardRecv_{name}")
            if not self.shape_traced:
                # Suppose to receive List(TensorMeta), python obj
                tensor_meta, py_obj = recv_python_object(method=self._metadata_comm_type)
                logger.debug(rmsg(f"recv tensor_meta {tensor_meta} py_obj {py_obj}"))
                inp.metadata = tensor_meta
                inp.obj = py_obj
            recvd = []
            # Receive the current stage inputs from previous stage
            for idx, meta in enumerate(inp.metadata):
                logger.debug(rmsg(f"fwd mb {self.current_mb} recv {name}'s {idx}th tensor meta {meta}"))
                recvd_tensor = self.recv_op(meta, tracing=self.tracing)
                recvd.append(recvd_tensor)
                # Collect the inputs that require grad for bwd
                if self.training and meta.requires_grad:
                    self.mb_to_inputs_for_grads[self.current_mb].append(recvd_tensor)
            inp_reconstructed = self.serialization_manager.deserialize(copy.deepcopy(inp.obj), recvd)
            if inp.input_idx is not None:
                # Current stage inputs
                assert inputs[inp.input_idx] is None
                inputs[inp.input_idx] = inp_reconstructed
            if (
                self.pipeline_parallel_rank != self.pipeline_parallel_size - 1
                and name in self.stage_id_to_IO_input_names[self.pipeline_parallel_rank + 1]
            ):
                # Pass along
                self.current_mb_pass_along_io[name] = inp_reconstructed
            self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardRecv_{name}")
        self.current_mb_stage_input = (tuple(inputs), self.current_mb)
        self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardPreprocessTask")

    def _fwd_postprocess_task(self):
        """
        Major duties of for this task:
        - Send the tensor shapes during tracing
        - Send the current stage IO to next stage
        """
        # Last stage do not need to send
        if self.pipeline_parallel_rank == self.pipeline_parallel_size - 1:
            return
        self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardPostprocessTask")

        if not self.tracing:
            if self.current_mb_stage_output is None:
                raise RuntimeError(rmsg("Running ForwardPostprocessTask but current_mb_stage_output is None"))
            if self.current_mb_stage_output[1] != self.current_mb:
                raise RuntimeError(
                    rmsg(
                        f"Running ForwardPostprocessTask for mb {self.current_mb} but current_mb_stage_output contains mb {self.current_mb_stage_output[1]}"  # noqa: E501
                    )
                )
        outputs = self.current_mb_stage_output[0]
        self.current_mb_stage_output = None

        current_stage_output_IO = self.stage_id_to_IO_input_names[self.pipeline_parallel_rank + 1]
        for name, out in current_stage_output_IO.items():
            self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardSend_{name}")
            if out.output_idx is not None:
                # Current stage outputs
                current_output = outputs[out.output_idx]
            else:
                # Tensors that needs to pass along
                if name not in self.current_mb_pass_along_io:
                    raise RuntimeError(
                        f"Pass along io {name} is missing, current_mb_pass_along_io {self.current_mb_pass_along_io.keys()}"  # noqa: E501
                    )
                current_output = self.current_mb_pass_along_io.pop(name)
            obj_stripped_of_tensors, tx_list, tensor_meta = self.serialization_manager.serialize(current_output)
            if not self.shape_traced:
                logger.debug(rmsg(f"send tensor_meta {tensor_meta} py_obj {obj_stripped_of_tensors}"))
                send_python_object((tensor_meta, obj_stripped_of_tensors), method=self._metadata_comm_type)
            for idx, t in enumerate(tx_list):
                logger.debug(rmsg(f"fwd mb {self.current_mb} send {name}'s {idx}th tensor meta {tensor_meta[idx]}"))
                self.send_op(t, tracing=self.tracing)
            self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardSend_{name}")
        if len(self.current_mb_pass_along_io) != 0:
            raise RuntimeError(f"Unprocessed passing along io: {self.current_mb_pass_along_io.keys()}")
        self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardPostprocessTask")

    def _bwd_preprocess_task(self):
        """
        Major duties of for this task:
        - Receive the grads for current mb backprop
        """
        self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardPreprocessTask")
        # Last stage do not need to recv
        # (fewu) Will we ever reach here?
        if self.pipeline_parallel_rank == self.pipeline_parallel_size - 1:
            return None, None

        if self.current_mb_grads is not None:
            raise RuntimeError("Running BackwardPreprocessTask but current_mb_grads is not None")
        if len(self.mb_to_outputs_for_grads[self.current_mb]) == 0:
            raise RuntimeError(
                "Running BackwardPreprocessTask but mb_to_outputs_for_grads does not contain grads for current mb"
            )

        grads = []
        current_mb_outputs = self.mb_to_outputs_for_grads[self.current_mb]
        logger.debug(rmsg(f"bwd mb {self.current_mb} recv grads count {len(current_mb_outputs)}"))
        for idx, t in enumerate(current_mb_outputs):
            self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardRecv_{idx}")
            meta = TensorMeta(tensor_index=-1, dtype=t.dtype, shape=t.size(), requires_grad=False, device=t.device)
            logger.debug(rmsg(f"bwd mb {self.current_mb} recv grad meta {meta}"))
            grads.append(self.recv_op(meta, recv_prev=False))
            self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardRecv_{idx}")
        self.current_mb_grads = (grads, self.current_mb)
        self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardPreprocessTask")

    def _bwd_postprocess_task(self):
        """
        Major duties of for this task:
        - Send grads for previous rank for backprop
        """
        self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardPostprocessTask")
        if len(self.mb_to_inputs_for_grads[self.current_mb]) == 0:
            raise RuntimeError(
                "Running BackwardPostprocessTask but mb_to_inputs_for_grads does not contain inputs for current mb"
            )

        current_mb_inputs = self.mb_to_inputs_for_grads.pop(self.current_mb)
        logger.debug(rmsg(f"bwd mb {self.current_mb} send grads count {len(current_mb_inputs)}"))
        for idx, t in enumerate(current_mb_inputs):
            self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardSend_{idx}")
            if not t.requires_grad or t.grad is None:
                raise RuntimeError(rmsg("Backward sending grads, but get None"))
            logger.debug(rmsg(f"bwd mb {self.current_mb} send grad shape {t.grad.size()}"))
            self.send_op(t.grad, send_next=False)
            self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardSend_{idx}")
        self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardPostprocessTask")

    def _reduce_grads_task(self):
        """
        Major duties of for this task:
        - Average the grads acorss data parallel ranks if both optimizer wrapper and ZeRO-1 not used
        - Add the grads for shared weights
        """
        self.timeline.mark_event_start(f"mb_{self.current_mb}_ReduceGradsTask")
        # For backward compatibility:
        #   If both optimizer wrapper and ZeRO-1 not used, average the grads acorss data parallel ranks
        if not self.use_zero1_optimizer and not self.use_optimizer_wrapper:
            # Group for different dtypes and all-reduce
            dtype_groups = {}
            for param in self.local_parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.dtype
                    if tp not in dtype_groups:
                        dtype_groups[tp] = []
                    dtype_groups[tp].append(param)

            logger.debug(
                rmsg(f"reduce grads dtype_groups counts {[(tp, len(group)) for tp, group in dtype_groups.items()]}")
            )
            # For each bucket, all-reduce and copy all-reduced grads.
            for tp in dtype_groups:
                bucket = dtype_groups[tp]
                grads = [param.grad.data for param in bucket]
                bucket_allreduce_gradients(grads)

        self._reduce_shared_weights()
        self._mark_step()
        self.timeline.mark_event_end(f"mb_{self.current_mb}_ReduceGradsTask")

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        scheduler.ForwardStepTask: _fwd_step_task,
        scheduler.BackwardStepTask: _bwd_step_task,
        scheduler.ForwardPostprocessTask: _fwd_postprocess_task,
        scheduler.ForwardPreprocessTask: _fwd_preprocess_task,
        scheduler.BackwardPostprocessTask: _bwd_postprocess_task,
        scheduler.BackwardPreprocessTask: _bwd_preprocess_task,
        scheduler.ReduceGradsTask: _reduce_grads_task,
    }

    def _exec_schedule(self, pipe_schedule):
        # For each step in the schedule
        for step_tasks in pipe_schedule:
            # For each instruction in the step
            for task in step_tasks:
                if type(task) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(f"{self.__class__.__name__} does not understand instruction {repr(task)}")

                logger.debug(rmsg(f"Run task {task}"))
                self.current_mb = task.mb
                # Equivalent to: self._fwd_step_task()
                self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(task)], self)
                self._exec_instr()
                logger.debug(rmsg(f"Task {task} finished"))

    def get_model_layers(self, module, target_module_type, current_path=None):
        current_path = current_path or []
        result = []
        for name, child_module in module.named_children():
            if isinstance(child_module, target_module_type):
                result.append(".".join(current_path + [name]))
            else:
                result.extend(self.get_model_layers(child_module, target_module_type, current_path + [name]))

        return result

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training."""
        raise RuntimeError(
            "model.forward() is not supported in pipeline model. \
                           Use model.run_train() and model.run_eval() instead."
        )

    def train(self, *args, **kwargs):
        """Disabled for pipeline parallel training."""
        raise RuntimeError(
            "model.train() is not supported in pipeline mode. \
                           Use model.run_train() and model.run_eval() instead."
        )

    def state_dict(self, *args, **kwargs):
        """Disabled for pipeline parallel training."""
        raise RuntimeError(
            "model.state_dict() is not supported in pipeline mode. \
                           Use model.local_state_dict() instead."
        )

    def named_parameters(self, *args, **kwargs):
        return self.original_torch_module.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.original_torch_module.parameters(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        return self.original_torch_module.named_buffers(*args, **kwargs)

    def buffers(self, *args, **kwargs):
        return self.original_torch_module.buffers(*args, **kwargs)

    """
    Below methods requires partition
    """

    def local_modules(self):
        self._validate_partitioned()
        for n, m in self.local_named_modules():
            yield m

    def local_named_modules(self, memo: Optional[Set[nn.Module]] = None, prefix: str = ""):
        self._validate_partitioned()
        for n, m in self.local_module.named_modules(memo=memo, prefix=prefix):
            yield n, m

    def local_children(self):
        self._validate_partitioned()
        for n, c in self.local_named_children():
            yield c

    def local_named_children(self):
        self._validate_partitioned()
        for n, c in self.local_module.named_children():
            yield n, c

    def local_parameters(self, recurse: bool = True):
        self._validate_partitioned()
        for n, p in self.local_named_parameters(recurse=recurse):
            yield p

    def local_named_parameters(self, recurse: bool = True):
        self._validate_partitioned()
        for n, p in self.local_module.named_parameters(recurse=recurse):
            original_name = self.local_name_to_original_name[n]
            yield original_name, p

    def local_buffers(self, recurse=True):
        self._validate_partitioned()
        for n, b in self.local_named_buffers(recurse=recurse):
            yield b

    def local_named_buffers(self, prefix: str = "", recurse: bool = True):
        self._validate_partitioned()
        for name, buf in self.local_module.named_buffers(prefix=prefix, recurse=recurse):
            original_name = self.local_name_to_original_name[name]
            yield original_name, buf

    def load_state_dict(self, state_dict, strict=True):
        self._validate_partitioned()
        local_state_dict = self.translate_origin_state_dict_to_local_state_dict(state_dict)
        self.local_module.load_state_dict(local_state_dict, strict=strict)

    def local_state_dict(self, *args, **kwargs):
        self._validate_partitioned()
        local_state_dict = self.local_module.state_dict(*args, **kwargs)
        local_state_dict = self.translate_local_state_dict_to_origin_state_dict(local_state_dict)
        return local_state_dict

    def translate_origin_state_dict_to_local_state_dict(self, origin_state_dict):
        self._validate_partitioned()
        local_dict = {}
        for n, p in origin_state_dict.items():
            if n in self.original_name_to_local_name:
                local_name = self.original_name_to_local_name[n]
                local_dict[local_name] = p
        return local_dict

    def translate_local_state_dict_to_origin_state_dict(self, local_state_dict):
        self._validate_partitioned()
        origin_state_dict = {}
        for n, p in local_state_dict.items():
            if n in self.local_name_to_original_name:
                origin_name = self.local_name_to_original_name[n]
                origin_state_dict[origin_name] = p
            else:
                raise ValueError(f"parameter {n} in local_state_dict does not exist in local module")
        return origin_state_dict

    """
    Below methods can be overwritten to support non-XLA devices
    """

    def _set_distributed(self):
        if self._metadata_comm_type == "gloo":
            parallel_state.initialize_pp_gloo_groups()
        self.send_op = send
        self.recv_op = recv_from

    def _mark_step(self):
        xm.mark_step()

    def _create_pg_with_ranks(self, ranks):
        pg = parallel_state.create_pg_with_ranks(ranks)
        return pg

    def _get_microbatch_dataloader(self, all_batches):
        return MpDeviceLoader(all_batches, xm.xla_device(), batches_per_execution=self.num_microbatches)

    def _maybe_materialize_local_module(self):
        """
        Check whether there's fake tensor in local module, if so materialize it to cpu
        """
        maybe_materalize_model(self.local_module)
        if self.param_init_fn is not None:
            reinit_model(self.local_module, torch.device("cpu"), self.param_init_fn)

    def move_model_to_device(self):
        if not self.model_moved_to_device:
            # Casting local module to xla device
            move_model_to_device(self.local_module, xm.xla_device())
            self._sync_shared_weights()
            self._mark_step()
            self.model_moved_to_device = True

    def _process_loss(self):
        """
        Average the losses of microbatches if not self.return_mb_loss
        """
        if self.pipeline_parallel_rank == self.pipeline_parallel_size - 1:
            loss_all = []
            for loss in self.losses:
                loss_all.append(loss)
            if not self.return_mb_loss:
                if not self.broadcast_and_average_loss:
                    # cast to cpu so the following operation will not create extra graph
                    loss_all = [loss.detach().cpu() if self.return_loss_on_cpu else loss.detach() for loss in loss_all]
                loss = torch.sum(torch.stack(loss_all), dim=0) / len(loss_all)
                if self.broadcast_and_average_loss:
                    loss /= parallel_state.get_data_parallel_size()
                    torch.distributed.all_reduce(loss, group=parallel_state.get_data_parallel_group())
                    torch.distributed.broadcast(
                        loss, torch.distributed.get_rank(), group=parallel_state.get_pipeline_model_parallel_group()
                    )
                    self._mark_step()
                return loss.detach().cpu() if self.return_loss_on_cpu else loss.detach()
            else:
                return loss_all
        elif self.broadcast_and_average_loss:
            loss = torch.tensor(0.0, device=xm.xla_device())
            pp_group = parallel_state.get_pipeline_model_parallel_group()
            src_rank = torch.distributed.distributed_c10d.get_global_rank(pp_group, self.pipeline_parallel_size - 1)
            torch.distributed.broadcast(loss, src_rank, group=pp_group)
            self._mark_step()
            return loss.detach().cpu()
        return None
