import copy
import logging
import time
from collections import defaultdict, OrderedDict
from types import MethodType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
from torch import nn
from torch.autograd.variable import Variable
from torch_xla.distributed.parallel_loader import MpDeviceLoader
from neuronx_distributed.utils import cpu_mode
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import _CHECKPOINT_PREFIX

if not cpu_mode():
    from torch_neuronx.xla_impl.ops import neuron_layer
else:
    def neuron_layer(x): return x

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
    stage_to_pipeline_parallel_rank,
)
from neuronx_distributed.pipeline.scheduler import (
    InferenceSchedule,
    Train1F1BSchedule,
    TrainInterleavedSchedule,
)
from neuronx_distributed.pipeline.timeline import PPTimeline
from neuronx_distributed.pipeline.trace import trace_model
from neuronx_distributed.utils.logger import get_logger, get_log_level
from neuronx_distributed.utils.model_utils import (
    maybe_materalize_model,
    move_model_to_device,
    reinit_model,
    get_delay_tracing,
)
from neuronx_distributed.utils.serialization import (
    SerializationManager,
    TensorMeta,
    find_loss_from_output_and_spec,
)
from neuronx_distributed.pipeline.manual_pipe_stage import PipelineStageModule
from neuronx_distributed.pipeline.partition import PipelineIO
from neuronx_distributed.utils import cpu_mode

from contextlib import contextmanager

logger = get_logger(rank0_only=(get_log_level() == logging.INFO))


@contextmanager
def mark_timeline(timeline, msg):
    try:
        timeline.mark_event_start(msg)
        yield
    finally:
        timeline.mark_event_end(msg)


INPUTS_ARG_NAME = "inputs"
LABLES_ARG_NAME = "labels"


class NxDPPModel(nn.Module):
    def __init__(
        self,
        module: Union[torch.nn.Module, list, tuple],
        transformer_layer_cls: Optional[Any] = None,
        num_microbatches: int = 1,
        virtual_pipeline_size: int = 1,
        output_loss_value_spec: Optional[Union[Dict, Tuple, bool]] = None,
        return_mb_loss: bool = False,
        broadcast_and_average_loss: bool = False,
        use_zero1_optimizer: bool = False,
        use_model_wrapper: bool = False,
        use_optimizer_wrapper: bool = False,
        pipeline_cuts: Optional[List[str]] = None,
        input_names: Optional[List[str]] = None,
        leaf_module_cls: Optional[List[Any]] = None,
        autowrap_functions: Optional[Tuple[Callable]] = None,
        autowrap_modules: Optional[Tuple[ModuleType, ...]] = None,
        autowrap_obj_methods: Optional[Dict[Any, List[Callable]]] = None,
        tracer_cls: Optional[Union[str, Any]] = None,
        param_init_fn: Optional[Any] = None,
        trace_file_path: Optional[str] = None,
        return_loss_on_cpu: Optional[bool] = True,
        deallocate_pipeline_outputs: bool = False,
        auto_partition: Optional[bool] = False,
        fuse_microbatches: Optional[bool] = False,
        manual_pp_partition: Optional[bool] = False,
        manual_pp_stage_partition_fn: Optional[Callable] = None,
        manual_pp_loss_fn: Optional[Callable] = None,
        _delay_tracing: Optional[bool] = False,
        _turn_off_odd_even_scheduler: Optional[bool] = False,
        _all_reduce_send_recv: Optional[bool] = False,
        _fused_send_recv: Optional[bool] = False,
        _fused_fwd_bwd: Optional[bool] = False,
        _mark_step_before_pp_runtime: Optional[bool] = True,
        _mark_step_after_pp_runtime: Optional[bool] = True,
        _deallocate_send_tensors: Optional[bool] = True,
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

            virtual_pipeline_size:
                Virtual pipeline size, if greater than 1 we will use the interleaved pipeline schedule.

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

            use_model_wrapper
                Whether model wrapper is used.

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

            return_loss_on_cpu:
                Whether to return loss on cpu

            _turn_off_odd_even_scheduler:
                Turn off odd/even scheduler regardless of PP distributed configs

            _all_reduce_send_recv:
                Use all_reduce instead of all_gather for send/recv

            _fused_send_recv:
                Whether to fuse the send and recv graph, currently only appliable to interleave pipeline schedule

            _fused_fwd_bwd:
                Whether to fuse the forward and backward graph, currently only appliable to interleave pipeline schedule

            deallocate_pipeline_outputs:
                Whether to deallocate the pipeline outputs after send. After send the output tensor is only useful for its '.grad_fn' field, and not its '.data'.

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
                "Model parallelism needs to be initialzed before applying NxDPPModel wrapper. Please call neuronx_distributed.parallel_layers.initialize_model_parallel(pipeline_model_parallel_size, tensor_model_parallel_size)"
                # noqa: E501
            )
        if transformer_layer_cls is None and manual_pp_partition is False:
            raise ValueError("NxDPPModel requires transformer_layer_cls as input")
        self.original_torch_module = module

        # Add layer boundary markers in HLO
        neuron_layer(transformer_layer_cls)

        # Public configs
        self.output_loss_value_spec = output_loss_value_spec
        self.output_loss = True
        self.transformer_layer_cls = transformer_layer_cls
        self.num_microbatches = num_microbatches
        self.return_mb_loss = return_mb_loss
        self.broadcast_and_average_loss = broadcast_and_average_loss
        self.use_zero1_optimizer = use_zero1_optimizer
        self.use_model_wrapper = use_model_wrapper
        self.use_optimizer_wrapper = use_optimizer_wrapper
        self.return_loss_on_cpu = return_loss_on_cpu
        self.deallocate_pipeline_outputs = deallocate_pipeline_outputs
        self.virtual_pipeline_size = virtual_pipeline_size
        self.param_init_fn = param_init_fn
        self._all_reduce_send_recv = _all_reduce_send_recv
        self.fuse_microbatches = fuse_microbatches
        self.manual_pp_partition = manual_pp_partition
        self.manual_pp_loss_fn = manual_pp_loss_fn
        if self.fuse_microbatches:
            if self.return_loss_on_cpu:
                logger.warning("return_loss_on_cpu will be set to False when fuse_microbatches is set to True.")
            self.return_loss_on_cpu = False
        if self.output_loss_value_spec is None:
            self.output_loss_value_spec = False
            self.output_loss = False

        # Non-public config
        self._mark_step_before_pp_runtime = _mark_step_before_pp_runtime
        self._mark_step_after_pp_runtime = _mark_step_after_pp_runtime
        self._fused_fwd_bwd = _fused_fwd_bwd
        self._fused_send_recv = _fused_send_recv
        self._deallocate_send_tensors = _deallocate_send_tensors

        if self.broadcast_and_average_loss:
            assert not self.return_mb_loss, "When broadcast_and_average_loss is True return_mb_loss must be False"

        self.pipeline_parallel_size = (
            parallel_state.get_pipeline_model_parallel_size() if not _debug_mode else _debug_pp_size
        )
        self.pipeline_parallel_rank = (
            parallel_state.get_pipeline_model_parallel_rank() if not _debug_mode else _debug_pp_rank
        )
        self.serialization_manager = SerializationManager()

        self._metadata_comm_type = "gloo" if _use_gloo_for_metadata_comm else "tcp"
        if not parallel_state.is_tcp_store_available() and self._metadata_comm_type == "tcp":
            logger.warning("Can not get default tcp_store, fall back to use gloo for metadata communication")
            self._metadata_comm_type = "gloo"

        # Use Odd/Even schedule when num_microbatches==pipeline_parallel_size
        # Odd/Even schedule could reduce overhead of all first pp_size-1 ranks waiting for the last PP rank to send/recv
        # Each rank only needs to wait for it's prev/next rank with Odd/Even
        if self.num_microbatches == self.pipeline_parallel_size and not _turn_off_odd_even_scheduler:
            # Only enable when mb_size==pp_size since we see more bubble introduced during steady state in odd/even cases
            self.use_odd_even_scheduler = True
        else:
            self.use_odd_even_scheduler = False

        # Internal attributes for amp
        self._autocast_enabled = False
        self._autocast_dtype = None

        # Pipeline status
        self.traced_model = None
        self.paritioned_model = None
        self.pipeline_cuts: Optional[List[str]] = []
        self.partitioned = False
        self.model_moved_to_device = False
        self.shape_traced = False
        self.input_names: Optional[List[str]] = None
        # Outputs from analyze_pipeline_module
        self.stage_id_to_IO_input_names: Optional[Dict] = None
        self.stage_id_to_model_input_names = None
        self.stage_id_to_input_count = None
        self.stage_id_to_output_count = None
        # Static states of pipeline
        self.shared_weights_name_to_pg: Dict[str, dist.ProcessGroup] = {}
        self.manually_shared_weights: Dict[str, List[Any]] = {}
        self.local_name_to_original_name: Dict[str, str] = {}
        self.original_name_to_local_name: Dict[str, str] = {}
        self.local_stage_modules: List[Any] = []
        self._delay_tracing = _delay_tracing
        self.meta_device_parameter_map = None
        self.leaf_module_cls = leaf_module_cls
        self.autowrap_functions: List[Callable] = []
        if autowrap_functions is not None:
            self.autowrap_functions.extend(autowrap_functions)
        self.autowrap_modules = [*autowrap_modules] if autowrap_modules else []
        if autowrap_modules is not None:
            self.autowrap_modules.extend(autowrap_modules)
        self.autowrap_obj_methods = autowrap_obj_methods
        self.tracer_cls = tracer_cls
        self.meta_named_parameters: Dict[str, torch.nn.Parameter] = {}

        self.clear_minibatch_state()
        if not _debug_mode:
            self._set_distributed()
            # timeline needs to be created after init dist
            # TODO reenable timeline
            trace_file_path = None
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
            logger.info("Model transformer layers are: \n%s", model_layers)
            num_partitions = self.pipeline_parallel_size * self.virtual_pipeline_size
            pipeline_cuts = create_partitions(num_partitions, model_layers)
            logger.info("Pipeline cuts are: \n%s", pipeline_cuts)

        # If pipeline cuts are set and input names are provided, directly run tracing and partition
        if pipeline_cuts is not None and len(pipeline_cuts) > 0 and not self._delay_tracing:
            if input_names is None:
                raise ValueError(
                    "Input names are required for tracing when the NxD optimizer and model wrapper are not used"
                )

            self.trace(
                args=None,
                kwargs=None,
                input_names=input_names,
                leaf_modules=leaf_module_cls,
                autowrap_functions=self.autowrap_functions,
                autowrap_modules=self.autowrap_modules,
                autowrap_obj_methods=autowrap_obj_methods,
                tracer_cls=tracer_cls,
            )
            for pp_cut in pipeline_cuts:
                self.cut_pipeline_stage(pp_cut)
            self.partition()

        if manual_pp_partition:
            self._create_manual_partition(module, manual_pp_stage_partition_fn) # type: ignore
            logger.debug(
                rmsg(
                    f"running manual PP with output_loss {self.output_loss}"
                    f" return_mb_loss {self.return_mb_loss}, "
                    f"broadcast_and_average_loss {self.broadcast_and_average_loss}"
                )
            )

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.original_torch_module, name)

    def __repr__(self):
        """ This is for pretty printing the model for debugging"""
        res = super().__repr__()
        outputs = [
            res,
            f"local_stage_modules: {self.local_stage_modules}",
            f"traced_model: {self.traced_model}",
        ]
        return "\n".join(outputs)

    def _create_manual_partition(
        self, module: Union[List[torch.nn.Module], Tuple[torch.nn.Module]], partition_fn: Optional[Callable]
    ):

        # disable the tracing step for shape inference
        self.tracing = False
        self.model_labels_iter = {model_chunk_id: None for model_chunk_id in range(self.virtual_pipeline_size)}
        if self.manual_pp_loss_fn is not None:
            self.output_loss = True

        assert type(module) is list, "module should be a list of layers"
        pp_stage = PipelineStageModule(
            layers=module,
            num_stages=self.pipeline_parallel_size,
            stage_index=self.pipeline_parallel_rank,
            partition_fn=partition_fn,
        )

        self.local_stage_modules.append(pp_stage)
        logger.debug(
            rmsg(f'pipeline_parallel_rank {self.pipeline_parallel_rank} '
                    f'local_modules {[mod for mod in self.local_stage_modules]}')
        )
        self._create_pg_for_shared_weights(self.local_stage_modules)

        # ==== Post partition
        self.partitioned = True

        # This is for handling the stage to stage IO
        self.stage_id_to_IO_input_names = defaultdict()
        for stage_id in range(self.pipeline_parallel_size):
            self.stage_id_to_IO_input_names[stage_id] = OrderedDict()
            if stage_id > 0:
                # other than the first stage
                # it receives data from other pipeline stages
                # the key name is dummy for
                self.stage_id_to_IO_input_names[stage_id][f'stage_{stage_id}_input'] = PipelineIO(
                    name="stage_io_dummy_name",
                    input_idx=0,
                    output_idx=0,
                    metadata=None,
                    obj=list(),
                )

        # This handles the model inputs loaded from data loader
        self.stage_id_to_model_input_names = defaultdict() # type: ignore
        for stage_id in range(self.pipeline_parallel_size):
            if stage_id == 0: 
                # first stage loads inputs
                self.stage_id_to_model_input_names[stage_id] = {INPUTS_ARG_NAME: 0} # type: ignore
            elif stage_id == self.pipeline_parallel_size - 1:
                # last stage load the labels
                self.stage_id_to_model_input_names[stage_id] = {LABLES_ARG_NAME: 1} # type: ignore
            else:
                self.stage_id_to_model_input_names[stage_id] = {} # type: ignore

        # NOTE: output and input always a tuple/list
        self.stage_id_to_input_count = { # type: ignore
            self.pipeline_parallel_rank: 1
        }

        self.stage_id_to_output_count = { # type: ignore
            self.pipeline_parallel_rank: 1
        }

        # TODO: Locate shared weights between stages
        # self.register_shared_weights()

        # Create pipeline schedule
        self.create_schedule()
        self.move_model_to_device()
    
    def _create_pg_for_shared_weights(self, local_stage_modules: List[PipelineStageModule]):
        """ This is for creating ProcessGroup in manual PP mode. 
        Enumerate local_stage_modules, for each module check the 
        shared_weights_on_pp_stages and create the ProcessGroup
        """
        for module in local_stage_modules:
            for shared_group_name in module.shared_weights_on_pp_stages:
                sharing_info = module.shared_weights_on_pp_stages[shared_group_name]
                shared_pp_stages, shared_layer_weight_path = sharing_info
                pg = self._create_pg_with_ranks(shared_pp_stages)
                if self.pipeline_parallel_rank in shared_pp_stages:
                    assert pg is not None, f"ProcessGroup is None for shared weight group {shared_group_name}"
                    self.manually_shared_weights[shared_group_name] = [pg, shared_layer_weight_path]
                    logger.debug(
                        rmsg(f'Created ProcessGroup for shared weight group {shared_group_name} with pp ranks {shared_pp_stages}'
                    ))

    def clear_minibatch_state(self):
        """
        Initialize/clear the runtime placeholders
        """
        self.training = None
        self.losses = []
        self.last_stage_outputs = [] # allow raw outputs without loss compute
        self.losses_for_printing = []
        self.model_inputs_iter = {model_chunk_id: None for model_chunk_id in range(self.virtual_pipeline_size)}
        # Introduce the labels iterator for manual pipeline parallel case
        self.model_labels_iter = {model_chunk_id: None for model_chunk_id in range(self.virtual_pipeline_size)}
        self.input_name_to_iter_idx = {model_chunk_id: None for model_chunk_id in range(self.virtual_pipeline_size)}
        self.current_mb = -1
        self.current_model_chunk = -1
        self.should_graph_break = True
        # Name map to object that will pass along the stage for current microbatch
        # format name: mb: {name: (io, model_chunk)}
        self.mb_pass_along_io = {mb: {} for mb in range(self.num_microbatches)}
        # for tracing purpose
        self.mb_pass_along_io[-1] = {}
        # Inputs that requires grads for each mb
        self.mb_to_inputs_for_grads = {
            model_chunk_id: {mb: [] for mb in range(self.num_microbatches)}
            for model_chunk_id in range(self.virtual_pipeline_size)
        }
        # Outputs that requires grads for each mb
        self.mb_to_outputs_for_grads = {
            model_chunk_id: {mb: [] for mb in range(self.num_microbatches)}
            for model_chunk_id in range(self.virtual_pipeline_size)
        }
        # Used to pass the stage input from _fwd_preprocess_task to _fwd_step_task
        # format (inputs, mb, model_chunk)
        self.current_mb_stage_input = None
        # Used to pass the stage output from _fwd_step_task to _fwd_postprocess_task
        # format (outputs, mb, model_chunk)
        self.current_mb_stage_output = None
        # Used to pass the grads recvd from _bwd_preprocess_task to _bwd_step_task
        # format (grads, mb, model_chunk)
        self.current_mb_grads = None

        # Store the tensor reference and release after mark step so that the send graph can be captured
        # TODO Revisit this once we are going to merge graph across MBs
        self.send_tensors = []

    def _empty_send_tensor_buffer(self):
        """
        Empty the send tensor buffers. when in xla mode, 
        it puts the reset op in a callback, so that the caller of this function 
        don't need to wrap this function in a callback closure
        """
        def _clear_send_tensors():
            self.send_tensors = []

        if not cpu_mode():
            xm.add_step_closure(_clear_send_tensors)
        else:
            _clear_send_tensors()

    def trace(
        self,
        args: Optional[List[str]] = None,
        kwargs: Optional[Dict[Any, Any]] = None,
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
        if self.transformer_layer_cls is not None:
            leaf_modules.append(self.transformer_layer_cls.__name__)
        else:
            raise RuntimeError('Expected to have transformer_layer_cls specified, but got None')
        self.traced_model = trace_model(
            self.original_torch_module,
            args=args,
            kwargs=kwargs,
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
        """
        Partition the model for pipeline execution
        Notations
        - stage: the stage under the context of the pipeline, so each core can have more than one stage
        - partition: same as stage
        - model_chunk: the model that belongs to current stage
        Be aware that this is different compared with scheduler.py where stage is used as pipeline parallel rank
        """
        if self.partitioned:
            logging.warning("NxDPPModel.partition() is called while model is already partitioned, skipping...")
            return
        num_stages = 1 + (len(self.pipeline_cuts) if self.pipeline_cuts is not None else 0)

        if (self.pipeline_parallel_size * self.virtual_pipeline_size) != num_stages:
            raise ValueError(
                f"User cut stages {num_stages} mismatch the initialized pipeline parallel size {self.pipeline_parallel_size}"
                # noqa: E501
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

        assert num_stages == len(
            self.partitions
        ), f"pipeline_parallel_size {num_stages} != number submodules {len(self.paritioned_model)}"

        for i in range(num_stages):
            if stage_to_pipeline_parallel_rank(i) == self.pipeline_parallel_rank:
                self.local_stage_modules.append(self.partitions[i])
        if parallel_state.get_tensor_model_parallel_rank() == 0 and parallel_state.get_data_parallel_rank() == 0:
            logger.debug(
                rmsg(f"pipeline_parallel_rank {self.pipeline_parallel_rank} local_modules {self.local_stage_modules}")
            )

        self._post_partition(qualname_map)

        # Execute steps to move model to device in case of delayed tracing flow
        if get_delay_tracing(self):
            self._create_meta_parameter_map()
            self.maybe_materialize_local_module()
            self.move_model_to_device()
            self._get_meta_device_parameter_map()

        # Execute post partition hooks
        from neuronx_distributed.trainer import hooks

        hooks.execute_all_hooks(self)

    def _post_partition(self, qualname_map):
        # Create name mapping between original parameter to partitioned parameter
        self._build_parameter_buffer_name_mapping(qualname_map)
        self.partitioned = True

        # Locate shared weights between stages
        self.register_shared_weights()

        # Create pipeline schedule
        self.create_schedule()

        # grad is only enabled for local parameters
        self._disable_grad_for_nonlocal()

        # Materialize local module to CPU then move to XLA device
        if not self.use_model_wrapper:
            self.maybe_materialize_local_module()

    def create_schedule(self):
        self.eval_scheduler = InferenceSchedule(
            self.num_microbatches, self.pipeline_parallel_size, self.pipeline_parallel_rank
        )
        if self.virtual_pipeline_size == 1:
            self.train_scheduler = Train1F1BSchedule(
                self.num_microbatches, self.pipeline_parallel_size, self.pipeline_parallel_rank
            )
        else:
            self.train_scheduler = TrainInterleavedSchedule(
                self.num_microbatches,
                self.virtual_pipeline_size,
                self.pipeline_parallel_size,
                self.pipeline_parallel_rank,
                fused_send_recv=self._fused_send_recv,
                fused_fwd_bwd=self._fused_fwd_bwd,
                use_odd_even_scheduler=self.use_odd_even_scheduler,
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
        orginal_state_dict = self.original_torch_module.state_dict(keep_vars=True)
        qualname_map_local = {}

        for local_name, origin_name in qualname_map.items():
            assert local_name.startswith("submod_"), f"Found local module name with out submod_ prefix {local_name}"
            local_name_split = local_name.split(".")
            submod_split = local_name_split[0]
            stage = int(submod_split[len("submod_") :])
            if stage_to_pipeline_parallel_rank(stage) == self.pipeline_parallel_rank:
                # remove the submod_ prefix
                local_name = ".".join(local_name_split[1:])
                qualname_map_local[local_name] = origin_name
        for local_module in self.local_stage_modules:
            local_state_dict = local_module.state_dict(keep_vars=True)
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
            for name, rank in weights:
                if rank in ranks_:
                    raise RuntimeError(
                        rmsg(f"Rank {rank} double registered for shared parameter! shared_weights {shared_weights}")
                    )
                ranks_.append(rank)
                if rank == self.pipeline_parallel_rank:
                    current_rank_name = name
            pg = self._create_pg_with_ranks(ranks_)
            logger.debug(
                rmsg(f"Found shared weights with (name, rank): {(current_rank_name, self.pipeline_parallel_rank)}")
            )
            if current_rank_name is not None:
                self.shared_weights_name_to_pg[current_rank_name] = pg
                logger.info(
                    rmsg(
                        f"Register shared weight with (name, rank): {(current_rank_name, self.pipeline_parallel_rank)}"
                    )
                )

    def _reduce_manually_shared_weights(self):
        for shared_name, pg_and_weight in self.manually_shared_weights.items():
            pg, layer_and_weight_path = pg_and_weight
            layer, weight_path = layer_and_weight_path
            weight = getattr(layer, weight_path)

            logger.debug(rmsg(f"Reduce shared weight {shared_name}"))
            if weight.grad is not None:
                torch.distributed.all_reduce(weight.grad, group=pg)
            elif hasattr(weight, "main_grad"):
                torch.distributed.all_reduce(weight.main_grad, group=pg)

    def _reduce_shared_weights(self):
        if self.manual_pp_partition:
            return self._reduce_manually_shared_weights()

        for shared_name, pg in self.shared_weights_name_to_pg.items():
            for local_module in self.local_stage_modules:
                for n, p in local_module.named_parameters():
                    if shared_name == n and p.requires_grad:
                        assert p.grad is not None or hasattr(p, "main_grad"), f"Found shared weight {n} has None grad"
                        logger.debug(rmsg(f"Reduce shared weight {shared_name}"))
                        if p.grad is not None:
                            torch.distributed.all_reduce(p.grad, group=pg)
                        elif hasattr(p, "main_grad"):
                            torch.distributed.all_reduce(p.main_grad, group=pg)
                        break

    def _sync_manually_shared_weights(self):
        for shared_name, pg_and_weight in self.manually_shared_weights.items():
            pg, layer_and_weight_path = pg_and_weight
            layer, weight_path = layer_and_weight_path
            weight = getattr(layer, weight_path)
            logger.info(rmsg(f"Sync shared weight {shared_name}"))
            with torch.no_grad():
                if torch.distributed.get_rank(group=pg) != 0:
                    weight.data.zero_()
                torch.distributed.all_reduce(weight.data, group=pg)

    def _sync_shared_weights(self):
        if self.manual_pp_partition:
            return self._sync_manually_shared_weights()

        for shared_name, pg in self.shared_weights_name_to_pg.items():
            for local_module in self.local_stage_modules:
                for n, p in local_module.named_parameters():
                    if shared_name == n:
                        logger.info(rmsg(f"Sync shared weight {shared_name}"))
                        with torch.no_grad():
                            # Set parameter data to zeros except for the first rank in the shared group
                            if torch.distributed.get_rank(group=pg) != 0:
                                p.data.zero_()
                            torch.distributed.all_reduce(p.data, group=pg)
                        break

    def _create_meta_parameter_map(self):
        for n, p in self.named_parameters():
            self.meta_named_parameters[n] = p
        return self.meta_named_parameters

    def _get_meta_device_parameter_map(self):
        if self.meta_device_parameter_map is not None:
            return self.meta_device_parameter_map
        else:
            self.meta_device_parameter_map = {}
            for name, param in self.named_parameters():
                if param.device.type == "xla":
                    self.meta_device_parameter_map[self.meta_named_parameters[name]] = param
                else:
                    self.meta_device_parameter_map[self.meta_named_parameters[name]] = self.meta_named_parameters[name]
            return self.meta_device_parameter_map

    def _mark_pipeline_cuts(self, cut_point):
        # Internal API to mark the cut in the graph
        for node in self.traced_model.graph.nodes:
            if node.op == "call_module" and node.target == cut_point:
                node.meta["partition"] = True
                return
        raise RuntimeError(f"cut_point {cut_point} does not exist in the graph")

    def cut_pipeline_stage(self, cut_point):
        # [TODO] make sure cut point is only at transformer layers
        if self.pipeline_cuts is not None:
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
                    f"train/eval inputs ({set(kwargs.keys())}) must be same as the tracing input names {set(self.input_names)}"
                    # noqa: E501
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
        if not self.manual_pp_partition:
            self._prepare_inputs_and_infer_shape(kwargs)
        else:
            self._create_model_inputs_iter(kwargs)

    def _prepare_inputs_and_infer_shape(self, kwargs):
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

            for model_chunk in range(self.virtual_pipeline_size):
                self.tracing = True
                self.should_graph_break = False
                self.current_model_chunk = model_chunk
                # We trace with grads set to true so that we can create inputs for backward pass too.
                with torch.set_grad_enabled(True):
                    # Currently model input iters will be created and destoried for every model chunk during tracing
                    self._create_model_inputs_iter(kwargs)
                    self._fwd_preprocess_task()
                    self._fwd_step_task()
                    self._fwd_postprocess_task()
                    self.clear_minibatch_state()
                self._mark_step()
                self.tracing = False
                self.should_graph_break = True
            self.shape_traced = True
            end = time.time()
            logger.info(rmsg(f"Tensor shapes inference finished, total consumed time {end - start}s"))
            for i in range(self.virtual_pipeline_size):
                stage_id = self.get_current_stage(i)
                logger.debug(
                    rmsg(
                        f"After tracing model chunk {i}'s stage_id_to_IO_input_names {self.stage_id_to_IO_input_names[stage_id]}"
                        # noqa: E501
                    )
                )

        # Need to create input iters again since the old one is garbage collected
        self._create_model_inputs_iter(kwargs)

    def perform_delayed_tracing_and_partition(self, args, kwargs):
        # Perform tracing
        model_layers = self.get_model_layers(self.original_torch_module, self.transformer_layer_cls)
        num_partitions = self.pipeline_parallel_size * self.virtual_pipeline_size
        pipeline_cuts = create_partitions(num_partitions, model_layers)
        if pipeline_cuts is not None and len(pipeline_cuts) > 0:
            self.trace(
                args=args,
                kwargs=kwargs,
                leaf_modules=self.leaf_module_cls,
                autowrap_functions=self.autowrap_functions,
                autowrap_modules=self.autowrap_modules,
                autowrap_obj_methods=self.autowrap_obj_methods,
                tracer_cls=self.tracer_cls,
            )
            for pp_cut in pipeline_cuts:
                self.cut_pipeline_stage(pp_cut)

        # Perform partition
        if not self.partitioned:
            self.partition()

    def run_train(self, *args, **kwargs):
        if self._mark_step_before_pp_runtime:
            self._mark_step()
        self._autocast_enabled = torch.is_autocast_enabled()
        self._autocast_dtype = torch.get_autocast_gpu_dtype()

        if not self.partitioned:
            self.perform_delayed_tracing_and_partition(args, kwargs)

        with torch.cuda.amp.autocast(enabled=False):
            loss = self._run_train(**kwargs)
        self._autocast_enabled = False
        self._autocast_dtype = None
        if self._mark_step_after_pp_runtime:
            self._mark_step()
        return loss

    def run_eval(self, *args, **kwargs):
        if self._mark_step_before_pp_runtime:
            self._mark_step()
        self._autocast_enabled = torch.is_autocast_enabled()
        self._autocast_dtype = torch.get_autocast_gpu_dtype()

        if not self.partitioned:
            self.perform_delayed_tracing_and_partition({}, kwargs)

        with torch.cuda.amp.autocast(enabled=False):
            loss_or_outputs = self._run_eval(**kwargs)
        self._autocast_enabled = False
        self._autocast_dtype = None
        if self._mark_step_after_pp_runtime:
            self._mark_step()
        return loss_or_outputs

    def _get_last_stage_outputs(self):
        """"""
        if len(self.last_stage_outputs) == 0:
            return None
        else:
            return self.last_stage_outputs

    def _run_train(self, **kwargs):
        self._prepare_run(kwargs)

        self.training = True
        for local_module in self.local_stage_modules:
            local_module.train()
        self._exec_schedule(self.train_scheduler)
        loss = self._process_loss()
        self.clear_minibatch_state()
        self.timeline.mark_step_end()
        return loss

    def _run_eval(self, **kwargs):
        self._prepare_run(kwargs, train=False)

        self.training = False
        for local_module in self.local_stage_modules:
            local_module.eval()
        with torch.no_grad():
            self._exec_schedule(self.eval_scheduler)
            loss = self._process_loss()
            outputs = self._get_last_stage_outputs()
            self.clear_minibatch_state()
            self.timeline.mark_step_end()

            if self.output_loss:
                return loss
            else:
                if outputs is not None:
                    assert(len(outputs) == 1), "outputs should have only one element in eval mode"
                    return outputs[0]
                else:
                    logger.warning(rmsg("No outputs found in eval mode"))
                    return None

    def get_current_stage(self, model_chunk_id):
        """
        Each stage group contains pipeline_parallel_size stages and there are virtual_pipeline_size stages
        For example with PP4 VPP4
        stages   0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  16
        PP ranks P0->P1->P2->P3->P0->P1->P2->P3->P0->P1->P2->P3->P0->P1->P2->P3
        """
        return self.pipeline_parallel_size * model_chunk_id + self.pipeline_parallel_rank

    def is_last_pp_rank_last_model_chunk(self, model_chunk_id):
        return (
            self.pipeline_parallel_rank == self.pipeline_parallel_size - 1
            and model_chunk_id == self.virtual_pipeline_size - 1
        )

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
        assert batch_size
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
        if self.manual_pp_partition:
            # The modification is to minimize the diff to traced based implementation
            self._create_model_inputs_and_labels_iter(input_kwargs)
            return

        for model_chunk_id in range(self.virtual_pipeline_size):
            inputs_: List[torch.Tensor] = []
            input_name_to_iter_idx = {}
            stage_id = self.get_current_stage(model_chunk_id)
            for inp_name in self.stage_id_to_model_input_names[stage_id].keys():
                assert (
                    inp_name in input_kwargs
                ), f"stage {stage_id} requires model input {inp_name}, which is not provided in the input list {input_kwargs.keys()}"  # noqa: E501
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
                self.model_inputs_iter[model_chunk_id] = iter(self.get_batch_iterator(inputs_))
                self.input_name_to_iter_idx[model_chunk_id] = input_name_to_iter_idx

    def _create_model_inputs_and_labels_iter(self, input_kwargs):
        # for first stage and last stage to get the input data
        # NOTE: not care about the input kwargs' key, as we assume the
        # input is a single list/tuple of tensors
        assert len(input_kwargs) == 2, f"input_kwargs needs to be a dict with 2 keys: {INPUTS_ARG_NAME} and {LABLES_ARG_NAME}"
        assert INPUTS_ARG_NAME in input_kwargs and LABLES_ARG_NAME in input_kwargs, "inputs and labels are required"
        inputs_ = input_kwargs[INPUTS_ARG_NAME]
        labels_ = input_kwargs[LABLES_ARG_NAME]
        model_chunk_id = 0

        if isinstance(inputs_, torch.Tensor):
            inputs_ = [inputs_]
        if isinstance(labels_, torch.Tensor):
            labels_ = [labels_]

        # NOTE: no virtual pipeline support yet, alway model_chunk_id = 0
        self.model_inputs_iter[model_chunk_id] = iter(self.get_batch_iterator(inputs_))

        if labels_ is not None:
            self.model_labels_iter[model_chunk_id] = iter(self.get_batch_iterator(labels_))
        else:
            self.model_labels_iter[model_chunk_id] = None
        logger.debug(rmsg(f'setting up the model inputs and labels for model chunk {self.current_model_chunk}'))
        self.input_name_to_iter_idx[model_chunk_id] = {INPUTS_ARG_NAME: 0, LABLES_ARG_NAME: 1}

    def _handle_stage_outputs(self, outputs, stage):
        # If there is only a single output from graph
        # make output a list so the indexing will be right
        if self.stage_id_to_output_count[stage] == 1:
            outputs = [outputs]
        else:
            if len(outputs) != self.stage_id_to_output_count[stage]:
                raise RuntimeError(
                    f"Stage {stage} number outputs ({len(outputs)}) mismatches with compiled result ({self.stage_id_to_output_count[stage]})"
                    # noqa: E501
                )
        return outputs

    # Adopted from https://github.com/NVIDIA/Megatron-LM/blob/5f9c870f9f24b482509699d206a9dbb00958f6fc/megatron/core/pipeline_parallel/schedules.py#L107
    def maybe_deallocate_output_tensor(self, model_chunk, mb):
        """Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

        This method should be called right after the output tensor has been
        sent to the next pipeline stage. At this point, the output tensor is
        only useful for its '.grad_fn' field, and not its '.data'.
        """
        if not self.deallocate_pipeline_outputs or self.tracing:
            return
        for out in self.mb_to_outputs_for_grads[model_chunk][mb]:
            assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
            assert out._base is None, "counter-productive to free a view of another tensor."
            out._origin_size = out.size()
            out.data = torch.empty(
                (1,),
                device=out.device,
                dtype=out.dtype,
            )

    def custom_backward(self, outputs, grad_outputs):
        """Directly call C++ autograd engine.

        To make the 'maybe_deallocate_output_tensor' (above) optimization work, the C++
        autograd engine must be called directly, bypassing Pytorch's
        torch.autograd.backward. Pytorch's 'backward' checks that the output and
        grad have the same shape, while C++'s 'backward' does not.
        """

        for idx, (output, grad_output) in enumerate(zip(outputs, grad_outputs)):
            assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
            assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
            assert isinstance(grad_output, (torch.Tensor, type(None))), (
                "grad_output == '%s'." % type(grad_output).__name__
            )

            # Handle scalar output
            if grad_output is None:
                grad_outputs[idx] = torch.ones_like(
                    output,
                    memory_format=torch.preserve_format,
                )

        # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
        Variable._execution_engine.run_backward(
            tensors=tuple(outputs),
            grad_tensors=tuple(grad_outputs),
            keep_graph=False,
            create_graph=False,
            inputs=tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )

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
                        f"Running ForwardStepTask for mb {self.current_mb} but current_mb_stage_input contains mb {self.current_mb_stage_input[1]}"
                        # noqa: E501
                    )
                )
            if self.current_mb_stage_input[2] != self.current_model_chunk:
                raise RuntimeError(
                    rmsg(
                        f"Running ForwardStepTask for model chunk {self.current_model_chunk} but current_mb_stage_input contains model chunk {self.current_mb_stage_input[2]}"
                        # noqa: E501
                    )
                )
            if self.current_mb_stage_output is not None:
                raise RuntimeError(rmsg("Running ForwardStepTask but current_mb_stage_output is not None"))
            if len(self.mb_to_outputs_for_grads[self.current_model_chunk][self.current_mb]) != 0:
                raise RuntimeError(
                    "Running ForwardStepTask but mb_to_outputs_for_grads already contains outputs for current mb"
                )
        inputs = self.current_mb_stage_input[0]
        self.current_mb_stage_input = None

        # Run local pipeline stage
        self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardStep")

        # [TODO]: From Torch 2.1 we can specify device_type='xla' or use torch_xla's autocast, need to revisit this once we have PT2.1 support
        with torch.autocast(enabled=self._autocast_enabled, dtype=self._autocast_dtype, device_type="cuda"):
            outputs = self.local_stage_modules[self.current_model_chunk](*inputs)

        self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardStep")

        if self.is_last_pp_rank_last_model_chunk(self.current_model_chunk):
            if self.output_loss and self.output_loss_value_spec:
                # when self.output_loss_value_spec is provided,
                # it falls into the case of using tracing based approach
                current_mb_loss = find_loss_from_output_and_spec(
                    outputs, self.output_loss_value_spec
                )
                if current_mb_loss is None:
                    raise RuntimeError(
                        f"User provided output_loss_value_spec "
                        f"{self.output_loss_value_spec} failed to fetch the loss"
                    )
                self.losses.append(current_mb_loss)
                self.losses_for_printing.append(current_mb_loss.detach().clone())
            elif self.output_loss:
                # require the loss_fn to compute the loss
                loss_fn = self.manual_pp_loss_fn
                labels = self.current_mb_stage_label[0][0]
                self.current_mb_stage_label = None
                current_mb_loss = loss_fn(outputs[0], labels[0])
                self.losses.append(current_mb_loss)
                self.losses_for_printing.append(current_mb_loss.detach().clone())
            else:
                # if no loss outputs, get the last outputs and store it.
                # useful to get raw outputs in inference/evaluation
                self.last_stage_outputs.append(outputs)
        else:
            current_stage = self.get_current_stage(self.current_model_chunk)
            outputs = self._handle_stage_outputs(outputs, current_stage)
            self.current_mb_stage_output = (outputs, self.current_mb, self.current_model_chunk)
            if self.training and not self.tracing:
                # Need to collect the mb_to_outputs_for_grads here since _fwd_postprocess_task sometimes run after _bwd_preprocess_task   # noqa: E501
                next_stage = current_stage + 1
                # Current stage IO output will be next stage IO input
                current_stage_output_IO = self.stage_id_to_IO_input_names[next_stage]
                # Iterate through current_stage_output_IO to enforce tensor order between PP ranks
                for name, out in current_stage_output_IO.items():
                    pass_along_io = False
                    if out.output_idx is not None:
                        # Current stage outputs
                        current_output = outputs[out.output_idx]
                    else:
                        current_output, model_chunk = self.mb_pass_along_io[self.current_mb][name]
                        if model_chunk != self.current_model_chunk:
                            raise RuntimeError(
                                f"[forward step]Process a pass_along_io model chunk {model_chunk} that does not belong to current model chunk {self.current_model_chunk}"
                            )  # noqa
                        pass_along_io = True
                    _, tx_list, tensor_meta = self.serialization_manager.serialize(current_output)
                    for idx, t in enumerate(tx_list):
                        # [TODO] Add support, requires for cross attention
                        if pass_along_io and t.requires_grad:
                            raise RuntimeError(
                                f"Does not support tensors that require grads to pass along! IO name {name} current stage {next_stage - 1}"
                                # noqa: E501
                            )
                        logger.debug(
                            rmsg(
                                f"fwd mb {self.current_mb} model chunk {self.current_model_chunk} collect {name}'s {idx}th tensor meta {tensor_meta[idx]} for bwd"
                                # noqa: E501
                            )
                        )
                        # Collect the outputs that require grad for bwd
                        # Current stage output and next stage input should match exactly
                        if self.training and t.requires_grad:
                            self.mb_to_outputs_for_grads[self.current_model_chunk][self.current_mb].append(t)
        if self.should_graph_break:
            self._mark_step()
        self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardStepTask")

    def _bwd_step_task(self):
        """
        Major duties of for this task:
        - Run current stage forward step for current mb
        """
        self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardStepTask")
        # Last pipeline stage will directly do backprop from loss
        if self.is_last_pp_rank_last_model_chunk(self.current_model_chunk):
            self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardStep")
            scaled_loss = self.losses[self.current_mb] / self.num_microbatches
            self.losses[self.current_mb] = None
            scaled_loss.backward()
            if self.should_graph_break:
                self._mark_step()
            self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardStep")
            self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardStepTask")
            return

        if self.current_mb_grads is None:
            raise RuntimeError("Running BackwardStepTask but current_mb_grads is None")
        if self.current_mb_grads[1] != self.current_mb:
            raise RuntimeError(
                f"Running BackwardStepTask for mb {self.current_mb} but current_mb_grads contains mb {self.current_mb_grads[1]}"
                # noqa: E501
            )
        if self.current_mb_grads[2] != self.current_model_chunk:
            raise RuntimeError(
                f"Running BackwardStepTask for model chunk {self.current_model_chunk} but current_mb_grads contains model chunk {self.current_mb_grads[2]}"
                # noqa: E501
            )
        if len(self.mb_to_outputs_for_grads[self.current_model_chunk][self.current_mb]) == 0:
            raise RuntimeError(
                "Running BackwardStepTask but mb_to_outputs_for_grads is does not contain outputs for current mb"
            )
        grads = self.current_mb_grads[0]
        self.current_mb_grads = None
        outputs = self.mb_to_outputs_for_grads[self.current_model_chunk].pop(self.current_mb)

        self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardStep")
        # Run local pipeline stage
        if not self.deallocate_pipeline_outputs:
            torch.autograd.backward(outputs, grad_tensors=grads)
        else:
            self.custom_backward(outputs, grads)
        if self.should_graph_break:
            self._mark_step()
        self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardStep")
        self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardStepTask")

    def _manual_pp_preprocess_stage_io(self):
        self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardPreprocessTask")
        current_stage = self.get_current_stage(self.current_model_chunk)
        is_last_stage = self.is_last_pp_rank_last_model_chunk(self.current_model_chunk)
        is_first_stage = current_stage == 0
        inputs = [None]
        labels = [None]

        # load data from input data iterator
        if is_first_stage:
            # first stage and last load input data
            with mark_timeline(self.timeline, f"mb_{self.current_mb}_ForwardFetchInput"):
                assert self.model_inputs_iter[self.current_model_chunk] is not None, "model_inputs_iter is None"
                model_inputs = next(self.model_inputs_iter[self.current_model_chunk])
                inputs = [model_inputs]
        if is_last_stage:
            with mark_timeline(self.timeline, f"mb_{self.current_mb}_ForwardFetchLabels"):
                if self.manual_pp_loss_fn is not None:
                    assert self.model_labels_iter[self.current_model_chunk] is not None, "model_labels_iter is None"
                    model_labels = next(self.model_labels_iter[self.current_model_chunk])
                    labels = [model_labels]
                else:
                    labels = [None]

        if not is_first_stage:
            # recv activations from previous stage
            for name, inp in self.stage_id_to_IO_input_names[current_stage].items():
                if not self.shape_traced:
                    tensor_meta, py_obj = recv_python_object(method=self._metadata_comm_type)
                    logger.debug(rmsg(f"recv for name {name} tensor_meta {tensor_meta} py_obj {py_obj}"))
                    inp.metadata = tensor_meta
                    inp.obj = py_obj

                self._receive_stage_input_to_buffer(inputs, current_stage, name, inp.input_idx, inp.obj, inp.metadata)

        # the last stage need to use the labels as inputs
        self.current_mb_stage_input = (tuple(inputs), self.current_mb, self.current_model_chunk)
        self.current_mb_stage_label = (tuple(labels), self.current_mb, self.current_model_chunk)

        if self.should_graph_break:
            # NOTE: this mark_step call is important to avoid
            # incorrect HLO graph tracing:
            # if mark_step is not called, there will be redundant recv CC op
            self._mark_step()

        self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardPreprocessTask")

    def _receive_stage_input_to_buffer(self, 
                                       inputs_buffer: list, 
                                       current_stage: int,
                                       input_name: str, 
                                       input_idx: int, 
                                       py_obj, 
                                       metadata):
        recvd = []
        # Receive the current stage inputs from previous stage
        for idx, meta in enumerate(metadata):
            logger.debug(
                rmsg(
                    f"fwd mb {self.current_mb} model chunk "
                    f"{self.current_model_chunk} "
                    f"recv {input_name}'s {idx}th tensor meta {meta}"
                )
            )
            recvd_tensor = self.recv_op(meta, all_reduce_send_recv=self._all_reduce_send_recv)
            recvd.append(recvd_tensor)
            # Collect the inputs that require grad for bwd
            if self.training and meta.requires_grad:
                self.mb_to_inputs_for_grads[self.current_model_chunk][self.current_mb].append(recvd_tensor)

        inp_reconstructed = self.serialization_manager.deserialize(copy.deepcopy(py_obj), recvd)
        if input_idx is not None:
            # Current stage inputs
            assert inputs_buffer[input_idx] is None
            inputs_buffer[input_idx] = inp_reconstructed

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
        if self.manual_pp_partition:
            return self._manual_pp_preprocess_stage_io()

        self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardPreprocessTask")
        if not self.tracing:
            if self.current_mb_stage_input is not None:
                raise RuntimeError(rmsg("Running ForwardPreprocessTask but current_mb_stage_input is not None"))
            if len(self.mb_to_inputs_for_grads[self.current_model_chunk][self.current_mb]) != 0:
                raise RuntimeError(
                    rmsg(
                        "Running ForwardPreprocessTask but mb_to_inputs_for_grads already contains inputs for current mb"
                        # noqa: E501
                    )
                )

        current_stage = self.get_current_stage(self.current_model_chunk)
        inputs = [None] * self.stage_id_to_input_count[current_stage]

        # Get the model inputs from model_inputs_iter
        # TODO revisit the graph breaker from the parallel dataloder
        if self.model_inputs_iter[self.current_model_chunk] is not None:
            self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardFetchInput")
            model_inputs = next(self.model_inputs_iter[self.current_model_chunk])
            for name, idx in self.stage_id_to_model_input_names[current_stage].items():
                inputs[idx] = model_inputs[self.input_name_to_iter_idx[self.current_model_chunk][name]]
            self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardFetchInput")

        for name, inp in self.stage_id_to_IO_input_names[current_stage].items():
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
                logger.debug(
                    rmsg(
                        f"fwd mb {self.current_mb} model chunk {self.current_model_chunk} recv {name}'s {idx}th tensor meta {meta}"
                    )
                )
                recvd_tensor = self.recv_op(meta, all_reduce_send_recv=self._all_reduce_send_recv)
                recvd.append(recvd_tensor)
                # Collect the inputs that require grad for bwd
                if self.training and meta.requires_grad:
                    self.mb_to_inputs_for_grads[self.current_model_chunk][self.current_mb].append(recvd_tensor)
            inp_reconstructed = self.serialization_manager.deserialize(copy.deepcopy(inp.obj), recvd)
            if inp.input_idx is not None:
                # Current stage inputs
                assert inputs[inp.input_idx] is None
                inputs[inp.input_idx] = inp_reconstructed
            # If next stage also requires this input
            if (
                current_stage != (len(self.stage_id_to_IO_input_names) - 1)
                and name in self.stage_id_to_IO_input_names[current_stage + 1]
            ):
                # Pass along
                self.mb_pass_along_io[self.current_mb][name] = (inp_reconstructed, self.current_model_chunk)
            self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardRecv_{name}")
        self.current_mb_stage_input = (tuple(inputs), self.current_mb, self.current_model_chunk)
        if self.should_graph_break:
            self._mark_step()
        self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardPreprocessTask")

    def _fwd_postprocess_task(self):
        """
        Major duties of for this task:
        - Send the tensor shapes during tracing
        - Send the current stage IO to next stage
        """
        # Last stage do not need to send
        if self.is_last_pp_rank_last_model_chunk(self.current_model_chunk):
            return
        self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardPostprocessTask")

        if not self.tracing:
            if self.current_mb_stage_output is None:
                raise RuntimeError(rmsg("Running ForwardPostprocessTask but current_mb_stage_output is None"))
            if self.current_mb_stage_output[1] != self.current_mb:
                raise RuntimeError(
                    rmsg(
                        f"Running ForwardPostprocessTask for mb {self.current_mb} but current_mb_stage_output contains mb {self.current_mb_stage_output[1]}"
                        # noqa: E501
                    )
                )
            if self.current_mb_stage_output[2] != self.current_model_chunk:
                raise RuntimeError(
                    rmsg(
                        f"Running ForwardPostprocessTask for model chunk {self.current_model_chunk} but current_mb_stage_output contains model chunk {self.current_mb_stage_output[2]}"
                        # noqa: E501
                    )
                )
        outputs = self.current_mb_stage_output[0]
        self.current_mb_stage_output = None

        current_stage = self.get_current_stage(self.current_model_chunk)
        current_stage_output_IO = self.stage_id_to_IO_input_names[current_stage + 1]
        for name, out in current_stage_output_IO.items():
            self.timeline.mark_event_start(f"mb_{self.current_mb}_ForwardSend_{name}")
            if out.output_idx is not None:
                # Current stage outputs
                current_output = outputs[out.output_idx]
            else:
                # Tensors that needs to pass along
                if name not in self.mb_pass_along_io[self.current_mb]:
                    raise RuntimeError(
                        f"Pass along io {name} is missing, current_mb_pass_along_io {self.mb_pass_along_io[self.current_mb].keys()}"
                        # noqa: E501
                    )
                current_output, model_chunk = self.mb_pass_along_io[self.current_mb].pop(name)
                if model_chunk != self.current_model_chunk:
                    raise RuntimeError(
                        f"[Forward post]Process a pass_along_io model chunk {model_chunk} that does not belong to current model chunk {self.current_model_chunk}"
                    )
            obj_stripped_of_tensors, tx_list, tensor_meta = self.serialization_manager.serialize(current_output)
            if not self.shape_traced:
                logger.debug(rmsg(f"send tensor_meta {tensor_meta} py_obj {obj_stripped_of_tensors}"))
                send_python_object((tensor_meta, obj_stripped_of_tensors), method=self._metadata_comm_type)
            for idx, t in enumerate(tx_list):
                logger.debug(
                    rmsg(
                        f"fwd mb {self.current_mb} model chunk {self.current_model_chunk} send {name}'s {idx}th tensor meta {tensor_meta[idx]}"
                    )
                )
                tensor = self.send_op(t, all_reduce_send_recv=self._all_reduce_send_recv)
                if self._deallocate_send_tensors:
                    # We need to capture the send op, if we do not keep the output in scope XLA will not capture the op
                    # This will keep a scalar instead of the full tensor, to save memory
                    with torch.no_grad():
                        tensor = torch.sum(tensor)
                self.send_tensors.append(tensor)
            self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardSend_{name}")
        self._empty_send_tensor_buffer()
        # deallocate the stage output tensor once the send is finished
        self.maybe_deallocate_output_tensor(self.current_model_chunk, self.current_mb)
        if self.should_graph_break:
            self._mark_step()
        if len(self.mb_pass_along_io[self.current_mb]) != 0:
            raise RuntimeError(f"Unprocessed passing along io: {self.mb_pass_along_io[self.current_mb].keys()}")
        self.timeline.mark_event_end(f"mb_{self.current_mb}_ForwardPostprocessTask")

    def _bwd_preprocess_task(self):
        """
        Major duties of for this task:
        - Receive the grads for current mb backprop
        """
        self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardPreprocessTask")
        # Last stage do not need to recv
        if self.is_last_pp_rank_last_model_chunk(self.current_model_chunk):
            return None, None

        if self.current_mb_grads is not None:
            raise RuntimeError("Running BackwardPreprocessTask but current_mb_grads is not None")
        if len(self.mb_to_outputs_for_grads[self.current_model_chunk][self.current_mb]) == 0:
            raise RuntimeError(
                "Running BackwardPreprocessTask but mb_to_outputs_for_grads does not contain grads for current mb"
            )

        grads = []
        current_mb_outputs = self.mb_to_outputs_for_grads[self.current_model_chunk][self.current_mb]
        logger.debug(
            rmsg(
                f"bwd mb {self.current_mb} model chunk {self.current_model_chunk} recv grads count {len(current_mb_outputs)}"
            )
        )  # noqa
        for idx, t in enumerate(current_mb_outputs):
            self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardRecv_{idx}")
            shape = t.size() if not self.deallocate_pipeline_outputs else t._origin_size
            meta = TensorMeta(tensor_index=-1, dtype=t.dtype, shape=shape, requires_grad=False, device=t.device)
            logger.debug(rmsg(f"bwd mb {self.current_mb} model chunk {self.current_model_chunk} recv grad meta {meta}"))
            grads.append(self.recv_op(meta, recv_prev=False, all_reduce_send_recv=self._all_reduce_send_recv))
            self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardRecv_{idx}")
        self.current_mb_grads = (grads, self.current_mb, self.current_model_chunk)
        if self.should_graph_break:
            self._mark_step()
        self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardPreprocessTask")

    def _bwd_postprocess_task(self):
        """
        Major duties of for this task:
        - Send grads for previous rank for backprop
        """
        self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardPostprocessTask")
        if len(self.mb_to_inputs_for_grads[self.current_model_chunk][self.current_mb]) == 0:
            raise RuntimeError(
                "Running BackwardPostprocessTask but mb_to_inputs_for_grads does not contain inputs for current mb"
            )

        current_mb_inputs = self.mb_to_inputs_for_grads[self.current_model_chunk].pop(self.current_mb)
        logger.debug(
            rmsg(
                f"bwd mb {self.current_mb} model chunk {self.current_model_chunk} send grads count {len(current_mb_inputs)}"
            )
        )
        for idx, t in enumerate(current_mb_inputs):
            self.timeline.mark_event_start(f"mb_{self.current_mb}_BackwardSend_{idx}")
            if not t.requires_grad or t.grad is None:
                raise RuntimeError(rmsg("Backward sending grads, but get None"))
            logger.debug(
                rmsg(f"bwd mb {self.current_mb} model chunk {self.current_model_chunk} send grad shape {t.grad.size()}")
            )
            tensor = self.send_op(t.grad, send_next=False, all_reduce_send_recv=self._all_reduce_send_recv)
            if self.deallocate_pipeline_outputs:
                tensor = torch.sum(tensor)
            self.send_tensors.append(tensor)
            self.timeline.mark_event_end(f"mb_{self.current_mb}_BackwardSend_{idx}")
        self._empty_send_tensor_buffer()
        if self.should_graph_break:
            self._mark_step()
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

        if self.manual_pp_partition:
            # at the grad reduction stage, the shape info is recorded
            self.shape_traced = True

        if self.should_graph_break:
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
                self.current_model_chunk = task.model_chunk
                self.should_graph_break = task.graph_break and (not self.fuse_microbatches) and (not cpu_mode())
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

    def local_named_modules(
        self, memo: Optional[Set[nn.Module]] = None, prefix: str = "", remove_duplicate: bool = True
    ):
        self._validate_partitioned()
        for local_module in self.local_stage_modules:
            for n, m in local_module.named_modules(memo=memo, prefix=prefix, remove_duplicate=remove_duplicate):
                yield n, m

    def local_children(self):
        self._validate_partitioned()
        for n, c in self.local_named_children():
            yield c

    def local_named_children(self):
        self._validate_partitioned()
        for local_module in self.local_stage_modules:
            for n, c in local_module.named_children():
                yield n, c

    def local_parameters(self, recurse: bool = True):
        self._validate_partitioned()
        for n, p in self.local_named_parameters(recurse=recurse):
            yield p

    def local_named_parameters(self, recurse: bool = True):
        self._validate_partitioned()
        for local_module in self.local_stage_modules:
            for n, p in local_module.named_parameters(recurse=recurse):
                if not self.manual_pp_partition:
                    # remove wrapper prefix for checkpointing RMSNorm to avoid mapping key error
                    updated_name = n.replace(_CHECKPOINT_PREFIX, "")
                    original_name = self.local_name_to_original_name[updated_name]
                    yield original_name, p
                else:
                    yield n, p

    def local_buffers(self, recurse=True):
        self._validate_partitioned()
        for n, b in self.local_named_buffers(recurse=recurse):
            yield b

    def local_named_buffers(self, prefix: str = "", recurse: bool = True):
        self._validate_partitioned()
        for local_module in self.local_stage_modules:
            for name, buf in local_module.named_buffers(prefix=prefix, recurse=recurse):
                original_name = self.local_name_to_original_name[name]
                yield original_name, buf

    def construct_state_dict_per_model_chunk(self, origin_state_dict_all_model_chunks, strict=True):
        """
        Split the state_dict for different model chunks
        """
        local_state_dict = self.translate_origin_state_dict_to_local_state_dict(origin_state_dict_all_model_chunks)
        model_chunk_to_param_names = {
            idx: set(local_stage_module.state_dict().keys())
            for idx, local_stage_module in enumerate(self.local_stage_modules)
        }
        model_chunk_to_state_dict = defaultdict(dict)
        for name, p in local_state_dict.items():
            found = False
            for model_chunk_id, param_names in model_chunk_to_param_names.items():
                if name in param_names:
                    model_chunk_to_state_dict[model_chunk_id][name] = p
                    found = True
                    break
            if not found:
                msg_ = rmsg(
                    f"local parameter {name} is missing, current pp rank model_chunk_to_param_names {model_chunk_to_param_names}"
                )
                if strict:
                    raise RuntimeError(msg_)
                else:
                    logger.warning(msg_)
        return model_chunk_to_state_dict

    """
    When calling local_state_dict, all model chunks' state_dicts will be merged into a single one
    When calling load_state_dict, we will first split the input state_dict for each model chunk,
    each model chunk will only load its own state_dict
    """

    def load_state_dict(self, state_dict, strict=True):
        self._validate_partitioned()
        model_chunk_to_state_dict = self.construct_state_dict_per_model_chunk(state_dict, strict=strict)
        load_results = []
        for idx, local_stage_module in enumerate(self.local_stage_modules):
            load_results.append(local_stage_module.load_state_dict(model_chunk_to_state_dict[idx], strict=strict))

        # combine the load_results from all modules
        for i in range(1, len(load_results)):
            load_results[0].missing_keys.extend(load_results[i].missing_keys)
            load_results[0].unexpected_keys.extend(load_results[i].unexpected_keys)

        return load_results[0]

    def local_state_dict(self, *args, **kwargs):
        self._validate_partitioned()
        local_state_dict = {}
        prefix = kwargs.get("prefix", "")
        for local_stage_module in self.local_stage_modules:
            state_dict = local_stage_module.state_dict(*args, **kwargs)
            state_dict = self.translate_local_state_dict_to_origin_state_dict(state_dict, prefix)
            local_state_dict.update(state_dict)
        return local_state_dict

    def translate_origin_state_dict_to_local_state_dict(self, origin_state_dict):
        self._validate_partitioned()
        local_dict = {}
        for n, p in origin_state_dict.items():
            if n in self.original_name_to_local_name:
                local_name = self.original_name_to_local_name[n]
                local_dict[local_name] = p
        return local_dict

    def translate_local_state_dict_to_origin_state_dict(self, local_state_dict, prefix=""):
        self._validate_partitioned()
        origin_state_dict = {}
        for n, p in local_state_dict.items():
            n = n.replace(prefix, "", 1)
            if n in self.local_name_to_original_name:
                origin_name = f"{prefix}{self.local_name_to_original_name[n]}"
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

    def _mark_step(self) -> None:
        if not cpu_mode():
            xm.mark_step()

    def _create_pg_with_ranks(self, ranks: List[int]) -> torch.distributed.ProcessGroup:
        """
        Args: 
            ranks: pp stage ranks
        """
        pg = parallel_state.create_pg_with_ranks(ranks)
        return pg

    def _get_microbatch_dataloader(self, all_batches):
        if not cpu_mode():
            # NOTE: the `self.num_microbatches + 1` is to
            # trick the MpDeviceLoader to avoid adding mark_step in the last microbatch
            return MpDeviceLoader(all_batches, self._get_device(), batches_per_execution=self.num_microbatches + 1)
        else:
            return all_batches

    def maybe_materialize_local_module(self):
        """
        Check whether there's fake tensor in local module, if so materialize it to cpu
        """
        for local_module in self.local_stage_modules:
            maybe_materalize_model(local_module)
            if self.param_init_fn is not None:
                reinit_model(local_module, torch.device("cpu"), self.param_init_fn)

    def move_model_to_device(self):
        if not self.model_moved_to_device:
            for local_module in self.local_stage_modules:
                # Casting local module to xla device
                move_model_to_device(local_module, self._get_device())
                self._sync_shared_weights()
                # We no longer need this mark_step as we are guarding
                # the fwd-bwd inside its own
                # self._mark_step()
                self.model_moved_to_device = True

    def _process_loss(self):
        """
        Average the losses of microbatches if not self.return_mb_loss
        """
        if not self.output_loss:
            return None
        if self.pipeline_parallel_rank == self.pipeline_parallel_size - 1:
            loss_all = []
            for loss in self.losses_for_printing:
                loss_all.append(loss)
            if not self.return_mb_loss:
                if not self.broadcast_and_average_loss:
                    # cast to cpu so the following operation will not create extra graph
                    loss_all = [loss.detach().cpu() if self.return_loss_on_cpu else loss.detach() for loss in loss_all]
                loss = torch.sum(torch.stack(loss_all), dim=0) / len(loss_all)
                if self.broadcast_and_average_loss:
                    loss = self._average_loss_across_cp_ranks(loss)
                    loss = self._average_loss_across_dp_ranks(loss)
                    loss = self._bcast_loss_across_pp_ranks(loss)
                return loss.detach().cpu() if self.return_loss_on_cpu else loss.detach()
            else:
                return loss_all
        elif self.broadcast_and_average_loss:
            loss = torch.tensor(0.0, device=self._get_device())
            loss = self._bcast_loss_across_pp_ranks(loss)
            return loss.detach().cpu()
        return None

    def _average_loss_across_cp_ranks(self, loss):
        cp_size = parallel_state.get_context_model_parallel_size()
        if cp_size > 1:
            loss /= cp_size
            torch.distributed.all_reduce(loss, group=parallel_state.get_context_model_parallel_group())
        return loss

    def _average_loss_across_dp_ranks(self, loss):
        loss /= parallel_state.get_data_parallel_size()
        torch.distributed.all_reduce(loss, group=parallel_state.get_data_parallel_group())
        return loss

    def _bcast_loss_across_pp_ranks(self, loss):
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        # NOTE: the broadcast is not working in xla mode, it will return 0.0
        if cpu_mode():
            src_rank = torch.distributed.distributed_c10d.get_global_rank(
                pp_group, self.pipeline_parallel_size - 1)
            torch.distributed.broadcast(
                loss, src_rank, group=parallel_state.get_pipeline_model_parallel_group()
            )
        else:
            torch.distributed.all_reduce(loss, group=pp_group)

        if self.should_graph_break:
            self._mark_step()
        return loss

    def _get_device(self,):
        if cpu_mode():
            return torch.device("cpu")
        else:
            return xm.xla_device()

    def is_last_stage(self):
        rst = self.is_last_pp_rank_last_model_chunk(self.virtual_pipeline_size - 1)
        return rst
