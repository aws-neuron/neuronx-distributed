import copy
import logging
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
    partition_traced_model,
)
from neuronx_distributed.pipeline.scheduler import InferenceSchedule, TrainSchedule
from neuronx_distributed.pipeline.trace import trace_model
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.utils.model_utils import (
    maybe_materalize_model,
    move_model_to_device,
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
        pipeline_cuts: Optional[List[str]] = None,
        input_names: Optional[List[str]] = None,
        leaf_module_cls: Optional[List[Any]] = None,
        autowrap_functions: Optional[Tuple[ModuleType]] = None,
        autowrap_modules: Optional[Tuple[Callable, ...]] = None,
        tracer_cls: Optional[Union[str, Any]] = None,
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

            tracer_cls:
                User provided tracer class for symbolic tracing. It can be "hf", "torch" or any tracer class
                user created.


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
        if not parallel_state.model_parallel_is_initialized():
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
        self.shape_traced = False
        self.return_mb_loss = return_mb_loss
        self.output_loss_value_spec = output_loss_value_spec
        self.transformer_layer_cls = transformer_layer_cls
        self.num_microbatches = num_microbatches
        self.pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_size()
        self.pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.serialization_manager = SerializationManager()
        self.input_names = None

        # Outputs from analyze_pipeline_module
        self.stage_id_to_IO_input_names = None
        self.stage_id_to_model_input_names = None
        self.stage_id_to_input_count = None
        self.stage_id_to_output_count = None

        self.shared_weights_name_to_pg = {}
        self.clear_minibatch_state()
        self._set_distributed()
        # If user set up the pipeline cuts in config, directly run tracing and partition
        if pipeline_cuts is not None and len(pipeline_cuts) > 0:
            self.trace(
                input_names=input_names,
                leaf_modules=leaf_module_cls,
                autowrap_functions=autowrap_functions,
                autowrap_modules=autowrap_modules,
                tracer_cls=tracer_cls,
            )
            for pp_cut in pipeline_cuts:
                self.cut_pipeline_stage(pp_cut)
            self.partition()

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
            tracer_cls=tracer_cls,
        )

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
        self.paritioned_model = partition_traced_model(self.traced_model)
        (
            self.stage_id_to_IO_input_names,
            self.stage_id_to_model_input_names,
            self.stage_id_to_input_count,
            self.stage_id_to_output_count,
        ) = analyze_pipeline_module(self.paritioned_model)
        self.partitioned = True

        # Extract the local module
        self.partitions = []
        for _, module in self.paritioned_model.named_children():
            self.partitions.append(module)
        self.register_shared_weights()
        assert self.pipeline_parallel_size == len(
            self.partitions
        ), f"pipeline_parallel_size {self.pipeline_parallel_size} != number submodules {len(self.paritioned_model)}"
        self.local_module = self.partitions[self.pipeline_parallel_rank]
        if parallel_state.get_tensor_model_parallel_rank() == 0 and parallel_state.get_data_parallel_rank() == 0:
            logger.debug(
                rmsg(f"pipeline_parallel_rank {self.pipeline_parallel_rank} local_module {self.local_module }")
            )
        self.eval_scheduler = InferenceSchedule(
            self.num_microbatches, self.pipeline_parallel_size, self.pipeline_parallel_rank
        )
        self.train_scheduler = TrainSchedule(
            self.num_microbatches, self.pipeline_parallel_size, self.pipeline_parallel_rank
        )
        logger.debug(rmsg(f"eval_schedule {[task for task in self.eval_scheduler.steps()]}"))
        logger.debug(rmsg(f"train_schedule {[task for task in self.train_scheduler.steps()]}"))

        # Materialize local module to CPU then move to XLA device
        self._maybe_materialize_local_module()

        self._disable_grad_for_nonlocal()

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
            for n, p in self.local_named_parameters():
                if shared_name == n and p.requires_grad:
                    assert p.grad is not None, f"Found shared weight {n} has None grad"
                    logger.debug(rmsg(f"Reduce shared weight {shared_name}"))
                    torch.distributed.all_reduce(p.grad, group=pg)
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
        for p in self.parameters():
            if p not in local_p:
                p.requires_grad = False
                p.grad = None

    def _maybe_materialize_local_module(self):
        """
        Check whether there's fake tensor in local module, if so materialize it to cpu
        """
        maybe_materalize_model(self.local_module)
        # Casting local module to xla device
        move_model_to_device(self.local_module, xm.xla_device())
        self._mark_step()

    def _prepare_run(self, kwargs, train=True):
        self._validate_partitioned()
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
            logger.info(rmsg("Tensor shapes inference finished..."))
            logger.debug(
                rmsg(
                    f"After tracing current stage's stage_id_to_IO_input_names {self.stage_id_to_IO_input_names[self.pipeline_parallel_rank]}"  # noqa: E501
                )
            )

        # Need to create input iters again since the old one is garbage collected
        self._create_model_inputs_iter(kwargs)

    def _process_loss(self):
        """
        Average the losses of microbatches if not self.return_mb_loss
        """
        if self.pipeline_parallel_rank == self.pipeline_parallel_size - 1:
            loss_all = []
            for loss in self.losses:
                loss_all.append(loss.detach().cpu())
            if not self.return_mb_loss:
                loss = torch.sum(torch.stack(loss_all), dim=0) / len(loss_all)
                return loss
            else:
                return loss_all
        return None

    def run_train(self, **kwargs):
        self._prepare_run(kwargs)

        self.training = True
        self.local_module.train()
        self._exec_schedule(self.train_scheduler)
        loss = self._process_loss()
        self.clear_minibatch_state()
        return loss

    def run_eval(self, **kwargs):
        self._prepare_run(kwargs, train=False)

        self.training = False
        self.local_module.eval()
        with torch.no_grad():
            self._exec_schedule(self.eval_scheduler)
            loss = self._process_loss()
            self.clear_minibatch_state()
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
            assert len(microbatch) > 0, "Microbatch lenght less than 0"
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

    def _fwd_step_task(self):
        """
        Major duties of for this task:
        - Run current stage forward step for current mb
        - Collect loss for the last pipeline stage
        - Collect the outputs that require grad for bwd
        """
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
        outputs = self.local_module(*inputs)

        if not self.tracing:
            self._mark_step()

        if self.pipeline_parallel_rank == self.pipeline_parallel_size - 1:
            # [TODO] Add inference support
            current_mb_loss = find_loss_from_output_and_spec(outputs, self.output_loss_value_spec)
            if current_mb_loss is None:
                raise RuntimeError(
                    f"User provided output_loss_value_spec {self.output_loss_value_spec} failed to fetch the loss"
                )
            self.losses.append(current_mb_loss)
        else:
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

    def _bwd_step_task(self):
        """
        Major duties of for this task:
        - Run current stage forward step for current mb
        """
        # Last pipeline stage will directly do backprop from loss
        if self.pipeline_parallel_rank == self.pipeline_parallel_size - 1:
            scaled_loss = self.losses[self.current_mb] / self.num_microbatches
            scaled_loss.backward()
            self._mark_step()
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

        # Run local pipeline stage
        torch.autograd.backward(outputs, grad_tensors=grads)
        self._mark_step()

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
            model_inputs = next(self.model_inputs_iter)
            for name, idx in self.stage_id_to_model_input_names[self.pipeline_parallel_rank].items():
                inputs[idx] = model_inputs[self.input_name_to_iter_idx[name]]

        for name, inp in self.stage_id_to_IO_input_names[self.pipeline_parallel_rank].items():
            if not self.shape_traced:
                # Suppose to receive List(TensorMeta), python obj
                tensor_meta, py_obj = recv_python_object()
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
        self.current_mb_stage_input = (tuple(inputs), self.current_mb)

    def _fwd_postprocess_task(self):
        """
        Major duties of for this task:
        - Send the tensor shapes during tracing
        - Send the current stage IO to next stage
        """
        # Last stage do not need to send
        if self.pipeline_parallel_rank == self.pipeline_parallel_size - 1:
            return

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

        # If there is only a single output
        if self.stage_id_to_output_count[self.pipeline_parallel_rank] == 1:
            outputs = [outputs]
        else:
            if len(outputs) != self.stage_id_to_output_count[self.pipeline_parallel_rank]:
                raise RuntimeError(
                    f"Stage {self.pipeline_parallel_rank} number outputs ({len(outputs)}) mismatches with compiled result ({self.stage_id_to_output_count[self.pipeline_parallel_rank]})"  # noqa: E501
                )

        current_stage_output_IO = self.stage_id_to_IO_input_names[self.pipeline_parallel_rank + 1]
        for name, out in current_stage_output_IO.items():
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
                send_python_object((tensor_meta, obj_stripped_of_tensors))
            for idx, t in enumerate(tx_list):
                logger.debug(rmsg(f"fwd mb {self.current_mb} send {name}'s {idx}th tensor meta {tensor_meta[idx]}"))
                self.send_op(t, tracing=self.tracing)
        if len(self.current_mb_pass_along_io) != 0:
            raise RuntimeError(f"Unprocessed passing along io: {self.current_mb_pass_along_io.keys()}")

    def _bwd_preprocess_task(self):
        """
        Major duties of for this task:
        - Receive the grads for current mb backprop
        """
        # Last stage do not need to recv
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
        for t in current_mb_outputs:
            meta = TensorMeta(tensor_index=-1, dtype=t.dtype, shape=t.size(), requires_grad=False, device=t.device)
            logger.debug(rmsg(f"bwd mb {self.current_mb} recv grad meta {meta}"))
            grads.append(self.recv_op(meta, recv_prev=False))
        self.current_mb_grads = (grads, self.current_mb)

    def _bwd_postprocess_task(self):
        """
        Major duties of for this task:
        - Send grads for previous rank for backprop
        """
        if len(self.mb_to_inputs_for_grads[self.current_mb]) == 0:
            raise RuntimeError(
                "Running BackwardPostprocessTask but mb_to_inputs_for_grads does not contain inputs for current mb"
            )

        current_mb_inputs = self.mb_to_inputs_for_grads.pop(self.current_mb)
        logger.debug(rmsg(f"bwd mb {self.current_mb} send grads count {len(current_mb_inputs)}"))
        for t in current_mb_inputs:
            if not t.requires_grad or t.grad is None:
                raise RuntimeError(rmsg("Backward sending grads, but get None"))
            logger.debug(rmsg(f"bwd mb {self.current_mb} send grad shape {t.grad.size()}"))
            self.send_op(t.grad, send_next=False)

    def _reduce_grads_task(self):
        """
        Major duties of for this task:
        - Average the grads acorss data parallel ranks
        - Add the grads for shared weights
        """
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

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training."""
        raise RuntimeError("Only run_train() and run_eval() are accessible when pipeline parallelism is enabled.")

    def load_state_dict(self, state_dict):
        self._validate_partitioned()
        print("loading state dict")
        self.local_module.load_state_dict(state_dict)

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
    [TODO] translate the module/parameter names
    """

    def local_modules(self):
        self._validate_partitioned()
        for m in self.local_module.modules():
            yield m

    def local_named_modules(self, memo: Optional[Set[nn.Module]] = None, prefix: str = ""):
        self._validate_partitioned()
        for n, m in self.local_module.named_modules(memo=memo, prefix=prefix):
            yield n, m

    def local_parameters(self, recurse: bool = True):
        self._validate_partitioned()
        for n, p in self.local_named_parameters(recurse=recurse):
            yield p

    def local_named_parameters(self, recurse: bool = True):
        self._validate_partitioned()
        for n, p in self.local_module.named_parameters(recurse=recurse):
            yield n, p

    def local_named_buffers(self, prefix: str = "", recurse: bool = True):
        self._validate_partitioned()
        for name, buf in self.local_module.named_buffers(prefix=prefix, recurse=recurse):
            yield name, buf

    def local_buffers(self, recurse=True):
        self._validate_partitioned()
        for n, b in self.local_named_buffers(recurse=recurse):
            yield b

    """
    Below methods can be overwritten to support non-XLA devices
    """

    def _set_distributed(self):
        parallel_state.set_gloo_group()
        self.send_op = send
        self.recv_op = recv_from

    def _mark_step(self):
        xm.mark_step()

    def _create_pg_with_ranks(self, ranks):
        pg = parallel_state.create_pg_with_ranks(ranks)
        return pg

    def _get_microbatch_dataloader(self, all_batches):
        return MpDeviceLoader(all_batches, xm.xla_device(), batches_per_execution=self.num_microbatches)
