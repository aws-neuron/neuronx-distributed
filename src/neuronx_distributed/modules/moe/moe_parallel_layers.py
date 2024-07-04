import math
from typing import Optional, Tuple

import torch
import torch.distributed
from torch import Tensor

from neuronx_distributed.modules.moe.model_utils import MoESequenceParallelMode
from neuronx_distributed.parallel_layers import layers, mappings, parallel_state, utils


class ExpertFusedLinearWithAsyncCommunication(torch.autograd.Function):
    """Linear layer execution with asynchronous communication, which handles the 3D weight tensor required for
    Mixture of Experts.

    The implementation largely mimics LinearWithAsyncCommunication, but is modified for the 3D weights.
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        async_grad_allreduce: bool,
        sequence_parallel_enabled: bool,
        save_for_backwards: bool = True,
    ):
        if bias is not None:
            raise NotImplementedError("Bias is not currently supported for MoE")
        if sequence_parallel_enabled:
            raise NotImplementedError("Since ExpertMLPs is executed only in TP mode, SP is not implemented")

        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel_enabled = sequence_parallel_enabled
        ctx.compute_weight_gradient = weight.requires_grad

        # E: num_experts, C: expert_capacity, H: input_size, I: intermediate/output_size
        # input: (E, C, H), weight: (E, H, I)

        if save_for_backwards:
            if ctx.compute_weight_gradient:
                ctx.save_for_backward(input, weight)
            else:
                ctx.save_for_backward(weight)

        # output: (E, C, I)
        output = torch.matmul(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # grad_output: (E, C, I)

        # input: (E, C, H), weight: (E, H, I)
        if ctx.compute_weight_gradient:
            input, weight = ctx.saved_tensors
        else:
            weight = ctx.saved_tensors[0]
            input = None

        # grad_input: (E, C, H)
        grad_input = grad_output.matmul(weight.transpose(-1, -2))

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input,
                group=parallel_state.get_tensor_model_parallel_group(),
            )

        # if no weight gradient, immediately return
        if not ctx.compute_weight_gradient:
            return grad_input, None, None, None, None, None, None

        # grad_weight: (E, H, I)
        grad_weight = torch.matmul(input.transpose(-1, -2), grad_output)

        return grad_input, grad_weight, None, None, None, None, None


class ExpertFusedColumnParallelLinear(layers.ColumnParallelLinear):
    """Specialized linear layer for MoE, supporting column parallelism for all experts simultaneously.

    This class inherits from ColumnParallelLinear, and over-rides certain attributes and functions needed to enable
    column-parallel linear layer computation for 3D weights.

    Bias is not currently supported for MoE.
    Sequence parallelism is handled independently of MLP computations in MoE, and therefore defaults to False.
    """

    autograd_func_class = ExpertFusedLinearWithAsyncCommunication

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        gather_output: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        stride: int = 1,
        init_method: torch.nn.init = None,
        keep_master_weight: bool = False,
    ):
        self.num_experts = num_experts

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            gather_output=gather_output,
            dtype=dtype,
            device=device,
            stride=stride,
            init_method=init_method,
            sequence_parallel_enabled=False,
            keep_master_weight=keep_master_weight,
            skip_bias_add=False,
        )

    def set_weight_and_bias_config(self):
        # Define 3D weight tensor, one linear layer per expert
        self.weight_shape = (self.num_experts, self.input_size, self.output_size_per_partition)
        # Column parallel partitioning for each expert
        self.weight_partition_dim = 2
        self.bias_shape = None

    def _init_weight(self, weight):
        # Initialize the linear layer of each expert separately
        assert len(weight.shape) == 3 and weight.shape[0] == self.num_experts
        for e in range(weight.shape[0]):
            if self.arg_init_method is None:
                torch.nn.init.kaiming_uniform_(weight[e], a=math.sqrt(5))
            else:
                self.arg_init_method(weight[e])


class ExpertFusedRowParallelLinear(layers.RowParallelLinear):
    """Specialized linear layer for MoE, supporting row parallelism for all experts simultaneously.

    This class inherits from RowParallelLinear, and over-rides certain attributes and functions needed to enable
    row-parallel linear layer computation for 3D weights. The forward pass of the parent class is over-ridden
    to avoid the output all-reduce depending on the sequence parallel mode.

    Bias is not currently supported for MoE.
    Sequence parallelism is handled independently of MLP computations in MoE, and therefore defaults to False.
    """

    autograd_func_class = ExpertFusedLinearWithAsyncCommunication

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        sequence_parallel_mode: MoESequenceParallelMode,
        input_is_parallel: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        stride: int = 1,
        init_method: torch.nn.init = None,
        keep_master_weight: bool = False,
    ):
        self.num_experts = num_experts
        if sequence_parallel_mode not in MoESequenceParallelMode:
            raise TypeError(f"Unknown sequence_parallel_mode: {sequence_parallel_mode}")
        self.sequence_parallel_mode = sequence_parallel_mode

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            input_is_parallel=input_is_parallel,
            dtype=dtype,
            device=device,
            stride=stride,
            init_method=init_method,
            sequence_parallel_enabled=False,
            keep_master_weight=keep_master_weight,
            skip_bias_add=False,
        )

    def set_weight_and_bias_config(self):
        # Define 3D weight tensor, one linear layer per expert
        self.weight_shape = (self.num_experts, self.input_size_per_partition, self.output_size)
        # Row parallel partitioning for each expert
        self.weight_partition_dim = 1
        self.bias_shape = None

    def _init_weight(self, weight):
        # Initialize the linear layer of each expert separately
        assert len(weight.shape) == 3 and weight.shape[0] == self.num_experts
        for e in range(weight.shape[0]):
            if self.arg_init_method is None:
                torch.nn.init.kaiming_uniform_(weight[e], a=math.sqrt(5))
            else:
                self.arg_init_method(weight[e])

    def forward(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel_enabled
            input_parallel = mappings.scatter_to_tensor_model_parallel_region(input_)

        # Matrix multiply.
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
            autograd_func_class=self.autograd_func_class,
        )

        if self.sequence_parallel_mode == MoESequenceParallelMode.EXIT_SP_ON_ENTRY_DELAY_MLP_AR:
            # Avoid the output all-reduce, in favor of a reduce-scatter at the end of the MoE layer instead
            output = output_parallel
        else:
            # All-reduce across all the partitions.
            output = mappings.reduce_from_tensor_model_parallel_region(output_parallel)

        return output


class LinearWithParallelInput(torch.autograd.Function):
    """Linear layer execution where the input is potentially parallel.
    Implements an all-reduce of weight gradients in the backward pass if necessary.
    """

    @staticmethod
    def forward(ctx, input, weight, reduce_weight_grad):
        # input: (S, B, H), weight: (E, H)
        ctx.reduce_weight_grad = reduce_weight_grad
        assert weight.requires_grad
        ctx.save_for_backward(input, weight)
        # output: (S, B, E)
        output = torch.matmul(input, weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (S, B, E)
        input, weight = ctx.saved_tensors
        reduce_weight_grad = ctx.reduce_weight_grad
        # grad_input: (S, B, H)
        grad_input = grad_output.matmul(weight)
        # grad_weight: (E, H)
        grad_weight = torch.einsum("sbe,sbh->eh", grad_output, input)
        if reduce_weight_grad and parallel_state.get_tensor_model_parallel_size() > 1:
            # All-reduce the gradients of the weight
            torch.distributed.all_reduce(grad_weight, group=parallel_state.get_tensor_model_parallel_group())
        return grad_input, grad_weight, None


class InputParallelLinear(torch.nn.Module):
    """Linear layer where the input is potentially in parallel.
    Used for defining the router in MoE when in certain SP modes. See routing.py for details.

    Arguments:
        input_size: Dimensionality of the input to the linear layer.
        output_size: Dimensionality of the output of the linear layer.
        reduce_weight_grad: Whether to all-reduce the gradients of the weights in the backward pass.
        dtype: Datatype for the layer weights.
        device: Device for the layer weights.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        reduce_weight_grad: bool,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        # Initialize the linear layer weight
        if device is None or device.type == "cpu":
            self.weight = torch.nn.Parameter(torch.empty(output_size, input_size, dtype=dtype))
        else:
            assert self.device.type == "xla", "Currently only xla device type is supported"
            self.weight = torch.nn.Parameter(torch.empty(output_size, input_size, device=device, dtype=dtype))

        if self.weight.device != torch.device("meta"):
            self.init_weight_cpu()

        self.reduce_weight_grad = reduce_weight_grad

    def forward(self, input):
        """Lightweight wrapper around the LinearWithParallelInput autograd function."""
        args = utils.cast_if_autocast_enabled(input, self.weight, self.reduce_weight_grad)
        utils.verify_casted_dtype(args)
        with torch.cuda.amp.autocast(enabled=False):
            return LinearWithParallelInput.apply(*args)

    def init_weight_cpu(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
