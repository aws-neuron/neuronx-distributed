from typing import Optional, Tuple
import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch_xla.core.xla_model as xm
from torch import nn
from torch.nn.parameter import Parameter

from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)
from .random import get_xla_rng_tracker
from .utils import EmbeddingUtility, divide

if "reduce_scatter_tensor" not in dir(torch.distributed):
    torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base
if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def param_is_not_tensor_parallel_duplicate(param: torch.Tensor) -> bool:
    return (
        hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
    ) or (get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(
    tensor: torch.Tensor, is_parallel: bool, dim: int, stride: int = 1
) -> None:
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(
    tensor: torch.Tensor,
) -> None:
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(
    destination_tensor: torch.Tensor, source_tensor: torch.Tensor
) -> None:
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_neuron(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on Neuron device.

    Args:
        weight (Parameter):
        init_method (Callable[[Tensor], None]): Taking a Tensor and initialize its elements.
        partition_dim (int): Dimension to apply partition.
    """

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    with get_xla_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    return_master_weight=False,
    *,
    params_dtype=torch.float32,
    stride=1,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    # Initialize master weight
    master_weight = Parameter(
        torch.empty(output_size, input_size, dtype=torch.float, requires_grad=False)
    )

    init_method(master_weight)

    master_weight = master_weight.to(dtype=params_dtype)
    
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(
        master_weight, per_partition_per_stride_size, dim=partition_dim
    )
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init_method: torch.nn.init = init.normal_,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.tensor_model_parallel_size = get_tensor_model_parallel_size()
        # Divide the weight matrix along the vocabulary dimension.
        (
            self.start_index,
            self.end_index,
        ) = EmbeddingUtility.range_from_global_vocab_size(
            self.num_embeddings,
            get_tensor_model_parallel_rank(),
            self.tensor_model_parallel_size,
        )
        self.num_embeddings_per_partition = self.end_index - self.start_index

        # Allocate weights and initialize.
        if device is None or device.type == "cpu":
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    dtype=dtype,
                )
            )
            _initialize_affine_weight_cpu(
                self.weight,
                self.num_embeddings,
                self.embedding_dim,
                self.num_embeddings_per_partition,
                0,
                init_method,
                params_dtype=dtype,
            )
        else:
            assert device.type == "xla", "Currently only xla device type is supported"
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=device,
                    dtype=dtype,
                )
            )
            _initialize_affine_weight_neuron(self.weight, init_method, partition_dim=0)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            input_mask = (input_ >= self.start_index) & (input_ < self.end_index)
            # Mask the input.
            masked_input = input_.clone() - self.start_index
            masked_input = torch.mul(masked_input, input_mask.long())
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(
            masked_input.long(),
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel = torch.mul(
                output_parallel, torch.unsqueeze(input_mask.float(), dim=-1)
            )
        # Reduce across all the model parallel Neuron devices.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class LinearWithAsyncCommunication(torch.autograd.Function):
    """Linear layer execution with asynchronous communication."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        async_grad_allreduce: bool,
    ):
        ctx.use_bias = bias is not None and weight.requires_grad
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.compute_weight_gradient = weight.requires_grad

        if ctx.compute_weight_gradient:
            ctx.save_for_backward(input, weight)
        else:
            ctx.save_for_backward(weight)

        total_input = input
        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.compute_weight_gradient:
            input, weight = ctx.saved_tensors
        else:
            weight = ctx.saved_tensors[0]
            input = None

        use_bias = ctx.use_bias

        handle = None
        if ctx.compute_weight_gradient:
            total_input = input

        grad_input = grad_output.matmul(weight)

        if handle is not None:
            handle.wait()

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )

        # if no weight gradient, immediately return
        if not ctx.compute_weight_gradient:
            if ctx.async_grad_allreduce:
                handle.wait()
            return grad_input, None, None, None, None, None, None

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
        total_input = total_input.view(
            total_input.shape[0] * total_input.shape[1], total_input.shape[2]
        )

        grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.async_grad_allreduce:
            handle.wait()
        return grad_input, grad_weight, grad_bias, None, None, None, None


def linear_with_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    async_grad_allreduce: bool,
) -> torch.Tensor:
    args = (
        input,
        weight,
        bias,
        async_grad_allreduce,
    )
    with torch.cuda.amp.autocast(enabled=False):
        return LinearWithAsyncCommunication.apply(*args)


class BaseParallelLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def _init_weight(self, weight):
        if self.arg_init_method is None:
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        else:
            self.arg_init_method(weight)

    def _init_bias(self):
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)


class ColumnParallelLinear(BaseParallelLinear):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    .. note::
        Input is supposed to be three dimensional and each dimension
        is expected to be batch,sequence and hidden feature, respectively.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all Neuron devices, otherwise, every Neuron device will have its output
                       which is Y_i = XA_i
        dtype: dtype of the weights
        device: Device on which the weights should be initialized.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        stride: int = 1,
        init_method: torch.nn.init = None
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.arg_init_method = init_method
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_size()
        self.output_size_per_partition = divide(output_size, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if device is None or device.type == "cpu":
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition, self.input_size, dtype=dtype
                )
            )
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight,
                self.output_size,
                self.input_size,
                self.output_size_per_partition,
                0,
                self._init_weight,
                params_dtype=dtype,
                stride=stride,
            )
        else:
            assert device.type == "xla", "Currently only xla device type is supported"
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device=device,
                    dtype=dtype,
                )
            )
            _initialize_affine_weight_neuron(self.weight, self._init_weight, partition_dim=0, stride=stride)

        if bias:
            self.bias_size = (
                self.output_size
                if self.gather_output
                else self.output_size_per_partition
            )
            if device is None or device.type == "cpu":
                self.bias = Parameter(torch.empty(self.bias_size, dtype=dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.bias_size,
                        device=device,
                        dtype=dtype,
                    )
                )
            self._init_bias()

            if not self.gather_output:
                set_tensor_model_parallel_attributes(self.bias, True, 0, stride=stride)
        else:
            self.register_parameter("bias", None)

        self.async_tensor_model_parallel_allreduce = world_size > 1

        self._forward_impl = linear_with_async_allreduce

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """

        if self.async_tensor_model_parallel_allreduce:
            input_parallel = input
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input)

        # Matrix multiply.
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
        )
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output = (output + self.bias) if self.bias is not None else output
        return output


class RowParallelLinear(BaseParallelLinear):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -

    .. note::
        Input is supposed to be three dimensional and each dimension
        is expected to be batch, sequence, and hidden feature, respectively.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the Neuron devices and we do not split
                           again.
        dtype: dtype of the weights
        device: Device on which the weights should be initialized.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        stride: int = 1,
        init_method: torch.nn.init = None
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.arg_init_method = init_method

        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if device is None or device.type == "cpu":
            self.weight = Parameter(
                torch.empty(
                    self.output_size, self.input_size_per_partition, dtype=dtype
                )
            )
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                self._init_weight,
                params_dtype=dtype,
                stride=stride,
            )
        else:
            assert device.type == "xla", "Currently only xla device type is supported"
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=device,
                    dtype=dtype,
                )
            )
            _initialize_affine_weight_neuron(
                self.weight, self._init_weight, partition_dim=1, stride=stride
            )
        if bias:
            if device is None or device.type == "cpu":
                self.bias = Parameter(torch.empty(self.output_size, dtype=dtype))
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=device,
                        dtype=dtype,
                    )
                )
            self._init_bias()
        else:
            self.register_parameter("bias", None)

        self._forward_impl = linear_with_async_allreduce
    

    def _init_bias(self):
        bound = 1 / math.sqrt(self.input_size_per_partition) if self.input_size_per_partition > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)


    def forward(
        self, input_: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            async_grad_allreduce=False,
        )
        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        output = (output_ + self.bias) if self.bias is not None else output_
        return output
