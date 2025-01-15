import math
import warnings
from typing import (
    Optional, Tuple, Union, Any, Callable, Dict, Type, cast
)
import os

import torch
import torch.distributed
import torch.nn.functional as F
import torch.nn.grad as grad
import torch.nn.init as init
import torch_xla.core.xla_model as xm
from torch.distributed import ProcessGroup
from torch.nn.parameter import Parameter
from .mappings import (
    _gather_along_dim,
    _reduce_scatter_along_dim,
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region_with_dim,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_input_channels_to_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
    get_aot_mode
)
from .random import get_xla_rng_tracker
from .utils import (
    EmbeddingUtility,
    cast_if_autocast_enabled,
    divide,
    get_padding_length,
    is_torch_version_greater_than_2,
    set_tensor_model_parallel_attributes,
    verify_casted_dtype,
)
from .utils import param_is_not_tensor_parallel_duplicate  # noqa: F401 # pylint: disable=W0611

if "reduce_scatter_tensor" not in dir(torch.distributed):
    torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base
if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base


def _initialize_affine_weight_neuron(
    weight: torch.Tensor,
    init_method: Callable[[torch.Tensor], None],
    partition_dim: int,
    num_partitions:int,
    stride: int = 1,
) -> None:
    """Initialize affine weight for model parallel on Neuron device.

    Args:
        weight (Parameter):
        init_method (Callable[[Tensor], None]): Taking a Tensor and initialize its elements.
        partition_dim (int): Dimension to apply partition.
    """

    set_tensor_model_parallel_attributes(
        tensor=weight,
        is_parallel=True,
        dim=partition_dim,
        stride=stride,
        num_partitions=num_partitions,
    )

    with get_xla_rng_tracker().fork():
        init_method(weight)


def create_local_weight(
    full_weight: torch.Tensor,
    partition_dim: int,
    per_partition_size: int,
    stride: int,
    out_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(full_weight, per_partition_per_stride_size, dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_size()
    my_weight_list = weight_list[rank::world_size]
    #TODO: Remove this check when torch 2.1 is default
    if stride == 1 and is_torch_version_greater_than_2() and full_weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        # torch cat is not supported for float8 dtypes from torch
        # Either way, here cat with stride = 1, is literally just a view[0]
        return my_weight_list[0]

    with torch.no_grad():
        return torch.cat(my_weight_list, dim=partition_dim, out=out_weight)


# Initialize a parameter with a given init_method on CPU
# Optionally return the un-partitioned parameter
def _initialize_parameter_cpu(
    param: torch.Tensor,  # shape should already be partitioned
    partition_dim: int,
    num_partitions: int,
    init_method: Callable[[torch.Tensor], None],
    return_master_param: bool = False,
    *,
    param_dtype: torch.dtype = torch.float32,
    stride: int = 1,
) -> Optional[torch.Tensor]:
    """Initialize a parameter for tensor parallelism

    During training, the primary purpose of this function is to ensure that
    weights are deterministically initialized regardless of the tensor
    parallel degree used. To ensure this, a master copy of the parameter is
    constructed on all processes first and then it is chunked and allocated to
    the relevant rank.

    When using a model architecture for inference using AOT tracing, all
    initialization logic can be skipped since only the weight shape is required
    to produce a compiled graph.
    """

    set_tensor_model_parallel_attributes(
        tensor=param, is_parallel=True, dim=partition_dim, stride=stride, num_partitions=num_partitions
    )

    # Skip all initialization logic during AOT tracing
    if get_aot_mode():
        return param

    # Create the master param
    master_param_shape = list(param.shape)
    master_param_shape[partition_dim] *= get_tensor_model_parallel_size()
    # Intentionally create the param as float32 so we do initialization in a high-precision datatype
    master_param = Parameter(torch.empty(master_param_shape, dtype=torch.float32, requires_grad=False))

    init_method(master_param)

    master_param_weight = master_param.to(dtype=param_dtype)
    create_local_weight(master_param_weight, partition_dim, param.shape[partition_dim], stride, out_weight=param)

    return master_param_weight if return_master_param else None


class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized across vocabulary/embedding dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
        shard_along_embedding: set true to parallelize across embedding dimension (default False)
        pad: set true to pad weights such that its divisible by tensor parallel degree
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init_method: Callable[[Any], Any] = init.normal_,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        shard_across_embedding: bool = False,
        pad: bool = False,
        sequence_parallel_enabled: bool = False,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        use_spmd_rank: bool = False
    ):
        super().__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.tensor_model_parallel_group = tensor_model_parallel_group if \
            tensor_model_parallel_group is not None else cast(ProcessGroup, get_tensor_model_parallel_group())
        self.tensor_model_parallel_size = self.tensor_model_parallel_group.size()
        self.shard_across_embedding = shard_across_embedding
        self.stride = 1
        self.pad = pad
        self.sequence_parallel_enabled = sequence_parallel_enabled

        self.rank_util: Optional[SPMDRank] = None
        if use_spmd_rank:
            self.rank_util = SPMDRank(self.tensor_model_parallel_size)

        if shard_across_embedding:
            # Divide the weight matrix along the embedding dimension.
            self.num_embeddings_per_partition = self.num_embeddings
            if pad:
                self.pad_size = get_padding_length(self.embedding_dim, self.tensor_model_parallel_size)
                self.embedding_dim = self.embedding_dim + self.pad_size
            self.embedding_dim_per_partition = divide(self.embedding_dim, self.tensor_model_parallel_size)
            self.padding_idx = padding_idx if padding_idx else None
            self.per_partition_size = self.embedding_dim_per_partition
            self.weight_partition_dim = 1
        else:
            if pad:
                self.pad_size = get_padding_length(self.num_embeddings, self.tensor_model_parallel_size)
                self.num_embeddings = self.num_embeddings + self.pad_size
            # Divide the weight matrix along the vocabulary dimension.
            (
                self.start_index,
                self.end_index,
            ) = EmbeddingUtility.range_from_global_vocab_size(
                self.num_embeddings,
                self.tensor_model_parallel_group.rank(),
                self.tensor_model_parallel_size,
            )
            self.num_embeddings_per_partition = self.end_index - self.start_index
            self.embedding_dim_per_partition = self.embedding_dim

            # only pad idx when it is in range
            if padding_idx and padding_idx >= self.start_index and padding_idx < self.end_index:
                self.padding_idx = padding_idx - self.start_index
            else:
                self.padding_idx = None

            self.per_partition_size = self.num_embeddings_per_partition
            self.weight_partition_dim = 0

        self.init_method = init_method
        self.dtype = dtype

        # Allocate weights and initialize.
        if device is None or device.type == "cpu" or device.type == "meta":
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim_per_partition,
                    dtype=dtype,
                )
            )
            if self.weight.device != torch.device("meta"):
                self.init_weight_cpu()
            else:
                set_tensor_model_parallel_attributes(
                    tensor=self.weight,
                    is_parallel=True,
                    dim=self.weight_partition_dim,
                    stride=1,
                    num_partitions=self.tensor_model_parallel_size,
                )
        else:
            assert device.type == "xla", "Currently only xla device type is supported"
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim_per_partition,
                    device=device,
                    dtype=dtype,
                )
            )
            _initialize_affine_weight_neuron(
                self.weight,
                init_method,
                partition_dim=self.weight_partition_dim,
                num_partitions=self.tensor_model_parallel_size,
            )

    def init_weight_cpu(self) -> None:
        _initialize_parameter_cpu(
            param=self.weight,
            partition_dim=self.weight_partition_dim,
            num_partitions=self.tensor_model_parallel_size,
            init_method=self.init_method,
            param_dtype=self.dtype,
        )

    def _forward_shard_across_vocab(self, input_: torch.Tensor) -> Any:
        start_index = self.start_index
        end_index = self.end_index
        if self.rank_util:
            start_index = self.num_embeddings_per_partition * self.rank_util.get_rank()
            end_index = self.num_embeddings_per_partition * (self.rank_util.get_rank() + 1)
        if self.tensor_model_parallel_size > 1:
            input_mask = (input_ >= start_index) & (input_ < end_index)
            # Mask the input.
            masked_input = input_.clone() - start_index
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
            output_parallel = torch.mul(output_parallel, torch.unsqueeze(input_mask.float(), dim=-1)).to(self.dtype)

        if self.sequence_parallel_enabled:
            # TODO: use sequence dimension instead of a manual transpose which assumes a layout
            return reduce_scatter_to_sequence_parallel_region(
                output_parallel.transpose(0, 1).contiguous(), process_group=self.tensor_model_parallel_group,
            )
        else:
            return reduce_from_tensor_model_parallel_region(
                output_parallel, process_group=self.tensor_model_parallel_group,
            )

    def _forward_shard_across_embed(self, input_: torch.Tensor) -> torch.Tensor:
        output_parallel = F.embedding(
            input_.long(),
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return gather_from_tensor_model_parallel_region(
            output_parallel, process_group=self.tensor_model_parallel_group,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.pad and self.training:
            raise RuntimeError("`pad=True` is only supported for inference. Set model.eval()")

        if self.shard_across_embedding:
            output = self._forward_shard_across_embed(input_)
            if self.pad and self.pad_size > 0:
                output = torch.narrow(output, -1, 0, self.embedding_dim - self.pad_size)
            return output
        else:
            output = self._forward_shard_across_vocab(input_)
            return output

    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        if not self.pad or self.pad_size == 0:
            return

        padding_dim = self.weight_partition_dim
        state_dict_size = model_state_dict[prefix].shape[padding_dim]

        if self.shard_across_embedding:
            if self.embedding_dim != state_dict_size + self.pad_size:
                raise RuntimeError(
                    f"State dict {prefix} is of an unexpected shape {state_dict_size} expected {state_dict_size - self.pad_size}"
                )
            model_state_dict[prefix] = torch.nn.functional.pad(model_state_dict[prefix], (0, self.pad_size))
            return
        else:
            if self.num_embeddings != state_dict_size + self.pad_size:
                raise RuntimeError(
                    f"State dict {prefix} is of an unexpected shape {state_dict_size} expected {state_dict_size - self.pad_size}"
                )
            model_state_dict[prefix] = torch.nn.functional.pad(model_state_dict[prefix], (0, 0, 0, self.pad_size))
            return


class LinearWithAsyncCommunication(torch.autograd.Function):
    """Linear layer execution with asynchronous communication."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        async_grad_allreduce: bool,
        sequence_parallel_enabled: bool,
        sequence_dimension: Optional[int] = 0,
        save_for_backward: bool = True,
        process_group: Optional[ProcessGroup] = None,
        reduce_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        ctx.use_bias = bias is not None and weight.requires_grad
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel_enabled = sequence_parallel_enabled
        ctx.sequence_dimension = sequence_dimension
        ctx.compute_weight_gradient = weight.requires_grad
        if process_group is None:
            process_group = get_tensor_model_parallel_group(as_list=True)
        ctx.process_group = process_group
        ctx.reduce_dtype = reduce_dtype

        if ctx.sequence_parallel_enabled:
            assert (
                ctx.sequence_dimension is not None
            ), "Found `sequence_parallel_enabled` set to True, but `sequence_dimension` was None, and this occured in an unexpected area"

        if save_for_backward:
            if ctx.compute_weight_gradient:
                ctx.save_for_backward(input, weight)
            else:
                ctx.save_for_backward(weight)

        if ctx.sequence_parallel_enabled:
            # `input` is supposed to be 3D and the optimal order of dimension is [sequence, batch, hidden]
            # If not SBH, the necessary transposes will be added
            total_input = _gather_along_dim(input, ctx.sequence_dimension, process_group=ctx.process_group)
        else:
            total_input = input

        output = torch.einsum('...m,mn->...n', total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0]
        if ctx.sequence_parallel_enabled:
            assert (
                ctx.sequence_dimension is not None
            ), "Found `sequence_parallel_enabled` set to True, but `sequence_dimension` was None, and this occured in an unexpected area"

        if ctx.compute_weight_gradient:
            input, weight = ctx.saved_tensors
        else:
            weight = ctx.saved_tensors[0]
            input = None

        use_bias = ctx.use_bias
        process_group = ctx.process_group

        handle = None
        if ctx.compute_weight_gradient:
            if ctx.sequence_parallel_enabled:
                # Optimal layout is SBH, but if not, transposes are added
                total_input = _gather_along_dim(input, ctx.sequence_dimension, process_group=process_group)
            else:
                total_input = input

        grad_input = grad_output.matmul(weight)

        if handle is not None:
            handle.wait()

        original_dtype = grad_input.dtype

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            grad_input = grad_input.to(ctx.reduce_dtype)
            handle = torch.distributed.all_reduce(grad_input, group=process_group, async_op=True)
            grad_input = grad_input.to(original_dtype)

        # if no weight gradient, immediately return
        if not ctx.compute_weight_gradient:
            if ctx.sequence_parallel_enabled:
                assert not ctx.async_grad_allreduce
                # Optimal layout is SBH, but if not, transposes are added
                sub_grad_input = _reduce_scatter_along_dim(grad_input.to(ctx.reduce_dtype), ctx.sequence_dimension, process_group=process_group)
                sub_grad_input = sub_grad_input.to(original_dtype)
                return sub_grad_input, None, None, None, None, None, None, None, None

            if ctx.async_grad_allreduce:
                assert handle
                handle.wait()
                grad_input = grad_input.to(original_dtype)
            return grad_input, None, None, None, None, None, None, None, None

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])

        if ctx.sequence_parallel_enabled:
            assert not ctx.async_grad_allreduce
            # optimal layout is SBH, but if not, transposes will be added
            sub_grad_input = _reduce_scatter_along_dim(grad_input.to(ctx.reduce_dtype), ctx.sequence_dimension, process_group=process_group)
            sub_grad_input=sub_grad_input.to(original_dtype)

        grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel_enabled:
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None

        if ctx.async_grad_allreduce:
            assert handle
            handle.wait()
            grad_input=grad_input.to(original_dtype)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


def linear_with_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
    sequence_dimension: Optional[int] = 0,
    autograd_func_class: Type[torch.autograd.Function] = LinearWithAsyncCommunication,
    save_for_backward: bool = True,
    process_group: Optional[ProcessGroup] = None,
    reduce_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    args = cast_if_autocast_enabled(
        input,
        weight,
        bias,
        async_grad_allreduce,
        sequence_parallel_enabled,
        sequence_dimension,
        save_for_backward,
        process_group,
        reduce_dtype,
    )
    verify_casted_dtype(args)
    with torch.cuda.amp.autocast(enabled=False):
        return autograd_func_class.apply(*args)


class BaseParallelLinear(torch.nn.Module):
    autograd_func_class: Type[torch.autograd.Function] = LinearWithAsyncCommunication

    def __init__(self):
        super().__init__()

    def _init_weight(self, weight: torch.Tensor) -> None:
        if self.arg_init_method is None:
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        else:
            self.arg_init_method(weight)

    def _init_bias(self) -> None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)


class ColumnParallelLinear(BaseParallelLinear):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p]. Here A is the weight matrix,
    X is the input.

    .. note::
        Input is supposed to be three dimensional and each dimension
        is expected to be batch,sequence and hidden feature, respectively.

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
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
        device: Optional[torch.device] = None,
        stride: int = 1,
        init_method: Optional[Callable[[Any], torch.Tensor]] = None,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        keep_master_weight: bool = False,
        skip_bias_add: bool = False,
        pad: bool = False,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        reduce_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.add_bias = bias
        self.gather_output = gather_output
        self.arg_init_method = init_method

        self.tensor_parallel_group = tensor_model_parallel_group if \
            tensor_model_parallel_group is not None else cast(ProcessGroup, get_tensor_model_parallel_group())

        world_size = torch.distributed.get_world_size(group=self.tensor_parallel_group)
        self.pad = pad
        if self.pad:
            self.pad_size = get_padding_length(self.output_size, world_size)
            self.output_size = self.output_size + self.pad_size

        # Divide the weight matrix along the last dimension.
        self.output_size_per_partition = divide(self.output_size, world_size)
        self.dtype = dtype
        self.device = device
        self.stride = stride
        self.keep_master_weight = keep_master_weight
        self.skip_bias_add = skip_bias_add
        self.bias_shape: Optional[Tuple[int]]

        self.initialize_weight_and_bias()

        self.async_tensor_model_parallel_allreduce = not sequence_parallel_enabled and world_size > 1
        if sequence_parallel_enabled:
            if world_size <= 1:
                warnings.warn(f"`sequence_parallel_enabled` is set to `True`, but got world_size of {world_size}")

            if sequence_dimension is None:
                warnings.warn(
                    "`sequence_parallel_enabled` is set to `True`, but got `sequence_dimension` as `None`. Defaulting `sequence_dimension` to 0."
                )
                sequence_dimension = 0

        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.sequence_dimension = sequence_dimension

        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel_enabled:
            raise RuntimeError(
                "`async_tensor_model_parallel_allreduce` and `sequence_parallel_enabled` cannot be enabled at the same time."
            )
        self.reduce_dtype = reduce_dtype
        self._forward_impl = linear_with_async_allreduce

    def set_weight_and_bias_config(self) -> None:
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight_shape = (self.output_size_per_partition, self.input_size)
        self.weight_partition_dim = 0

        if self.add_bias:
            bias_size = self.output_size if self.gather_output else self.output_size_per_partition
            self.bias_shape = (bias_size,)
        else:
            self.bias_shape = None

    def initialize_weight_and_bias(self):
        self.set_weight_and_bias_config()
        init_device = self.device

        # Get torch init device if device is not explicitly mentioned
        self.weight = Parameter(torch.empty(*self.weight_shape, device=init_device, dtype=self.dtype))
        # Mark the true device after weight initialization
        self.device = self.weight.device

        # Initialize weight.
        if self.device.type == "cpu":
            self.init_weight_cpu()
        elif self.device.type == "meta":
            set_tensor_model_parallel_attributes(
                tensor=self.weight, is_parallel=True, dim=self.weight_partition_dim,
                stride=self.stride, num_partitions=self.tensor_parallel_group.size(),
            )
        else:
            _initialize_affine_weight_neuron(
                self.weight, self._init_weight, partition_dim=self.weight_partition_dim,
                num_partitions=self.tensor_parallel_group.size(),
                stride=self.stride
            )
        if self.add_bias:
            assert self.bias_shape
            if self.device is None or self.device.type == "cpu":
                self.bias = Parameter(torch.empty(*self.bias_shape, dtype=self.dtype))
            else:
                self.bias = Parameter(torch.empty(*self.bias_shape, device=self.device, dtype=self.dtype))
            if self.bias.device != torch.device("meta"):
                self._init_bias()

            if not self.gather_output:
                set_tensor_model_parallel_attributes(
                    self.bias, True, 0, stride=self.stride, num_partitions=self.tensor_parallel_group.size(),
                )
        else:
            self.register_parameter("bias", None)

    def init_weight_cpu(self) -> None:
        self.master_weight = _initialize_parameter_cpu(
            param=self.weight,
            partition_dim=self.weight_partition_dim,
            num_partitions=self.tensor_parallel_group.size(),
            init_method=self._init_weight,
            param_dtype=self.dtype,
            stride=self.stride,
            return_master_param=self.keep_master_weight,
        )

    def forward(self, input: torch.Tensor, *_: Any) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """
        if self.pad and self.training:
            raise RuntimeError("`pad=True` is only supported for inference. Set model.eval()")

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input, process_group=self.tensor_parallel_group)

        # Matrix multiply.
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            autograd_func_class=self.autograd_func_class,
            process_group=self.tensor_parallel_group,
            reduce_dtype = self.reduce_dtype,
        )
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel_enabled
            output = gather_from_tensor_model_parallel_region(output_parallel, process_group=self.tensor_parallel_group)
            if self.pad and self.pad_size > 0:
                output = torch.narrow(output, -1, 0, self.output_size - self.pad_size)
        else:
            output = output_parallel
        if self.skip_bias_add:
            return output, self.bias
        output = (output + self.bias) if self.bias is not None else output
        return output

    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        if not self.pad or self.pad_size == 0:
            return
        if self.output_size != model_state_dict[prefix].shape[0] + self.pad_size:
            size = model_state_dict[prefix].shape[0]
            raise RuntimeError(f"State dict {prefix} is of an unexpected size {size} expected {size - self.pad_size}")
        model_state_dict[prefix] = torch.nn.functional.pad(model_state_dict[prefix], (0, 0, 0, self.pad_size))


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
        Input(X) is supposed to be three dimensional and each dimension
        is expected to be batch, sequence, and hidden feature, respectively.
        A is the weight matrix.

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
        device: Optional[torch.device] = None,
        stride: int = 1,
        init_method: Optional[Callable[..., Any]] = None,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        keep_master_weight: bool = False,
        skip_bias_add: bool = False,
        pad: bool = False,
        reduce_output: bool = True,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        reduce_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.add_bias = bias
        self.input_is_parallel = input_is_parallel
        self.pad = pad
        self.reduce_output = reduce_output
        self.tensor_parallel_group = tensor_model_parallel_group if \
            tensor_model_parallel_group is not None else cast(ProcessGroup, get_tensor_model_parallel_group())
        self.reduce_dtype = reduce_dtype

        world_size = self.tensor_parallel_group.size()
        if self.pad:
            self.pad_size = get_padding_length(self.input_size, world_size)
            self.input_size = self.input_size + self.pad_size
        # Divide the weight matrix along the last dimension.
        self.input_size_per_partition = divide(self.input_size, world_size)
        self.arg_init_method = init_method
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.sequence_parallel_enabled and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel_enabled`, `input_is_parallel` must be `True`")

        if self.sequence_parallel_enabled and sequence_dimension is None:
            warnings.warn(
                "`sequence_parallel_enabled` is set to `True`, but got `sequence_dimension` as `None`. Defaulting `sequence_dimension` to 0."
            )
            sequence_dimension = 0

        self.sequence_dimension: int = sequence_dimension  # type: ignore
        self.dtype = dtype
        self.device = device
        self.stride = stride
        self.keep_master_weight = keep_master_weight
        self.skip_bias_add = skip_bias_add
        self.bias_shape: Optional[Tuple[int]]
        self.initialize_weight_and_bias()
        self.reduce_dtype = reduce_dtype
        self._forward_impl = linear_with_async_allreduce

    def set_weight_and_bias_config(self) -> None:
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight_shape = (self.output_size, self.input_size_per_partition)
        self.weight_partition_dim = 1

        if self.add_bias:
            self.bias_shape = (self.output_size,)
        else:
            self.bias_shape = None

    def initialize_weight_and_bias(self) -> None:
        self.set_weight_and_bias_config()
        init_device = self.device

        # Get torch init device if device is not explicitly mentioned
        self.weight = Parameter(torch.empty(*self.weight_shape, device=init_device, dtype=self.dtype))
        self.device = self.weight.device

        # Initialize weight.
        if self.device.type == "cpu":
            self.init_weight_cpu()
        elif self.device.type == "meta":
            set_tensor_model_parallel_attributes(
                tensor=self.weight, is_parallel=True, dim=self.weight_partition_dim,
                stride=self.stride, num_partitions=self.tensor_parallel_group.size(),
            )
        else:
            _initialize_affine_weight_neuron(
                self.weight, self._init_weight, partition_dim=self.weight_partition_dim,
                num_partitions=self.tensor_parallel_group.size(),
                stride=self.stride,
            )

        if self.add_bias:
            assert self.bias_shape
            if self.device is None or self.device.type == "cpu":
                self.bias = Parameter(torch.empty(*self.bias_shape, dtype=self.dtype))
            else:
                self.bias = Parameter(torch.empty(*self.bias_shape, device=self.device, dtype=self.dtype))
            if self.bias.device != torch.device("meta"):
                self._init_bias()
            setattr(self.bias, "sequence_parallel_enabled", self.sequence_parallel_enabled)
        else:
            self.register_parameter("bias", None)

    def init_weight_cpu(self) -> None:
        self.master_weight = _initialize_parameter_cpu(
            param=self.weight,
            partition_dim=self.weight_partition_dim,
            num_partitions=self.tensor_parallel_group.size(),
            init_method=self._init_weight,
            param_dtype=self.dtype,
            stride=self.stride,
            return_master_param=self.keep_master_weight,
        )

    def _init_bias(self) -> None:
        bound = 1 / math.sqrt(self.input_size_per_partition) if self.input_size_per_partition > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """
        if self.pad and self.training:
            raise RuntimeError("`pad=True` is only supported for inference. Set model.eval()")

        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            if self.pad and self.pad_size > 0:
                input_ = torch.nn.functional.pad(input_, (0, self.pad_size))
            assert not self.sequence_parallel_enabled
            input_parallel = scatter_to_tensor_model_parallel_region(input_, process_group=self.tensor_parallel_group)

        # Matrix multiply.
        output_ = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
            sequence_dimension=self.sequence_dimension,
            autograd_func_class=self.autograd_func_class,
            process_group=self.tensor_parallel_group,
            reduce_dtype = self.reduce_dtype,
        )

        if self.reduce_output:
            # All-reduce across all the partitions.
            original_dtype = output_.dtype

            output_ = output_.to(self.reduce_dtype)

            if self.sequence_parallel_enabled:
                output_ = reduce_scatter_to_sequence_parallel_region(
                    output_, self.sequence_dimension, process_group=self.tensor_parallel_group,
                )
            else:
                output_ = reduce_from_tensor_model_parallel_region(
                    output_, process_group=self.tensor_parallel_group,
                )

            output_ = output_.to(original_dtype)

        if self.skip_bias_add:
            return output_, self.bias
        output = (output_ + self.bias) if self.bias is not None else output_
        return output

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> None:
        if not self.pad or self.pad_size == 0:
            return
        if self.input_size != model_state_dict[prefix].shape[1] + self.pad_size:
            size = model_state_dict[prefix].shape[1]
            raise RuntimeError(f"State dict {prefix} is of an unexpected size {size} expected {size - self.pad_size}")
        model_state_dict[prefix] = torch.nn.functional.pad(model_state_dict[prefix], (0, self.pad_size))


class Conv2dWithInputGradAllReduce(torch.autograd.Function):
    """Conv2D layer execution with optional all-reduce on input gradient"""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        allreduce_weight_grad: bool,
    ):
        ctx.allreduce_weight_grad = allreduce_weight_grad

        ctx.save_for_backward(input, weight)

        ctx.stride = stride
        ctx.padding = padding
        # We don't need the bias to compute its gradient, but want to know if it's None
        # because if it is we can skip computing its gradient
        ctx.bias_is_none = bias is None

        # TODO: add support for non-zero padding types
        output = F.conv2d(input, weight, bias, stride, padding)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # Adapted from https://stackoverflow.com/questions/74949892/implementing-a-conv2d-backward-in-pytorch
        grad_output = grad_outputs[0]
        input, weight = ctx.saved_tensors

        handle = None

        stride = ctx.stride
        padding = ctx.padding
        dilation = 1
        groups = 1

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)

            # Allreduce on input grad while we calculate weight and bias grads
            if ctx.allreduce_weight_grad:
                handle = torch.distributed.all_reduce(
                    grad_input, group=get_tensor_model_parallel_group(), async_op=True
                )

        # Calculate weight grad
        if ctx.needs_input_grad[1]:
            grad_weight = grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)

        # Calculate bias grad
        if not ctx.bias_is_none and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        if ctx.needs_input_grad[0] and ctx.allreduce_weight_grad:
            assert handle
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


def conv2d_with_weight_grad_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    allreduce_weight_grad: bool,
) -> torch.Tensor:
    args = cast_if_autocast_enabled(
        input,
        weight,
        bias,
        stride,
        padding,
        allreduce_weight_grad,
    )
    verify_casted_dtype(args)
    with torch.cuda.amp.autocast(enabled=False):
        return Conv2dWithInputGradAllReduce.apply(*args)


# Convolution kernels have shape [c_out, c_int, *kernel_spatial_dims]
# Where kernel_spatial_dims is a tuple of the same length as the dimensionality of the conv
CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION = 0
CONV_KERNEL_INPUT_CHANNEL_DIMENSION = 1


class BaseParallelConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
        groups: int,
        bias: bool,
        padding_mode: str,
        partition_dim: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
        init_method: Optional[Callable[[Any], torch.Tensor]] = None,
        keep_master_params: bool = False,
        partition_pad: bool = False,
    ):
        if not all(d == 1 for d in _as_tuple2(dilation)):
            raise NotImplementedError(f"Non-1 dilation is not yet supported. Received: {dilation}")
        if groups != 1:
            raise NotImplementedError(f"Non-1 groups is not yet supported. Received: {groups}")
        if padding_mode != "zeros":
            raise NotImplementedError(f"Non-zeros padding is not yet supported. Received: {padding_mode}")

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.partition_dim = partition_dim
        self.arg_init_method = init_method
        self.dtype = dtype
        self.keep_master_params = keep_master_params
        self.partition_pad = partition_pad

        world_size = get_tensor_model_parallel_size()

        # Parameters
        # Convs have a kernel shape of [c_out, c_in, *kernel_dims]
        if partition_dim == CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION:
            if self.partition_pad:
                self.partition_pad_size = get_padding_length(self.out_channels, world_size)
                self.out_channels = self.out_channels + self.partition_pad_size

            self.channels_per_partition = divide(self.out_channels, world_size)
            weight_shape = [self.channels_per_partition, in_channels, *_as_tuple2(kernel_size)]
        elif partition_dim == CONV_KERNEL_INPUT_CHANNEL_DIMENSION:
            if self.partition_pad:
                self.partition_pad_size = get_padding_length(self.in_channels, world_size)
                self.in_channels = self.in_channels + self.partition_pad_size

            self.channels_per_partition = divide(self.in_channels, world_size)
            weight_shape = [out_channels, self.channels_per_partition, *_as_tuple2(kernel_size)]
        else:
            assert False, f"Unsupported partition dim: {partition_dim}"

        # Weight/bias initialization has a weird behaviour:
        #   If initializing the param on CPU: initialize, split, keep master param if requested
        #   If initializing the param on XLA: initialize (but don't split)
        # This behavior is consistent with how BaseParallelLinear does it

        # Initialize weight
        self.weight = Parameter(torch.empty(weight_shape, dtype=dtype, device=device))
        if self.weight.device.type == "cpu":
            if self.weight.device != torch.device("meta"):
                self.master_weight = _initialize_parameter_cpu(
                    self.weight,
                    partition_dim=partition_dim,
                    num_partitions=world_size,
                    init_method=self._init_weight,
                    return_master_param=self.keep_master_params,
                    param_dtype=self.dtype,
                    stride=1,
                )
        elif self.weight.device.type == "meta":
            set_tensor_model_parallel_attributes(
                tensor=self.weight, is_parallel=True, dim=partition_dim, stride=1, num_partitions=world_size,
            )
        else:
            assert device and device.type == "xla", "Currently only xla device type is supported"
            _initialize_affine_weight_neuron(
                self.weight,
                self._init_weight,
                partition_dim=partition_dim,
                num_partitions=world_size,
                stride=1,
            )

        # Initialize bias
        if bias:
            # If this is an OutputChannelConv2d then we want to use the partitioned
            # number of channels because we do bias add before the AllGather
            bias_channels = (
                self.channels_per_partition
                if partition_dim == CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION
                else self.out_channels
            )
            self.bias = Parameter(torch.empty(bias_channels, dtype=dtype, device=device))
            if self.bias.device.type != "meta":
                # If we're sharding the bias, use _initialize_parameter_cpu to shard it and keep the master bias if requested
                # Note: we only shard the param if on CPU. This is inherited/copied behaviour from BaseParallelLinear. See
                #       comment above weight/bias init block
                if self.bias.device.type == "cpu" and partition_dim == CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION:
                    self.master_bias = _initialize_parameter_cpu(
                        self.bias,
                        CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION,
                        num_partitions=world_size,
                        init_method=self._init_bias,
                        return_master_param=self.keep_master_params,
                        param_dtype=self.dtype,
                        stride=1,
                    )
                # TODO: _initialize_affine_weight_neuron forks the XLA RNG tracker when initializing weights
                #       do we need to do that for bias too?
                else:
                    self._init_bias(self.bias)
                    self.master_bias = self.bias if self.keep_master_params else None
            else:
                # If we have a meta device type then nothing to do, initialization would be a no-op
                self.master_bias = self.bias if self.keep_master_params else None
        else:
            self.register_parameter("bias", None)

        self._forward_impl = conv2d_with_weight_grad_allreduce

    def _init_weight(self, weight):
        if self.arg_init_method is None:
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        else:
            self.arg_init_method(weight)

    def _init_bias(self, bias):
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(bias, -bound, bound)


# Convolutions can take an int or tuple for most of their __init__ args
# This function broadcasts the given arg to a tuple if it's not a tuple already
def _as_tuple2(arg: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(arg, tuple):
        return arg
    if isinstance(arg, int):
        return (arg, arg)
    raise TypeError(f"Arg should be int or tuple of int, but received {type(arg)}")


class OutputChannelParallelConv2d(BaseParallelConv):
    """Conv2d layer with parallelism on its output channels

    The definition of a Conv2d layer can be found at https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    This layer parallelizes the Conv2d along the output channel dimension

    .. note::
        Input is expected to be four dimensional, in order [N, C, H, W]

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels in the original Conv that is being parallelized. Parallelization is handled internally by this class
        kernel_size: Size of the kernel. Can be a single number for a square kernel or a tuple of two numbers
        stride: Stride of the convolution. Can be a single number for uniform H/W stride or a tuple of two numbers
        padding: Padding of the convolution. Can be a single number for uniform H/W padding or a tuple of two numbers
        bias: If true, add bias
        gather_output: If true, call all-gather on the output to assemble the partial outputs produced by each Neuron device into the full output, and make the full output available on all Neuron devices
        dtype: Datatype of the weights
        device: Device on which the weights should be initialized
        init_method: Method for initializing the weight
        keep_master_weight: If device="cpu", whether to keep the original ("master") weight the per-worker weights are split from
        partition_pad: Pad the output channel dimension if needed to make the output channel count divisible by the tensor model parallel size
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        gather_output: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        init_method: Optional[Callable[[Any], torch.Tensor]] = None,
        keep_master_weight: bool = False,
        partition_pad: bool = False,
    ):
        # Base class expects these all to be tuples so it can support N-dimensional convs
        kernel_size = _as_tuple2(kernel_size)
        stride = _as_tuple2(stride)
        padding = _as_tuple2(padding)
        dilation = _as_tuple2(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            CONV_KERNEL_OUTPUT_CHANNEL_DIMENSION,
            dtype,
            device,
            init_method,
            keep_master_weight,
            partition_pad,
        )
        self.kernel_size: Tuple[int, int]
        self.stride: Tuple[int, int]
        self.padding: Tuple[int, int]
        self.dilation: Tuple[int, int]

        self.allreduce_weight_grad = get_tensor_model_parallel_size() > 1
        self.gather_output = gather_output

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Forward of OutputChannelParallelConv2d

        Args:
            in_tensor: 4D tensor in order [N, C, H ,W]

        Returns:
            - output
        """

        if self.allreduce_weight_grad:
            input_parallel = in_tensor
        else:
            input_parallel = copy_to_tensor_model_parallel_region(in_tensor)

        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            allreduce_weight_grad=self.allreduce_weight_grad,
        )

        # We intentionally did the bias add in _forward_impl to do less work overall
        # This way, each worker only has to do 1/world_size of the bias add
        if self.gather_output:
            # All-gather across the partitions
            output = gather_from_tensor_model_parallel_region_with_dim(output_parallel, gather_dim=1)
            if self.partition_pad and self.partition_pad_size > 0:
                output = torch.narrow(output, 1, 0, self.out_channels - self.partition_pad_size)
        else:
            output = output_parallel

        return output

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> None:
        if not self.partition_pad or self.partition_pad_size == 0:
            return
        if self.out_channels != model_state_dict[prefix].shape[0] + self.partition_pad_size:
            size = model_state_dict[prefix].shape[0]
            raise RuntimeError(
                f"State dict {prefix} is of an unexpected size {size} expected {size - self.partition_pad_size}"
            )
        model_state_dict[prefix] = torch.nn.functional.pad(
            model_state_dict[prefix], (0, 0, 0, 0, 0, 0, 0, self.partition_pad_size)
        )


class InputChannelParallelConv2d(BaseParallelConv):
    """Conv2d layer with parallelism on its input channels

    The definition of a Conv2d layer can be found at https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    This layer parallelizes the Conv2d along the input channel dimension

    .. note::
        Input is expected to be four dimensional, in order [N, C, H, W]

    Arguments:
        in_channels: Number of input channels in the original Conv that is being parallelized. Parallelization is handled internally by this class
        out_channels: Number of output channels
        device: Device on which the weights should be initialized
        kernel_size: Size of the kernel. Can be a single number for a square kernel or a tuple of two numbers
        stride: Stride of the convolution. Can be a single number for uniform H/W stride or a tuple of two numbers
        padding: Padding of the convolution. Can be a single number for uniform H/W padding or a tuple of two numbers
        bias: If true, add bias
        input_is_parallel: Whether the input to this layer is already split among workers, e.g. by being the output of a OutputChannelParallelConv2d
        dtype: Datatype of the weights
        init_method: Method for initializing the weight
        keep_master_weight: If device="cpu", whether to keep the original ("master") weight the per-worker weights are split from
        partition_pad: Pad the input channel dimension if needed to make the input channel count divisible by the tensor model parallel size
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        input_is_parallel: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        init_method: Optional[Callable[[Any], torch.Tensor]] = None,
        keep_master_weight: bool = False,
        partition_pad: bool = False,
    ):
        # Base class expects these all to be tuples so it can support N-dimensional convs
        kernel_size = _as_tuple2(kernel_size)
        stride = _as_tuple2(stride)
        padding = _as_tuple2(padding)
        dilation = _as_tuple2(dilation)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            CONV_KERNEL_INPUT_CHANNEL_DIMENSION,
            dtype,
            device,
            init_method,
            keep_master_weight,
            partition_pad,
        )
        self.kernel_size: Tuple[int, int]
        self.stride: Tuple[int, int]
        self.padding: Tuple[int, int]
        self.dilation: Tuple[int, int]

        self.input_is_parallel = input_is_parallel

    def forward(self, in_tensor: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of InputChannelParallelConv2d

        Args:
            in_tensor: 4D tensor in order [N, C, H ,W]

        Returns:
            - output
        """

        if self.input_is_parallel:
            input_parallel = in_tensor
        else:
            input_parallel = scatter_input_channels_to_tensor_model_parallel_region(in_tensor)

        # Intentionally pass in bias=None here. We want to delay bias add to after
        # the AllReduce for better perf
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            # Don't allreduce the weight grad, each worker produces the weight grad
            # for its own channels
            allreduce_weight_grad=False,
        )

        # All-reduce across the partitions
        output = reduce_from_tensor_model_parallel_region(output_parallel)

        # Reshape bias to output size and add
        if self.bias is not None:
            output = output + self.bias[:, None, None]

        return output


class SPMDRank(torch.nn.Module):
    """
        This is a temporary workaround for getting rank in traced SPMD model. To be replaced
        with ReplicaID in HLO once compiler adds support.

        To use this temporary fix,
        1. Register SPMDRank as a module where you need the rank.
        2. Save weight as arange tensor in model checkpoint

        >>> self.spmd_rank = SPMDRank(world_rank)
        >>> ckpt['spmd_rank.rank'] = torch.arange(0, world_size, dtype=torch.int32)
    """
    def __init__(self, world_size: int):
        super().__init__()
        self.world_size = world_size
        self.rank = torch.nn.Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)
        set_tensor_model_parallel_attributes(
            self.rank,
            is_parallel=True,
            dim=0,
            stride=1,
            num_partitions=self.world_size,
        )

    def get_rank(self) -> torch.Tensor:
        return self.rank
