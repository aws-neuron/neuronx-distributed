import torch
import torch_xla.core.xla_model as xm

from .parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)
from .utils import split_tensor_along_last_dim, split_tensor_along_dim

if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base
if "reduce_scatter_tensor" not in dir(torch.distributed):
    torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base


def nonzero_partition_dim_swap(
    func: Callable[[Tensor, int], Tensor],
) -> Callable[[Tensor, int], Tensor]:
    """ Decorator that internally swaps the partition/gather dim with 0-dimension. To the
    outside the partition_dim appears to be the (arbitrary) partition dimension. Internally,
    partition/split dimension is always 0 - which is achieved by pre- and post-transpose. """

    @functools.wraps(func)
    def wrapped_fn(x: Tensor, partition_dim: int) -> Tensor:
        x_t = x.transpose(0, partition_dim) if partition_dim != 0 else x
        y_t: Tensor = func(x_t, partition_dim=0)
        y = y_t.transpose(0, partition_dim) if partition_dim != 0 else y_t

        return y

    return wrapped_fn


def _reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 device.
    if get_tensor_model_parallel_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_


def _split_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    return _split_along_dim(input_, len(input_.shape)-1)

def _split_along_first_dim(input_: Tensor) -> Tensor:
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    return _split_along_dim(input_, 0)

def _split_along_dim(input_: Tensor, partition_dim: int) -> Tensor:
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_size()
    # Bypass the function if we are using only 1 device.
    if world_size == 1:
        return input_

    # Split along partition dimension.
    input_list = split_tensor_along_dim(input_, partition_dim, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


@nonzero_partition_dim_swap
def _gather_along_dim(x: Tensor, partition_dim: int) -> Tensor:
    """Given a tensor partitioned across the specified dimension,
    gather and concatenate along partition dimension (using TP/SP group).
    """
    tp_group = get_tensor_model_parallel_group()

    # bpyass the function if we only have 1 TP rank.
    if tp_group.size() == 1:
        return x

    output = xm.all_gather(
        x,
        dim=partition_dim,
        groups=get_tensor_model_parallel_group(as_list=True),
        pin_layout=False,
    )

    return output.contiguous()


def _gather_along_last_dim(x: Tensor) -> Tensor:
    return _gather_along_dim(x, partition_dim=len(x.shape)-1)

def _reduce_scatter_along_first_dim(x: Tensor) -> Tensor:
    return _reduce_scatter_along_dim(x, 0)

def _reduce_scatter_along_last_dim(x: Tensor) -> Tensor:
    return _reduce_scatter_along_dim(x, len(x.shape)-1)

@nonzero_partition_dim_swap
def _reduce_scatter_along_dim(x: Tensor, partition_dim: int) -> Tensor:
    """Reduce-scatter the input tensor across model parallel group."""
    tp_group = get_tensor_model_parallel_group()

    world_size = get_tensor_model_parallel_size()
    # Bypass the function if we are using only 1 device.
    if world_size == 1:
        return input_

    # Size and dimension.
    rank = get_tensor_model_parallel_rank()

    xm.reduce_scatter(
        xm.REDUCE_SUM,
        x.contiguous(),
        scatter_dim=partition_dim,
        shard_count=tp_group.size(),
        scale=1,
        output=output,
        groups=get_tensor_model_parallel_group(as_list=True),
        pin_layout=False,
    )

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=1).contiguous()

    return output


def _split_along_first_dim(input_: torch.Tensor) -> torch.Tensor:
    """Split the tensor along its first dimension and keep the corresponding slice."""
    world_size = get_tensor_model_parallel_size()
    # Bypass the function if we are using only 1 GPU for tensor model parallel.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size(0)
    assert dim_size % world_size == 0
    local_dim_size = dim_size // world_size
    dim_offset = get_tensor_model_parallel_rank() * local_dim_size
    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()
    return output


def _gather_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatenate along the last dimension."""

    world_size = get_tensor_model_parallel_size()
    # Bypass the function if we are using only 1 device.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _gather_along_first_dim(input_: torch.Tensor) -> torch.Tensor:
    """Gather tensors and concatenate along the first dimension."""
    world_size = get_tensor_model_parallel_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    output = xm.all_gather(input_, groups=get_tensor_model_parallel_group(as_list=True), pin_layout=False)

    return xm.all_to_all(
        x,
        split_dimension=split_dim,
        concat_dimension=concat_dim,
        split_count=ep_group.size(),
        groups=get_expert_model_parallel_group(as_list=True),
        pin_layout=False,
    )

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the tensor model parallel region."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the tensor model parallel region."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from tensor model parallel region and concatenate."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input into TP/SP partitions along specified sequence dimension,
    only keep the corresponding chunk for the current TP rank."""

    @staticmethod
    def symbolic(graph, input_: Tensor, partition_dim: int) -> Tensor:
        return _split_along_dim(input_, partition_dim=partition_dim)

    @staticmethod
    def forward(ctx, input_: Tensor, partition_dim: int) -> Tensor:
        ctx.partition_dim = partition_dim
        return _split_along_dim(input_, partition_dim=partition_dim)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        return _gather_along_dim(grad_output, partition_dim=ctx.partition_dim), None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather input partitions across TP/SP group and concatenate along specified
    sequence dimension."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_second_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_second_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_second_dim(grad_output)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chunk to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_, to_model_parallel: bool = True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, to_model_parallel: bool = True):
        ctx.to_model_parallel = to_model_parallel
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.to_model_parallel:
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            return _split_along_first_dim(grad_output), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _ScatterInputChannelsToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_second_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_second_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_second_dim(grad_output)


class _ScatterInputChannelsToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_dim(input_, 1)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_dim(input_, 1)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_dim(grad_output, 1)


# -----------------
# Helper functions.
# -----------------


def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_input_channels_to_tensor_model_parallel_region(input_):
    return _ScatterInputChannelsToModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    return _ScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region(input_, to_model_parallel=True):
    return _GatherFromSequenceParallelRegion.apply(input_, to_model_parallel)


def reduce_scatter_to_sequence_parallel_region(input_: Tensor) -> Tensor:
    return _ReduceScatterToSequenceParallelRegion.apply(input_, 0)


def reduce_scatter_to_tensor_model_parallel_region_with_dim(
    input_: Tensor,
    partition_dim: int,
) -> Tensor:
    """performs a reduce-scatter within TP group, with the scatter happening across
    the user-specified dimension."""
    return _ReduceScatterToSequenceParallelRegion.apply(input_, partition_dim)


def gather_from_tensor_model_parallel_region_with_dim(
    input_: Tensor,
    gather_dim: int,
) -> Tensor:
    """performs a all-gather within TP group, with the gather happening across
    the user-specified dimension."""
    return _GatherFromSequenceParallelRegion.apply(input_, gather_dim, False)


def enter_expert_parallel_region(x: Tensor, scatter_gather: bool) -> Tensor:
    """used to enter expert-parallel (EP) region.

    parallelism dimensions:
    * before (non-expert region): [PP, DP,        TP]
    * after  (expert region)    : [PP, DPEXP, EP, TP]
    * satisfy DP == DPEXP * EP

    Args:
        x: (e, c, h) or (e, c/sp, h) if SP. routed activations, where index along
            the e dimension determines which expert the activation needs to go to. \
            contains a subset of tokens to be handled by each expert.
        scatter_gather: whether to apply scatter-gather optimization to reduce
            communication volume. currently this should be set to True when sequence
            length is divisible by tp degree.

    Returns:
        x: (e/ep, ep, c, h): contains a subset of tokens (partitioned w/ DP_EXP) \
            only for the experts that are associated with this EP rank.

    """
    e, c, h = x.shape

    # add dimension to make it easier to track tokens
    x = x.view(e, 1, c, h)

    # DROP DUPLICATE_TOKENS: (e, 1, c, h) -> (e, 1, c/sp, h)
    if scatter_gather:
        x = _ScatterToSequenceParallelRegion.apply(x, 2)

    # SWAP PARTITION DIMENSION, ENTER EP: (e, 1, c/sp, h) -> (e/ep, ep, c/sp, h)
    x = _AllToAllInExpertParallelRegion.apply(x, 0, 1)

    # REGATHER DUPLICATE TOKENS: (e/ep, ep, c/sp, h) -> (e/ep, ep, c, h)
    if scatter_gather:
        x = _GatherFromSequenceParallelRegion.apply(x, 2, False)

    return x


def exit_expert_parallel_region(x: Tensor, scatter_gather: bool) -> Tensor:
    """used to exit expert-parallel (EP) region.

    parallelism dimensions:
    * before (expert region)    : [PP, DPEXP, EP, TP]
    * after  (non-expert region): [PP, DP,        TP]
    * and satisfy DP == DPEXP * EP

    Args:
        x: (e/ep, ep, c, h): contains a subset of tokens  \
           that are assigned to the subset of experts that are associated with \
           this EP rank.
        scatter_gather: whether to apply scatter-gather optimization to reduce
            communication volume. currently this should be set to True when sequence

    Returns:
        x: (e, c, h)
    """
    e, p, c, h = x.shape

    # DROP DUPLICATE_TOKENS: (e/ep, ep, c, h) -> (e/ep, ep, c/sp, h)
    if scatter_gather:
        x = _ScatterToSequenceParallelRegion.apply(x, 2)

    # SWAP PARTITION DIMENSION, EXIT EP: (e/ep, ep, c/sp, h) -> (e, 1, c/sp, h)
    x = _AllToAllInExpertParallelRegion.apply(x, 1, 0)

    # REGATHER DUPLICATE TOKENS: (e, 1, c/sp, h) -> (e, 1, c, h)
    if scatter_gather:
        x = _GatherFromSequenceParallelRegion.apply(x, 2, False)

    # drop the extra dimension: (e, c, h)
    x = x.squeeze(1)

    return x
