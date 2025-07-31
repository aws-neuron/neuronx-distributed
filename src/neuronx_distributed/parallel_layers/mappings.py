import functools
from typing import Any, Callable, Optional, Tuple, Sequence, cast

import torch
from torch import Tensor
from torch.autograd import Function
import torch.distributed
from torch.distributed import ProcessGroup
import torch_xla.core.xla_model as xm
from torch_neuronx.xla_impl.ops import nki_jit

import neuronxcc.nki.nccl as nccl
import neuronxcc.nki.language as nl
from neuronxcc.nki.compiler.backends.neuron.dimensions import CCPipeline  # noqa: N811
import numpy as np

from . import parallel_state

from .parallel_state import (
    get_expert_model_parallel_replica_groups,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_size,
)
from .utils import split_tensor_along_last_dim, split_tensor_along_dim, divide
from .comm import all_gather, reduce_scatter, all_reduce

if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base
if "reduce_scatter_tensor" not in dir(torch.distributed):
    torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base


def nonzero_partition_dim_swap(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that internally swaps the partition/gather dim with 0-dimension. To the
    outside the partition_dim appears to be the (arbitrary) partition dimension. Internally,
    partition/split dimension is always 0 - which is achieved by pre- and post-transpose."""

    @functools.wraps(func)
    def wrapped_fn(x: Tensor, partition_dim: int, *args, **kwargs) -> Tensor:
        x_t = x.transpose(0, partition_dim) if partition_dim != 0 else x
        y_t: Tensor = func(x_t, 0, *args, **kwargs)
        y = y_t.transpose(0, partition_dim) if partition_dim != 0 else y_t

        return y

    return wrapped_fn


def _reduce(input_: torch.Tensor, computation=xm.REDUCE_SUM, process_group: Optional[ProcessGroup] = None) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""

    group: ProcessGroup = process_group if process_group is not None else get_tensor_model_parallel_group()
    # Bypass the function if we are using only 1 device.
    if group.size() == 1:
        return input_

    tensor_bucket = all_reduce(computation, input_, groups=group)
    return tensor_bucket[0]


def _split_along_last_dim(input_: torch.Tensor, process_group: Optional[ProcessGroup] = None) -> torch.Tensor:
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    return _split_along_dim(input_, len(input_.shape)-1, process_group=process_group)


def _split_along_first_dim(input_: Tensor, process_group: Optional[ProcessGroup] = None) -> Tensor:
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    return _split_along_dim(input_, 0, process_group=process_group)


def _split_along_dim(input_: Tensor, partition_dim: int, process_group: Optional[ProcessGroup] = None) -> Tensor:
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    group: ProcessGroup = process_group if process_group is not None else get_tensor_model_parallel_group()
    world_size = group.size()
    # Bypass the function if we are using only 1 device.
    if world_size == 1:
        return input_

    rank = group.rank()
    dim_size = divide(input_.shape[partition_dim], world_size)
    rank_range = torch.arange(dim_size, device=input_.device) + rank * dim_size
    return torch.index_select(input_, partition_dim, rank_range)


@nonzero_partition_dim_swap
def _gather_along_dim(x: Tensor, partition_dim: int, process_group: Optional[ProcessGroup] = None,
                      tile_cc=False) -> Tensor:
    """Given a tensor partitioned across the specified dimension,
    gather and concatenate along partition dimension (using TP/SP group).
    """
    tp_group = process_group if process_group is not None else get_tensor_model_parallel_group()

    # bpyass the function if we only have 1 TP rank.
    if tp_group.size() == 1:  # type: ignore
        return x

    if tile_cc:
        tp_size = tp_group.size()  # type: ignore
        shape = list(x.shape)
        shape[partition_dim] *= tp_size
        output = torch.empty(shape, dtype=x.dtype, device=x.device)

        _traced_tiled_ag[(CCPipeline(1),)](x, output, cc_dim=partition_dim, tp_rank=tp_size)

        return output

    output = all_gather(
        x,
        dim=partition_dim,
        groups=tp_group,
        pin_layout=False,
    )

    return output.contiguous()


def _gather_along_first_dim(x: Tensor, process_group: Optional[ProcessGroup] = None) -> Tensor:
    return _gather_along_dim(x, partition_dim=0, process_group=process_group)


def _gather_along_last_dim(x: Tensor, process_group: Optional[ProcessGroup] = None) -> Tensor:
    return _gather_along_dim(x, partition_dim=len(x.shape) - 1, process_group=process_group)


def _reduce_scatter_along_first_dim(x: Tensor, computation=xm.REDUCE_SUM,
                                    process_group: Optional[ProcessGroup] = None) -> Tensor:
    return _reduce_scatter_along_dim(x, partition_dim=0, computation=computation, process_group=process_group)

# NKI cc tile related functions
xm_compute_to_np = {
    xm.REDUCE_MAX: np.max,
    xm.REDUCE_SUM: np.add,
}

def _get_tensor_refs_tiled_nki_cc(multi_rank_tensor, single_rank_tensor, cc_dim, tp_rank, num_tiles, tile_id):
    mr_shape = multi_rank_tensor.shape
    assert len(mr_shape) == 3
    assert (*mr_shape[:cc_dim], mr_shape[cc_dim] // tp_rank, *mr_shape[cc_dim + 1:]) == single_rank_tensor.shape

    S = mr_shape[cc_dim]
    assert S % (num_tiles * tp_rank) == 0, \
        f'tiled CC expects S % (num_tiles * tp_rank) == 0; but got ({S}) % ({num_tiles} * {tp_rank}).  ' \
        f'multi_rank_tensor.shape={multi_rank_tensor.shape}  single_rank_tensor.shape={single_rank_tensor.shape}'

    mr_tile_size = S // num_tiles
    sr_tile_size = S // tp_rank // num_tiles

    if cc_dim == 0:
        mr_ref = multi_rank_tensor[
            nl.arange(mr_tile_size)[:, None, None] + tile_id * mr_tile_size,
            nl.arange(mr_shape[1])[None, :, None],
            nl.arange(mr_shape[2])[None, None, :]]
        sr_ref = single_rank_tensor[
            nl.arange(sr_tile_size)[:, None, None] + tile_id * sr_tile_size,
            nl.arange(mr_shape[1])[None, :, None],
            nl.arange(mr_shape[2])[None, None, :]]
    elif cc_dim == 1:
        mr_ref = multi_rank_tensor[
            nl.arange(mr_shape[0])[:, None, None],
            nl.arange(mr_tile_size)[None, :, None] + tile_id * mr_tile_size,
            nl.arange(mr_shape[2])[None, None, :]]
        sr_ref = single_rank_tensor[
            nl.arange(mr_shape[0])[:, None, None],
            nl.arange(sr_tile_size)[None, :, None] + tile_id * sr_tile_size,
            nl.arange(mr_shape[2])[None, None, :]]
    else:
        raise NotImplementedError

    return mr_ref, sr_ref


def tiled_nki_rs(src_tensor, dst_tensor, cc_dim, tp_rank, op=xm.REDUCE_SUM):
    """
    NKI kernel doing reduce-scatter with a compiler attribute to request tiling on the src/dst
    tensors' outermost dimension.
    The kernel should be compiled with NKI's 'early-inline' experimental flag so that compiler's
    collective tiling logic will transform the collective instruction accordingly and perform fusion
    with the tiled upstream/downstream compute.
    """
    cc = nccl.reduce_scatter(
        op=xm_compute_to_np[op], srcs=[src_tensor], dsts=[dst_tensor],
        replica_groups=[list(range(tp_rank))], reduce_scatter_dim=cc_dim)
    cc.set_attr('tile_outermost_dim', 1)
    return


def spmd_tiled_nki_rs(src_tensor, dst_tensor, cc_dim, tp_rank, op=xm.REDUCE_SUM):
    """
    NKI kernel doing reduce-scatter that supports explicit tiling specified via SPMD grid.
    In comparison to tiled_nki_rs, this kernel does manual tiling as opposed to the compiler doing
    the tiling.  So this kernel should not be attached with 'early-inline' attribute.
    """
    num_tiles = nl.num_programs(axes=0)
    tile_id = nl.program_id(axis=0)
    grid_ndim = nl.program_ndim()
    assert grid_ndim == 1, f"tiled_nki_rs only supports specialization along one axis , got grid {grid_ndim}"

    src_ref, dst_ref = _get_tensor_refs_tiled_nki_cc(src_tensor, dst_tensor, cc_dim, tp_rank, num_tiles, tile_id)
    _ = nccl.reduce_scatter(
        op=xm_compute_to_np[op], srcs=[src_ref], dsts=[dst_ref],
        replica_groups=[list(range(tp_rank))], reduce_scatter_dim=cc_dim)


def tiled_nki_ag(src_tensor, dst_tensor, cc_dim, tp_rank):
    """
    NKI kernel doing all-gather with a compiler attribute to request tiling on the src/dst
    tensors' outermost dimension.
    The kernel should be compiled with NKI's 'early-inline' experimental flag so that compiler's
    collective tiling logic will transform the collective instruction accordingly and perform fusion
    with the tiled upstream/downstream compute.
    """
    cc = nccl.all_gather(
        op=np.max, srcs=[src_tensor], dsts=[dst_tensor],
        replica_groups=[list(range(tp_rank))], all_gather_dim=cc_dim)
    cc.set_attr('tile_outermost_dim', 1)
    return


_traced_tiled_rs = nki_jit(experimental_flags='early-inline')(tiled_nki_rs)
_traced_tiled_ag = nki_jit(experimental_flags='early-inline')(tiled_nki_ag)
_traced_spmd_tiled_rs = nki_jit()(spmd_tiled_nki_rs)


def _reduce_scatter_along_last_dim(x: Tensor, computation=xm.REDUCE_SUM,
                                   process_group: Optional[ProcessGroup] = None) -> Tensor:
    return _reduce_scatter_along_dim(x, partition_dim=len(x.shape) - 1, computation=computation,
                                     process_group=process_group)


@nonzero_partition_dim_swap
def _reduce_scatter_along_dim(
        x: Tensor, partition_dim: int, computation=xm.REDUCE_SUM, process_group: Optional[ProcessGroup] = None,
        tile_cc: bool = False,
) -> Tensor:
    """Reduce-scatter the input tensor across model parallel group."""
    tp_group = process_group if process_group is not None else cast(ProcessGroup, get_tensor_model_parallel_group())
    tp_size = tp_group.size()

    # bypass the function if we only have 1 TP/SP rank
    if tp_size == 1:
        return x

    shape = list(x.shape)
    if shape[partition_dim] % tp_size != 0:
        raise RuntimeError(
            f"unable to create {tp_size} partitions along dim {partition_dim} "
            f"of tensor of shape {tuple(x.shape)} due to not being evenly divisible."
        )
    shape[partition_dim] //= tp_size
    output = torch.empty(shape, dtype=x.dtype, device=x.device)

    if tile_cc:
        _traced_tiled_rs[(CCPipeline(1),)](
            x, output, cc_dim=partition_dim, tp_rank=tp_size, op=computation)
        return output

    reduce_scatter(
        computation,
        x.contiguous(),
        scatter_dim=partition_dim,
        shard_count=tp_size,
        scale=1,
        output=output,
        groups=tp_group,
        pin_layout=False,
    )
    return output


def _all_to_all_in_expert_parallel_region(x: Tensor, split_dim: int, concat_dim: int) -> Tensor:
    emp_size = parallel_state.get_expert_model_parallel_size()
    if emp_size == 1:  # bypass if there's no EP.
        return x

    return xm.all_to_all(
        x,
        split_dimension=split_dim,
        concat_dimension=concat_dim,
        split_count=emp_size,
        groups=get_expert_model_parallel_replica_groups(),
        pin_layout=False,
    )


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the tensor model parallel region."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_, process_group):
        return input_

    @staticmethod
    def forward(ctx, input_, process_group: Optional[ProcessGroup] = None):
        ctx.process_group = process_group if process_group is not None \
            else get_tensor_model_parallel_group()
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        process_group = ctx.process_group
        return _reduce(grad_output, process_group=process_group), None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the tensor model parallel region."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_, process_group):
        return _reduce(input_, process_group)

    @staticmethod
    def forward(ctx, input_, process_group: Optional[ProcessGroup] = None):
        return _reduce(input_, process_group=process_group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(graph, input_, process_group):
        return _split_along_last_dim(input_, process_group)

    @staticmethod
    def forward(ctx, input_, process_group: Optional[ProcessGroup] = None):
        process_group = process_group if process_group is not None \
            else get_tensor_model_parallel_group()
        ctx.process_group = process_group
        return _split_along_last_dim(input_, process_group=process_group)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output, process_group=ctx.process_group), None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from tensor model parallel region and concatenate."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(ctx, input_, process_group):
        return _gather_along_last_dim(input_, process_group=process_group)

    @staticmethod
    def forward(ctx, input_, process_group: Optional[ProcessGroup] = None):
        process_group = process_group if process_group is not None \
            else get_tensor_model_parallel_group()
        ctx.process_group = process_group
        return _gather_along_last_dim(input_, process_group=process_group)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output, process_group=ctx.process_group), None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input into TP/SP partitions along specified sequence dimension,
    only keep the corresponding chunk for the current TP rank."""

    @staticmethod
    def symbolic(
        graph, input_: Tensor, partition_dim: int, process_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        return _split_along_dim(input_, partition_dim=partition_dim, process_group=process_group)

    @staticmethod
    def forward(
        ctx, input_: Tensor, partition_dim: int, process_group: Optional[ProcessGroup] = None,
    ) -> Tensor:
        ctx.partition_dim = partition_dim
        process_group = process_group if process_group is not None else get_tensor_model_parallel_group()
        ctx.process_group = process_group
        return _split_along_dim(input_, partition_dim=partition_dim, process_group=process_group)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return _gather_along_dim(grad_outputs[0], ctx.partition_dim, process_group=ctx.process_group,), None, None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather input partitions across TP/SP group and concatenate along specified
    sequence dimension."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(
            graph,
            input_: Tensor,
            partition_dim: int,
            to_model_parallel: bool = True,
            process_group: Optional[ProcessGroup] = None,
            tile_cc: bool = False,
    ) -> Tensor:
        return _gather_along_dim(input_, partition_dim=partition_dim, process_group=process_group, tile_cc=tile_cc)

    @staticmethod
    def forward(
            ctx,
            input_: Tensor,
            partition_dim: int,
            to_model_parallel: bool = True,
            process_group: Optional[ProcessGroup] = None,
            tile_cc: bool = False,
    ) -> Tensor:
        ctx.partition_dim = partition_dim
        ctx.to_model_parallel = to_model_parallel
        ctx.process_group = process_group if process_group is not None else get_tensor_model_parallel_group()
        return _gather_along_dim(input_, partition_dim=partition_dim, process_group=ctx.process_group, tile_cc=tile_cc)

    @staticmethod
    def backward(ctx, *grad_outputs: Any) -> Any:
        partition_dim: int = ctx.partition_dim
        grad_input = (
            _reduce_scatter_along_dim(grad_outputs[0], partition_dim=partition_dim, process_group=ctx.process_group)
            if ctx.to_model_parallel
            else _split_along_dim(grad_outputs[0], partition_dim)
        )
        return grad_input, None, None, None, None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the sequence parallel region and concatenate."""

    # FIXME(mkozuki): Definition of static symbolic methods don't look correct according to
    # https://pytorch.org/docs/stable/onnx.html#static-symbolic-method
    @staticmethod
    def symbolic(
            graph, input_: Tensor, partition_dim: int, process_group: Optional[ProcessGroup] = None, tile_cc: bool = False, dtype: Optional[torch.dtype] = None
    ) -> Tensor:
        process_group = process_group if process_group is not None else get_tensor_model_parallel_group()
        return _reduce_scatter_along_dim(input_, partition_dim=partition_dim, process_group=process_group,
                                         tile_cc=tile_cc)

    @staticmethod
    def forward(
        ctx, input_: Tensor, partition_dim: int, process_group: Optional[ProcessGroup] = None, tile_cc=False, dtype: Optional[torch.dtype] = None
    ) -> Tensor:
        ctx.partition_dim = partition_dim
        process_group if process_group is not None else get_tensor_model_parallel_group()
        ctx.process_group = process_group
        ctx.dtype = dtype
        return _reduce_scatter_along_dim(input_, partition_dim=partition_dim, process_group=process_group,
                                         tile_cc=tile_cc)

    @staticmethod
    def backward(ctx, *grad_outputs: Any) -> Any:
        if ctx.dtype is not None:
            grad_outputs = (grad_outputs[0].to(ctx.dtype),) + grad_outputs[1:]
        return _gather_along_dim(
            grad_outputs[0], partition_dim=ctx.partition_dim, process_group=ctx.process_group,
        ), None, None, None, None


class _AllToAllInExpertParallelRegion(Function):
    @staticmethod
    def symbolic(graph, input_: Tensor, split_dim: int, concat_dim: int) -> Tensor:
        return _all_to_all_in_expert_parallel_region(
            input_,
            split_dim=split_dim,
            concat_dim=concat_dim,
        )

    @staticmethod
    def forward(ctx, input_: Tensor, split_dim: int, concat_dim: int) -> Tensor:
        ctx.split_dim = split_dim
        ctx.concat_dim = concat_dim
        return _all_to_all_in_expert_parallel_region(
            input_,
            split_dim=split_dim,
            concat_dim=concat_dim,
        )

    @staticmethod
    def backward(ctx, *grad_outputs: Any) -> Any:
        # all2all as before but with concat/split dims inverted.
        grad_input = _all_to_all_in_expert_parallel_region(
            grad_outputs[0],
            split_dim=ctx.concat_dim,
            concat_dim=ctx.split_dim,
        )
        return grad_input, None, None


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


def copy_to_tensor_model_parallel_region(input_, process_group: Optional[ProcessGroup] = None):
    return _CopyToModelParallelRegion.apply(input_, process_group)


def reduce_from_tensor_model_parallel_region(input_, process_group: Optional[ProcessGroup] = None):
    return _ReduceFromModelParallelRegion.apply(input_, process_group)


def reduce_from_context_model_parallel_region(input_, process_group: Optional[ProcessGroup] = None):
    return _ReduceFromModelParallelRegion.apply(input_, process_group)


def scatter_input_channels_to_tensor_model_parallel_region(input_):
    return _ScatterInputChannelsToModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_, process_group: Optional[ProcessGroup] = None):
    return _ScatterToModelParallelRegion.apply(input_, process_group)


def gather_from_tensor_model_parallel_region(
        input_, process_group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(input_, process_group)


def scatter_to_sequence_parallel_region(
    input_: torch.Tensor,
    sequence_dimension: int = 0,
    process_group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    return _ScatterToSequenceParallelRegion.apply(input_, sequence_dimension, process_group)


def gather_from_sequence_parallel_region(
        input_: torch.Tensor,
        sequence_dimension: int = 0,
        to_model_parallel: bool = True,
        process_group: Optional[ProcessGroup] = None,
        tile_cc: bool = False,
) -> Tensor:
    return _GatherFromSequenceParallelRegion.apply(
        input_, sequence_dimension, to_model_parallel, process_group, tile_cc
    )  # type: ignore


def reduce_scatter_to_sequence_parallel_region(
        input_: Tensor, sequence_dimension: int = 0, process_group: Optional[ProcessGroup] = None, dtype: Optional[torch.dtype] = None,
) -> Tensor:
    return _ReduceScatterToSequenceParallelRegion.apply(
        input_, sequence_dimension, process_group, False, dtype
    )  # type: ignore

def reduce_scatter_to_sequence_parallel_region_tiled(
        input_: Tensor, sequence_dimension: int = 0, process_group: Optional[ProcessGroup] = None,
) -> Tensor:
    return _ReduceScatterToSequenceParallelRegion.apply(
        input_, sequence_dimension, process_group, True
    )  # type: ignore

def reduce_scatter_to_tensor_model_parallel_region_with_dim(
    input_: Tensor,
    partition_dim: int,
    process_group: Optional[ProcessGroup] = None
) -> Tensor:
    """performs a reduce-scatter within TP group, with the scatter happening across
    the user-specified dimension."""
    return _ReduceScatterToSequenceParallelRegion.apply(
        input_, partition_dim, process_group,
    ) # type: ignore


def gather_from_tensor_model_parallel_region_with_dim(
    input_: Tensor,
    gather_dim: int,
    process_group: Optional[ProcessGroup] = None
) -> Tensor:
    """performs a all-gather within TP group, with the gather happening across
    the user-specified dimension."""
    return _GatherFromSequenceParallelRegion.apply(input_, gather_dim, False, process_group)


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


class _ScatterToProcessGroupSPMD(torch.autograd.Function):
    """
        This is a SPMD compatible implementation of Scatter op, where each process will
        get i'th chunk based on input rank. This is used mainly in inference.

        This is same as _ScatterToModelParallelRegion but with rank passed as input.
    """
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        partition_dim: int,
        rank: torch.Tensor,
        process_group: Optional[ProcessGroup] = None,
    ) -> torch.Tensor:
        ctx.partition_dim = partition_dim
        ctx.process_group = process_group if process_group is not None \
            else cast(ProcessGroup, get_tensor_model_parallel_group())
        ctx.save_for_backward(rank)

        assert input_.size(partition_dim) % ctx.process_group.size() == 0, \
            f"tensor dim to scatter ({input_.size(partition_dim)}) must be "\
            f"multiple of group size ({ctx.process_group.size()})"

        numel_per_partition = input_.size(partition_dim) // ctx.process_group.size()
        # indices = [0, 1, 2, ..., numel_per_partition]
        indices = torch.arange(0, numel_per_partition, device=input_.device)

        # rank0 = [0, 1, 2, ...]
        # rank1 = [8, 9, 10, ...]
        indices = indices + (rank * numel_per_partition)

        input_ = torch.index_select(
            input_,
            dim=partition_dim,
            index=indices
        )
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        # we should be using _ScatterToModelParallelRegion
        raise NotImplementedError("backward pass for ScatterToProcessGroupSPMD not implemented")


class _RoundRobinScatterToProcessGroupSPMD(torch.autograd.Function):
    """
        This is a SPMD compatible implementation of Scatter op, where each process will
        get a chunk in which each entry will be offseted by process-group size.

        e.g.
        > input: [1, 2, 3, 4, 5, 6, 7, 8]
        > process_group_size = 2
        > rank 0: [1, 3, 5, 7]
        > rank 1: [2, 4, 6, 8]
    """
    @staticmethod
    def forward(ctx, input_: torch.Tensor, partition_dim: int, rank: torch.Tensor, process_group: Optional[ProcessGroup] = None) -> torch.Tensor:
        ctx.partition_dim = partition_dim
        ctx.process_group = process_group if process_group is not None \
            else cast(ProcessGroup, get_tensor_model_parallel_group())
        ctx.save_for_backward(rank)

        assert input_.size(partition_dim) % ctx.process_group.size() == 0, \
            f"tensor dim to scatter ({input_.size(partition_dim)}) must be "\
            f"multiple of group size ({ctx.process_group.size()})"

        # indices = [0, 2, 4, ..., seq_len]
        indices = torch.arange(0,
                               input_.size(1),       # seq_length
                               ctx.process_group.size(), # stride = process group size
                               device=input_.device)

        # rank0 = [0, 2, 4, ...]
        # rank1 = [1, 3, 5, ...]
        indices = indices + rank

        repeat_shape = []
        for i, s in enumerate(input_.shape):
            if i == partition_dim:
                repeat_shape.append(1)
            else:
                repeat_shape.append(s)

        input_ = torch.gather(
            input_, partition_dim, indices.unsqueeze(-1).repeat(*repeat_shape),
        )
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        raise NotImplementedError("backward pass for RoundRobinScatterToProcessGroupSPMD not implemented")



def scatter_to_process_group_spmd(
    input_: torch.Tensor,
    partition_dim: int,
    rank: torch.Tensor,
    process_group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    return _ScatterToProcessGroupSPMD.apply(
        input_,
        partition_dim,
        rank,
        process_group,
    ) # type: ignore


def round_robin_scatter_to_process_group_spmd(
    input_: torch.Tensor,
    partition_dim: int,
    rank: torch.Tensor,
    process_group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    return _RoundRobinScatterToProcessGroupSPMD.apply(
        input_,
        partition_dim,
        rank,
        process_group,
    ) # type: ignore
