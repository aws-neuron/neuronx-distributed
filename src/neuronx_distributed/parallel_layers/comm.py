from neuronx_distributed.utils import cpu_mode
from neuronx_distributed.utils.logger import get_logger
import torch_xla.core.xla_model as xm

import torch.distributed as dist
import torch
from torch.distributed import ProcessGroup
from typing import List, Union, Optional

if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base
if "reduce_scatter_tensor" not in dir(torch.distributed):
    torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base

_reduce_type_mapping = {
    xm.REDUCE_SUM: dist.ReduceOp.SUM,
    xm.REDUCE_MAX: dist.ReduceOp.MAX,
    xm.REDUCE_MIN: dist.ReduceOp.MIN,
}

logger = get_logger(rank0_only=True)


def _get_group_mesh(groups: Union[ProcessGroup, List[List[int]]]) -> List[List[int]]:
    if isinstance(groups, List):
        return groups
    else:
        return groups._mesh


def cpu_reduce_scatter(
    reduce_type: str,
    input: torch.Tensor,
    scale: int = 1,
    scatter_dim: int = 0,
    shard_count: Optional[int] = None,
    groups: Union[ProcessGroup, List[List[int]]] = None,
    output: torch.Tensor = None,
    pin_layout: bool = True,
):
    """Gloo backend for CPU CC ops doesn't reduce_scatter.
    Instead we use torch.distributed.reduce and scatter to
    perform the same operation.
    """
    reduce_type = _reduce_type_mapping[reduce_type]
    world_size = dist.get_world_size(group=groups)
    input_shape = list(input.shape)

    if output is None:
        output_shape = input_shape[::]
        output_shape[scatter_dim] = input_shape[scatter_dim] // world_size
        output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    if shard_count is not None:
        assert (
            shard_count == world_size
        ), "shard_count must be equal to world_size of communication group"

    if input_shape[scatter_dim] % world_size != 0:
        raise RuntimeError(
            f"Cannot reduce on dim {scatter_dim} with input shape {input_shape}"
            f"and world_size {world_size}"
        )

    if scatter_dim != 0:
        input.transpose_(0, scatter_dim)
        output.transpose_(0, scatter_dim)

    gloo_reduce_scatter(output, input, reduce_type, groups, async_op=False)

    if scatter_dim != 0:
        input.transpose_(0, scatter_dim)
        output.transpose_(0, scatter_dim)

    if scale != 1:
        output *= scale

    return output


def gloo_reduce_scatter(
    output: torch.Tensor,
    input: torch.Tensor,
    op: Union[str, dist.ReduceOp],
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
):
    """Reduce the results to rank 0 and scatter the results to all ranks.

    Need to convert the input tensor and scatter tensor to contiguous
    otherwise running into slient data corruption issue.
    """

    world_size = dist.get_world_size(group=group)
    global_rank = dist.get_rank()

    # NOTE: has to have contiguous input tensor
    input = input.contiguous()
    group_ranks = dist.get_process_group_ranks(
        group=group if group is not None else dist.group.WORLD
    )

    # NOTE: here for some reason the dst ranks must be the global rank:
    # cause there is a `get_group_rank(group, dst)` before calling
    # group.reduce(...)
    dist.reduce(input, dst=group_ranks[0], op=op, group=group, async_op=async_op)

    if global_rank == group_ranks[0]:
        # always assume the tensor is transpose to scatter on dim 0
        scattered_tensors = list(torch.split(input, input.size(0) // world_size, dim=0))
        scattered_tensors = [t.contiguous() for t in scattered_tensors]
    else:
        scattered_tensors = None

    # NOTE: same as dist.reduce function, it requires global rank as the src
    dist.scatter(
        output, scattered_tensors, src=group_ranks[0], group=group, async_op=async_op
    )

    return output


def reduce_scatter(
    reduce_type: Union[str, dist.ReduceOp],
    input: torch.Tensor,
    scale: int,
    scatter_dim: int,
    shard_count: int,
    groups: Union[ProcessGroup, List[List[int]]] = None,
    output: torch.Tensor = None,
    pin_layout: bool = True,
):
    """wrapper function to handle the reduce scatter for CPU mode
    and XLA mode. Aligned the function signature with the xla_model.reduce_scatter
    """
    if not cpu_mode():
        groups = _get_group_mesh(groups)

        return xm.reduce_scatter(
            reduce_type=reduce_type,
            input=input,
            scale=scale,
            scatter_dim=scatter_dim,
            shard_count=shard_count,
            groups=groups,
            output=output,
            pin_layout=pin_layout,
        )
    else:
        cpu_reduce_scatter(
            reduce_type=reduce_type,
            input=input,
            scale=scale,
            scatter_dim=scatter_dim,
            shard_count=shard_count,
            groups=groups,
            output=output,
            pin_layout=pin_layout,
        )


def all_gather(
    input: torch.Tensor,
    dim: int = 0,
    groups: Union[ProcessGroup, List[List[int]]] = None,
    output: torch.Tensor = None,
    pin_layout: bool = True,
):
    """wrapper function to handle the all gather for CPU mode
    and XLA mode. Aligned the function signature with the xla_model.all_gather
    """
    if not cpu_mode():
        groups = _get_group_mesh(groups)

        return xm.all_gather(
            input,
            dim=dim,
            groups=groups,
            output=output,
            pin_layout=pin_layout,
        )
    else:
        # cpu mode, need to have the output tensor as a list
        # if there is a output provided, then we copy the data into the
        # output tensor

        world_size = dist.get_world_size(group=groups)
        gathered_tensors = [torch.empty_like(input) for _ in range(world_size)]

        dist.all_gather(gathered_tensors, input, group=groups)

        if output is not None:
            output.copy_(torch.cat(gathered_tensors, dim=dim).contiguous())
            return output
        else:
            return torch.cat(gathered_tensors, dim=dim).contiguous()


def all_reduce(
    op_type: str,
    tensor_bucket: Union[torch.Tensor, List[torch.Tensor]],
    groups: Union[ProcessGroup, List[List[int]]],
    pin_layout: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """wrapper function to handle the all-reduce for both CPU mode and XLA mode
    It is an in-place operation, so the input tensor will be modified.
    """
    if cpu_mode():
        op_type = _reduce_type_mapping[op_type]
        if isinstance(tensor_bucket, torch.Tensor):
            tensor_bucket = [tensor_bucket]
        torch.distributed.all_reduce_coalesced(tensor_bucket, op=op_type, group=groups)
    else:
        if isinstance(tensor_bucket, torch.Tensor):
            tensor_bucket = [tensor_bucket]
        groups = _get_group_mesh(groups)
        tensor_bucket = xm.all_reduce(op_type, tensor_bucket, groups=groups, pin_layout=pin_layout)
    return tensor_bucket
