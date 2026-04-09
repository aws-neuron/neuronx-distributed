from typing import Optional

import torch

from torch_neuronx.xla_impl.ops import Argmax
from torch_neuronx.utils import get_platform_target

from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_group,
)
from neuronx_distributed.parallel_layers.mappings import _gather_along_dim
from neuronx_distributed.utils.utils import hardware

import neuronxcc.nki.language as nl
from nkilib.core.max.cascaded_max import cascaded_max as nki_max


def _can_use_nki_max(
    tensor: torch.Tensor, dim: int, disable_argmax_kernel: bool = False
) -> bool:
    """
    Check if we can use the NKI max kernel.

    Requirements:
    - Hardware: Trn2 or Trn3
    - Tensor: 2D or 3D with shape[0] == 1
    - dim: Must be the last dimension
    - Size: At least 128 elements in reduction dimension

    TODO: Remove these guardrails as kernel support expands.
    """
    # Check if kernel is manually disabled
    if disable_argmax_kernel:
        return False

    # Check hardware compatibility
    hw_type = hardware(get_platform_target())
    if hw_type not in (hardware.TRN2, hardware.TRN3):
        return False

    # Check dimension requirements
    shape = tensor.shape
    num_dims = len(shape)
    if dim != num_dims - 1:
        return False

    # Check minimum reduction size
    if shape[dim] < 128:
        return False

    # Check tensor dimensionality
    return num_dims == 2 or (num_dims == 3 and shape[0] == 1)


def argmax(
    tensor: torch.Tensor,
    dim: int,
    gather_dim: int,
    keepdim: bool = False,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
    disable_argmax_kernel: bool = False,
) -> torch.Tensor:
    """Performs distributed argmax on sharded tensors.

    Calculates argmax across all sharded tensors in a distributed environment.
    Similar to torch.argmax but handles tensor-parallel sharding.

    Args:
        tensor: Input tensor to perform argmax on.
        dim: Dimension along which to find argmax.
        gather_dim: Dimension the tensor is sharded on.
        keepdim: Whether to keep the reduced dimension. Defaults to False.
        process_group: Process group for distributed operations.
            Uses tensor model parallel group if None.
        disable_argmax_kernel: Whether to use torch.argmax instead of the NKI
            argmax kernel. Defaults to False.

    Returns:
        Tensor with global argmax indices across all shards.

    Example:
        Sharded tensor shape (1, 4), dim=1, tp_degree=2, keepdim=False, gather_dim=1
        Returns tensor of shape (1,).
    """
    # NxD distributed state
    process_group = process_group or get_tensor_model_parallel_group(as_list=False)
    tp_degree = torch.distributed.get_world_size(group=process_group)

    # Fast path for single LNC
    if tp_degree == 1:
        return Argmax.apply(tensor, dim, keepdim)

    # Find local max values and indices
    local_value, local_index = _compute_local_max(tensor, dim, disable_argmax_kernel)

    # Gather results from all ranks
    global_values = _gather_along_dim(
        local_value, gather_dim, process_group=process_group
    )
    global_indices = _gather_along_dim(
        local_index, gather_dim, process_group=process_group
    )

    # Correct indices for sharding offset when gather_dim == dim
    if gather_dim == dim:
        global_indices = _apply_sharding_offset(
            global_indices, dim, tensor.shape[gather_dim], tp_degree
        )

    # Find global argmax and extract final indices
    global_argmax = Argmax.apply(global_values, dim=dim, keepdim=True)
    final_indices = torch.gather(global_indices, dim, global_argmax)

    if not keepdim:
        return final_indices.squeeze(dim)
    return final_indices


def _compute_local_max(
    tensor: torch.Tensor, dim: int, disable_argmax_kernel: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local max using NKI kernel when possible, otherwise torch.max."""
    if _can_use_nki_max(tensor, dim, disable_argmax_kernel):
        # NKI kernel requires 2D input, squeeze if needed
        # TODO: remove this when kernel support is expanded
        is_3d = len(tensor.shape) == 3
        input_tensor = tensor.squeeze(0) if is_3d else tensor

        value, index = nki_max[2](input_tensor)

        # Restore dimension if squeezed
        if is_3d:
            value = value.unsqueeze(0)
            index = index.unsqueeze(0)
    else:
        # Fallback to torch.max for cases that don't meet nki_max criteria
        value, index = torch.max(tensor, dim=dim, keepdim=True)

    return value, index


def _apply_sharding_offset(
    indices: torch.Tensor, dim: int, shard_size: int, tp_degree: int
) -> torch.Tensor:
    """Apply offset to indices to account for tensor sharding."""
    offset_shape = [1] * len(indices.shape)
    offset_shape[dim] = tp_degree

    offset = torch.arange(0, shard_size * tp_degree, shard_size)
    offset = offset.view(offset_shape)

    return indices + offset
