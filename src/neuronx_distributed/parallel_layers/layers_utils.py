import torch
from typing import (
    Optional, Tuple, Any
)
from torch.distributed import ProcessGroup

from .parallel_state import (
    get_tensor_model_parallel_group,
)

from .mappings import (
    _gather_along_dim,
    _reduce_scatter_along_dim,
)

def _linear_autograd_base_setup_fwd(
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
    '''This function performs the setup related tasks which are part
    of the forward autograd of the linearAsyncCommunication function.
    It saves attributes and tensors to the context so they can be used
    during the backward pass. This is done to reduce code redundancy when
    inheriting the layer.'''

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
    # If sequence parallel is enabled: `input` is supposed to be 3D and the optimal order of dimension is 
    # [sequence, batch, hidden] If not SBH, the necessary transposes will be added
    return _gather_along_dim(input, ctx.sequence_dimension, process_group=ctx.process_group) if sequence_parallel_enabled else input


def _linear_autograd_base_setup_bwd(
    ctx,
    grad_outputs: Any,
) ->  Tuple[torch.tensor, torch.tensor, torch.tensor]:
    '''This function performs the setup related tasks which are part of
    the backward autograd of the linearAsyncCommunication function.
    It saves attributes and tensors to the context so they can be used
    during the backward pass. This is done to reduce code redundancy when
    inheriting the layer.'''
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

    #Setting this to None initially in case compute_weight_gradient is False
    total_input = None
    if ctx.compute_weight_gradient:
        if ctx.sequence_parallel_enabled:
            # Optimal layout is SBH, but if not, transposes are added
            total_input = _gather_along_dim(input, ctx.sequence_dimension, process_group=ctx.process_group)
        else:
            total_input = input

    return total_input, weight, grad_output

def _linear_autograd_bwd_grad_reduce(
    ctx,
    grad_input: torch.Tensor,
    original_dtype
) -> Optional[Any]:
    ''' This function performs the all reduce of the input gradient'''
    handle=None
    if ctx.async_grad_allreduce:
        # Asynchronous all-reduce
        grad_input = grad_input.to(ctx.reduce_dtype)
        handle = torch.distributed.all_reduce(grad_input, group=ctx.process_group, async_op=True)
        grad_input = grad_input.to(original_dtype)
    return handle

def _linear_autograd_bwd_no_weight_grad(
    ctx,
    grad_input,
    original_dtype
) -> torch.Tensor:
    ''' This function returns the input gradient scatter as per the TP
    degree as we dont want to calculate the weight grad.'''
    
    assert not ctx.async_grad_allreduce
    # Optimal layout is SBH, but if not, transposes are added
    sub_grad_input = _reduce_scatter_along_dim(grad_input.to(ctx.reduce_dtype), ctx.sequence_dimension, process_group=ctx.process_group)
    sub_grad_input = sub_grad_input.to(original_dtype)
    return sub_grad_input

def _linear_autograd_bwd_input_grad(
    ctx,
    grad_input,
    handle,
    original_dtype
) -> Tuple[Any, Any]:
    '''This function does the input grad scatter or just converts
    the dtype of the input grad depending on whether sequence parallel is enabled
    or not'''
    sub_grad_input = None
    if ctx.sequence_parallel_enabled:
        assert not ctx.async_grad_allreduce
        # optimal layout is SBH, but if not, transposes will be added
        sub_grad_input = _reduce_scatter_along_dim(grad_input.to(ctx.reduce_dtype), ctx.sequence_dimension, process_group=ctx.process_group)
        sub_grad_input = sub_grad_input.to(original_dtype)

    if ctx.async_grad_allreduce:
        assert handle
        handle.wait()
        grad_input = grad_input.to(original_dtype)

    return sub_grad_input, grad_input