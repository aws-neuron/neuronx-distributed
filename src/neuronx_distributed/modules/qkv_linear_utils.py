import torch
import torch_xla.core.xla_model as xm

from typing import (
    Optional, Tuple, Any
)
from torch.distributed import ProcessGroup
from neuronx_distributed.parallel_layers import parallel_state
from ..parallel_layers.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_replica_groups,
    get_tensor_model_parallel_size,
)

from ..parallel_layers.mappings import (
    _gather_along_dim,
    _reduce_scatter_along_dim,
)

def check_use_bias(weight_qkv, fuse_qkv, weight_q, bias_q, bias_qkv):
    return (bias_qkv is not None if fuse_qkv else bias_q is not None) and check_requires_grad(
        weight_qkv, fuse_qkv, weight_q
    )

def check_requires_grad(weight_qkv, fuse_qkv, weight_q):
    return weight_qkv.requires_grad if fuse_qkv else weight_q.requires_grad

def _qkvlinear_autograd_base_setup_fwd(
        ctx,
        input: torch.Tensor,
        weight_q: Optional[torch.Tensor],
        weight_k: Optional[torch.Tensor],
        weight_v: Optional[torch.Tensor],
        bias_q: Optional[torch.Tensor],
        bias_k: Optional[torch.Tensor],
        bias_v: Optional[torch.Tensor],
        async_grad_allreduce: bool,
        sequence_parallel_enabled: bool,
        kv_size_multiplier: int,
        weight_qkv: Optional[torch.Tensor] = None,
        bias_qkv: Optional[torch.Tensor] = None,
        fuse_qkv: bool = False,
        output_size_q: Optional[int] = None,
        output_size_kv: Optional[int] = None,
        reduce_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    '''This function performs the setup related tasks which are part
    of the forward autograd of the gqaqkvlinearAsyncCommunication function.
    It saves attributes and tensors to the context so they can be used
    during the backward pass. This is done to reduce code redundancy when
    inheriting the layer.'''

    ctx.use_bias = check_use_bias(weight_qkv, fuse_qkv, weight_q, bias_q, bias_qkv)
    ctx.async_grad_allreduce = async_grad_allreduce
    ctx.sequence_parallel_enabled = sequence_parallel_enabled
    ctx.compute_weight_gradient = check_requires_grad(weight_qkv, fuse_qkv, weight_q)
    ctx.kv_size_multiplier = kv_size_multiplier
    ctx.fuse_qkv = fuse_qkv
    ctx.reduce_dtype = reduce_dtype

    if ctx.compute_weight_gradient:
        if ctx.fuse_qkv:
            ctx.save_for_backward(input, weight_qkv)
        else:
            ctx.save_for_backward(input, weight_q, weight_k, weight_v)
    else:
        if ctx.fuse_qkv:
            ctx.save_for_backward(weight_qkv)
        else:
            ctx.save_for_backward(weight_q, weight_k, weight_v)

    if ctx.sequence_parallel_enabled:
        # `input` is supposed to be 3D and its order of dimension is [sequence, batch, hidden]
        total_input = xm.all_gather(
            input,
            groups=get_tensor_model_parallel_replica_groups(),
            pin_layout=False,
        )
    else:
        total_input = input

    return total_input

def _qkvlinear_autograd_base_setup_bwd(
    ctx,
    grad_output_q: torch.Tensor,
    grad_output_k: torch.Tensor,
    grad_output_v: torch.Tensor,
) -> Tuple[torch.Tensor, Any, Any, Any, Any, Any, Any]:
    '''This function performs the setup related tasks which are part of
    the backward autograd of the gqaqkvlinearAsyncCommunication function.
    It saves attributes and tensors to the context so they can be used
    during the backward pass. This is done to reduce code redundancy when
    inheriting the layer.'''

    input = None
    weight_qkv = None
    weight_q = None
    weight_k = None
    weight_v = None
    total_input = None
    
    if ctx.compute_weight_gradient:
        if ctx.fuse_qkv:
            input, weight_qkv = ctx.saved_tensors
        else:
            input, weight_q, weight_k, weight_v = ctx.saved_tensors
    else:
        if ctx.fuse_qkv:
            weight_qkv = ctx.saved_tensors[:1]
        else:
            weight_q, weight_k, weight_v = ctx.saved_tensors[:3]
        input = None

    if ctx.compute_weight_gradient:
        if ctx.sequence_parallel_enabled:
            total_input = xm.all_gather(
                input,
                groups=get_tensor_model_parallel_replica_groups(),
                pin_layout=False,
            )
        else:
            total_input = input
    
    if ctx.kv_size_multiplier > 1:
        # Since we repeat the K and V by a factor of kv_size_multipler, we need to
        # sum up the gradients from the repeated portions. get_kv_shared_group()
        # returns the ranks which have the same K and V heads, and hence allows us to
        # sum up from the distributed ranks.
        original_dtype = grad_output_k.dtype
        grad_output_k = grad_output_k.to(ctx.reduce_dtype)
        grad_output_v = grad_output_v.to(ctx.reduce_dtype)

        outs = xm.all_reduce(
            xm.REDUCE_SUM,
            [grad_output_k, grad_output_v],
            scale=1.0,
            groups=parallel_state.get_kv_shared_replica_groups(),
            pin_layout=False
        )
        grad_output_k, grad_output_v = outs[0].to(original_dtype), outs[1].to(original_dtype)

    
    return total_input, weight_qkv, weight_q, weight_k, weight_v, grad_output_k, grad_output_v

def _qkvlinear_autograd_bwd_grad_reduce(
    ctx,
    grad_input: torch.Tensor,
    original_dtype
) -> Optional[Any]:
    ''' This function performs the all reduce of the input gradient
    if async_grad_allreduce is enabled'''
    
    if ctx.async_grad_allreduce:
        # Asynchronous all-reduce
        grad_input = grad_input.to(ctx.reduce_dtype)
        torch.distributed.all_reduce(grad_input, group=get_tensor_model_parallel_group())
        grad_input=grad_input.to(original_dtype)
    
    return grad_input

def _qkvlinear_autograd_bwd_no_weight_grad(
    ctx,
    grad_input,
    original_dtype
) -> torch.Tensor:
    ''' This function returns the input gradient scatter as per the TP
    degree as we dont want to calculate the weight grad.'''
    
    assert not ctx.async_grad_allreduce
    world_size = get_tensor_model_parallel_size()
    shape = list(grad_input.shape)
    shape[0] //= world_size

    grad_input = grad_input.to(ctx.reduce_dtype)

    sub_grad_input = torch.empty(
        torch.Size(shape),
        dtype=grad_input.dtype,
        device=grad_input.device,
        requires_grad=False,
    )

    xm.reduce_scatter(
        xm.REDUCE_SUM,
        grad_input,
        output=sub_grad_input,
        groups=get_tensor_model_parallel_replica_groups(),
        shard_count=world_size,
        scatter_dim=0,
        scale=1,
        pin_layout=False,
    )

    sub_grad_input=sub_grad_input.to(original_dtype)

    return sub_grad_input

def _qkvlinear_autograd_bwd_input_grad(
    ctx,
    grad_input,
    original_dtype
) -> torch.Tensor:
    ''' This function returns the input gradient scatter as per the TP
    degree.'''

    grad_input = grad_input.to(ctx.reduce_dtype)
    sub_grad_input = torch.empty(grad_input.shape, dtype=grad_input.dtype, device=grad_input.device, requires_grad=False)
    xm.reduce_scatter(
        xm.REDUCE_SUM,
        grad_input,
        output=sub_grad_input,
        groups=get_tensor_model_parallel_replica_groups(),
        shard_count=get_tensor_model_parallel_size(),
        scatter_dim=0,
        scale=1,
        pin_layout=False,
    )
    sub_grad_input=sub_grad_input.to(original_dtype)

    return sub_grad_input