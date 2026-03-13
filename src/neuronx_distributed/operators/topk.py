import torch
import torch_xla.core.xla_model as xm
from torch_neuronx.xla_impl.ops import TopK
from torch_neuronx.utils import get_platform_target

import math
from itertools import islice

import neuronxcc.nki.language as nl

from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_group
from neuronx_distributed.parallel_layers.mappings import  _gather_along_dim
from neuronx_distributed.kernels.topk import topk_rotated
from neuronx_distributed.utils.utils import hardware

try:
    from neuronxcc.nki._private_kernels.topk.topk import topk as nki_topk
except ImportError:
    from neuronxcc.nki._pre_prod_kernels.topk.topk import topk as nki_topk


def _is_nki_topk_available():
    hardware_type = hardware(get_platform_target())
    return hardware_type == hardware.TRN2 or hardware_type == hardware.TRN3

def _get_topk_wrappers(topk_implementation, lnc):
    """
    Returns two wrapper functions for topk implementation:
    1. topk_without_sorted: calls topk with sorting disabled (if supported)
    2. topk_with_sorted: calls topk with sorting enabled
    
    Args:
        topk_implementation: The topk implementation to wrap
        
    Returns:
        tuple: (topk_without_sorted_func, topk_with_sorted_func)
    """
    
    # Detect implementation type once
    func_name = getattr(topk_implementation, 'func_name', None)
    is_topk_rotated = func_name == getattr(topk_rotated, 'func_name', None)
    is_nki_topk = func_name == getattr(nki_topk, 'func_name', None)
    
    def create_wrapper(sorted_output):
        """Factory to create topk wrapper with specified sorting behavior"""
        def wrapper(tensor, k, dim):
            if is_topk_rotated:
                return topk_implementation[(lnc,)](tensor, k, dim=dim, sorted=sorted_output)
            elif is_nki_topk:
                # FIXME: sorted=False gives error, need to fix it in kernel
                return topk_implementation[(lnc,)](tensor, k, sorted=True)
            else:
                return topk_implementation(tensor, k, dim=dim)
        
        return wrapper
    
    return create_wrapper(False), create_wrapper(True)


def get_topk_implementation(use_topk_rotated_kernel=False, lnc=2, dim=-1, tensor=None, stages=1):
    hardware_type = hardware(get_platform_target())
    is_trn1 = hardware_type == hardware.TRN1
    lnc = lnc if is_trn1 else nl.nc(lnc)
    if use_topk_rotated_kernel:
        topk_implementation = topk_rotated
        assert stages == 1, "stages other than 1 is not supported when using topk_rotated kernel"
    else:
        # check if nki topk kernel is available, if so, always prefer 1 stage (k%8==0 will be removed after kernel update)
        can_use_nki_topk = dim in (-1, len(tensor.shape) - 1) and _is_nki_topk_available()
        if can_use_nki_topk:
            stages = 1
            topk_implementation = nki_topk
        else:
            topk_implementation = TopK.apply

    topk_implementation, call_topk_kernel_with_sorted_parameter = _get_topk_wrappers(topk_implementation, lnc)

    return topk_implementation, call_topk_kernel_with_sorted_parameter, stages


def topk(tensor, k, dim, gather_dim, process_group=None, stages=1, rank_id=None, use_topk_rotated_kernel=False, lnc=2):
    """
    This function performs a distributed topk.
    This function will take in a sharded tensor,
    and then calculate topk amongst all the
    sharded tensors in the distributed environment,
    along the provided `dim`. The signature is
    similar to torch.topk, except it also includes
    a parameter called `gather_dim`.

    Example: Given a sharded tensor of shape (1,16) and k=4
    where the dim to find the argmax is 1, tp_degree
    is 2, keepdim is False, and the dim that's been
    sharded is 1. The returned shape of this function
    will be (1,4).

    Arguments:
    1. tensor: the tensor to perform the argmax call
    2. dim: the dimension to find topk along.
    3. gather_dim: the dimension to gather on. This
    should be the dimension the tensor was sharded on.

    Note: stages > 1 enables cascaded reduction. However,
    if nki topk kernel is available, this is always
    overwritten to 1 as nki kernel does internal cascading
    without global cc overhead.

    Note: This method can use different underlying implementations
    for per shard topk calculation based on the the parameters provided
    and nki kernel availability on hardware.
    Refer to the method source code to understand
    when the different implementations will be used
    and the constraints around using those implementations.

    Returns: A tensor representing the global topk
    amongst the sharded tensors.
    """
    # nxd distributed state
    process_group = process_group if process_group is not None else get_tensor_model_parallel_group(as_list=False)
    tp_degree = torch.distributed.get_world_size(group=process_group)
    hardware_type = hardware(get_platform_target())
    is_trn1 = hardware_type == hardware.TRN1
    is_trn2_or_trn3 = (hardware_type == hardware.TRN2 or hardware_type == hardware.TRN3)

    topk_implementation, call_topk_kernel_with_sorted_parameter, stages = get_topk_implementation(use_topk_rotated_kernel, lnc, dim, tensor, stages)

    if stages > 1:
        if is_trn2_or_trn3:
            assert tp_degree == 64, f"tp degree other than 64 is not supported got {tp_degree} on {hardware_type}"
            assert stages == 3, f"stages other than 3 is not supported got {stages} on {hardware_type}"
        else:
            assert tp_degree == 32, f"tp degree other than 32 is not supported got {tp_degree} on {hardware_type}"
            assert stages == 2, f"stages other than 2 is not supported got {stages} on {hardware_type}"

        mesh = []
        if is_trn2_or_trn3: # 4x4x4 topology
            group_size = int(math.ceil(math.pow(tp_degree,1/stages)))
            n_groups = tp_degree//group_size

            mesh.append([list(range(i, i + group_size)) for i in range(0, tp_degree, group_size)])
            mesh.append([list(islice(range(j,tp_degree,group_size),i,i+group_size)) for i in range(0,n_groups,group_size) for j in range(group_size)])
            mesh.append([list(range(i,tp_degree,n_groups)) for i in range(n_groups)])
        elif is_trn1: #8x4 topology
            group_size = 8
            n_groups = tp_degree//group_size

            mesh.append([list(range(i, i + group_size)) for i in range(0, tp_degree, group_size)])
            mesh.append([list(islice(range(j,tp_degree,group_size),i,i+group_size)) for i in range(0,n_groups,group_size) for j in range(group_size)])
        else:
            raise NotImplementedError(f"Unsupported hardware type {hardware_type}")

        stage_pg=[]
        for i in range(stages):
            stage_pg.append(torch.distributed.new_group(mesh[i], pg_options={"xla_pg_options": {"mesh": mesh[i]}}))

    if tp_degree == 1:
        return call_topk_kernel_with_sorted_parameter(tensor, k, dim)

    sharded_size = tensor.shape[gather_dim]
    num_dims = len(tensor.shape)

    # find local rank max value and index
    local_k = k
    if gather_dim == dim:
        local_k = min(k, sharded_size)
    local_value, local_index = topk_implementation(tensor, local_k, dim=dim)

    if stages > 1:
        if gather_dim == dim:
            rank_offset = rank_id*sharded_size
            local_index = local_index + rank_offset.to(torch.int32)
        else:
            raise NotImplementedError
        for i in range(stages):
            local_value = _gather_along_dim(local_value, gather_dim, process_group=stage_pg[i])
            local_index_ = _gather_along_dim(local_index, gather_dim, process_group=stage_pg[i])

            local_value, local_index = topk_implementation(local_value, k, dim=dim)

            local_index = torch.gather(local_index_, dim, local_index)

        return local_value, local_index
    else:
        # perform all-gather on the local rank topk values and indices to get global topk and indices
        global_values = _gather_along_dim(local_value, gather_dim, process_group=process_group)
        global_indices = _gather_along_dim(local_index, gather_dim, process_group=process_group)

    # indices are based on local shard, so we need to correct it by applying
    # an offset derived from tp degree and sharded size. This is only applicable
    # when the gather_dim is equal to the topk dim.
    if gather_dim == dim:
        full_size = sharded_size * tp_degree
        offset = torch.arange(0, full_size, sharded_size)
        offset = offset.repeat_interleave(local_k)
        offset = offset.view([1 if i != dim else -1 for i in range(num_dims)])
        corrected_global_indices = global_indices + offset
    else:
        corrected_global_indices = global_indices

    # calculate the global topk based on the local topk from the global topk values
    # and then retrieve the corrected indices
    values, global_max_local_index = call_topk_kernel_with_sorted_parameter(global_values, k, dim=dim)

    final_indices = torch.gather(corrected_global_indices, dim, global_max_local_index)

    return values, final_indices
