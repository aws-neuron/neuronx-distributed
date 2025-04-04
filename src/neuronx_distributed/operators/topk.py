import torch
from torch_neuronx.xla_impl.ops import TopK
from torch_neuronx.utils import get_platform_target 

import math
from itertools import islice

from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_group
from neuronx_distributed.parallel_layers.mappings import  _gather_along_dim


def topk(tensor, k, dim, gather_dim, process_group=None, stages=1, rank_id=None):
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

    Returns: A tensor representing the global topk
    amongst the sharded tensors.
    """
    # nxd distributed state
    
    process_group = process_group if process_group is not None else get_tensor_model_parallel_group(as_list=False)
    tp_degree = torch.distributed.get_world_size(group=process_group)
    hardware = get_platform_target()

    if stages > 1:
        if hardware == "trn2":
            assert tp_degree == 64, f"tp degree other than 64 is not supported got {tp_degree} on {hardware}"
            assert stages == 3, f"stages other than 3 is not supported got {stages} on {hardware}"
        else:
            assert tp_degree == 32, f"tp degree other than 32 is not supported got {tp_degree} on {hardware}"
            assert stages == 2, f"stages other than 2 is not supported got {stages} on {hardware}"


        mesh = []
        if hardware == "trn2": # 4x4x4 topology
            group_size = int(math.ceil(math.pow(tp_degree,1/stages)))
            n_groups = tp_degree//group_size

            mesh.append([list(range(i, i + group_size)) for i in range(0, tp_degree, group_size)])
            mesh.append([list(islice(range(j,tp_degree,group_size),i,i+group_size)) for i in range(0,n_groups,group_size) for j in range(group_size)])
            mesh.append([list(range(i,tp_degree,n_groups)) for i in range(n_groups)])
        elif hardware == "trn1": #8x4 topology
            group_size = 8
            n_groups = tp_degree//group_size

            mesh.append([list(range(i, i + group_size)) for i in range(0, tp_degree, group_size)])
            mesh.append([list(islice(range(j,tp_degree,group_size),i,i+group_size)) for i in range(0,n_groups,group_size) for j in range(group_size)])
        else:
            raise NotImplementedError(f"Unsupported hardware type {hardware}")

        stage_pg=[]
        for i in range(stages):
            stage_pg.append(torch.distributed.new_group(mesh[i], pg_options={"xla_pg_options": {"mesh": mesh[i]}}))

    if tp_degree == 1:
        return TopK.apply(tensor, k, dim)

    sharded_size = tensor.shape[gather_dim]
    num_dims = len(tensor.shape)

    # find local rank max value and index
    local_value, local_index = TopK.apply(tensor, k, dim=dim)

    if stages > 1:
        if gather_dim == dim:
            rank_offset = rank_id*sharded_size
            local_index = local_index + rank_offset.to(torch.int32)
        else:
            raise NotImplementedError
        for i in range(stages):
            local_value = _gather_along_dim(local_value, gather_dim, process_group=stage_pg[i])
            local_index_ = _gather_along_dim(local_index, gather_dim, process_group=stage_pg[i])

            local_value, local_index = TopK.apply(local_value, k, dim=dim) 
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
        offset = offset.repeat_interleave(k)
        offset = offset.view([1 if i != dim else -1 for i in range(num_dims)])
        corrected_global_indices = global_indices + offset
    else:
        corrected_global_indices = global_indices

    # calculate the global topk based on the local topk from the global topk values
    # and then retrieve the corrected indices
    values, global_max_local_index = TopK.apply(global_values, k, dim=dim)

    final_indices = torch.gather(corrected_global_indices, dim, global_max_local_index)

    return values, final_indices
