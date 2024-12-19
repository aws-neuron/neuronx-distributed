import torch
from torch_neuronx.xla_impl.ops import TopK

from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_group
from neuronx_distributed.parallel_layers.mappings import  _gather_along_dim

def topk(tensor, k, dim, gather_dim, process_group=None):
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

    if tp_degree == 1:
        return TopK.apply(tensor, k, dim)

    sharded_size = tensor.shape[gather_dim]
    num_dims = len(tensor.shape)

    # find local rank max value and index
    local_value, local_index = TopK.apply(tensor, k, dim=dim)

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
