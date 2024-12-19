import torch
import math
import torch.nn.functional as F
from torch_neuronx.xla_impl.ops import Argmax

from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_group,
)
from neuronx_distributed.parallel_layers.mappings import _gather_along_dim


def argmax(tensor, dim, gather_dim, keepdim=False, process_group=None):
    """
    This function performs a distributed argmax.
    This function will take in a sharded tensor,
    and then calculate the max amongst all the
    sharded tensors in the distributed environment,
    along the provided `dim`. The signature is
    similar to torch.argmax, except it also includes
    a parameter called `gather_dim`.

    Example: Given a sharded tensor of shape (1,4)
    where the dim to find the argmax is 1, tp_degree
    is 2, keepdim is False, and the dim that's been
    sharded is 1. The returned shape of this function
    will be (1,).

    Arguments:
    1. tensor: the tensor to perform the argmax call
    2. dim: the dimension to find the argmax along.
    3. gather_dim: the dimension to gather on. This
    should be the dimension the tensor was sharded on.
    4. keepdim: whether to keep or drop the dim
    specified. The default is False.

    Returns: A tensor representing the global argmax
    amongst the sharded tensors.
    """
    
    process_group = process_group if process_group is not None else get_tensor_model_parallel_group(as_list=False)
    
    # nxd distributed state
    tp_degree = torch.distributed.get_world_size(group=process_group)

    if tp_degree == 1:
        return Argmax.apply(tensor, dim, keepdim)

    sharded_size = tensor.shape[gather_dim]
    num_dims = len(tensor.shape)

    # find local rank max value and index
    local_value, local_index = torch.max(tensor, dim=dim, keepdim=True)

    # perform all-gather on the local rank max values and indices to get global max and indices
    global_values = _gather_along_dim(local_value, gather_dim, process_group=process_group)
    global_indices = _gather_along_dim(local_index, gather_dim, process_group=process_group)

    # indices are based on local shard, so we need to correct it by applying
    # an offset derived from tp degree and sharded size. This is only applicable
    # when the gather_dim is equal to the argmax dim.
    
    if gather_dim == dim:
        full_size = sharded_size * tp_degree
        offset = torch.arange(0, full_size, sharded_size)
        offset = offset.view([1 if i != dim else -1 for i in range(num_dims)])
        corrected_global_indices = global_indices + offset
    else:
        corrected_global_indices = global_indices

    # calculate the global argmax based on the local argmax from the global max values
    # and then retrieve the corrected indices
    global_max_local_index = Argmax.apply(global_values, dim=dim, keepdim=True)

    final_indices = torch.gather(corrected_global_indices, dim, global_max_local_index)

    if not keepdim:
        return final_indices.squeeze(dim)

    return final_indices
