import torch

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
_TENSOR_MODEL_PARALLEL_GROUP_SPMD = None

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_SPMD = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None

def initialize_model_parallel(tensor_model_parallel_size: int = 1) -> None:
    """
    Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size: number of Neuron devices used to parallelize model tensor.

    Let's say we have a total of 16 Neuron devices denoted by n0 ... n15 and we
    use 2 Neuron devices to parallelize the data. The present function will
    create 8 data-parallel groups, and 2 tensor model-parallel groups as:
        8 data_parallel groups:
            [n0, n1], [n2, n3], [n4, n5], [n6, n7], [n8, n9], [n10, n11], [n12, n13], [n14, n15]
        4 tensor model-parallel groups:
            [n0, n2, n4, n6], [n1, n3, n5, n7]
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    
    world_size: int = torch.distributed.get_world_size()
    tensor_model_parallel_size: int = min(tensor_model_parallel_size, world_size)
    data_parallel_size: int = world_size // tensor_model_parallel_size
    if torch.distributed.get_rank() == 0:
        print(
            "> initializing tensor model parallel with size {}".format(
                tensor_model_parallel_size
            )
        )
        print(
            "> initializing data parallel with size {}".format(data_parallel_size)
        )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_data_parallel_groups: int = world_size // data_parallel_size

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_SPMD
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    all_data_parallel_group_ranks = []
    for i in range(tensor_model_parallel_size):
        ranks = range(i, world_size,
                        tensor_model_parallel_size)
        all_data_parallel_group_ranks.append(list(ranks))
    
    _DATA_PARALLEL_GROUP_SPMD = all_data_parallel_group_ranks
    for ranks in all_data_parallel_group_ranks:
        pg_options = {'xla_pg_options' : {'mesh' : _DATA_PARALLEL_GROUP_SPMD}}
        group = torch.distributed.new_group(ranks, pg_options=pg_options)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GROUP_SPMD
    all_tensor_parallel_group_ranks = []
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)
        all_tensor_parallel_group_ranks.append(ranks)
    _TENSOR_MODEL_PARALLEL_GROUP_SPMD = all_tensor_parallel_group_ranks
    for ranks in all_tensor_parallel_group_ranks:
        pg_options = {'xla_pg_options' : {'mesh' : _TENSOR_MODEL_PARALLEL_GROUP_SPMD}}
        group = torch.distributed.new_group(ranks, pg_options=pg_options)
        if rank in ranks:
           _TENSOR_MODEL_PARALLEL_GROUP = group           


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True

def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP

def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP

def get_tensor_model_parallel_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())

def set_tensor_model_parallel_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size
    
def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank
    
def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())

def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_size()
    return (global_rank // local_world_size) * local_world_size

def get_data_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank in the data parallel group."""
    global_rank = torch.distributed.get_rank()
    data_parallel_size: int = get_data_parallel_size()
    num_data_parallel_groups = torch.distributed.get_world_size() // data_parallel_size
    return global_rank % num_data_parallel_groups

def get_data_parallel_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())

def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())

def destroy_model_parallel():
    """Set the groups to none."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None