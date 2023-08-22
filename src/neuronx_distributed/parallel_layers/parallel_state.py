import torch

from ..utils.logger import get_logger

logger = get_logger()

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
    pipeline_model_parallel_size: int = min(pipeline_model_parallel_size, world_size)
    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
        raise RuntimeError(
            (
                f"`world_size` ({world_size}) is not divisible by tensor_model_parallel_size"
                f" ({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size})"
            )
        )
        print("> initializing data parallel with size {}".format(data_parallel_size))

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_data_parallel_groups: int = world_size // data_parallel_size

    rank = torch.distributed.get_rank()

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_SPMD
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    all_data_parallel_group_ranks = []
    for i in range(tensor_model_parallel_size):
        ranks = range(i, world_size, tensor_model_parallel_size)
        all_data_parallel_group_ranks.append(list(ranks))

    _DATA_PARALLEL_GROUP_SPMD = all_data_parallel_group_ranks
    for ranks in all_data_parallel_group_ranks:
        pg_options = {"xla_pg_options": {"mesh": _DATA_PARALLEL_GROUP_SPMD}}
        group = torch.distributed.new_group(ranks, pg_options=pg_options)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GROUP_SPMD
    all_tensor_parallel_group_ranks = []
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), "tensor model parallel group is already initialized"
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(
            i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size
        )
        all_tensor_parallel_group_ranks.append(ranks)
    _TENSOR_MODEL_PARALLEL_GROUP_SPMD = all_tensor_parallel_group_ranks
    for ranks in all_tensor_parallel_group_ranks:
        pg_options = {"xla_pg_options": {"mesh": _TENSOR_MODEL_PARALLEL_GROUP_SPMD}}
        group = torch.distributed.new_group(ranks, pg_options=pg_options)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups.
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    global _PIPELINE_MODEL_PARALLEL_GROUP_SPMD
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, "pipeline model parallel group is already initialized"
    all_pipeline_parallel_group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        all_pipeline_parallel_group_ranks.append(list(ranks))
    _PIPELINE_MODEL_PARALLEL_GROUP_SPMD = all_pipeline_parallel_group_ranks
    for ranks in _PIPELINE_MODEL_PARALLEL_GROUP_SPMD:
        pg_options = {"xla_pg_options": {"mesh": _PIPELINE_MODEL_PARALLEL_GROUP_SPMD}}
        if rank in ranks:
            group = torch.distributed.new_group(ranks, pg_options=pg_options)
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks

    global _NEXT_RANK_GROUP_SPMD
    global _PREV_RANK_GROUP_SPMD
    global _NEXT_RANK_GROUP
    global _PREV_RANK_GROUP
    parity = bool(get_pipeline_model_parallel_rank() % 2)
    _NEXT_RANK_GROUP_SPMD = get_pipeline_model_parallel_sr_group(parity)
    _PREV_RANK_GROUP_SPMD = get_pipeline_model_parallel_sr_group(not parity)
    for ranks in _NEXT_RANK_GROUP_SPMD:
        pg_options = {"xla_pg_options": {"mesh": _NEXT_RANK_GROUP_SPMD}}
        if rank in ranks:
            group = torch.distributed.new_group(ranks, pg_options=pg_options)
            _NEXT_RANK_GROUP = group
    for ranks in _PREV_RANK_GROUP_SPMD:
        pg_options = {"xla_pg_options": {"mesh": _PREV_RANK_GROUP_SPMD}}
        if rank in ranks:
            group = torch.distributed.new_group(ranks, pg_options=pg_options)
            _PREV_RANK_GROUP = group
    if torch.distributed.get_rank() == 0:
        logger.debug(rmsg(f"_PIPELINE_MODEL_PARALLEL_GROUP_SPMD {_PIPELINE_MODEL_PARALLEL_GROUP_SPMD}"))
        logger.debug(rmsg(f"_TENSOR_MODEL_PARALLEL_GROUP_SPMD {_TENSOR_MODEL_PARALLEL_GROUP_SPMD}"))
        logger.debug(rmsg(f"_DATA_PARALLEL_GROUP_SPMD {_DATA_PARALLEL_GROUP_SPMD}"))


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_tensor_model_parallel_group(as_list=False):
    """Get the tensor model parallel group the caller rank belongs to."""
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is not None
    ), "intra_layer_model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP._mesh if as_list else _TENSOR_MODEL_PARALLEL_GROUP


def get_data_parallel_group(as_list=False):
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP._mesh if as_list else _DATA_PARALLEL_GROUP


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
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = None
    global _PIPELINE_MODEL_PARALLEL_GROUP_SPMD
    _PIPELINE_MODEL_PARALLEL_GROUP_SPMD = None
    global _NEXT_RANK_GROUP
    _NEXT_RANK_GROUP = None
    global _PREV_RANK_GROUP
    _PREV_RANK_GROUP = None
    global _NEXT_RANK_GROUP_SPMD
    _NEXT_RANK_GROUP_SPMD = None
    global _PREV_RANK_GROUP_SPMD
    _PREV_RANK_GROUP_SPMD = None


def set_gloo_group():
    global _GLOBAL_GLOO_GROUP
    if _GLOBAL_GLOO_GROUP is None:
        _GLOBAL_GLOO_GROUP = torch.distributed.new_group(backend="gloo")
    else:
        logger.warning("gloo group is already created, skipping...")


def get_gloo_group():
    if _GLOBAL_GLOO_GROUP is None:
        raise RuntimeError("Trying to get gloo group when not initialized")
    return _GLOBAL_GLOO_GROUP


def create_pg_with_ranks(ranks):
    """
    Create a SPMD process group based on input pp ranks.
    This can be used to create process group to average grads for shared weights betweenn PP ranks
    Input:
    - ranks: a list of ranks that will be used to create the process group
    """
    world_size = torch.distributed.get_world_size()
    world_rank = torch.distributed.get_rank()
    pipeline_model_parallel_size = get_pipeline_model_parallel_size()
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
    all_shared_ranks_spmd = []

    # Collect the share ranks for each PP group
    for i in range(num_pipeline_model_parallel_groups):
        pp_group_ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        shared_global_ranks = [pp_group_ranks[k] for k in ranks]
        all_shared_ranks_spmd.append(shared_global_ranks)

    # For each PP groups, create the same pg for every PP rank.
    # The pg will only contain the shared ranks
    # This is because that torch.distributed.new_group requires all processes in main group to enter
    pp_model_parallel_group_spmd = get_pipeline_model_parallel_group(as_list=True)
    for ranks, current_shared_ranks in zip(pp_model_parallel_group_spmd, all_shared_ranks_spmd):
        pg_options = {"xla_pg_options": {"mesh": all_shared_ranks_spmd}}
        if world_rank in ranks:
            logger.debug(
                rmsg(
                    f"creating pg based on ranks {ranks}, all_shared_ranks_spmd {all_shared_ranks_spmd}, current_shared_ranks {current_shared_ranks}"  # noqa: E501
                )
            )
            group = torch.distributed.new_group(current_shared_ranks, pg_options=pg_options)
    return group


def gather_python_object(obj, group):
    """
    Eagerly gather python object for a group
    Usually used to collect timeline events
    """
    object_gather_list = None
    if torch.distributed.get_rank(group=group) == 0:
        object_gather_list = [None] * torch.distributed.get_world_size(group=group)
    torch.distributed.gather_object(obj, object_gather_list=object_gather_list, group=group)
    return object_gather_list


def rmsg(msg):
    """
    Return a message with parallel ranking information
    """
    try:
        pp_rank = get_pipeline_model_parallel_rank()
        tp_rank = get_tensor_model_parallel_rank()
        dp_rank = get_data_parallel_rank()
    except AssertionError:
        # Parallel state is not initialized
        pp_rank, tp_rank, dp_rank = -1, -1, -1
    global_rank = torch.distributed.get_rank()
    return f"[rank_{global_rank}_pp{pp_rank}_tp{tp_rank}_dp{dp_rank}] {msg}"
