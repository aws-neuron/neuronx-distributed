import os
import itertools
from typing import Any, List, Optional, TYPE_CHECKING

import torch
import torch.distributed
from torch.distributed import ProcessGroup
from ..utils.logger import get_logger

if TYPE_CHECKING:
    from torch._C._distributed_c10d import Store

try:
    # Method exists at least from PT 1.13-2.1
    from torch.distributed.distributed_c10d import _get_default_store

    TCP_STORE_AVAILABLE = True
except ImportError:
    TCP_STORE_AVAILABLE = False

logger = get_logger()

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP: Optional[ProcessGroup] = None
_TENSOR_MODEL_PARALLEL_GROUP_SPMD: Optional[ProcessGroup] = None

# Expert model parallel group that the current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP: Optional[ProcessGroup] = None
_EXPERT_MODEL_PARALLEL_GROUP_SPMD: Optional[ProcessGroup] = None

# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP: Optional[ProcessGroup] = None
_PIPELINE_GLOBAL_RANKS: Optional[List[int]] = None
_PIPELINE_MODEL_PARALLEL_GROUP_SPMD: Optional[ProcessGroup] = None
_NEXT_RANK_GROUP_SPMD: Optional[ProcessGroup] = None
_PREV_RANK_GROUP_SPMD: Optional[ProcessGroup] = None
_NEXT_RANK_GROUP: Optional[ProcessGroup] = None
_PREV_RANK_GROUP: Optional[ProcessGroup] = None

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP: Optional[ProcessGroup] = None
_DATA_PARALLEL_GROUP_SPMD: Optional[ProcessGroup] = None

# Expert data parallel group that the current rank belongs to.
_EXP_DATA_PARALLEL_GROUP: Optional[ProcessGroup] = None
_EXP_DATA_PARALLEL_GROUP_SPMD: Optional[ProcessGroup] = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE: Optional[int] = None
_MPU_TENSOR_MODEL_PARALLEL_RANK: Optional[int] = None

_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE: Optional[int] = None
_MPU_EXPERT_MODEL_PARALLEL_RANK: Optional[int] = None

# A CPU group that contains ranks from current rank's PP group\
# Used for PP metadata transmission
PP_GROUP_PG_GLOO: Optional[ProcessGroup] = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
) -> None:
    """
    Initialize model data parallel groups.

    Arguments:
        pipeline_model_parallel_size: number of Neuron devices used to parallelize model layer.
        tensor_model_parallel_size: number of Neuron devices used to parallelize model tensor.

        expert_model_parallel_size: number of Neuron devices used to parallelize MoE experts.

    mental model:
    WITHOUT EXPERT PARALLELISM (EP)
    imagine an array filled with worker global ranks [1, 2, .., PP*DP*TP],
    reshaped to a (contiguous, row-major) tensor of shape [PP, DP, TP]
    (for now we are ignoring EP by using EP = 1, this will be discussed later).
    indices along the final dimension (TP) have stride of 1 (contiguous)
        NOTE: this is important because it ensures as much TP communication as
        possible is intra-node, as workers in the same node have contiguous
        global ranks.
    indices along the 2nd to last dimension (DP) have stride of TP
    indices along the 3rd to last dimension (PP) have stride of DP * TP

    WITH EXPERT PARALLELISM (EP):
    the tensor from before can have two shapes
    [PP, DP_exp,    EP, TP] - in expert regions (MLP)
    [PP, DP_nonexp    , TP] - everywhere else.
    since DP_exp * EP == DP_nonexp, we can view switches between expert and nonexpert
    regions as a reshaping of this tensor, and regardless of which mode we're in:
    * the stride of earlier dimensions (in this case only PP) remains DP_exp * EP * TP.
    * the stride of later dimensions (in this case only TP) remains 1.
    importantly, this means that when switching between nonexpert and expert regions,
    any given worker will retain the same PP and TP ranks.

    EXAMPLE 1 (NO EP)
    ----------------------------------------------------------------------------------
    Let's say:
    * we have a total of 32 Neuron devices denoted by n0 ... n32
    * user specifies TP=8, PP=4
    From this we can derive that DP = N / (TP * PP) = 1

    The function will create:
    * 8 pipeline model-parallel groups of size PP=4.
      Stride is 8, since the product of all subsequent parallelism dimensions is 8.
      [
        [n00, n08, n16, n24],  # (DP=0, TP=0)
        [n01, n09, n17, n25],  # (DP=0, TP=1)
        ...
        [n06, n14, n22, n30],  # (DP=0, TP=6)
        [n07, n15, n23, n31]   # (DP=0, TP=7)
      ]
    * 32 data-parallel groups of size DP=1 (meaning no data parallelism).
      [
        [n00],  # (PP=0, TP=0)
        [n01],  # (PP=0, TP=1)
        ...
        [n30],  # (PP=3, TP=6)
        [n31]   # (PP=3, TP=7)
      ]
    * 4 tensor model-parallel groups of size TP=8
      Stride is 1 since this is the final parallelism dimension.
      [
        [n00, n01, n02, n03, n04, n05, n06, n07],  # (PP=0, DP=0)
        [n08, n09, n10, n11, n12, n13, n14, n15],  # (PP=1, DP=0)
        [n16, n17, n18, n19, n20, n21, n22, n23],  # (PP=2, DP=0)
        [n24, n25, n26, n27, n28, n29, n30, n31],  # (PP=3, DP=0)
      ]

    EXAMPLE 2 (WITH EP)
    ----------------------------------------------------------------------------------
    Lets say:
    * we have a total of 128 neuron devices denoted by n0 ... n128
    * user specifies TP=8, PP=4, EP=2
    From this we can derive that DP_nonexp = 4, and DP_exp = 2

    The function will create:
    * 32 pipeline model parallel groups of size PP=4 each.
      stride is 32, because product of all subsequent parallelism dimensions is 32.
      [
        [n000, n032, n064, n096],  # (DP=0, TP=0) or (DP_EXP=0, EP=0, TP=0)
        [n001, n033, n065, n097],  # (DP=0, TP=1) or (DP_EXP=0, EP=0, TP=1)
        ...
        [n030, n062, n094, n126],  # (DP=3, TP=6) or (DP_EXP=1, EP=1, TP=6)
        [n031, n063, n095, n127]   # (DP=3, TP=7) or (DP_EXP=1, EP=1, TP=7)
      ]
    * 32 DP_nonexp groups of size DP_nonexp=4 each.
      stride is 8 (TP)
      [
        [n000, n008, n016, n024],  # (PP=0, TP=0)
        [n001, n009, n017, n025],  # (PP=0, TP=1)
        ...
        [n102, n110, n118, n126],  # (PP=3, TP=6)
        [n103, n111, n119, n127],  # (PP=3, TP=7)
      ]
    * 64 DP_exp groups of size DP_exp=2 each.
      stride is 16 (EP * TP)
      [
        [n000, n016],  # (PP=0, EP=0, TP=0)
        [n001, n017],  # (PP=0, EP=0, TP=1)
        ...
        [n110, n126],  # (PP=3, EP=1, TP=6)
        [n111, n127]   # (PP=3, EP=1, TP=7)
      ]
    * 64 expert model parallel groups of size EP=2 each.
      stride is 8 (TP)
      [
        [n000, n008],  # (PP=0, DP_EXP=0, TP=0)
        [n001, n009],  # (PP=0, DP_EXP=0, TP=1)
        ...
        [n118, n126],  # (PP=3, DP_EXP=1, TP=6)
        [n119, n127]   # (PP=3, DP_EXP=1, TP=7)
      ]
    * 16 TP groups of size TP=8 each.
      stride is 1, contiguousness prioritizes TP communication happening within
      ranks on same node.
      [
        [n000, n001, n002, n003, n004, n005, n006, n007],  # (PP=0, DP=0) or (PP=0, DP_EXP=0, EP=0)
        [n008, n009, n010, n011, n012, n013, n014, n015],  # (PP=0, DP=1) or (PP=0, DP_EXP=0, EP=1)
        ...
        [n112, n113, n114, n115, n116, n117, n118, n119],  # (PP=3, DP=2) or (PP=3, DP_EXP=1, EP=0)
        [n120, n121, n122, n123, n124, n125, n126, n127]   # (PP=3, DP=3) or (PP=3, DP_EXP=1, EP=1)
      ]
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    tensor_model_parallel_size: int = min(tensor_model_parallel_size, world_size)
    pipeline_model_parallel_size: int = min(pipeline_model_parallel_size, world_size)
    expert_model_parallel_size: int = min(expert_model_parallel_size, world_size)

    # compute implied data parallel degrees for both expert and non-expert regions,
    # in both cases making sure implied data parallel size is an integer.
    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
        raise RuntimeError(
            f"invalid implied data parallel degree: "
            f"`world_size` ({world_size}) is not divisible by "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )
    data_parallel_size: int = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size)

    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size * expert_model_parallel_size) != 0:
        raise RuntimeError(
            f"invalid implied expert data parallel degree: "
            f"`world_size` ({world_size}) is not divisible by "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size}) x "
            f"expert_model_parallel_size ({expert_model_parallel_size})"
        )
    exp_data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size * expert_model_parallel_size
    )

    if tensor_model_parallel_size == 4:
        # On trn1, TP=4 is a special case where each TP group consists of locally connected,
        # non-contiguous ranks grouped within each node to avoid cross-node TP.
        # Ex: for TP=4 PP=1 on 2 trn1.32xl nodes (64 NeuronCores):
        #   16 TP groups: [ [0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], ... [7, 15, 23, 31],
        #                   [32, 40, 48, 56], [33, 41, 49, 57], [34, 42, 50, 58], ... [39, 47, 55, 63] ]
        #    4 DP groups: [ [0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39]
        #                   [8, 9, 10, 11, 12, 13, 14, 15, 40, 41, 42, 43, 44, 45, 46, 47]
        #                   [16, 17, 18, 19, 20, 21, 22, 23, 48, 49, 50, 51, 52, 53, 54, 55]
        #                   [24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63] ]
        #   64 PP groups: [ [0], [1], [2] .. [63] ]  (No pipeline parallelism)
        if expert_model_parallel_size > 1:
            raise NotImplementedError("TP=4 case not yet implemented for expert parallelism")

        cluster_ranks = torch.arange(0, world_size)
        cluster_ranks_exp = (
            cluster_ranks.reshape([pipeline_model_parallel_size, data_parallel_size // 8, 4, 8])
            .transpose(-1, -2)
            .reshape(
                pipeline_model_parallel_size, data_parallel_size, expert_model_parallel_size, tensor_model_parallel_size
            )
        )
        cluster_ranks_nonexp = (
            cluster_ranks.reshape([pipeline_model_parallel_size, data_parallel_size // 8, 4, 8])
            .transpose(-1, -2)
            .reshape(pipeline_model_parallel_size, data_parallel_size, tensor_model_parallel_size)
        )
    else:
        cluster_ranks = torch.arange(0, world_size)
        cluster_ranks_exp = cluster_ranks.reshape(
            [
                pipeline_model_parallel_size,
                exp_data_parallel_size,
                expert_model_parallel_size,
                tensor_model_parallel_size,  # important: contiguous parallelism dimension
            ]
        )
        cluster_ranks_nonexp = cluster_ranks.reshape(
            [
                pipeline_model_parallel_size,
                data_parallel_size,
                tensor_model_parallel_size,  # important: contiguous parallelism dimension
            ]
        )

    logger.info("> initializing tensor model parallel with size %d", tensor_model_parallel_size)
    logger.info("> initializing pipeline model parallel with size %d", pipeline_model_parallel_size)
    logger.info("> initializing data parallel with size %d", data_parallel_size)
    logger.info("> initializing world size to %d", world_size)
    if expert_model_parallel_size > 1:
        logger.info("> initializing expert model parallel with size %d", expert_model_parallel_size)
        logger.info("> initializing data parallel (exp) with size %d", exp_data_parallel_size)

    # We create a dummy neff and execute it across all workers in the world.
    # This is done to initialize the collectives. Collectives initialization
    # requires all workers in the world to participate and this soometimes
    # may not be guranteed. Hence as a workaround, we run this dummy neff, and
    # get the collectives initialized.
    temp = torch.rand([1], device="xla")
    torch.distributed.all_reduce(temp, group=torch.distributed.group.WORLD)
    import torch_xla.core.xla_model as xm

    xm.mark_step()

    rank = torch.distributed.get_rank()
    compress_rg = int(os.getenv("NEURON_EXPERIMENTAL_COMPRESS_RG", "0"))

    # Build the data-parallel groups.
    all_data_parallel_group_ranks = [
        cluster_ranks_nonexp[pp_rank, :, tp_rank].tolist()
        for pp_rank, tp_rank in itertools.product(
            range(pipeline_model_parallel_size),
            range(tensor_model_parallel_size),
        )
    ]
    _build_and_assign_groups(
        group_name="_DATA_PARALLEL_GROUP",
        spmd_group_name="_DATA_PARALLEL_GROUP_SPMD",
        mesh=all_data_parallel_group_ranks,
        compress_rg=False,
    )

    # Build the expert data-parallel groups.
    all_exp_data_parallel_group_ranks = [
        cluster_ranks_exp[pp_rank, :, ep_rank, tp_rank].tolist()
        for pp_rank, ep_rank, tp_rank in itertools.product(
            range(pipeline_model_parallel_size),
            range(expert_model_parallel_size),
            range(tensor_model_parallel_size),
        )
    ]
    _build_and_assign_groups(
        group_name="_EXP_DATA_PARALLEL_GROUP",
        spmd_group_name="_EXP_DATA_PARALLEL_GROUP_SPMD",
        mesh=all_exp_data_parallel_group_ranks,
        compress_rg=False,
    )

    # Build the tensor model-parallel groups.
    all_tensor_parallel_group_ranks = [
        cluster_ranks_nonexp[pp_rank, dp_rank, :].tolist()
        for pp_rank, dp_rank in itertools.product(
            range(pipeline_model_parallel_size),
            range(data_parallel_size),
        )
    ]
    _build_and_assign_groups(
        group_name="_TENSOR_MODEL_PARALLEL_GROUP",
        spmd_group_name="_TENSOR_MODEL_PARALLEL_GROUP_SPMD",
        mesh=all_tensor_parallel_group_ranks,
        compress_rg=compress_rg,
    )

    # Build the expert model-parallel groups
    all_expert_parallel_group_ranks = [
        cluster_ranks_exp[pp_rank, dp_exp_rank, :, tp_rank].tolist()
        for pp_rank, dp_exp_rank, tp_rank in itertools.product(
            range(pipeline_model_parallel_size),
            range(exp_data_parallel_size),
            range(tensor_model_parallel_size),
        )
    ]
    _build_and_assign_groups(
        group_name="_EXPERT_MODEL_PARALLEL_GROUP",
        spmd_group_name="_EXPERT_MODEL_PARALLEL_GROUP_SPMD",
        mesh=all_expert_parallel_group_ranks,
        compress_rg=False,
    )

    # Build the pipeline model-parallel groups.
    all_pipeline_parallel_group_ranks = [
        cluster_ranks_nonexp[:, dp_rank, tp_rank].tolist()
        for dp_rank, tp_rank in itertools.product(
            range(data_parallel_size),
            range(tensor_model_parallel_size),
        )
    ]
    _build_and_assign_groups(
        group_name="_PIPELINE_MODEL_PARALLEL_GROUP",
        spmd_group_name="_PIPELINE_MODEL_PARALLEL_GROUP_SPMD",
        mesh=all_pipeline_parallel_group_ranks,
        compress_rg=False,
    )

    for ranks in _PIPELINE_MODEL_PARALLEL_GROUP_SPMD:
        if rank in ranks:
            global _PIPELINE_GLOBAL_RANKS
            _PIPELINE_GLOBAL_RANKS = ranks

    # Only create pre/next groups if PP is enabled
    if pipeline_model_parallel_size > 1:
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


def _build_and_assign_groups(
    group_name: str,
    spmd_group_name: str,
    mesh: List[List[int]],
    compress_rg: bool,
) -> None:
    def __set_global_var(key: str, val: Any) -> None:
        if key not in globals():
            raise RuntimeError(f"expected {key} to be in globals but was undefined")
        # if globals()[key] is not None:
        #    raise RuntimeError(f"expected {key} to be uninitialized but was set to {globals()[key]}")

        globals()[key] = val

    if compress_rg:
        # When scaling to large number of nodes, the size of the replica groups becomes huge.
        # This increases the overall HLO hashing time which in turn causes framework overhead.
        # This can be reduced by passing the first tp replica only. All the other ranks would
        # infer their groups depending on the size of the replica group and the start and end ranks
        # Note: this works only for cases where the ranks are continuous. It won't work for TP=4 case.
        mesh = [mesh[0]]

    __set_global_var(key=spmd_group_name, val=mesh)
    for group_ranks in mesh:
        group = torch.distributed.new_group(
            group_ranks,
            pg_options={"xla_pg_options": {"mesh": mesh}},
        )
        if torch.distributed.get_rank() in group_ranks:
            __set_global_var(key=group_name, val=group)

    if globals()[group_name] is None:
        raise RuntimeError(f"expected {group_name} to be initialized but was not. mesh: {mesh}")

    try_set_nki_parallel_state()


def try_set_nki_parallel_state() -> None:
    """
    Inject parallel state information into NkiKernel, if compatible torch_neuronx exists.
    """
    try:
        from torch_neuronx.xla_impl.ops import NkiKernel

        NkiKernel._parallel_state = dict(
            parallel_group=get_tensor_model_parallel_group(as_list=True),
            rank=get_tensor_model_parallel_rank(),
            world_size=get_tensor_model_parallel_size(),
        )
        logger.debug(rmsg("Successfully initialized NKI parallel state."))
    except Exception as e:
        logger.warning(
            rmsg(
                f"Failed to initialize NKI parallel state with exception {e}."
                "Proceeding without distributed NKI support."
            )
        )


def model_parallel_is_initialized() -> bool:
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_tensor_model_parallel_group(as_list: bool = False) -> ProcessGroup:
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, "intra_layer_model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP._mesh if as_list else _TENSOR_MODEL_PARALLEL_GROUP


def get_expert_model_parallel_group(as_list: bool = False) -> ProcessGroup:
    assert _EXPERT_MODEL_PARALLEL_GROUP is not None, "expert model parallel group is not initialized"
    return _EXPERT_MODEL_PARALLEL_GROUP._mesh if as_list else _EXPERT_MODEL_PARALLEL_GROUP


def get_data_parallel_group(as_list: bool = False) -> ProcessGroup:
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP._mesh if as_list else _DATA_PARALLEL_GROUP


def get_expert_data_parallel_group(as_list: bool = False) -> ProcessGroup:
    """Get the expert data parallel group the caller rank belongs to."""
    assert _EXP_DATA_PARALLEL_GROUP is not None, "expert data parallel group is not initialized"
    return _EXP_DATA_PARALLEL_GROUP._mesh if as_list else _EXP_DATA_PARALLEL_GROUP


def get_tensor_model_parallel_size() -> int:
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def set_tensor_model_parallel_size(world_size: int) -> None:
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_tensor_model_parallel_rank(rank: int) -> None:
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def get_tensor_model_parallel_rank() -> int:
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_size()
    return (global_rank // local_world_size) * local_world_size


def set_expert_model_parallel_size(world_size: int) -> None:
    """Set the expert model parallel size."""
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_expert_model_parallel_size() -> int:
    """Return world size for the expert model parallel group."""
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_expert_model_parallel_group())


def set_expert_model_parallel_rank(rank: int) -> None:
    """Set the expert model parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def get_expert_model_parallel_rank() -> int:
    """Return my rank for the expert model parallel group."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    if _MPU_EXPERT_MODEL_PARALLEL_RANK is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_expert_model_parallel_group())


def get_data_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to the first local rank in the data parallel group."""
    global_rank = torch.distributed.get_rank()
    data_parallel_size: int = get_data_parallel_size()
    num_data_parallel_groups = torch.distributed.get_world_size() // data_parallel_size
    return global_rank % num_data_parallel_groups


def get_data_parallel_size() -> int:
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank() -> int:
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


def get_expert_data_parallel_size() -> int:
    """Return world size for the expert data parallel group."""
    return torch.distributed.get_world_size(group=get_expert_data_parallel_group())


def get_expert_data_parallel_rank() -> int:
    """Return my rank for the expert data parallel group."""
    return torch.distributed.get_rank(group=get_expert_data_parallel_group())


def get_pipeline_model_parallel_group(as_list: bool = False) -> ProcessGroup:
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, "pipeline_model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP._mesh if as_list else _PIPELINE_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_rank() -> int:
    """Return my rank for the pipeline model parallel group."""
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def get_pipeline_model_parallel_sr_group(parity: bool) -> ProcessGroup:
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    world_size = get_pipeline_model_parallel_size()

    def subgroup(r, ranks):
        return [ranks[r], ranks[(r + 1) % world_size]]

    group = list()
    for ranks in _PIPELINE_MODEL_PARALLEL_GROUP_SPMD:
        for i in range(parity, world_size, 2):
            group.append(subgroup(i, ranks))
    return group


def get_pipeline_model_parallel_size() -> int:
    """Return world size for the pipeline model parallel group."""
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def get_next_rank_group(as_list: bool = False) -> ProcessGroup:
    """Get the next tensor model parallel group the caller rank belongs to."""
    assert _NEXT_RANK_GROUP is not None, "intra_layer_model parallel group is not initialized"
    return _NEXT_RANK_GROUP._mesh if as_list else _NEXT_RANK_GROUP


def get_prev_rank_group(as_list: bool = False) -> ProcessGroup:
    """Get the previous tensor model parallel group the caller rank belongs to."""
    assert _PREV_RANK_GROUP is not None, "intra_layer_model parallel group is not initialized"
    return _PREV_RANK_GROUP._mesh if as_list else _PREV_RANK_GROUP


def get_pipeline_model_parallel_next_rank() -> int:
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank() -> int:
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def destroy_model_parallel() -> None:
    """Set the groups to none."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _EXPERT_MODEL_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = None
    global _EXPERT_MODEL_PARALLEL_GROUP_SPMD
    _EXPERT_MODEL_PARALLEL_GROUP_SPMD = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _EXP_DATA_PARALLEL_GROUP
    _EXP_DATA_PARALLEL_GROUP = None
    global _EXP_DATA_PARALLEL_GROUP_SPMD
    _EXP_DATA_PARALLEL_GROUP_SPMD = None
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


def is_tcp_store_available() -> bool:
    return TCP_STORE_AVAILABLE


def get_tcp_store() -> "Store":
    """
    Getting the default tcp_store from the global group initialization
    """
    assert is_tcp_store_available(), "Can not import _get_default_store from distributed_c10d"
    return _get_default_store()


def initialize_pp_gloo_groups() -> None:
    global PP_GROUP_PG_GLOO
    assert PP_GROUP_PG_GLOO is None, "pp gloo groups are already initialized!"
    logger.info("initialize_pp_gloo_groups...")
    pp_group_spmd = get_pipeline_model_parallel_group(as_list=True)
    rank = torch.distributed.get_rank()
    for pp_group in pp_group_spmd:
        pg = torch.distributed.new_group(ranks=pp_group, backend="gloo")
        if rank in pp_group:
            PP_GROUP_PG_GLOO = pg


def get_pp_gloo_group() -> ProcessGroup:
    global PP_GROUP_PG_GLOO
    assert PP_GROUP_PG_GLOO is not None, "pp gloo groups are not initialized!"
    return PP_GROUP_PG_GLOO


def is_global_rank_zero() -> bool:
    # TODO: Change this to torch.distributed.get_rank when PTL fix of init_process
    # before nxd_config is added.
    import torch_xla.core.xla_model as xm

    return xm.get_ordinal() == 0


def create_pg_with_ranks(ranks: List[int]) -> ProcessGroup:
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


def gather_python_object(obj: Any, group: ProcessGroup) -> List[Any]:
    """
    Eagerly gather python object for a group
    Usually used to collect timeline events
    """
    object_gather_list = None
    if torch.distributed.get_rank(group=group) == 0:
        object_gather_list = [None] * torch.distributed.get_world_size(group=group)
    torch.distributed.gather_object(obj, object_gather_list=object_gather_list, group=group)
    return object_gather_list


def rmsg(msg: str) -> str:
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
    try:
        global_rank = torch.distributed.get_rank()
    except RuntimeError:
        # torch distributed not initialized, mainly in PTL case
        import torch_xla.core.xla_model as xm

        global_rank = xm.get_ordinal()
    return f"[rank_{global_rank}_pp{pp_rank}_tp{tp_rank}_dp{dp_rank}] {msg}"


def rmsg_ep(msg: str) -> str:
    pp_rank = get_pipeline_model_parallel_rank()
    ep_rank = get_expert_model_parallel_rank()
    tp_rank = get_tensor_model_parallel_rank()
    dp_rank = get_data_parallel_rank()
    return f"[pp{pp_rank}|ep{ep_rank}|tp{tp_rank}|dp{dp_rank}] {msg}"
