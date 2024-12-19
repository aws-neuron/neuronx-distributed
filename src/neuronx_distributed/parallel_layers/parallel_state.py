import os
import itertools
from typing import Any, List, Optional, Union, TYPE_CHECKING, Callable

import torch
import torch.distributed
import torch_xla.core.xla_model as xm
from torch.distributed import ProcessGroup
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.utils.utils import hardware
from ..utils import cpu_mode
from enum import Enum
from collections import namedtuple
from torch_neuronx.utils import get_platform_target


if TYPE_CHECKING:
    from torch._C._distributed_c10d import Store

try:
    # Method exists at least from PT 1.13-2.1
    from torch.distributed.distributed_c10d import _get_default_store

    TCP_STORE_AVAILABLE = True
except ImportError:
    TCP_STORE_AVAILABLE = False

logger = get_logger()

# copy of world process group
_WORLD_GROUP: Optional[ProcessGroup] = None
_WORLD_GROUP_SPMD: Optional[ProcessGroup] = None

# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP: Optional[ProcessGroup] = None
_TENSOR_MODEL_PARALLEL_GROUP_SPMD: Optional[List[List[int]]] = None

# Expert model parallel group that the current rank belongs to.
_EXPERT_MODEL_PARALLEL_GROUP: Optional[ProcessGroup] = None
_EXPERT_MODEL_PARALLEL_GROUP_SPMD: Optional[List[List[int]]] = None

# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP: Optional[ProcessGroup] = None
_PIPELINE_GLOBAL_RANKS: Optional[List[int]] = None
_PIPELINE_MODEL_PARALLEL_GROUP_SPMD: Optional[List[List[int]]] = None
_NEXT_RANK_GROUP_SPMD: Optional[List[List[int]]] = None
_PREV_RANK_GROUP_SPMD: Optional[List[List[int]]] = None
_NEXT_RANK_GROUP: Optional[ProcessGroup] = None
_PREV_RANK_GROUP: Optional[ProcessGroup] = None

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP: Optional[ProcessGroup] = None
_DATA_PARALLEL_GROUP_SPMD: Optional[List[List[int]]] = None

# Expert data parallel group that the current rank belongs to.
_EXP_DATA_PARALLEL_GROUP: Optional[ProcessGroup] = None
_EXP_DATA_PARALLEL_GROUP_SPMD: Optional[List[List[int]]] = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE: Optional[int] = None
_MPU_TENSOR_MODEL_PARALLEL_RANK: Optional[int] = None

_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE: Optional[int] = None
_MPU_EXPERT_MODEL_PARALLEL_RANK: Optional[int] = None

# Shuffle group that the current rank belongs to.
_TOKEN_SHUFFLE_GROUP: Optional[ProcessGroup] = None
_TOKEN_SHUFFLE_GROUP_SPMD: Optional[List[List[int]]] = None
_TOKEN_SHUFFLE_GROUP_SIZE: Optional[int] = None

_KV_SHARED_GROUP: Optional[ProcessGroup] = None
_KV_SHARED_GROUP_SPMD: Optional[List[List[int]]] = None
_KV_SHARED_GROUP_SIZE: Optional[int] = None

# Speculative decoding parallel group
_SPECULATIVE_DRAFT_GROUP: Optional[ProcessGroup] = None
_SPECULATIVE_DRAFT_GROUP_SPMD: Optional[List[List[int]]] = None
_SPECULATIVE_DRAFT_GROUP_SIZE: Optional[int] = None

# global hardware type to use everywhere
_HARDWARE_TYPE: hardware = hardware.TRN1

# A CPU group that contains ranks from current rank's PP group\
# Used for PP metadata transmission
PP_GROUP_PG_GLOO: Optional[ProcessGroup] = None

# Indidcator for shape based tracing.
_AOT_MODE = False


ParallelGroups = namedtuple('ParallelGroups', ['tp_groups', 'dp_groups', 'pp_groups', 'ep_model_groups', 'ep_data_groups'])

def ascending_ring_PG_group(lnc_size: int, cluster_ranks_nonexp: torch.tensor,
                            cluster_ranks_exp: torch.tensor,  tp: int, dp: int, pp: int, 
                            ep_model_degree: int, ep_data_degree: int) -> ParallelGroups:
    # this function never uses lnc_size but passed along to support the fn pointer logic,
    # so its value doesnt matter eg: in case of trn1, this value doesnt matter.
    # Logic 1: Group tensor parallel group in ascending ring fashion 0 to n-1 consecutive ranks 
    # belong to one TP group -> n is the tp size

    # Build the tensor model-parallel groups.
    tp_groups = [
        cluster_ranks_nonexp[pp_rank, dp_rank, :].tolist()
        for pp_rank, dp_rank in itertools.product(
            range(pp),
            range(dp),
        )
    ]

    # Build the data parallel groups.
    dp_groups = [
        cluster_ranks_nonexp[pp_rank, :, tp_rank].tolist()
        for pp_rank, tp_rank in itertools.product(
            range(pp),
            range(tp),
        )
    ]

    # Build the pipeline model-parallel groups.
    pp_groups = [
        cluster_ranks_nonexp[:, dp_rank, tp_rank].tolist()
        for dp_rank, tp_rank in itertools.product(
            range(dp),
            range(tp),
        )
    ]

    # Build the expert model-parallel groups
    ep_model_groups = [
        cluster_ranks_exp[pp_rank, dp_exp_rank, :, tp_rank].tolist()
        for pp_rank, dp_exp_rank, tp_rank in itertools.product(
            range(pp),
            range(ep_data_degree),
            range(tp),
        )
    ]

    # Build the expert data-parallel groups.
    ep_data_groups = [
        cluster_ranks_exp[pp_rank, :, ep_rank, tp_rank].tolist()
        for pp_rank, ep_rank, tp_rank in itertools.product(
            range(pp),
            range(ep_model_degree),
            range(tp),
        )
    ]

    return ParallelGroups(tp_groups, dp_groups, pp_groups, ep_model_groups, ep_data_groups)

def ascending_descending_ring_PG_group(lnc_size: int, cluster_ranks_nonexp: torch.tensor,
                                       cluster_ranks_exp: torch.tensor,  tp: int, dp: int, pp: int,
                                       ep_model_degree: int, ep_data_degree: int) -> ParallelGroups:    
    # Logic 2: Group tensor parallel group in ascending ring fashion for first half and
    # for second half choose the consective ranks in opposite direction. (0 to n/2-1) + (lastRank-1 to lastRank-1+ n/2-1 )
    # eg: tp32 on 64 devices will have [0,1,..15] + [63,62,..48] and so on
    # Added only for TP, DP, PP in order not for EP. EP is still pending

    # Build the tensor model-parallel groups.
    tp_groups = [
        cluster_ranks_nonexp[pp_rank, dp_rank, :].tolist()
        for pp_rank, dp_rank in itertools.product(
            range(pp),
            range(dp),
        )
    ]
        
    total_ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 64))
    
    num_rows = 4 # from trn2 topology TODO: Replace 4 with master list like [0,1,2,...15,48,49,...63,16,....]
    num_ranks_per_row = total_ranks_per_node//num_rows
    
    ranks_start = [int(i*num_ranks_per_row) for i in range(num_rows)] # [0,16,32,48] for vnc2
    ranks_end = [int(val+num_ranks_per_row) for val in ranks_start] # [16,32,48,64] for vnc2
    world_size: int = torch.distributed.get_world_size()

    nodes = world_size//total_ranks_per_node
    
    tp_groups=[]
    for node in range(nodes):
        node_skip_val = node * total_ranks_per_node # temp variable to jump all ranks in n nodes
        tp_groups.append([i for i in range(ranks_start[0] + node_skip_val, ranks_end[0] + node_skip_val)]+
                                               [i for i in range(ranks_start[3] + node_skip_val, ranks_end[3] + node_skip_val)]) # first row and last row are one group in Logic2
        tp_groups.append([i for i in range(ranks_start[1] + node_skip_val, ranks_end[1] + node_skip_val)]+
                                               [i for i in range(ranks_start[2] + node_skip_val, ranks_end[2] + node_skip_val)]) # second and third row are one group in Logic2

    def _combine_tp_groups_for_dp(tp_groups, dp_degree):
        # Slice the first `dp_degree` groups from `tp_groups`
        result = []
        for i in range(len(tp_groups)//dp_degree):
            groups = tp_groups[i*dp_degree:dp_degree+i*dp_degree]
            # Use zip to combine elements in an element-wise fashion
            result += [list(combo) for combo in zip(*groups)]
        return result

    def _combine_dp_groups_for_pp(dp_groups, tp_degree):
        # Step 1: Determine the number of parts needed
        num_parts = len(dp_groups) // tp_degree

        # Step 2: Split the result into multiple parts
        parts = [dp_groups[i * tp_degree:(i + 1) * tp_degree] for i in range(num_parts)]        

        # Step 3: Combine corresponding elements from each part
        combined_result = [[list(sublist) for sublist in zip(*tup)] for tup in zip(*parts)]
        
        # Step 4:  Remove one dimension
        combined_result = [item for sublist in combined_result for item in sublist]

        return combined_result

    dp_groups = _combine_tp_groups_for_dp(tp_groups, dp)

    if pp>1:
        pp_groups=_combine_dp_groups_for_pp(dp_groups=dp_groups, tp_degree=tp) # given tp and dp, pp is auto calculated
    else:
        pp_groups = [[i] for i in range((world_size))] # might not be required and above code should handle this also, but keeping it till its tested

    # Build the expert model-parallel groups
    if ep_model_degree>1:
        ep_model_groups = [
            cluster_ranks_exp[pp_rank, dp_exp_rank, :, tp_rank].tolist()
            for pp_rank, dp_exp_rank, tp_rank in itertools.product(
                range(pp),
                range(ep_data_degree),
                range(tp),
            )
        ]
    else:
        ep_model_groups = [[i] for i in range((world_size))]

    # Build the expert data-parallel groups.
    if ep_data_degree>1:
        ep_data_groups = [
            cluster_ranks_exp[pp_rank, :, ep_rank, tp_rank].tolist()
            for pp_rank, ep_rank, tp_rank in itertools.product(
                range(pp),
                range(ep_model_degree),
                range(tp),
            )
        ]
    else:
        ep_data_groups = [[i] for i in range((world_size))]

    return ParallelGroups(tp_groups, dp_groups, pp_groups, ep_model_groups, ep_data_groups)


class PG_Group_Logic(Enum):
    LOGIC1 = (ascending_ring_PG_group, "Ascending Ring PG Group")
    LOGIC2 = (ascending_descending_ring_PG_group, "Ascending Descending Ring PG Group")

    def __init__(self, func: Callable[..., Any], description: str):
        self.func = func
        self.description = description
        
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

# Setting default replica group selection logic. Default is LOGIC1
_REPLICA_LOGIC: PG_Group_Logic = PG_Group_Logic.LOGIC1


def get_logic_chosen(lnc_size: int, hardware_type: hardware, tp: int)-> PG_Group_Logic:
    '''
    All the below are TP groups, if tp group is more than 1 then that means two tp groups talk to each other for DP/PP and should have communication between them,
    Logic 1: Group tensor parallel group in ascending ring fashion 0 to n-1 consecutive ranks belong to one TP group -> n is the tp size
    Logic 2: Group tensor parallel group in ascending ring fashion for first half and for second half choose the consective ranks in opposite direction. (0 to n/2-1) + (lastRank-1 to lastRank-1+ n/2-1 )
    
    NOTE: Custom devices chosen by cutomer would not work here, needs change. for eg: on trn2 if customer chooses 32 devices only and then does tp32 or tp16pp2 doesnt work and will choose logic1 as default

    VNC1: 128 cores
    1. 128 x 1 : Logic1 of (0,1,2,3,..127) should work
    2. 64 x 2  : Logic1 of (0,1,2,..63),(64,65,66,..127) doesnt work as 0 and 64 cant communicate directly, so need Logic2
    3. 32 x 4  : Logic1 of (0,1,2,..31),(32,33,34....63),(64,65,...95),(96,97,...127) should work 
    4. 8 x 16  : Logic1 of (0,1,2...7),(8,..15)....(120,...127) should work 
    5. 4 x 32  : Logic1 of (0,1,2,3)(4,5,6,7)......(124,125,126,127) should also work

    VNC2: 64 cores
    1. 4x16  : Logic1 of (0,1,2,3)(4,5,6,7),,,(60,61,62,63) should work
    2. 16x4  : Logic1 of (0,1,...15),....(48,49,....63) should work
    3. 32x2  : Logic1 of (0,1,...31)(32,33,...63) doesnt work as 0 and 32 cant communicate directly, so need Logic2
    4. 64x1  : Logic1 of (0,1,2,,,63) should work
    5. 4x4x4 : Logic1 of (0,1,2,3)(4,5,6,7)...(62,63,64,65,66) for tp and dp is (0,4,8,12)(1,5,9,13).. per row. PP group is (0,16,32,48)(4,20,36,52).. column wise. Logic 1 should work

    Not Supported ::
    VNC1:
    1. 16 x 8  : Logic1 of (0,1,2,..15),(16,..31)...(112,...127) doesnt work as 0 and 16 cant communicate directly, also logic3 wont work, we need diff logic - (0,1,2,..16)(33,34 ...48) and so on

    VNC2:
    1. 8x8   : Logic1 of (0,1,2,..7),(9,10,...15) ... doesnt work, Logic 2 is also not supporting this and we need different logic - (0,1,..7)(16,17,...23) and so on.

    Expert Parallelism is not supported
    '''
    
    routing_logic = {
        (hardware.TRN2, 1, 64) : PG_Group_Logic.LOGIC2,
        (hardware.TRN2, 2, 32) : PG_Group_Logic.LOGIC2,
    }

    ret_logic = routing_logic.get((hardware_type, lnc_size, tp), PG_Group_Logic.LOGIC1)
    
    # special case where only 32 out of 64 devices are used in vnc2 scenario of trn2, fallback to logic1
    if hardware_type == hardware.TRN2 and lnc_size==2 and torch.distributed.get_world_size() < 64:
        ret_logic = PG_Group_Logic.LOGIC1
    
    global _REPLICA_LOGIC
    _REPLICA_LOGIC = ret_logic

    logger.info(rmsg(f"Chosen Logic for replica groups {ret_logic=}"))
    return ret_logic


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    skip_collective_init: bool = False,
    lnc_size: int = 1,
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

    # set global _HARDWARE_TYPE variable from the arg passed in
    global _HARDWARE_TYPE
    _HARDWARE_TYPE = hardware(get_platform_target())

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    tensor_model_parallel_size = min(tensor_model_parallel_size, world_size)
    pipeline_model_parallel_size = min(pipeline_model_parallel_size, world_size)
    expert_model_parallel_size = min(expert_model_parallel_size, world_size)

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
    expert_data_parallel_size: int = world_size // (
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

        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        num_local_ranks = local_world_size // tensor_model_parallel_size
        cluster_ranks = torch.arange(0, world_size).reshape(
            pipeline_model_parallel_size, data_parallel_size // num_local_ranks, tensor_model_parallel_size, num_local_ranks
        )
        cluster_ranks_exp = cluster_ranks.transpose(-1, -2).reshape(
            pipeline_model_parallel_size, data_parallel_size, expert_model_parallel_size, tensor_model_parallel_size
        )
        cluster_ranks_nonexp = cluster_ranks.transpose(-1, -2).reshape(
            pipeline_model_parallel_size, data_parallel_size, tensor_model_parallel_size
        )
    else:
        cluster_ranks = torch.arange(0, world_size)
        cluster_ranks_exp = cluster_ranks.reshape(
            [
                pipeline_model_parallel_size,
                expert_data_parallel_size,
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
        logger.info("> initializing data parallel (exp) with size %d", expert_data_parallel_size)

    if not skip_collective_init and not cpu_mode():
        # cut graph because compiler birsim cannot verify graph with rand()
        xm.mark_step()
        # We create a dummy neff and execute it across all workers in the world.
        # This is done to initialize the collectives. Collectives initialization
        # requires all workers in the world to participate and this soometimes
        # may not be guranteed. Hence as a workaround, we run this dummy neff, and
        # get the collectives initialized.
        temp = torch.rand([1], device="xla")
        torch.distributed.all_reduce(temp, group=torch.distributed.group.WORLD)
        xm.mark_step()

    rank = torch.distributed.get_rank()
    compress_rg = not skip_collective_init and os.environ.get("NEURON_EXPERIMENTAL_COMPRESS_RG", "0") == "1"

    world_ranks = [list(range(0, world_size))]
    _build_and_assign_groups(
        group_name="_WORLD_GROUP",
        spmd_group_name="_WORLD_GROUP_SPMD",
        mesh=world_ranks,
        compress_rg=False,
    )
    
    allocate_ranks_fn = get_logic_chosen(lnc_size, _HARDWARE_TYPE, tp=tensor_model_parallel_size)

    replica_groups = allocate_ranks_fn(lnc_size, cluster_ranks_nonexp, cluster_ranks_exp,
                                        tensor_model_parallel_size, data_parallel_size, 
                                        pipeline_model_parallel_size, expert_model_parallel_size,
                                        expert_data_parallel_size)
    

    logger.info(rmsg(f"tp_groups: {replica_groups.tp_groups=}"))
    logger.info(rmsg(f"dp_groups: {replica_groups.dp_groups=}"))
    logger.info(rmsg(f"pp_groups: {replica_groups.pp_groups=}"))
    logger.info(rmsg(f"ep_model_groups: {replica_groups.ep_model_groups=}"))
    logger.info(rmsg(f"ep_data_groups: {replica_groups.ep_data_groups=}"))

    _build_and_assign_groups(
        group_name="_TENSOR_MODEL_PARALLEL_GROUP",
        spmd_group_name="_TENSOR_MODEL_PARALLEL_GROUP_SPMD",
        mesh=replica_groups.tp_groups,
        compress_rg=compress_rg,
    )

    _build_and_assign_groups(
        group_name="_DATA_PARALLEL_GROUP",
        spmd_group_name="_DATA_PARALLEL_GROUP_SPMD",
        mesh=replica_groups.dp_groups,
        compress_rg=False,
    )    

    _build_and_assign_groups(
        group_name="_PIPELINE_MODEL_PARALLEL_GROUP",
        spmd_group_name="_PIPELINE_MODEL_PARALLEL_GROUP_SPMD",
        mesh=replica_groups.pp_groups,
        compress_rg=False,
    )

    _build_and_assign_groups(
        group_name="_EXP_DATA_PARALLEL_GROUP",
        spmd_group_name="_EXP_DATA_PARALLEL_GROUP_SPMD",
        mesh=replica_groups.ep_data_groups,
        compress_rg=False,
    )
    
    _build_and_assign_groups(
        group_name="_EXPERT_MODEL_PARALLEL_GROUP",
        spmd_group_name="_EXPERT_MODEL_PARALLEL_GROUP_SPMD",
        mesh=replica_groups.ep_model_groups,
        compress_rg=False,
    )

    assert _PIPELINE_MODEL_PARALLEL_GROUP_SPMD
    for ranks in _PIPELINE_MODEL_PARALLEL_GROUP_SPMD:
        if rank in ranks:
            global _PIPELINE_GLOBAL_RANKS
            _PIPELINE_GLOBAL_RANKS = ranks
            break

    # Only create pre/next groups if PP is enabled
    if pipeline_model_parallel_size > 1:
        _create_pipeline_parallel_sr_groups(rank)

def _create_pipeline_parallel_sr_groups(rank: int) -> None:
    if not cpu_mode():
        parity = bool(get_pipeline_model_parallel_rank() % 2)
        _build_and_assign_groups(
            group_name="_NEXT_RANK_GROUP",
            spmd_group_name="_NEXT_RANK_GROUP_SPMD",
            mesh=get_pipeline_model_parallel_sr_group(parity),
            compress_rg=False,
        )
        _build_and_assign_groups(
            group_name="_PREV_RANK_GROUP",
            spmd_group_name="_PREV_RANK_GROUP_SPMD",
            mesh=get_pipeline_model_parallel_sr_group(not parity),
            compress_rg=False,
        )
    else:
        global _NEXT_RANK_GROUP_SPMD
        global _PREV_RANK_GROUP_SPMD
        global _NEXT_RANK_GROUP
        global _PREV_RANK_GROUP
        # cpu mode creation
        _SEND_RECV_PAIRS = get_pipeline_model_parallel_sr_group()
        _NEXT_RANK = get_pipeline_model_parallel_next_rank()
        _PREV_RANK = get_pipeline_model_parallel_prev_rank()
        logger.debug(rmsg(f"next ranks: {_NEXT_RANK_GROUP_SPMD}, prev ranks: {_PREV_RANK_GROUP_SPMD}"))
        for ranks in _SEND_RECV_PAIRS:
            logger.debug(rmsg(f"creating send/recv group with ranks {ranks}"))
            group = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks and _NEXT_RANK in ranks:
                _NEXT_RANK_GROUP = group
                _NEXT_RANK_GROUP_SPMD = group
            if rank in ranks and _PREV_RANK in ranks:
                _PREV_RANK_GROUP = group
                _PREV_RANK_GROUP_SPMD = group


def _build_and_assign_groups(
    group_name: str,
    spmd_group_name: str,
    mesh: List[List[int]],
    compress_rg: bool,
) -> None:
    def __set_global_var(key: str, val: Any) -> None:
        if key not in globals():
            raise RuntimeError(f"expected {key} to be in globals but was undefined")
        globals()[key] = val

    __set_global_var(key=spmd_group_name, val=mesh)
    rank = torch.distributed.get_rank()
    for group_ranks in mesh:
        if cpu_mode():
            # In CPU mode, all ranks need to participate every group creation
            group = torch.distributed.new_group(group_ranks, backend="gloo")
            if rank in group_ranks:
                __set_global_var(key=group_name, val=group)
                logger.debug(
                    rmsg(f"assigned {group_name} with "
                         f"group ranks {torch.distributed.get_process_group_ranks(group)}"
                    )
                )
        elif rank in group_ranks:
            group = torch.distributed.new_group(
                group_ranks,
                # When scaling to large number of nodes, the size of the replica groups becomes huge.
                # This increases the overall HLO hashing time which in turn causes framework overhead.
                # This can be reduced by passing the first tp replica only. All the other ranks would
                # infer their groups depending on the size of the replica group and the start and end ranks
                # Note: this works only for cases where the ranks are continuous. It won't work for TP=4 case.
                pg_options={"xla_pg_options": {"mesh": [mesh[0]] if compress_rg else mesh}},
            )
            __set_global_var(key=group_name, val=group)
            break

    if globals()[group_name] is None:
        raise RuntimeError(f"expected {group_name} to be initialized but was not. mesh: {mesh} and cur_rank={torch.distributed.get_rank()}")


def model_parallel_is_initialized() -> bool:
    """Check if model and data parallel groups are initialized."""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_world_group(as_list: bool = False) -> ProcessGroup:
    assert _WORLD_GROUP is not None, "intra_layer_model parallel group is not initialized"
    assert (group := getattr(_WORLD_GROUP, "_mesh") if as_list else _WORLD_GROUP)
    return group


def get_tensor_model_parallel_group(as_list: bool = False) -> Union[ProcessGroup, List[List[int]]]:
    """Get the tensor model parallel group the caller rank belongs to."""
    if as_list:
        return get_tensor_model_parallel_replica_groups()
    assert _TENSOR_MODEL_PARALLEL_GROUP, "intra_layer_model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_replica_groups() -> List[List[int]]:
    """Get the tensor model parallel replica groups."""
    assert _TENSOR_MODEL_PARALLEL_GROUP_SPMD is not None, "intra_layer_model parallel group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP_SPMD


def get_expert_model_parallel_group(as_list: bool = False) -> Union[ProcessGroup, List[List[int]]]:
    """Get the expert model parallel group the caller rank belongs to."""
    if as_list:
        return get_expert_model_parallel_replica_groups()
    assert _EXPERT_MODEL_PARALLEL_GROUP, "expert model parallel group is not initialized"
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_expert_model_parallel_replica_groups() -> List[List[int]]:
    """Get the expert model parallel replica groups."""
    assert _EXPERT_MODEL_PARALLEL_GROUP_SPMD is not None, "expert model parallel group is not initialized"
    return _EXPERT_MODEL_PARALLEL_GROUP_SPMD


def get_data_parallel_group(as_list: bool = False) -> Union[ProcessGroup, List[List[int]]]:
    """Get the data parallel group the caller rank belongs to."""
    if as_list:
        return get_data_parallel_replica_groups()
    assert _DATA_PARALLEL_GROUP, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_data_parallel_replica_groups() -> List[List[int]]:
    """Get the data parallel replica groups."""
    assert _DATA_PARALLEL_GROUP_SPMD is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP_SPMD


def get_expert_data_parallel_group(as_list: bool = False) -> Union[ProcessGroup, List[List[int]]]:
    """Get the expert data parallel group the caller rank belongs to."""
    if as_list:
        return get_expert_data_parallel_replica_groups()
    assert _EXP_DATA_PARALLEL_GROUP, "expert data parallel group is not initialized"
    return _EXP_DATA_PARALLEL_GROUP


def get_expert_data_parallel_replica_groups() -> List[List[int]]:
    """Get the expert data parallel replica groups."""
    assert _EXP_DATA_PARALLEL_GROUP_SPMD is not None, "expert data parallel group is not initialized"
    return _EXP_DATA_PARALLEL_GROUP_SPMD


def get_tensor_model_parallel_size() -> int:
    """Return world size for the tensor model parallel group."""
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    group = get_tensor_model_parallel_group()
    assert isinstance(group, ProcessGroup)
    tp_size = torch.distributed.get_world_size(group=group)
    set_tensor_model_parallel_size(tp_size)
    return tp_size


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
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    group = get_tensor_model_parallel_group()
    assert isinstance(group, ProcessGroup)
    tp_rank = torch.distributed.get_rank(group=group)
    set_tensor_model_parallel_rank(tp_rank)
    return tp_rank


def get_tensor_model_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    tp_size = get_tensor_model_parallel_size()
    if tp_size == 4:
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        offset = local_world_size * (global_rank // local_world_size)
        src_rank = offset + global_rank % tp_size
    else:
        src_rank = (global_rank // tp_size) * tp_size
    return src_rank


def set_expert_model_parallel_size(world_size: int) -> None:
    """Set the expert model parallel size."""
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_expert_model_parallel_size() -> int:
    """Return world size for the expert model parallel group."""
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    group = get_expert_model_parallel_group()
    assert isinstance(group, ProcessGroup)
    emp_size = torch.distributed.get_world_size(group=group)
    set_expert_model_parallel_size(emp_size)
    return emp_size


def set_expert_model_parallel_rank(rank: int) -> None:
    """Set the expert model parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def get_expert_model_parallel_rank() -> int:
    """Return my rank for the expert model parallel group."""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    group = get_expert_model_parallel_group()
    assert isinstance(group, ProcessGroup)
    emp_rank = torch.distributed.get_rank(group=group)
    set_expert_model_parallel_rank(emp_rank)
    return emp_rank


def get_data_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to the first local rank in the data parallel group."""
    global_rank = torch.distributed.get_rank()
    num_data_parallel_groups = len(get_data_parallel_replica_groups())
    return global_rank % num_data_parallel_groups


def get_data_parallel_size() -> int:
    """Return world size for the data parallel group."""
    group = get_data_parallel_group()
    assert isinstance(group, ProcessGroup)
    return torch.distributed.get_world_size(group=group)


def get_data_parallel_rank() -> int:
    """Return my rank for the data parallel group."""
    
    global _DATA_PARALLEL_GROUP_SPMD
    global _REPLICA_LOGIC

    if _DATA_PARALLEL_GROUP_SPMD and _REPLICA_LOGIC == PG_Group_Logic.LOGIC2:
        # get the global rank and then search through the DP group manually on what index does this rank belong to in each DP replica group. Required for Logic 2
        cur_rank = torch.distributed.get_rank()
        for groups in _DATA_PARALLEL_GROUP_SPMD:
            if cur_rank in groups:            
                return groups.index(cur_rank)
        assert False, f"Should not reach here, Cur_rank={cur_rank}, mesh={_DATA_PARALLEL_GROUP_SPMD} "
    else:
        # above manual search vs this code searches via torch, the problem that we have with this approach is when torch creates a group it always constructs the 
        # group of [48,32] as {32:0, 48:1} meaning 32 will send data to 48, but in asc desc ring topology its the reverse where 48 is the sender and 32 is the receiver
        # Since torch sorts it automatically its not possible to get the PP ranks correctly while calculating next and prev ranks for PP, thus using above manual logic where
        # we scal [48,32] group and then assign {48:0, 32:1} so on and so forth via indexing the original PP group in _DATA_PARALLEL_GROUP_SPMD mesh
        group = get_data_parallel_group()
        assert isinstance(group, ProcessGroup)
        return torch.distributed.get_rank(group=group)


def get_expert_data_parallel_size() -> int:
    """Return world size for the expert data parallel group."""
    group = get_expert_data_parallel_group()
    assert isinstance(group, ProcessGroup)
    return torch.distributed.get_world_size(group=group)


def get_expert_data_parallel_rank() -> int:
    """Return my rank for the expert data parallel group."""
    group = get_expert_data_parallel_group()
    assert isinstance(group, ProcessGroup)
    return torch.distributed.get_rank(group=group)


def get_pipeline_model_parallel_group(as_list: bool = False) -> Union[ProcessGroup, List[List[int]]]:
    """Get the pipeline model parallel group the caller rank belongs to."""
    if as_list:
        return get_pipeline_model_parallel_replica_groups()
    assert _PIPELINE_MODEL_PARALLEL_GROUP, "pipeline_model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_replica_groups() -> List[List[int]]:
    """Get the pipeline model parallel replica groups."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP_SPMD is not None, "pipeline_model parallel group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP_SPMD


def get_pipeline_model_parallel_rank() -> int:
    """Return my rank for the pipeline model parallel group."""

    global _PIPELINE_MODEL_PARALLEL_GROUP_SPMD
    global _REPLICA_LOGIC

    if _PIPELINE_MODEL_PARALLEL_GROUP_SPMD and _REPLICA_LOGIC == PG_Group_Logic.LOGIC2:
        # get the global rank and then search through the PP group manually on what index does this rank belong to in each PP replica group. 
        # This is for next and prev ranks of PP group selection
        cur_rank = torch.distributed.get_rank()
        for groups in _PIPELINE_MODEL_PARALLEL_GROUP_SPMD:
            if cur_rank in groups:            
                return groups.index(cur_rank)
        assert False, f"Should not reach here, Cur_rank={cur_rank}, mesh={_PIPELINE_MODEL_PARALLEL_GROUP_SPMD} "
    else:
        # above manual search vs this code searches via torch, the problem that we have with this approach is when torch creates a group it always constructs the 
        # group of [48,32] as {32:0, 48:1} meaning 32 will send data to 48, but in asc desc ring topology its the reverse where 48 is the sender and 32 is the receiver
        # Since torch sorts it automatically its not possible to get the PP ranks correctly while calculating next and prev ranks for PP, thus using above manual logic where
        # we scal [48,32] group and then assign {48:0, 32:1} so on and so forth via indexing the original PP group in _PIPELINE_MODEL_PARALLEL_GROUP_SPMD mesh
        group = get_pipeline_model_parallel_group()
        assert isinstance(group, ProcessGroup)
        return torch.distributed.get_rank(group=group)


def get_pipeline_model_parallel_sr_group(parity: Optional[bool] = None) -> List[List[int]]:
    world_size = get_pipeline_model_parallel_size()

    def subgroup(r, ranks):
        return [ranks[r], ranks[(r + 1) % world_size]]

    group = list()
    for ranks in get_pipeline_model_parallel_replica_groups():
        if not cpu_mode():
            # for xla mode, that we separate the pre and next groups
            assert parity is not None, "must provide argument `parity` for xla mode"
            for i in range(parity, world_size, 2):
                group.append(subgroup(i, ranks))
        else:
            # for cpu mode, we create all pairs
            for i in range(world_size):
                # ranks include all ranks in a pipeline group
                group.append(subgroup(i, ranks))
    return group


def get_pipeline_model_parallel_size() -> int:
    """Return world size for the pipeline model parallel group."""
    group = get_pipeline_model_parallel_group()
    assert isinstance(group, ProcessGroup)
    return torch.distributed.get_world_size(group=group)


def get_next_rank_group(as_list: bool = False) -> Union[ProcessGroup, List[List[int]]]:
    """Get the next tensor model parallel group the caller rank belongs to."""
    if as_list:
        return get_next_rank_replica_groups()
    assert _NEXT_RANK_GROUP, "intra_layer_model parallel group is not initialized"
    return _NEXT_RANK_GROUP


def get_next_rank_replica_groups() -> List[List[int]]:
    """Get the next tensor model parallel replica groups."""
    assert _NEXT_RANK_GROUP_SPMD is not None, "intra_layer_model parallel group is not initialized"
    return _NEXT_RANK_GROUP_SPMD


def get_prev_rank_group(as_list: bool = False) -> Union[ProcessGroup, List[List[int]]]:
    """Get the previous tensor model parallel group the caller rank belongs to."""
    if as_list:
        return get_prev_rank_replica_groups()
    assert _PREV_RANK_GROUP, "intra_layer_model parallel group is not initialized"
    return _PREV_RANK_GROUP


def get_prev_rank_replica_groups() -> List[List[int]]:
    """Get the previous tensor model parallel replica groups."""
    assert _PREV_RANK_GROUP_SPMD is not None, "intra_layer_model parallel group is not initialized"
    return _PREV_RANK_GROUP_SPMD


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
    global _WORLD_GROUP
    _WORLD_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP_SPMD
    _TENSOR_MODEL_PARALLEL_GROUP_SPMD = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None

    global _EXPERT_MODEL_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = None
    global _EXPERT_MODEL_PARALLEL_GROUP_SPMD
    _EXPERT_MODEL_PARALLEL_GROUP_SPMD = None
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = None

    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP_SPMD
    _DATA_PARALLEL_GROUP_SPMD = None

    global _EXP_DATA_PARALLEL_GROUP
    _EXP_DATA_PARALLEL_GROUP = None
    global _EXP_DATA_PARALLEL_GROUP_SPMD
    _EXP_DATA_PARALLEL_GROUP_SPMD = None

    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP_SPMD
    _PIPELINE_MODEL_PARALLEL_GROUP_SPMD = None
    global _PIPELINE_GLOBAL_RANKS
    _PIPELINE_GLOBAL_RANKS = None

    global _NEXT_RANK_GROUP
    _NEXT_RANK_GROUP = None
    global _NEXT_RANK_GROUP_SPMD
    _NEXT_RANK_GROUP_SPMD = None

    global _PREV_RANK_GROUP
    _PREV_RANK_GROUP = None
    global _PREV_RANK_GROUP_SPMD
    _PREV_RANK_GROUP_SPMD = None


def initialize_token_shuffle_group(token_shuffle_group_size: int = 1):
    """
    Initialize communication groups for token shuffling.

    Arguments:
        token_shuffle_group_size: the group size of devices conducting token shuffling collectively. It is a subset of the data parallel group. 1 means not shuffle.
    """
    global _TOKEN_SHUFFLE_GROUP_SIZE
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    if _TOKEN_SHUFFLE_GROUP is not None:
        assert (
            token_shuffle_group_size == _TOKEN_SHUFFLE_GROUP_SIZE
        ), "Currently the library supports only token shuffle group size for all layers."
        return
    _TOKEN_SHUFFLE_GROUP_SIZE = token_shuffle_group_size

    data_parallel_size = get_data_parallel_size()
    tensor_model_parallel_size = get_tensor_model_parallel_size()
    pipeline_model_parallel_size = get_pipeline_model_parallel_size()
    world_size = torch.distributed.get_world_size()
    assert (
        token_shuffle_group_size <= data_parallel_size
    ), "token_shuffle_group_size should be less than or equal to data_parallel_size"

    if data_parallel_size % token_shuffle_group_size != 0:
        raise RuntimeError(
            f"invalid implied shuffle group size: "
            f"`data_parallel_size` ({data_parallel_size}) is not divisible by "
            f"token_shuffle_group_size ({token_shuffle_group_size})"
        )
    logger.info(rmsg(f"> initializing token shuffle group with size {token_shuffle_group_size}"))

    cluster_ranks = torch.arange(0, world_size)
    cluster_ranks_shuffle = cluster_ranks.reshape(
        [
            pipeline_model_parallel_size,
            data_parallel_size // token_shuffle_group_size,
            token_shuffle_group_size,
            tensor_model_parallel_size,  # important: contiguous parallelism dimension
        ]
    )

    # Build the shuffle groups. Assign related variables.
    all_token_shuffle_group_ranks = [
        cluster_ranks_shuffle[pp_rank, non_shuffle_rank, :, tp_rank].tolist()
        for pp_rank, non_shuffle_rank, tp_rank in itertools.product(
            range(pipeline_model_parallel_size),
            range(data_parallel_size // token_shuffle_group_size),
            range(tensor_model_parallel_size),
        )
    ]

    _build_and_assign_groups(
        group_name="_TOKEN_SHUFFLE_GROUP",
        spmd_group_name="_TOKEN_SHUFFLE_GROUP_SPMD",
        mesh=all_token_shuffle_group_ranks,
        compress_rg=False,
    )


def get_token_shuffle_group(as_list: bool = False) -> Union[ProcessGroup, List[List[int]]]:
    """Get the token shuffle group the caller rank belongs to."""
    if as_list:
        return get_token_shuffle_replica_groups()
    assert _TOKEN_SHUFFLE_GROUP, "token_shuffle parallel group is not initialized"
    return _TOKEN_SHUFFLE_GROUP


def get_token_shuffle_replica_groups() -> List[List[int]]:
    """Get the token shuffle replica groups."""
    assert _TOKEN_SHUFFLE_GROUP_SPMD is not None, "token_shuffle parallel group is not initialized"
    return _TOKEN_SHUFFLE_GROUP_SPMD


def get_token_shuffle_group_size() -> int:
    """Return world size for the token shuffle group."""
    assert _TOKEN_SHUFFLE_GROUP_SIZE, "token_shuffle parallel group is not initialized"
    return _TOKEN_SHUFFLE_GROUP_SIZE


def destroy_token_shuffle_group():
    global _TOKEN_SHUFFLE_GROUP
    _TOKEN_SHUFFLE_GROUP = None
    global _TOKEN_SHUFFLE_GROUP_SPMD
    _TOKEN_SHUFFLE_GROUP_SPMD = None
    global _TOKEN_SHUFFLE_GROUP_SIZE
    _TOKEN_SHUFFLE_GROUP_SIZE = None


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
    if PP_GROUP_PG_GLOO is not None:
        logger.error("pp gloo groups are already initialized!")
        return
    pp_group_spmd = get_pipeline_model_parallel_replica_groups()
    logger.debug(f"initialize_pp_gloo_groups... {pp_group_spmd}")
    rank = torch.distributed.get_rank()
    for pp_group in pp_group_spmd:
        if rank in pp_group:
            PP_GROUP_PG_GLOO = torch.distributed.new_group(ranks=pp_group, backend="gloo")
            break


def get_pp_gloo_group() -> ProcessGroup:
    assert PP_GROUP_PG_GLOO, "pp gloo groups are not initialized!"
    return PP_GROUP_PG_GLOO


def is_global_rank_zero() -> bool:
    # TODO: Change this to torch.distributed.get_rank when PTL fix of init_process
    # before nxd_config is added.
    if cpu_mode():
        return torch.distributed.get_rank() == 0
    else:
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
    pp_model_parallel_group_spmd = get_pipeline_model_parallel_replica_groups()
    saved_group = None
    for ranks, current_shared_ranks in zip(pp_model_parallel_group_spmd, all_shared_ranks_spmd):
        if cpu_mode():
            group = torch.distributed.new_group(ranks, backend="gloo")
            if world_rank in ranks:
                saved_group = group
                logger.debug(
                    rmsg(
                        f"creating pg based on ranks {ranks}, "
                        f"all_shared_ranks_spmd {all_shared_ranks_spmd}, "
                        f"current_shared_ranks {current_shared_ranks}"  # noqa: E501
                    )
                )
        elif world_rank in ranks:
            # in xla mode, only relevant ranks join the group creation once
            logger.debug(
                rmsg(
                    f"creating pg based on ranks {ranks}, "
                    f"all_shared_ranks_spmd {all_shared_ranks_spmd}, "
                    f"current_shared_ranks {current_shared_ranks}"  # noqa: E501
                )
            )
            pg_options = {"xla_pg_options": {"mesh": all_shared_ranks_spmd}}
            saved_group = torch.distributed.new_group(current_shared_ranks, pg_options=pg_options)
            break
    return saved_group


def initialize_kv_group(kv_shared_group_size: int = 1, sequential_ranks_in_group: bool = False):
    # Build the kv-shared model-parallel groups.
    global _KV_SHARED_GROUP_SIZE
    global _HARDWARE_TYPE
    _HARDWARE_TYPE = hardware(get_platform_target())

    if _KV_SHARED_GROUP is not None:
        assert (
            kv_shared_group_size == _KV_SHARED_GROUP_SIZE
        ), "Currently the library supports only single KV size for all layers"
        return

    tensor_model_parallel_size = get_tensor_model_parallel_size()
    assert tensor_model_parallel_size % kv_shared_group_size == 0, (
        f"kv_shared_group_size: {kv_shared_group_size}, "
        f"should divide tensor model parallel group {tensor_model_parallel_size} "
    )
    _KV_SHARED_GROUP_SIZE = kv_shared_group_size
    logger.info(rmsg("> initializing kv group with size {}".format(kv_shared_group_size)))
    world_size = torch.distributed.get_world_size()
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    all_kv_shred_group_ranks = arrange_kv_groups(num_tensor_model_parallel_groups, tensor_model_parallel_size,
                                                 kv_shared_group_size, sequential_ranks_in_group, _HARDWARE_TYPE)
    _build_and_assign_groups(
        group_name="_KV_SHARED_GROUP",
        spmd_group_name="_KV_SHARED_GROUP_SPMD",
        mesh=all_kv_shred_group_ranks,
        compress_rg=False,
    )


def get_kv_shared_group(as_list: bool = False) -> Union[ProcessGroup, List[List[int]]]:
    """Get the KV shared group the caller rank belongs to."""
    if as_list:
        return get_kv_shared_replica_groups()
    assert _KV_SHARED_GROUP, "kv_shared parallel group is not initialized"
    return _KV_SHARED_GROUP


def get_kv_shared_replica_groups() -> List[List[int]]:
    """Get the KV shared replica groups."""
    assert _KV_SHARED_GROUP_SPMD is not None, "kv_shared parallel group is not initialized"
    return _KV_SHARED_GROUP_SPMD


def get_kv_shared_group_size() -> int:
    """Get the KV shared group size."""
    assert _KV_SHARED_GROUP_SIZE, "kv_shared parallel group is not initialized"
    return _KV_SHARED_GROUP_SIZE


def destroy_kv_group() -> None:
    global _KV_SHARED_GROUP
    _KV_SHARED_GROUP = None
    global _KV_SHARED_GROUP_SPMD
    _KV_SHARED_GROUP_SPMD = None
    global _KV_SHARED_GROUP_SIZE
    _KV_SHARED_GROUP_SIZE = None


def initialize_speculative_draft_group(group_size: int = 1):
    # Build the kv-shared model-parallel groups.
    global _SPECULATIVE_DRAFT_GROUP_SIZE
    global _SPECULATIVE_DRAFT_GROUP
    if _SPECULATIVE_DRAFT_GROUP is not None:
        assert (
            group_size == _SPECULATIVE_DRAFT_GROUP_SIZE
        ), "Currently the library supports only single speculative draft group size"
        return

    tensor_model_parallel_size = get_tensor_model_parallel_size()
    assert tensor_model_parallel_size % group_size == 0, (
        f"draft_group_size: {group_size}, "
        f"should divide tensor model parallel group {tensor_model_parallel_size} "
    )
    _SPECULATIVE_DRAFT_GROUP_SIZE = group_size
    logger.info(rmsg("> initializing draft group with size {}".format(group_size)))
    world_size = torch.distributed.get_world_size()
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    
    draft_groups = []
    total_ranks = num_tensor_model_parallel_groups * tensor_model_parallel_size
    for i in range(0, total_ranks, group_size):
        ranks = list(range(i, i + group_size))
        draft_groups.append(ranks)
    
    _build_and_assign_groups(
        group_name="_SPECULATIVE_DRAFT_GROUP",
        spmd_group_name="_SPECULATIVE_DRAFT_GROUP_SPMD",
        mesh=draft_groups,
        compress_rg=False,
    )


def get_speculative_draft_group(as_list: bool = False) -> Union[ProcessGroup, List[List[int]]]:
    """Get the draft group the caller rank belongs to."""
    if as_list:
        return get_speculative_draft_replica_groups()
    assert _SPECULATIVE_DRAFT_GROUP, "draft parallel group is not initialized"
    return _SPECULATIVE_DRAFT_GROUP


def get_speculative_draft_replica_groups() -> List[List[int]]:
    """Get the draft replica groups."""
    assert _SPECULATIVE_DRAFT_GROUP_SPMD is not None, "draft parallel group is not initialized"
    return _SPECULATIVE_DRAFT_GROUP_SPMD



def gather_python_object(obj: Any, group: ProcessGroup) -> List[Any]:
    """
    Eagerly gather python object for a group
    Usually used to collect timeline events
    """
    object_gather_list: List[Any] = []
    if torch.distributed.get_rank(group=group) == 0:
        object_gather_list = [None] * torch.distributed.get_world_size(group=group)
    torch.distributed.gather_object(obj, object_gather_list=object_gather_list, group=group)
    return object_gather_list


def set_aot_mode(mode: bool):
    """Set AOT mode used for tracing"""
    global _AOT_MODE
    _AOT_MODE = mode


def get_aot_mode() -> bool:
    """Get AOT mode used for tracing"""
    global _AOT_MODE
    return _AOT_MODE


def arrange_kv_groups(num_tensor_model_parallel_groups: int = 1, tensor_model_parallel_size: int = 1,
                      kv_shared_group_size: int = 1, sequential_ranks_in_group: bool = False, hardware_type: hardware = hardware.TRN1) -> List[List[int]]:
    """
    E.g. one num_tensor_model_parallel_groups contains [0,1,2,3] and every 2 ranks are grouped (kv_shared_group_size)
    when sequential_ranks_in_group is True => groups = [[0,1], [2,3]
    when sequential_ranks_in_group is False, it is interleaved => groups = [[0,2], [1,3]]
    On Trn1:
        For TP32 with kv_replication 4: (K0,K1,K2,K3)(K0,K1,K2,K3)…. 4 times and each nearby ranks holding different K heads
    """
    groups = []
    if hardware_type == hardware.TRN2:
        """
        Replicate single kv head on adjacent ranks (specific to Trn2 currently)
            For TP64 with kv_replication 8: (K0,K0,K0….. 8 times)(K1,K1,……. 8 times) … for all K heads
            For TP128 with kv_replication 16: (K0,K0,K0….. 16 times)(K1,K1,……. 16 times) … for all K heads
        """
        num_kv_heads = tensor_model_parallel_size // kv_shared_group_size
        for i in range(num_tensor_model_parallel_groups):
            for k in range(num_kv_heads):
                group = []
                for j in range(kv_shared_group_size):
                    rank = i * tensor_model_parallel_size + k * kv_shared_group_size + j
                    group.append(rank)
                groups.append(group)
    elif sequential_ranks_in_group:
        total_ranks = num_tensor_model_parallel_groups * tensor_model_parallel_size
        for i in range(0, total_ranks, kv_shared_group_size):
            ranks = list(range(i, i + kv_shared_group_size))
            groups.append(ranks)
    else:
        for i in range(num_tensor_model_parallel_groups):
            for j in range(tensor_model_parallel_size // kv_shared_group_size):
                ranks = list(
                    range(
                        i * tensor_model_parallel_size + j,
                        (i + 1) * tensor_model_parallel_size,
                        tensor_model_parallel_size // kv_shared_group_size,
                        )
                )
                groups.append(ranks)
    return groups


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
