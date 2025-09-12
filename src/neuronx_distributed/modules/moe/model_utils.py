from typing import Dict, Callable, Any

import torch
import torch.nn.functional as F
from enum import Enum
from neuronx_distributed.utils.model_utils import get_platform_lnc
from neuronx_distributed.utils.logger import get_logger

logger = get_logger()

ACT2FN: Dict[str, Callable] = {
    "gelu": F.gelu,
    "leaky_relu": F.leaky_relu,
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "silu": F.silu,
    "tanh": torch.tanh,
}

class GLUType(Enum):
    """
    Enum class defined all supported GLU type supported in MoE module in NxD
    """
    GLU = "glu"
    SWIGLU = "swiglu"

    @classmethod
    def validate(cls, glu_type):
        if isinstance(glu_type, cls):
            return glu_type
        if glu_type is None:
            logger.warning("glu_type is None, default to basic GLU")
            glu_type = "glu"
        if glu_type not in [e.value for e in cls]:
            raise ValueError(f"glu_type={glu_type} not supported, must be one of {[e.value for e in cls]}")
        return cls(glu_type)

class ACTFunc(Enum):
    """
    Enum class defined all supported activation functions supported in MoE module in NxD
    Example:
    ACTFunc.SILU.value = 0
    ACTFunc.SILU.name_str = "silu"
    """
    def __new__(cls, id, name):
        obj = object.__new__(cls)
        obj._value_ = id
        obj.id = id
        obj.name_str = name
        return obj

    # The enum values to match with kernel enum values
    SILU = (0, "silu")
    GELU = (1, "gelu")
    GELU_TANH_APPROX = (2, "gelu_tanh_approx")
    SIGMOID = (3, "sigmoid")
    RELU = (4, "relu")
    TANH = (5, "tanh")
    LEAKY_RELU = (6, "leaky_relu")

    @classmethod
    def validate(cls, act_func):
        if isinstance(act_func, cls):
            return act_func

        if act_func is None:
            logger.warning("act_func is None, default to SIGMOID")
            return cls.SIGMOID

        if isinstance(act_func, str):
            for member in cls:
                if member.name_str == act_func:
                    return member
            raise ValueError(f"act_func={act_func} not supported, must be one of: {[e.name_str for e in cls]}")

        raise ValueError(f"Invalid type for act_func: {type(act_func)}")

def get_kernel_activation_func_id(act_fn: ACTFunc, glu_type: GLUType):
    """
    This function is used to check which activation function will be used in NKI blockwise kernel.
    The corresponding INT value of activation function id will be returned and used to construct
    ActivationFunction(id) in blockwise.py
    """
    if glu_type == GLUType.GLU:
        if act_fn == ACTFunc.SILU:
            return ACTFunc.SILU.value
    if glu_type == GLUType.SWIGLU:
        if act_fn == ACTFunc.SIGMOID:
            return ACTFunc.SIGMOID.value
    raise ValueError(f"Unsupported ACTFunc and GLUType combination in NKI kernel flow: {act_fn}, {glu_type}.")

# Used to determine when to use selective loading for token generation. See forward() for more details.
DEFAULT_SELECTIVE_LOADING_THRESHOLD = 1.0
DEFAULT_BLOCK_SIZE = 512
DEFAULT_SKIP_MODE = (False, False)
DEFAULT_LNC_SIZE = get_platform_lnc()
DEFAULT_PADDING_VALUE = -1
# This number is the hardcoded hidden_act scaling factor in shard-on-intermediate dynamic kernel, so in can_use_blockwise_matmul_nki, if
# user want to use shard-on-intermediation dynamic kernel, then the scaling factor should be 1.702
DEFAULT_HIDDEN_ACT_SCALING_FACTOR = 1.702

def create_spmd_ranks(
    model_state_dict: Dict[str, Any],
    prefix: str,
    world_size,
):
    # add weight for spmd rank
    model_state_dict[f"{prefix}spmd_rank.rank"] = torch.arange(
        0, world_size, dtype=torch.int32
    )
