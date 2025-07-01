from typing import Dict, Callable

import torch
import torch.nn.functional as F
from neuronxcc.nki._private_kernels.blockwise_mm import SkipMode

from neuronx_distributed.utils.model_utils import get_platform_lnc

ACT2FN: Dict[str, Callable] = {
    "gelu": F.gelu,
    "leaky_relu": F.leaky_relu,
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "silu": F.silu,
    "tanh": torch.tanh,
}

# Used to determine when to use selective loading for token generation. See forward() for more details.
DEFAULT_SELECTIVE_LOADING_THRESHOLD = 1.0
DEFAULT_BLOCK_SIZE = 512
DEFAULT_SKIP_MODE = (False, False)
DEFAULT_LNC_SIZE = get_platform_lnc()
DEFAULT_PADDING_VALUE = -1
