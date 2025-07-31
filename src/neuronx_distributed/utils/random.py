import random

import neuronx_distributed as nxd
import numpy as np
import torch


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if nxd.parallel_layers.parallel_state.model_parallel_is_initialized():
        nxd.parallel_layers.random.model_parallel_xla_manual_seed(seed)
