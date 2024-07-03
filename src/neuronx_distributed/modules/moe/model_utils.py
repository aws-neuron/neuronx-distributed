import enum
import torch
import torch.nn.functional as F

ACT2FN = {
    "gelu": F.gelu,
    "leaky_relu": F.leaky_relu,
    "relu": F.relu,
    "sigmoid": torch.sigmoid,
    "silu": F.silu,
    "tanh": torch.tanh,
}


class MoESequenceParallelMode(str, enum.Enum):
    """Defines the modes of sequence parallelism used by in MoE."""

    # No sequence parallel
    NO_SP = "NO_SP"

    # Exit SP on entry to MoE layer, scatter before exiting
    EXIT_SP_ON_ENTRY = "EXIT_SP_ON_ENTRY"

    # Exit SP on entry to MoE layer, don't do the all-reduce in down_proj MLP, reduce-scatter before exiting
    EXIT_SP_ON_ENTRY_DELAY_MLP_AR = "EXIT_SP_ON_ENTRY_DELAY_MLP_AR"

    # Use SP optimizations for matmul-based permute/unpermute
    OPTIMIZED_SP_MATMUL = "OPTIMIZED_SP_MATMUL"
