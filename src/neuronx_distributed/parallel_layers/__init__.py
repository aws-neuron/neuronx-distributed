"""Model parallel utility interface."""

from .checkpointing import save, load
from .grads import clip_grad_norm
from .layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    copy_tensor_model_parallel_attributes,
    set_defaults_if_not_set_tensor_model_parallel_attributes,
    set_tensor_model_parallel_attributes,
)
from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from .parallel_state import initialize_model_parallel
from .random import get_xla_rng_tracker, model_parallel_xla_manual_seed
from .utils import (
    copy_tensor_model_parallel_attributes,
    set_defaults_if_not_set_tensor_model_parallel_attributes,
    set_tensor_model_parallel_attributes,
    split_tensor_along_last_dim,
)

# Specify parallel modules and functions so that we can skip tracing them during PP partition
PARALLEL_MODULES = [ColumnParallelLinear, RowParallelLinear, ParallelEmbedding]
PARALLEL_FUNCTIONS = [
    parallel_cross_entropy,
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
]
