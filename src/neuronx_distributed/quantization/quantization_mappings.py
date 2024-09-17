"""
Module to map the quantization layers to their corresponding non quantized versions
"""
from typing import Any, Callable, Dict

import neuronx_distributed.modules.moe.moe_parallel_layers as moe_parallel_layers
import neuronx_distributed.parallel_layers.layers as parallel_layers
import neuronx_distributed.quantization.quantization_layers as q_layers

# Default map for swapping dynamic modules
DEFAULT_QUANT_MODULE_MAPPINGS: Dict[Callable, Any] = {
    parallel_layers.ColumnParallelLinear: q_layers.QuantizedColumnParallel,
    parallel_layers.RowParallelLinear: q_layers.QuantizedRowParallel,
    moe_parallel_layers.ExpertFusedColumnParallelLinear: q_layers.QuantizedExpertFusedColumnParallel,
    moe_parallel_layers.ExpertFusedRowParallelLinear: q_layers.QuantizedExpertFusedRowParallel,
}


def get_default_quant_module_mappings() -> Dict[Callable, Any]:
    """Get module mapping for dynamic quantization"""
    return DEFAULT_QUANT_MODULE_MAPPINGS
