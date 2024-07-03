import copy
from typing import Any, Callable, Dict, TypedDict

from neuronx_distributed.quantization.quantization_layers import (
    QuantizationType,
    QuantizedDtype,
)
from neuronx_distributed.quantization.quantization_mappings import (
    get_default_quant_module_mappings,
)


class CONFIG_DICT_TYPE(TypedDict):
    quantization_type: QuantizationType
    quantized_dtype: QuantizedDtype


_DEFAULT_CUSTOM_CONFIG_DICT: CONFIG_DICT_TYPE = {
    "quantization_type": QuantizationType.SCALAR,
    "quantized_dtype": QuantizedDtype.INT8,
}


def get_default_custom_config_dict() -> CONFIG_DICT_TYPE:
    r"""Defines the default custom config dict."""
    return _DEFAULT_CUSTOM_CONFIG_DICT


def convert(
    module: Any, q_config: CONFIG_DICT_TYPE = None, inplace: bool = False, mapping: Dict[Callable, Any] = None
) -> Any:
    """Funtion to convert a Non quantized module to its quantized version based on the q_config

    Args:
        module (Any): Module to be quantized
        q_config (CONFIG_DICT_TYPE, optional): Q Config to be used to convert the module. Defaults to None.
        inplace (bool, optional): Replace module inplace. Defaults to False.
        mapping (Dict[Callable, Any], optional): A mapping to be used for Module swapping. Defaults to None.

    Returns:
        Any: Swapped Module
    """
    if not inplace:
        module = copy.deepcopy(module)
    if q_config is None:
        q_config = get_default_custom_config_dict()
    if mapping is None:
        mapping = get_default_quant_module_mappings()

    _convert_initialized_float_to_initialized_quantized(module=module, q_config=q_config, mapping=mapping)
    return module


def _convert_initialized_float_to_initialized_quantized(module, q_config, mapping):
    """
    A function to convert an non-quantized model with default values to its corresponding quantized model with its default values.

    NOTE: This is not a function to transfer the actual weights, and this will be done subsequently
    """
    reassign = {}

    for name, mod in module.named_children():
        if not type(mod) in mapping:
            _convert_initialized_float_to_initialized_quantized(module=mod, q_config=q_config, mapping=mapping)
        if type(mod) in mapping:
            quantized_class = mapping[type(mod)]
            reassign[name] = quantized_class.from_float(
                mod=mod,
                quantization_type=q_config.get("quantization_type"),
                quantized_dtype=q_config.get("quantized_dtype"),
            )
        # Currently there is a bug in quantize.convert function where even though
        # Parallel embedding has set for tensor_model_parallel attribute, it does not show
        # up in the converted module. Detailed investigation to be performed here: NAPP-2474
        if type(mod).__name__ == "ParallelEmbedding":
            setattr(mod.weight, "tensor_model_parallel", True)
            setattr(mod.weight, "partition_dim", mod.stride)
            setattr(mod.weight, "partition_stride", mod.weight_partition_dim)

    for key, value in reassign.items():
        module._modules[key] = value

    return module
