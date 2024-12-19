import copy
from fnmatch import fnmatch
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from neuronx_distributed.quantization.quantization_config import (
    BASE_QCONFIG_DICT_TYPE,
    get_default_custom_qconfig_dict,
)
from neuronx_distributed.quantization.quantization_mappings import (
    get_default_quant_module_mappings,
)
from neuronx_distributed.utils.logger import get_logger

logger = get_logger("Neuron")

def convert(
    module: torch.nn.Module,
    q_config: Optional[BASE_QCONFIG_DICT_TYPE] = None,
    inplace: bool = False, mapping: Optional[Dict[Callable, Any]] = None,
    include: Optional[Union[str, List[str]]] = None,
    modules_to_not_convert: Optional[List[str]] = None,
) -> torch.nn.Module:
    """Funtion to convert a Non quantized module to its quantized version based on the q_config

    Args:
        module (Any): Module to be quantized
        q_config (CONFIG_DICT_TYPE, optional): Q Config to be used to convert the module. Defaults to None.
        inplace (bool, optional): Replace module inplace. Defaults to False.
        mapping (Dict[Callable, Any], optional): A mapping to be used for Module swapping. Defaults to None.
        include (Optional[Union[str, List[str]]], optional): Patterns constituting the allowlist. If provided, module names must match at
            least one pattern from the allowlist.. Defaults to None.
        modules_to_not_convert (Optional[List[str]], optional):
            A list of modules to not convert. If a module name is in the list (e.g. `lm_head`), it will not be
            converted.

        Example of include = ["*mlp.linear_block.lay2"]
        It will select all the layers with the suffix. Here if lay2 is a module not present
        in mapping, it will recursively parse though the lay2, and replace all the layers present
        in the mapping. If lay2 is in the mapping, it will swap out the layer.

    Returns:
        Any: Swapped Module
    """
    if not inplace:
        module = copy.deepcopy(module)
    if q_config is None:
        q_config = get_default_custom_qconfig_dict()
    if mapping is None:
        mapping = get_default_quant_module_mappings()

    assert (include is None) and (modules_to_not_convert is None) or ((include is None) ^ (modules_to_not_convert is None)), f"Either include and modules_to_not_convert both should be None, or only one of them should be not-None, not both. Provided values include: {include} , modules_to_not_convert: {modules_to_not_convert}"

    if include is None:
        _convert_initialized_float_to_initialized_quantized(module=module, q_config=q_config, mapping=mapping, modules_to_not_convert=modules_to_not_convert)
    else:
        logger.info("Will swap from include list")

        for name, module_to_swap in module.named_modules():
            if not any(fnmatch(name, pattern) for pattern in include):
                continue
            if type(module_to_swap) in mapping:
                # Directly swap the current module
                logger.info(f"Will be directly swapping the module : {name}")
                _swap_module(root_module=module, module_to_swap=module_to_swap,module_name_to_swap=name, q_config=q_config, mapping=mapping)
            else:
                # Recursively  swap modules from the current root
                logger.info(f"Will be recursively swapping the module within : {name}")
                _convert_initialized_float_to_initialized_quantized(module=module_to_swap, q_config=q_config, mapping=mapping)
    return module

def _swap_module(
        root_module: torch.nn.Module,
        module_to_swap: torch.nn.Module,
        module_name_to_swap: str,
        q_config: BASE_QCONFIG_DICT_TYPE,
        mapping: Dict[Callable, Any],

    ) -> None:
    module_names = module_name_to_swap.split(".")
    parent_module_name = module_name_to_swap[: module_name_to_swap.rindex(".")]
    parent_module = root_module.get_submodule(parent_module_name)

    quantized_class = mapping[type(module_to_swap)]
    quant_module = quantized_class.from_float(
        mod=module_to_swap,
        q_config=q_config,
    )
    setattr(parent_module, module_names[-1], quant_module)


def _convert_initialized_float_to_initialized_quantized(
        module: torch.nn.Module,
        q_config: BASE_QCONFIG_DICT_TYPE,
        mapping: Dict[Callable, Any],
        prefixes: Optional[List[str]] = None,
        modules_to_not_convert: Optional[List[str]] = None,
    ) -> torch.nn.Module:
    """
    A function to convert an non-quantized model with default values to its corresponding quantized model with its default values.

    NOTE: This is not a function to transfer the actual weights, and this will be done subsequently
    """
    reassign = {}
    if modules_to_not_convert is None:
        modules_to_not_convert = []

    for name, mod in module.named_children():
        if prefixes is None:
            prefixes = []
        prefixes.append(name)

        if type(mod) not in mapping:
            _convert_initialized_float_to_initialized_quantized(
                module=mod,
                q_config=q_config,
                mapping=mapping,
                prefixes=prefixes,
                modules_to_not_convert=modules_to_not_convert
            )
        if type(mod) in mapping and name not in modules_to_not_convert:
            if not any(key in ".".join(prefixes) for key in modules_to_not_convert):
                quantized_class = mapping[type(mod)]
                reassign[name] = quantized_class.from_float(
                    mod=mod,
                    q_config=q_config,
                )
        # Currently there is a bug in quantize.convert function where even though
        # Parallel embedding has set for tensor_model_parallel attribute, it does not show
        # up in the converted module. Detailed investigation to be performed here: NAPP-2474
        if type(mod).__name__ == "ParallelEmbedding":
            setattr(mod.weight, "tensor_model_parallel", True)
            setattr(mod.weight, "partition_dim", mod.stride)
            setattr(mod.weight, "partition_stride", mod.weight_partition_dim)
            setattr(mod.weight, "num_partitions", mod.tensor_model_parallel_size)

        # Remove the last key for recursion
        prefixes.pop(-1)

    for key, value in reassign.items():
        module._modules[key] = value

    return module

def direct_cast_quantize(tensor: torch.Tensor, downcast_dtype: torch.dtype) -> torch.Tensor:
    """
    A utility function to quantize a tensor from higher dtype to downcast dtype without any scaling factor

    Args:
        tensor (torch.Tensor): tensor to be dequantized
        downcast_dtype (torch.dtype): downcast dtype

    Returns:
        torch.Tensor: downcasted tensor
    """
    downcast_tensor = tensor.to(downcast_dtype)
    return downcast_tensor
