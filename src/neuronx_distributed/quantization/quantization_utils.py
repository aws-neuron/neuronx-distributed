import torch
import torch.ao.nn.quantized.dynamic as nnqd
import torch.nn as nn
from torch.ao.quantization.qconfig import default_dynamic_qconfig


def convert_qint8_to_int8_state_dict(state_dict: dict) -> dict:
    """A utility function to convert a qint8 type state dict to int8 type state dict.
    This conversion is done `inplace`

    NOTE: This is not a good practice to translate the checkpoints, rather the logic should reside
    in the Adaptor: QuantizedParallelLinearLayerStateDictAdaptor (see its usage)

    Args:
        state_dict (dict): Input state dict in qint8 format

    Returns:
        dict: Output state dict in int8 format
    """
    prefixes = set()
    for name in state_dict.keys():
        if "_packed_params.dtype" in name:
            prefixes.add(name.split("_packed_params.dtype")[0])

    for prefix in prefixes:
        state_dict[prefix + "weight"] = torch.int_repr(state_dict[prefix + "_packed_params._packed_params"][0])
        state_dict[prefix + "scale"] = torch.tensor([state_dict[prefix + "_packed_params._packed_params"][0].q_scale()])

        if len(state_dict[prefix + "_packed_params._packed_params"]) == 2:
            # Bias is included
            bias = state_dict[prefix + "_packed_params._packed_params"][1]
            if isinstance(bias, torch.nn.Parameter):
                bias = bias.data
        state_dict[prefix + "bias"] = bias

        state_dict.pop(prefix + "_packed_params._packed_params")
        state_dict.pop(prefix + "_packed_params.dtype")
        state_dict.pop(prefix + "zero_point")


def convert_float_model_to_pytorch_int8_model(float_model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    qconfig_spec = {torch.nn.Linear: default_dynamic_qconfig}
    mapping = {nn.Linear: nnqd.Linear}
    quant_model = torch.quantization.quantize_dynamic(
        float_model, qconfig_spec=qconfig_spec, mapping=mapping, dtype=torch.qint8, inplace=inplace
    )
    return quant_model
