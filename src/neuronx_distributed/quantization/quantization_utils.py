import torch
import torch.ao.nn.quantized.dynamic as nnqd
import torch.nn as nn
from torch.ao.nn.quantized.dynamic.modules.linear import _quantize_weight
from torch.ao.quantization.qconfig import QConfig, default_dynamic_qconfig
from torch.quantization import MinMaxObserver, default_observer

from neuronx_distributed.quantization.observer import PerChannelAbsMaxObserver


def extract_q_scale_per_tensor(q_tensor: torch.Tensor) -> torch.Tensor:
    """Extract scales per tensor.

    Args:
        q_tensor (torch.Tensor): Input torch.qint8/torch.quint8

    Returns:
        torch.Tensor: returns the tensor of shape torch.Size([1])
    """
    assert q_tensor.qscheme() == torch.per_tensor_affine
    return torch.tensor([q_tensor.q_scale()])


def extract_q_scale_per_channel(q_tensor: torch.Tensor) -> torch.Tensor:
    """Extract the scale for per channel quantization

    Ideally scales would be a 1D tensor. But we want to multiply the scales with the weight to dequantize.
    So we simply view the scale so that its broadcastable to the weight shape.

    Args:
        q_tensor (torch.Tensor): Input quantized tensor

    Returns:
        torch.Tensor: scale with all the shape broadcastable wrt weight
    """
    assert q_tensor.qscheme() == torch.per_channel_affine
    per_channel_axis = q_tensor.q_per_channel_axis()
    q_tensor_shape = q_tensor.shape
    # The shape here would be [1, 1, ....]
    scale_shape = [1] * len(q_tensor_shape)
    # The shape now would be [1, C, 1, .....], so makes it broadcastble
    scale_shape[per_channel_axis] = q_tensor_shape[per_channel_axis]
    return q_tensor.q_per_channel_scales().to(torch.float32).view(scale_shape)


def extract_q_scale(q_tensor: torch.Tensor):
    if q_tensor.qscheme() == torch.per_tensor_affine:
        return extract_q_scale_per_tensor(q_tensor)
    elif q_tensor.qscheme() == torch.per_channel_affine:
        return extract_q_scale_per_channel(q_tensor)
    else:
        raise (f"qscheme: {q_tensor.qscheme()} is not supported")


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
        state_dict[prefix + "scale"] = extract_q_scale(state_dict[prefix + "_packed_params._packed_params"][0])

        if len(state_dict[prefix + "_packed_params._packed_params"]) == 2:
            # Bias is included
            bias = state_dict[prefix + "_packed_params._packed_params"][1]
            if isinstance(bias, torch.nn.Parameter):
                bias = bias.data
        state_dict[prefix + "bias"] = bias

        state_dict.pop(prefix + "_packed_params._packed_params")
        state_dict.pop(prefix + "_packed_params.dtype")
        state_dict.pop(prefix + "zero_point")


def quantize_pytorch_model_per_tensor_symmetric(float_model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    qconfig_spec = {torch.nn.Linear: default_dynamic_qconfig}
    mapping = {nn.Linear: nnqd.Linear}
    quant_model = torch.quantization.quantize_dynamic(
        float_model, qconfig_spec=qconfig_spec, mapping=mapping, dtype=torch.qint8, inplace=inplace
    )
    return quant_model


def quantize_pytorch_model_per_channel_symmetric(float_model: torch.nn.Module, inplace=False) -> torch.nn.Module:
    q_config = QConfig(
        activation=default_observer,
        weight=PerChannelAbsMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0),
    )

    qconfig_spec = {nn.Linear: q_config}
    mapping = {nn.Linear: nnqd.Linear}

    quant_model = torch.quantization.quantize_dynamic(
        float_model, qconfig_spec=qconfig_spec, mapping=mapping, dtype=torch.qint8, inplace=inplace
    )
    return quant_model


def quantize_per_tensor_symmetric(tensor):
    tensor_observer = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)()
    tensor_observer(tensor)
    q_tensor = _quantize_weight(tensor, tensor_observer)
    return q_tensor


def quantize_per_channel_symmetric(tensor: torch.Tensor, channel_axis: int):
    tensor_observer = PerChannelAbsMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=channel_axis
    )()
    tensor_observer(tensor)
    q_tensor = _quantize_weight(tensor, tensor_observer)
    return q_tensor
