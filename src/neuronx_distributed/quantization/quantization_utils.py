from typing import Any, Dict
import torch
import torch.ao.nn.quantized.dynamic as nnqd
import torch.nn as nn
from torch.ao.nn.quantized.dynamic.modules.linear import _quantize_weight
from torch.ao.quantization.qconfig import QConfig, default_dynamic_qconfig
from torch.quantization import MinMaxObserver, default_observer

from neuronx_distributed.quantization.observer import PerChannelAbsMaxObserver
from neuronx_distributed.quantization.quantization_config import DtypeBound


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


def extract_q_scale(q_tensor: torch.Tensor) -> torch.Tensor:
    if q_tensor.qscheme() == torch.per_tensor_affine:
        return extract_q_scale_per_tensor(q_tensor)
    elif q_tensor.qscheme() == torch.per_channel_affine:
        return extract_q_scale_per_channel(q_tensor)
    else:
        raise ValueError(f"qscheme: {q_tensor.qscheme()} is not supported")


def convert_qint8_to_int8_state_dict(state_dict: Dict[str, Any]) -> None:
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

#### TODO: Deprecated after testing with pytorch 2.3+ and adapt to https://github.com/huggingface/optimum-quanto

def quantize_fp8_per_channel(tensor: torch.Tensor, dtype: torch.dtype, channel_axis: int):
    assert dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    fp8_max, fp8_min = DtypeBound.from_torch_dtype(dtype)
    dim = tuple(d for d in range(len(tensor.shape)) if d != channel_axis)
    max_values = torch.amax(torch.abs(tensor), dim=dim, keepdim=True)
    scales = max_values / fp8_max
    quantized_weights = tensor / scales
    quantized_weights = torch.clamp(quantized_weights, fp8_min, fp8_max)
    quantized_weights = quantized_weights.to(dtype)

    scale_shape = [1] * len(quantized_weights.shape)
    scale_shape[channel_axis] = quantized_weights.shape[channel_axis]
    return quantized_weights, scales.to(torch.float32).view(scale_shape)

def quantize_fp8_per_tensor(tensor, dtype):
    assert dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    fp8_max, fp8_min = DtypeBound.from_torch_dtype(dtype)
    max_value = torch.max(torch.abs(tensor))
    scale = max_value / fp8_max
    quantized_weights = tensor / scale
    quantized_weights = torch.clamp(quantized_weights, fp8_min, fp8_max)
    quantized_weights = quantized_weights.to(dtype)
    return quantized_weights, scale

class QuantizedLinear(torch.nn.Module):
    """
    Creating a layer to quantize the weight of torch.nn.Linear layer
    """

    def __init__(self) -> None:
        super().__init__()
    
    def set_weight_and_scale(self, weight, scale):
        self.weight = torch.nn.Parameter(weight)
        self.scale = torch.nn.Parameter(scale)

    def forward(self, input: torch.Tensor):
        raise NotImplementedError()

    @classmethod
    def from_float(cls, mod):
        q_layer = QuantizedLinear()
        qconfig = mod.qconfig
        if qconfig.weight['qscheme'] == torch.per_tensor_symmetric:
            weight, scale = quantize_fp8_per_tensor(mod.weight, qconfig.weight["dtype"])
        elif qconfig.weight['qscheme'] == torch.per_channel_symmetric:
            weight, scale = quantize_fp8_per_channel(mod.weight, qconfig.weight["dtype"], qconfig.weight["ch_axis"])
        else:
            raise RuntimeError()
        q_layer.set_weight_and_scale(weight, scale)
        return q_layer

"""
Why do we have separate code flow for fp8 and int8?
Pytorch has extensive support for int8/uint8 quantization. But it does not have support for fp8 quantization.
Two ways to solve the issue
1. Upstream fp8 support to PyTorch
2. Adapt to https://github.com/huggingface/optimum-quanto
"""

def quantize_pytorch_model_per_tensor_symmetric(float_model: torch.nn.Module, inplace: bool = False, dtype=torch.qint8) -> torch.nn.Module:
    if dtype == torch.qint8 or dtype == torch.int8:
        # Both int8 and quint8 have the same quantization for pytorch which is torch.qint8
        qconfig_spec = {torch.nn.Linear: default_dynamic_qconfig}
        mapping = {nn.Linear: nnqd.Linear}
    elif dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        # This is not the best practice to monkey patch weight field in Qconfig. Again adapt to optimum-quanto
        qconfig_spec = {torch.nn.Linear: QConfig(
            activation=None,
            weight=dict(qscheme=torch.per_tensor_symmetric, dtype=dtype))}
        mapping = {nn.Linear: QuantizedLinear}
    else:
        raise ValueError(f"dtype: {dtype} is not supported to quantize model on CPU")

    quant_model = torch.quantization.quantize_dynamic(
        float_model, qconfig_spec=qconfig_spec, mapping=mapping, dtype=torch.qint8, inplace=inplace
    )
    return quant_model


def quantize_pytorch_model_per_channel_symmetric(float_model: torch.nn.Module, inplace: bool = False, dtype=torch.qint8) -> torch.nn.Module:
    if dtype == torch.qint8 or dtype == torch.int8:
        q_config = QConfig(
            activation=default_observer,
            weight=PerChannelAbsMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0),
        )
        qconfig_spec = {nn.Linear: q_config}
        mapping = {nn.Linear: nnqd.Linear}
    elif dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        # This is not the best practice to monkey patch weight field in Qconfig. Again adapt to optimum-quanto
        qconfig_spec = {torch.nn.Linear: QConfig(activation=None, weight=dict(qscheme=torch.per_channel_symmetric, dtype=dtype, ch_axis=0))}
        mapping = {nn.Linear: QuantizedLinear}
    else:
        raise ValueError(f"dtype: {dtype} is not supported to quantize model on CPU")

    quant_model = torch.quantization.quantize_dynamic(
        float_model, qconfig_spec=qconfig_spec, mapping=mapping, dtype=torch.qint8, inplace=inplace
    )
    return quant_model


def quantize_per_tensor_symmetric(tensor: torch.Tensor):
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
