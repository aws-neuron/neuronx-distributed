### Create Enum to define the type of quantization possible
import enum
from enum import Enum
from typing import TypedDict

import torch


class MyEnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        else:
            return True


class QuantizationType(Enum, metaclass=MyEnumMeta):
    PER_TENSOR_SYMMETRIC = "per_tensor_symmetric"
    PER_CHANNEL_SYMMETRIC = "per_channel_symmetric"


class QuantizedDtype(Enum, metaclass=MyEnumMeta):
    INT8 = torch.int8


class BASE_QCONFIG_DICT_TYPE(TypedDict):
    quantization_type: QuantizationType
    quantized_dtype: QuantizedDtype


class PER_CHANNEL_QCONFIG_DICT_TYPE(BASE_QCONFIG_DICT_TYPE):
    quantization_per_channel_axis: int


_DEFAULT_CUSTOM_QCONFIG_DICT: BASE_QCONFIG_DICT_TYPE = {
    "quantization_type": QuantizationType.PER_TENSOR_SYMMETRIC,
    "quantized_dtype": QuantizedDtype.INT8,
}

_DEFAULT_PER_CHANNEL_QCONFIG_DICT: PER_CHANNEL_QCONFIG_DICT_TYPE = {
    "quantization_type": QuantizationType.PER_CHANNEL_SYMMETRIC,
    "quantized_dtype": QuantizedDtype.INT8,
    "quantization_per_channel_axis": 0,
}


def get_default_custom_qconfig_dict() -> BASE_QCONFIG_DICT_TYPE:
    r"""Defines the default custom config dict."""
    return dict(_DEFAULT_CUSTOM_QCONFIG_DICT)


def get_default_per_channel_custom_qconfig_dict() -> PER_CHANNEL_QCONFIG_DICT_TYPE:
    """Defines the default custom per channel config dict"""
    return dict(_DEFAULT_PER_CHANNEL_QCONFIG_DICT)
