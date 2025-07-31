### Create Enum to define the type of quantization possible
import enum
from enum import Enum
from typing import TypedDict, Optional

import torch


class MyEnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        else:
            return True


class DtypeBound(Enum):
    """Enum representing the bounds for data types."""

    # We need this check to be compatible with pytorch version < 2.1
    if torch.__version__ >= '2.1':
        F8E4M3_MAX = 240.0
        F8E4M3_MIN = -240.0
        F8E5M2_MAX = torch.finfo(torch.float8_e5m2).max
        F8E5M2_MIN = torch.finfo(torch.float8_e5m2).min

    @staticmethod
    def from_torch_dtype(dtype):
        """Map PyTorch data type to the corresponding DtypeBound."""
        if dtype == torch.float8_e4m3fn:
            return (DtypeBound.F8E4M3_MAX.value, DtypeBound.F8E4M3_MIN.value)
        elif dtype == torch.float8_e5m2:
            return (DtypeBound.F8E5M2_MAX.value, DtypeBound.F8E5M2_MIN.value)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

class QuantizationType(Enum, metaclass=MyEnumMeta):
    PER_TENSOR_SYMMETRIC = "per_tensor_symmetric"
    PER_CHANNEL_SYMMETRIC = "per_channel_symmetric"
    EXPERT_WISE_PER_CHANNEL_SYMMETRIC = "expert_wise_per_channel_symmetric"

class ActivationQuantizationType(Enum, metaclass=MyEnumMeta):
    DYNAMIC = "dynamic"
    NONE = None


class QuantizedDtype(Enum, metaclass=MyEnumMeta):
    INT8 = torch.int8
    if torch.__version__ >= '2.1':
        F8E4M3 = torch.float8_e4m3fn
        F8E5M2 = torch.float8_e5m2

    @classmethod
    def has_dtype(cls, dtype_string: str) -> None:
        """Check if the dtype string is a valid QuantizedDtype."""
        assert dtype_string.upper() in cls.__members__, f"{dtype_string} is not a valid QuantizedDtype."

    @classmethod
    def get_dtype(cls, dtype_string: str) -> torch.dtype:
        """Return the value for the given dtype string (in uppercase) if it exists."""
        QuantizedDtype.has_dtype(dtype_string=dtype_string)
        return cls[dtype_string.upper()].value


class BASE_QCONFIG_DICT_TYPE(TypedDict):
    quantization_type: QuantizationType
    quantized_dtype: QuantizedDtype
    activation_quantization_type: ActivationQuantizationType
    clamp_bound: float


class PER_CHANNEL_QCONFIG_DICT_TYPE(BASE_QCONFIG_DICT_TYPE):
    quantization_per_channel_axis: Optional[int]

class EXPERT_WISE_PER_CHANNEL_QCONFIG_DICT_TYPE(BASE_QCONFIG_DICT_TYPE):
    quantization_per_channel_axis: Optional[int]

_DEFAULT_CUSTOM_QCONFIG_DICT: BASE_QCONFIG_DICT_TYPE = {
    "quantization_type": QuantizationType.PER_TENSOR_SYMMETRIC,
    "quantized_dtype": QuantizedDtype.INT8,
    "activation_quantization_type": ActivationQuantizationType.NONE,
    "clamp_bound": float('inf'),
}

_DEFAULT_PER_CHANNEL_QCONFIG_DICT: PER_CHANNEL_QCONFIG_DICT_TYPE = {
    "quantization_type": QuantizationType.PER_CHANNEL_SYMMETRIC,
    "quantized_dtype": QuantizedDtype.INT8,
    # Each quantized layer sets its own the default channel
    "quantization_per_channel_axis": None,
    "activation_quantization_type": ActivationQuantizationType.NONE,
    "clamp_bound": float('inf'),
}

_DEFAULT_EXPERT_WISE_PER_CHANNEL_QCONFIG_DICT: EXPERT_WISE_PER_CHANNEL_QCONFIG_DICT_TYPE = {
    "quantization_type": QuantizationType.EXPERT_WISE_PER_CHANNEL_SYMMETRIC,
    "quantized_dtype": QuantizedDtype.F8E4M3,
    # Each quantized layer sets its own the default channel
    "quantization_per_channel_axis": None,
    "activation_quantization_type": ActivationQuantizationType.NONE,
    "clamp_bound": float('inf'),
}


def get_default_custom_qconfig_dict() -> BASE_QCONFIG_DICT_TYPE:
    r"""Defines the default custom config dict."""
    return BASE_QCONFIG_DICT_TYPE(**_DEFAULT_CUSTOM_QCONFIG_DICT)


def get_default_per_channel_custom_qconfig_dict() -> PER_CHANNEL_QCONFIG_DICT_TYPE:
    """Defines the default custom per channel config dict"""
    return PER_CHANNEL_QCONFIG_DICT_TYPE(**_DEFAULT_PER_CHANNEL_QCONFIG_DICT)

def get_default_expert_wise_per_channel_custom_qconfig_dict() -> EXPERT_WISE_PER_CHANNEL_QCONFIG_DICT_TYPE:
    """
    Defines the default custom expert wise per channel config dict
    Limitations:
        - Cannot run multiple quantization type with expert wise per channel quantization
        - Does not work with lnc=1 kernel 
        - lnc=2 blockwise kernel does not support multi expert per token
    """
    return PER_CHANNEL_QCONFIG_DICT_TYPE(**_DEFAULT_EXPERT_WISE_PER_CHANNEL_QCONFIG_DICT)