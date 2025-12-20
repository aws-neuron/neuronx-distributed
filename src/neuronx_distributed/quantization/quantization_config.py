### Create Enum to define the type of quantization possible
import os
import enum
from enum import Enum
from typing import Tuple, TypedDict, Optional, List, cast

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

        F8E4M3FN_MAX = torch.finfo(torch.float8_e4m3fn).max
        F8E4M3FN_MIN = torch.finfo(torch.float8_e4m3fn).min

        F8E5M2_MAX = torch.finfo(torch.float8_e5m2).max
        F8E5M2_MIN = torch.finfo(torch.float8_e5m2).min

        F4E2M1FN_X2_MAX = +6.0
        F4E2M1FN_X2_MIN = -6.0

        F8E8M0_MAX = 2.0 ** (255.0 - 127.0)
        F8E8M0_MIN = 2.0 ** (-127.0)

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
    BLOCKWISE_SYMMETRIC = "blockwise_symmetric"
    EXPERT_WISE_PER_CHANNEL_SYMMETRIC = "expert_wise_per_channel_symmetric"

class ActivationQuantizationType(Enum, metaclass=MyEnumMeta):
    DYNAMIC = "dynamic"
    NONE = None


def get_float4x4_torch_dtype():
    """
    A limitation of Torch XLA is that u16 is not supported. This dtype works for the full model,
    but not for the module level integration tests, so the environment variable `NEURON_FLOAT4X4_IS_FLOAT16`
    can be used to make the F4E2M1FN_X4 quantized dtype enum point to float16 instead.
    """
    if int(os.environ.get("NEURON_FLOAT4X4_IS_FLOAT16", "0")) > 0:
        return torch.float16
    else:
        return torch.uint16


class QuantizedDtype(Enum, metaclass=MyEnumMeta):
    INT8 = torch.int8
    if torch.__version__ >= '2.1':
        F8E4M3 = torch.float8_e4m3fn
        F8E4M3FN = torch.float8_e4m3fn
        F8E5M2 = torch.float8_e5m2

    F4E2M1FN_X4 = get_float4x4_torch_dtype()
    F8E4M3FN_X4 = torch.uint32
    F8E5M2_X4 = torch.uint32

    @classmethod
    def has_dtype(cls, dtype_string: str) -> None:
        """Check if the dtype string is a valid QuantizedDtype."""
        assert dtype_string.upper() in cls.__members__, f"{dtype_string} is not a valid QuantizedDtype."

    @classmethod
    def get_dtype(cls, dtype_string: str) -> torch.dtype:
        """Return the value for the given dtype string (in uppercase) if it exists."""
        QuantizedDtype.has_dtype(dtype_string=dtype_string)
        return cls[dtype_string.upper()].value

    def is_float(self) -> bool:
        return self != QuantizedDtype.INT8
    
    def get_packed_count(self) -> int:
        if self in [QuantizedDtype.F4E2M1FN_X4, QuantizedDtype.F8E4M3FN_X4, QuantizedDtype.F8E5M2_X4]:
            return 4
        else:
            return 1


class ScaleDtype(Enum, metaclass=MyEnumMeta):
    F32 = torch.float32
    F8E8M0 = torch.uint8
    
    def get_default_scale(self): 
        if self == ScaleDtype.F8E8M0:
            return 127
        else:
            return 1.0

    @classmethod
    def has_dtype(cls, dtype_string: str) -> None:
        """Check if the dtype string is a valid ScaleDtype."""
        assert dtype_string.upper() in cls.__members__, f"{dtype_string} is not a valid ScaleDtype."

    @classmethod
    def get_dtype(cls, dtype_string: str) -> torch.dtype:
        """Return the value for the given dtype string (in uppercase) if it exists."""
        ScaleDtype.has_dtype(dtype_string=dtype_string)
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


class BLOCKWISE_QCONFIG_DICT_TYPE(BASE_QCONFIG_DICT_TYPE):
    block_axis: Optional[List[int]]
    block_size: Optional[List[int]]
    scale_dtype: ScaleDtype


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

_DEFAULT_BLOCKWISE_QCONFIG_DICT: BLOCKWISE_QCONFIG_DICT_TYPE = {
    "quantization_type": QuantizationType.BLOCKWISE_SYMMETRIC,
    "quantized_dtype": QuantizedDtype.F8E4M3,
    "block_axis": None,
    "block_size": None,
    "scale_dtype": ScaleDtype.F32,
    "activation_quantization_type": ActivationQuantizationType.NONE,
    "clamp_bound": float('inf'),
}

def validate_block_axis_size(block_axis: Optional[List[int]], block_size: Optional[List[int]]) -> Tuple[List[int], List[int]]:
    assert block_size is not None and block_axis is not None, "block_axis and block_size must be specified for blockwise quantization"
    assert len(block_size) == len(block_axis), "block_axis and block_size list arguments must have the same length"
    return cast(List[int], block_axis), cast(List[int], block_size)

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


def get_default_blockwise_custom_qconfig_dict() -> BLOCKWISE_QCONFIG_DICT_TYPE:
    """Defines the default blockwise config dict"""
    return BLOCKWISE_QCONFIG_DICT_TYPE(**_DEFAULT_BLOCKWISE_QCONFIG_DICT)


def get_default_expert_wise_per_channel_custom_qconfig_dict() -> EXPERT_WISE_PER_CHANNEL_QCONFIG_DICT_TYPE:
    """
    Defines the default custom expert wise per channel config dict
    Limitations:
        - Cannot run multiple quantization type with expert wise per channel quantization
        - Does not work with lnc=1 kernel 
        - lnc=2 blockwise kernel does not support multi expert per token
    """
    return PER_CHANNEL_QCONFIG_DICT_TYPE(**_DEFAULT_EXPERT_WISE_PER_CHANNEL_QCONFIG_DICT)


def is_ocp_mx_quantized(
    q_type: QuantizationType, 
    q_dtype: QuantizedDtype | torch.dtype, 
    scale_dtype: ScaleDtype | torch.dtype
):
    """
    Conditions for OCP Microscaling (MX) quantization scheme on Neuron
    - Blockwise
    - Blocks are FP4 or FP8 packed x4
    - Scales are F8E8M0
    
    Block size (32) is not validated here
    """
    return q_type == QuantizationType.BLOCKWISE_SYMMETRIC \
        and QuantizedDtype(q_dtype) in [QuantizedDtype.F4E2M1FN_X4, QuantizedDtype.F8E4M3FN_X4, QuantizedDtype.F8E5M2_X4] \
        and ScaleDtype(scale_dtype) == ScaleDtype.F8E8M0