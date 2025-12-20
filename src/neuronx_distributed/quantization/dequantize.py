from typing import List, Tuple
import torch
import math

from neuronx_distributed.quantization.microscaling.transform_weights import split_byte_4bit_tensor, apply_lut_byte_4bit_tensor
from neuronx_distributed.quantization.quantization_config import QuantizationType, QuantizedDtype, ScaleDtype, is_ocp_mx_quantized

DEFAULT_BLOCK_SIZE = 128
OCP_MX_BLOCK_SIZE = 32

def direct_cast_dequantize(tensor: torch.Tensor, upcast_dtype: torch.dtype) -> torch.Tensor:
    """
    A utility function to dequantize a tensor from lower dtype to upcast dtype without any scaling factor

    Args:
        tensor (torch.Tensor): tensor to be dequantized
        upcast_dtype (torch.dtype): upcast dtype

    Returns:
        torch.Tensor: upcasted tensor
    """
    upcast_tensor = tensor.to(upcast_dtype)
    return upcast_tensor

def scale_dequantize(tensor: torch.Tensor, scale: torch.Tensor, upcast_dtype: torch.dtype) -> torch.Tensor:
    """
    A utility function to dequantize a tensor from lower dtype to upcast dtype based on its corresponding scale
    Note: It will not convert back the tensor to its existing dtype

    Args:
        tensor (torch.Tensor): tensor to be dequantized
        scale (torch.Tensor): scale to be used for dequantization

    Returns:
        torch.Tensor: upcasted tensor multiplied with scale
    """
    upcast_tensor = tensor.to(torch.float32)
    upcast_tensor *= scale.unsqueeze(len(scale.shape)-1)
    upcast_tensor = upcast_tensor.to(upcast_dtype)
    return upcast_tensor


def get_broadcastable_shapes_for_blockwise_scale_dequantize(tensor_shape, scale_shape):
    tensor_shape = list(tensor_shape)
    scale_shape = list(scale_shape)

    scale_ndim = len(scale_shape)
    tensor_ndim = len(tensor_shape)
    
    assert scale_ndim <= tensor_ndim, f"Cannot have more scale dimensions than tensor dimensions: tensor shape {tensor_shape}, scale_shape {scale_shape}"
    block_axis = []
    for d in range(tensor_ndim):
        if d < scale_ndim:
            assert scale_shape[d] <= tensor_shape[d], "Scale dimensions cannot be larger than tensor dimensions"
            if scale_shape[d] != tensor_shape[d]:
                assert tensor_shape[d] % scale_shape[d] == 0, "Tensor dimensions must be divisible by scale dimensions for a blocked dimension"
                block_axis.append(d)
        else:
            # right pad 1s to scale shape
            scale_shape.append(1)
            block_axis.append(d)

    assert len(block_axis) > 0, "Scale dimension sizes cannot be the same as tensor dimension sizes: no block dimensions found"

    # add dimensions of size block_size[i] to tensor_shape that are tiled from the axis at block_axis[i]
    # add dimensions of size 1 to scale_shape broadcast with the block_size[i] dimension in tensor_shape
    for ax in reversed(block_axis):
        # size of the block along this axis
        size = tensor_shape[ax] // scale_shape[ax]
        if size == tensor_shape[ax]:
            continue
        tensor_shape = tensor_shape[:ax] + [tensor_shape[ax] // size, size] + tensor_shape[ax+1:]
        scale_shape = scale_shape[:ax] + [scale_shape[ax], 1] + scale_shape[ax+1:]

    return tuple(tensor_shape), tuple(scale_shape)



def blockwise_scale_dequantize(tensor:torch.Tensor, scale:torch.Tensor, upcast_dtype: torch.dtype, mx_swizzle: bool = False) -> torch.Tensor:
    """
    Perform block-wise multiplication between weight and scale tensors.

    Args:
        tensor (torch.Tensor): Weight tensor
        scale (torch.Tensor): Scale tensor
        upcast_dtype (torch.dtype): upcast dtype

    Returns:
        torch.Tensor: Resulting tensor after block-wise dequantization.
    """
    if is_ocp_mx_quantized(QuantizationType.BLOCKWISE_SYMMETRIC, tensor.dtype, scale.dtype):
        assert tensor.device == torch.device('cpu'), f"FP4 dequantization in torch is only supported on CPU, attempted to run on {tensor.device=}"    
    if mx_swizzle:
        raise ValueError("Transformed weight layout for MXFP4 is not supported by torch blockwise dequantization flow")

    if tensor.dtype == QuantizedDtype.F4E2M1FN_X4.value:
        assert tensor.device == torch.device('cpu'), f"FP4 dequantization in torch is only supported on CPU, attempted to run on {tensor.device=}"
        # view as bytes
        tensor = tensor.contiguous().view(torch.uint8)
        # adjust shape for packing factor of 2 for FP4
        split_shape = [*tensor.shape[:-1], tensor.shape[-1] * 2]
        # split each 4 bit element into its own byte
        tensor = split_byte_4bit_tensor(tensor)
        # use the fp4 lookup table to convert the 4 bit elements to float values
        tensor = apply_lut_byte_4bit_tensor(tensor, dtype=torch.float32)
        # preserve original shape, adjusted for packing factor
        tensor = tensor.view(split_shape)

    if scale.dtype == ScaleDtype.F8E8M0.value:
        # apply uint8 to e8m0 conversion
        scale = (2.0 ** (scale.view(torch.uint8).to(torch.int32) - 127)).to(torch.float32)

    # tensor_bc_shape and scale_bc_shape are broadcastable to each other such that
    # the scale dim is 1 where the blocks are, in order to scale those elements
    tensor_bc_shape, scale_bc_shape = get_broadcastable_shapes_for_blockwise_scale_dequantize(tensor.shape, scale.shape)

    orig_tensor_shape = tensor.shape
    tensor = tensor.reshape(tensor_bc_shape)
    scale = scale.reshape(scale_bc_shape)

    result = tensor.to(torch.float32) * scale
    return result.reshape(orig_tensor_shape).to(upcast_dtype)