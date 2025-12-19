# Copyright (c) 2025, OpenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Handles weight packing/unpacking, undo interleaving, reshaping.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from neuronx_distributed.quantization.quantization_config import QuantizedDtype


FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def get_mxfp4_tensor(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 16384 * 512,
) -> torch.Tensor:
    """
    Method to dequantize MXFP4 tensor packed as fp4_x2 into uint8. Lightly edited version of OpenAI implementation:
    https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py#L68

    Inputs:
        blocks: quantized fp4 tensor packed into uint8 as fp4_x2, of shape [*shape, G, 16] -> 32 fp4 values per block
        scales: uint8 MX scales, of shape [*shape, G]
        dtype=torch.bfloat16: dtype to dequantize to
        rows_per_chunk=16384 * 512: chunk size to use when processing quantized tensor
    Outputs:
        out: dequantized tensor in target dtype, of shape [*shape, G * 32]
    """

    scales = scales.to(torch.int32) - 127

    assert blocks.shape[:-1] == scales.shape, (
        f"{blocks.shape=} does not match {scales.shape=}"
    )

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

def split_byte_4bit_tensor(
    tensor: torch.Tensor,
    *,
    rows_per_chunk: int = 16384 * 512
):
    *prefix_shape, packed_dim = tensor.shape
    rows_total = math.prod(prefix_shape)
    tensor = tensor.reshape(rows_total, packed_dim).view(torch.uint8)

    out = torch.empty(rows_total, packed_dim * 2, dtype=torch.uint8, device=tensor.device)
    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = tensor[r0:r1]
        lo = blk & 0x0F
        hi = blk >> 4
        out[r0:r1, 0::2] = lo
        out[r0:r1, 1::2] = hi
        del lo, hi, blk

    return out.reshape(*prefix_shape, packed_dim * 2)


def dequant_byte_4bit_tensor(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    rows_per_chunk: int = 16384 * 512
):
    lut = torch.tensor(FP4_VALUES, dtype=torch.bfloat16, device=blocks.device)
    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)
    scales = scales.to(torch.int32) - 127

    out = torch.empty(rows_total, B, dtype=torch.bfloat16, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = blocks[r0:r1]
        exp = scales[r0:r1]
        sub = out[r0:r1]
        sub[:, :] = lut[blk]
        torch.ldexp(sub, exp, out=sub)
        del blk, exp

    return out.reshape(*prefix_shape, G, B).view(*prefix_shape, G * B)


def apply_lut_byte_4bit_tensor(
    blocks: torch.Tensor,
    *,
    rows_per_chunk: int = 16384 * 512,
    dtype=torch.float32
):
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    out = torch.empty(rows_total, B, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = blocks[r0:r1]
        blk = blk.to(torch.long)
        sub = out[r0:r1]
        sub[:, :] = lut[blk]
        del blk, sub

    return out.reshape(*prefix_shape, G, B).view(*prefix_shape, G * B)


def pack_byte_4bit_tensor(
    tensor: torch.Tensor,
    *,
    rows_per_chunk: int = 16384 * 512
):
    *prefix_shape, split_dim = tensor.shape
    assert tensor.dtype == torch.uint8, f"{tensor.dtype=} must be uint8"
    assert split_dim % 2 == 0, f"{split_dim=} must be even"
    assert split_dim >= 2, f"{split_dim=} must be at least 2"
    rows_total = math.prod(prefix_shape)
    tensor = tensor.reshape(rows_total, split_dim)

    out = torch.empty(rows_total, split_dim // 2, dtype=torch.uint8, device=tensor.device)
    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = tensor[r0:r1, :]
        lo = blk[:, 0::2]
        hi = blk[:, 1::2] << 4
        
        out[r0:r1, :] = hi | lo
    
    return out.reshape(*prefix_shape, split_dim // 2)


def get_mxfp4_tensor_from_uint16(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 16384 * 512,
    output_quad_row: bool = False,
) -> torch.Tensor:
    """
    Method to dequantize MXFP4 tensor packed as fp4_x4 into uint16.

    Inputs:
        blocks: quantized fp4 tensor packed as fp4_x4 into uint16, of shape [*shape, G, 8] -> 32 fp4 values per block
        scales: uint8 MX scales, of shape [*shape, G]
        dtype=torch.bfloat16: dtype to dequantize to
        rows_per_chunk=16384 * 512: chunk size to use when processing quantized tensor
        output_quad_row=False: flag to indicates whether to return quad row format
    Outputs:
        out: dequantized tensor in target dtype, of shape [*shape, G * 32] OR [*shape, G * 8, 4]
    """

    scales = scales.to(torch.int32) - 127

    assert blocks.shape[:-1] == scales.shape, (
        f"{blocks.shape=} does not match {scales.shape=}"
    )

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape  # B = 8x uint16 values
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.to(torch.int32).reshape(rows_total, B)  # Convert to int32 for bit ops
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 4, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_0 = ((blk >> 0) & 0x0F).to(torch.long)   # bits 3-0
        idx_1 = ((blk >> 4) & 0x0F).to(torch.long)   # bits 7-4
        idx_2 = ((blk >> 8) & 0x0F).to(torch.long)   # bits 11-8
        idx_3 = ((blk >> 12) & 0x0F).to(torch.long)  # bits 15-12

        sub = out[r0:r1]

        sub[:, 0::4] = lut[idx_0]  # positions 0, 4, 8, 12, 16, 20, 24, 28
        sub[:, 1::4] = lut[idx_1]  # positions 1, 5, 9, 13, 17, 21, 25, 29
        sub[:, 2::4] = lut[idx_2]  # positions 2, 6, 10, 14, 18, 22, 26, 30
        sub[:, 3::4] = lut[idx_3]  # positions 3, 7, 11, 15, 19, 23, 27, 31

        torch.ldexp(sub, exp, out=sub)

        del idx_0, idx_1, idx_2, idx_3, blk, exp

    if output_quad_row:
        return out.reshape(*prefix_shape, G, B, 4).view(*prefix_shape, G * B, 4)
    else:
        return out.reshape(*prefix_shape, G, B * 4).view(*prefix_shape, G * B * 4)


def get_mxfp8_tensor_from_uint32(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    rows_per_chunk: int = 16384 * 512,
    output_quad_row: bool = False,
    replace_nan_with_zeros: bool = False,
) -> torch.Tensor:
    """
    Method to dequantize MXFP8 tensor packed as fp8_x4 into uint32.

    Inputs:
        blocks: quantized fp4 tensor packed as fp8_x4 into uint32, of shape [*shape, G, 8] -> 32 fp4 values per block
        scales: uint8 MX scales, of shape [*shape, G]
        dtype=torch.bfloat16: dtype to dequantize to
        fp8_dtype=torch.float8_e4m3fn: fp8 dtype to view packed values as
        rows_per_chunk=16384 * 512: chunk size to use when processing quantized tensor
        output_quad_row=False: flag to indicates whether to return quad row format
        replace_nan_with_zeros=False: flag to indicate whether to replace nan values with 0s
    Outputs:
        out: dequantized tensor in target dtype, of shape [*shape, G * 32] OR [*shape, G * 8, 4)
    """

    scales = scales.to(torch.int32) - 127

    assert blocks.shape[:-1] == scales.shape, (
        f"{blocks.shape=} does not match {scales.shape=}"
    )

    *prefix_shape, G, B = blocks.shape  # B = 8x uint32 values
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.to(torch.int32).reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 4, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]  # Shape: [chunk_size, 8]
        exp = scales[r0:r1]

        # Extract 8-bit values and convert to uint8 first, then view as fp8
        idx_0 = ((blk >> 0) & 0xFF).to(torch.uint8).view(fp8_dtype).to(dtype)
        idx_1 = ((blk >> 8) & 0xFF).to(torch.uint8).view(fp8_dtype).to(dtype)
        idx_2 = ((blk >> 16) & 0xFF).to(torch.uint8).view(fp8_dtype).to(dtype)
        idx_3 = ((blk >> 24) & 0xFF).to(torch.uint8).view(fp8_dtype).to(dtype)

        sub = out[r0:r1]  # Shape: [chunk_size, 32]

        # Each idx_X should have shape [chunk_size, 8] after conversion
        sub[:, 0::4] = idx_0  # positions 0, 4, 8, 12, 16, 20, 24, 28
        sub[:, 1::4] = idx_1  # positions 1, 5, 9, 13, 17, 21, 25, 29
        sub[:, 2::4] = idx_2  # positions 2, 6, 10, 14, 18, 22, 26, 30
        sub[:, 3::4] = idx_3  # positions 3, 7, 11, 15, 19, 23, 27, 31

        torch.ldexp(sub, exp, out=sub)

        del idx_0, idx_1, idx_2, idx_3, blk, exp

    if replace_nan_with_zeros:
        out = torch.nan_to_num(out, nan=0.0)

    if output_quad_row:
        return out.reshape(*prefix_shape, G, B, 4).view(*prefix_shape, G * B, 4)
    else:
        return out.reshape(*prefix_shape, G, B * 4).view(*prefix_shape, G * B * 4)


def pack_fp4_x4_uint16(X):
    """
    Assumes:
    - X.dtype == uint8
    - uint8 represents fp4_x2
    - dim to pack is final dim

    Note: this layout is pre-swizzled if we unpack and place all four packed values along another dim.
    - Example: if we have [128, 16] and pack this to [128, 8], we can unpack this to [128, 8, 4] and tranpose / reshape to [512, 8] which is swizzled.
    """

    if isinstance(X, torch.Tensor):
        assert X.dtype == torch.uint8, f"expected uint8, got {X.dtype}"
        repacked = X.view(QuantizedDtype.F4E2M1FN_X4.value)

    elif isinstance(X, np.ndarray):
        assert X.dtype == np.uint8
        repacked = X.view(np.uint16)

    else:
        raise ValueError("Unsupported input dtype!")

    return repacked


def split_gate_up(W_gate_up, scale_gate_up, bias_gate_up):
    """
    Gate and up proj weights, scales, biases are interleaved in 2I dim.

    W.shape = [E, 2I, H/32, 16] (16 fp4_x2/uint8 values = block of 32 fp4 values)
    scale.shape = [E, 2I, H/32]
    bias.shape = [E, 2I]
    """

    W_gate, W_up = W_gate_up[:, ::2, :, :], W_gate_up[:, 1::2, :, :]
    scale_gate, scale_up = scale_gate_up[:, ::2, :], scale_gate_up[:, 1::2, :]
    bias_gate, bias_up = bias_gate_up[:, ::2], bias_gate_up[:, 1::2]

    # Assume all inputs are either torch.Tensor or np.ndarray
    if isinstance(W_gate_up, torch.Tensor):
        return W_gate.contiguous(), scale_gate.contiguous(), bias_gate.contiguous(), W_up.contiguous(), scale_up.contiguous(), bias_up.contiguous()
    else:
        return W_gate, scale_gate, bias_gate, W_up, scale_up, bias_up


def _pad_tensor(X, pad_to, pad_value=0):
    """
    Pads dim i to pad_to[i] using constant fill value. Uses RHS padding only.
    """

    if isinstance(X, torch.Tensor):
        padding = []
        for i in reversed(range(len(X.shape))):
            pad_left = 0
            pad_right = pad_to[i] - X.shape[i]
            padding.extend([pad_left, pad_right])

        padding = tuple(padding)

        return F.pad(X, padding, "constant", pad_value)

    elif isinstance(X, np.ndarray):
        padding = []
        for i in range(len(X.shape)):
            pad_left = 0
            pad_right = max(0, pad_to[i] - X.shape[i])
            padding.append((pad_left, pad_right))

        return np.pad(X, padding, mode='constant', constant_values=pad_value)

    else:
        raise ValueError("Invalid input type!")


def reshape_pad_proj(W, scale, bias):
    """
    W.shape = [E, <I or H>, <I or H>/32, 8]
    scale.shape = [E, <I or H>, <I or H>/32]
    bias.shape = [E, <I or H>]
    """

    # gate/up: [E, H, I/32, 8] -> [E, H, I/4]
    # down: [E, I, H/32, 8] -> [E, I, H/4]
    W_reshaped = W.reshape((128, 2880, 2880 // 4))

    # Pad I, H to 3072, use E8M0 bias for scale padding
    E8M0_BIAS = 127  # pad
    W_padded = _pad_tensor(W_reshaped, (128, 3072, 3072 // 4))
    scale_padded = _pad_tensor(scale, (128, 3072, 3072 // 32), E8M0_BIAS)
    bias_padded = _pad_tensor(bias, (128, 3072))

    return W_padded, scale_padded, bias_padded
