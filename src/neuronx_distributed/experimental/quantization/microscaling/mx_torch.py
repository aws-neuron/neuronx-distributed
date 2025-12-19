# TODO[release]: move from experimental to testing or utils folder, as this code is not intended to be a prod API

"""
Torch implementation of microscaling (MX) quantization and MX matrix multiplication.
"""

import torch

from neuronx_distributed.quantization.microscaling.transform_weights import get_mxfp4_tensor_from_uint16, get_mxfp8_tensor_from_uint32


# MXFP4 and MXFP8 are supported packed dtypes
VALID_MX_TYPES = (torch.uint16, torch.uint32)
# input to QMX must be fp16 or bf16
VALID_QMX_INPUT_TYPE = (torch.bfloat16, torch.float16)
# only MXFP8 online quantization is supported
VALID_QMX_OUTPUT_TYPE = torch.uint32


def _get_ieee_frexp(tensor):
    """
    Extract IEEE 754 exponent from fp32 tensor.
    Equivalent to np.frexp()[1]
    """
    # Convert to int32 to extract bits
    int_view = tensor.to(torch.float32).view(torch.int32)

    # Extract exponent bits (bits 30:23) and subtract bias
    exp_bits = (int_view >> 23) & 0xFF

    # Handle special cases (zero, denormal)
    is_zero = (tensor == 0.0)
    exp = exp_bits - 127  # Remove IEEE bias

    zero_exp = torch.full_like(exp, -126)
    exp = torch.where(is_zero, zero_exp, exp)

    return exp


def _get_mx_max_exp(dtype, use_unbiased_scale=False):
    # fp8_e4m3fn_x4
    if dtype == torch.uint32:
        # Scales block max to [128, 256), avoiding clamping when quantizing. Avoids bias but does not use full fp8 range.
        if use_unbiased_scale:
            return 7
        # Scales values to [256, 512). Introduces bias if values are scaled to (448, 512).
        else:
            return 8
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _get_mx_fp_max(dtype):
    # fp4_e2m1fn_x4
    if dtype == torch.uint16:
        return 6.0
    # fp8_e4m3fn_x4
    elif dtype == torch.uint32:
        return 448.0
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def quantize_mxfp8(in_tensor, out_x4_dtype=torch.uint32, fp8_dtype=torch.float8_e4m3fn, use_unbiased_scale=False):
    """
    Torch implementation of MX quantization, following OCP specifications:
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.

    Inputs:
        in_tensor: unquantized input tensor, must be bf16 or fp16
        out_x4_dtype=torch.uint32: x4 packed dtype to store result in
        fp8_dtype=torch.float8_e4m3fn: fp8 dtype to use when quantizing
        use_unbiased_scale=False: whether to use scale suggested in OCP spec or use unbiased scale that sacrifices some precision when scaling.
    Returns:
        mx_data_packed: quantized tensor stored x4 packed in target dtype
        mx_scale: uint8 MX scale tensor
    """

    assert in_tensor.dtype in VALID_QMX_INPUT_TYPE, f"Expected input type in {VALID_QMX_INPUT_TYPE}, got {in_tensor.dtype}"
    assert out_x4_dtype == VALID_QMX_OUTPUT_TYPE, f"Expected quantized dtype: {VALID_QMX_OUTPUT_TYPE}, got {out_x4_dtype}"
    assert in_tensor.ndim == 2, f"Expected 2D in_tensor, got {in_tensor.ndim}D"

    max_exp = _get_mx_max_exp(out_x4_dtype, use_unbiased_scale=use_unbiased_scale)
    max_val = _get_mx_fp_max(out_x4_dtype)
    float32_exp_bias = 127

    P, F = in_tensor.shape
    SP, SF = P // 8, F // 4

    exp = _get_ieee_frexp(in_tensor)

    # Scale is based on max val in block of [8, 4]
    exp_reshaped = exp.view(SP, 8, SF, 4)  # [SP, 8, SF, 4]
    max_exp_per_block = torch.amax(exp_reshaped, dim=(1, 3))  # [SP, SF]
    mx_scale = (max_exp_per_block + float32_exp_bias - max_exp).to(torch.uint8)

    # Compute scale tensor for division
    scale_int32 = mx_scale.to(torch.int32) - float32_exp_bias
    scale_blocks = torch.pow(2.0, scale_int32.float())  # [SP, SF]

    # Expand scale to match input tensor shape
    scale = scale_blocks.unsqueeze(1).unsqueeze(3).expand(SP, 8, SF, 4).contiguous().view(P, F)

    # Quantize: divide input by scale
    mx_data = in_tensor / scale

    # Clamp to [-max_val, max_val], then downcast
    mx_data = torch.clamp(mx_data, -max_val, max_val).to(fp8_dtype)

    # Ensure that tensor is contiguous; view contiguous values as fp8_x4/int32
    mx_data_packed = mx_data.contiguous().view(torch.uint32)

    return mx_data_packed, mx_scale


def dequantize_mx_tensor(tensor, scale, dtype, input_is_transposed=False, output_is_transposed=False, output_quad_row=False):
    """
    Descales MX tensor target dtype and shape.

    Inputs:
        tensor: fp4_x4/uint16 or fp8_x4/uint32 packed quantized values
        scale: uint8 MX scale
        dtype: target descaled dtype
        input_is_transposed=False: flag that indicates whether final 2 dims of tensor are transposed
        output_is_transposed=False: flat that indicates whether tensor should be returned with final 2 dims transposed
        output_quad_row=False: flag that indicates whether to return quad row format (*shape, M, K, 4)
    Outputs:
        descaled: descaled tensor in target dtype
    """

    # Transpose from [*shape, K, M] to [*shape, M, K] to put block dim on final dim
    if input_is_transposed:
        tensor = torch.transpose(tensor, -2, -1)
        scale = torch.transpose(scale, -2, -1)

    # Unpack block dim
    *shape, M, K = tensor.shape
    blocks = tensor.reshape((*shape, M, K // 8, 8))
    scales = scale

    # fp4_x4
    if tensor.dtype == torch.uint16:
        descaled = get_mxfp4_tensor_from_uint16(blocks, scales, dtype=dtype, output_quad_row=output_quad_row)

    # fp8_x4
    elif tensor.dtype == torch.uint32:
        descaled = get_mxfp8_tensor_from_uint32(blocks, scales, dtype=dtype, output_quad_row=output_quad_row)

    else:
        ValueError(f"Unsupported dtype: {tensor.dtype}")

    # Format output
    assert not (output_quad_row and output_is_transposed)
    if output_quad_row:
        # [*shape, M, K, 4] -> [*shape, K, M, 4]
        return torch.transpose(descaled, -3, -2)
    elif output_is_transposed:
        # [*shape, M, K*4] -> [*shape, K*4, M]
        return torch.transpose(descaled, -2, -1)
    else:
        return descaled


def matmul_mx_single_tile(stationaryT_x4, moving_x4, stationaryT_scale, moving_scale, output_dtype=torch.float32):
    """
    Torch implementation of a single tile of stationary.T @ moving in MX.

    Inputs:
        stationaryT_x4: 2D transposed stationary tensor of shape [K, M]
        moving_x4: 2D moving tensor of shape [K, N]
        stationaryT_scale: 2D scale tensor of shape [K//8, M]
        moving_scale: 2D scale tensor of shape [K//8, N]
        output_dtype=torch.float32: dtype to cast matmul output to
    Outputs:
        output: result of stationary @ moving ([M, K] @ [K, N])
    """

    assert stationaryT_x4.dtype in VALID_MX_TYPES, f"Expected stationaryT_x4 type in: {VALID_MX_TYPES}, got {stationaryT_x4.dtype=}"
    assert moving_x4.dtype in VALID_MX_TYPES, f"Expected moving_x4 type in: {VALID_MX_TYPES}, got {moving_x4.dtype=}"

    assert len(stationaryT_x4.shape) == 2, f"Expected 2D stationaryT_x4 tensor, got {len(stationaryT_x4.shape)}D"
    assert len(moving_x4.shape) == 2, f"Expected 2D moving_x4 tensor, got {len(moving_x4.shape)}D"

    K, I = stationaryT_x4.shape  # noqa: E741
    K_, J = moving_x4.shape

    assert K <= 128
    assert I <= 128
    assert J <= 512
    assert K == K_, f"Expected matching contraction dims, got {K}, {K_}"

    # Descale to fp32, mimicing hardware
    descaled_dtype = torch.float32
    stationary = dequantize_mx_tensor(stationaryT_x4, stationaryT_scale, descaled_dtype, input_is_transposed=True, output_quad_row=True)
    moving = dequantize_mx_tensor(moving_x4, moving_scale, descaled_dtype, input_is_transposed=True, output_quad_row=True)

    # Matmul, accumulates over K, quad dims
    output = torch.einsum("kiq,kjq->ij", stationary, moving).to(output_dtype)
    return output


def matmul_mx(stationaryT_x4, moving_x4, stationaryT_scale, moving_scale, accumulation_dtype=torch.float32, output_dtype=torch.bfloat16):
    """
    Torch implementation of stationary.T @ moving in MX. Uses tiling to simulate hardware behavior.

    Inputs:
        stationaryT_x4: 2D transposed stationary tensor of shape [K, M]
        moving_x4: 2D moving tensor of shape [K, N]
        stationaryT_scale: 2D scale tensor of shape [K//8, M]
        moving_scale: 2D scale tensor of shape [K//8, N]
        accumulation_dtype=torch.float32: dtype to accumulate tile matmul results in
        output_dtype=torch.bfloat16: dtype to cast matmul output to
    Outputs:
        output: result of stationary @ moving ([M, K] @ [K, N])
    """

    K, I = stationaryT_x4.shape  # noqa: E741
    K_, J = moving_x4.shape
    assert K == K_, f"Expected equal contraction dims, recieved {K}, {K_}"

    TILE_K, TILE_I, TILE_J = 128, 128, 512
    output = torch.zeros(I, J, dtype=accumulation_dtype)

    for k_start in range(0, K, TILE_K):
        k_end = min(k_start + TILE_K, K)
        for i_start in range(0, I, TILE_I):
            i_end = min(i_start + TILE_I, I)
            for j_start in range(0, J, TILE_J):
                j_end = min(j_start + TILE_J, J)

                # pad tiles with zeros
                stat_tile = torch.zeros(TILE_K, TILE_I, dtype=stationaryT_x4.dtype)
                mov_tile = torch.zeros(TILE_K, TILE_J, dtype=moving_x4.dtype)

                k_actual, i_actual, j_actual = k_end - k_start, i_end - i_start, j_end - j_start
                stat_tile[:k_actual, :i_actual] = stationaryT_x4[k_start:k_end, i_start:i_end]
                mov_tile[:k_actual, :j_actual] = moving_x4[k_start:k_end, j_start:j_end]

                # pad scales with 127s
                scale_TILE_K = (TILE_K + 7) // 8
                stat_scale_tile = torch.full((scale_TILE_K, TILE_I), 127, dtype=stationaryT_scale.dtype)
                mov_scale_tile = torch.full((scale_TILE_K, TILE_J), 127, dtype=moving_scale.dtype)

                scale_k_start, scale_k_end = k_start // 8, (k_end + 7) // 8
                scale_k_actual = scale_k_end - scale_k_start
                stat_scale_tile[:scale_k_actual, :i_actual] = stationaryT_scale[scale_k_start:scale_k_end, i_start:i_end]
                mov_scale_tile[:scale_k_actual, :j_actual] = moving_scale[scale_k_start:scale_k_end, j_start:j_end]

                tile_result = matmul_mx_single_tile(stat_tile, mov_tile, stat_scale_tile, mov_scale_tile, output_dtype=accumulation_dtype)
                output[i_start:i_end, j_start:j_end] += tile_result[:i_actual, :j_actual]

    return output.to(output_dtype)
