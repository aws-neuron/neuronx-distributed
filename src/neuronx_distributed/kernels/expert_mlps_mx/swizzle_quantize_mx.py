"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Layout and microscaling (MX) quantization utilities token-generation with microscaling format (MX) weights.
"""

import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki._private.private_api import quantize_mx, float8_e4m3fn_x4

from neuronx_distributed.kernels.expert_mlps_mx.constants import SUPPORTED_QMX_OUTPUT_DTYPES, MX_SCALE_DTYPE


def swizzle_quantize_mx_input(input_sb):
    """
    Function to transform input tensor into "swizzled" layout and perform quantization to MXFP8.

    Input shape:
        input_sb: [T_32 * 4_H, T/32, H/512, 16_H * 8_H]

    Output shapes:
        output_quant_sb: [16_H * 8_H, H/512, T], 4_H is packed in x4 dtype
        output_scale_sb: [16_H * 8_H, H/512, T], scales located in leading 4P of each SBUF quadrant
    """

    # TODO: add output_x4_dtype as optional arg to API once NKI supports simulating w/ x4 dtypes as kernel args / if we want to use float8_e5m2_x4
    output_x4_dtype = float8_e4m3fn_x4
    assert output_x4_dtype in SUPPORTED_QMX_OUTPUT_DTYPES, f"Got {output_x4_dtype=}, expected output_x4_dtype in {SUPPORTED_QMX_OUTPUT_DTYPES}"

    # Perform T/32 * H/512 transposes to achieve swizzled layout
    T32_H4, n_T32_tiles, n_H512_tiles, H128 = input_sb.shape
    SWIZZLE_SHAPE = (H128, n_H512_tiles, n_T32_tiles, T32_H4)
    input_swizzled_sb = nl.ndarray(SWIZZLE_SHAPE, dtype=input_sb.dtype, buffer=nl.sbuf)
    for tile_T32 in nl.affine_range(n_T32_tiles):
        for tile_H512 in nl.affine_range(n_H512_tiles):
            input_swizzled_sb[:, tile_H512, tile_T32, :] = nisa.tensor_copy(nisa.nc_transpose((input_sb[:, tile_T32, tile_H512, :])))

    # View swizzled shape as [128_H * 8_H, H/512, T * H_4]
    T_H4 = n_T32_tiles * T32_H4
    T = T_H4 // 4
    input_swizzled_sb = input_swizzled_sb.reshape((H128, n_H512_tiles, T_H4))

    # Quantize to MXFP8
    OUTPUT_QMX_SHAPE = (H128, n_H512_tiles, T)
    output_quant_sb = nl.ndarray(OUTPUT_QMX_SHAPE, dtype=output_x4_dtype, buffer=nl.sbuf)
    output_scale_sb = nl.ndarray(OUTPUT_QMX_SHAPE, dtype=MX_SCALE_DTYPE, buffer=nl.sbuf)
    # TODO: update indexing util function to accept ND dst_scale mgrid
    s_p, s_H512_f, s_T_f = nl.mgrid[0:H128, 0:n_H512_tiles, 0:T]
    s_p_0, _, s_p_1 = s_p.split(4, 8, 4)
    s_p = s_p_0 * 32 + s_p_1
    quantize_mx(
        src=input_swizzled_sb[...],
        dst=output_quant_sb[...],
        dst_scale=output_scale_sb[s_p, s_H512_f, s_T_f],
    )

    return output_quant_sb, output_scale_sb


def quantize_mx_activation(act_res_sb):
    """
    Function to quantize activations to MXFP8.

    Input shape:
        activation_res_sb: [16_I * 8_I, I/512, T, 4_I]

    Output shapes:
        act_quant_sb: [16_I * 8_I, I/512, T], 4_I is packed in x4 dtype
        act_scale_sb: [16_I * 8_I, I/512, T], scales located in leading 4P of each SBUF quadrant
    """

    TILE_I, n_I512_tiles, T, I_4 = act_res_sb.shape
    act_quant_shape = (TILE_I, n_I512_tiles, T)
    act_quant_sb = nl.ndarray(act_quant_shape, dtype=float8_e4m3fn_x4, buffer=nl.sbuf)
    act_scale_sb = nl.ndarray(act_quant_shape, dtype=nl.uint8, buffer=nl.sbuf)
    s_p, s_I512_f, s_T_f = nl.mgrid[0:TILE_I, 0:n_I512_tiles, 0:T]
    # TODO: change below indexing when util function accepts ND dst_scale grid + when we support I % 512 != 0
    s_p_0, _, s_p_1 = s_p.split(4, 8, 4)
    s_p = s_p_0 * 32 + s_p_1
    quantize_mx(
        src=act_res_sb[...],
        dst=act_quant_sb[...],
        dst_scale=act_scale_sb[s_p, s_I512_f, s_T_f]
    )

    return act_quant_sb, act_scale_sb