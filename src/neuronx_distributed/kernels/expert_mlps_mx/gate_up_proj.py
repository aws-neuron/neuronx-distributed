"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Gate and up projection compute for token-generation with microscaling format (MX) weights.
"""

import numpy as np
import neuronxcc.nki.typing as nt
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki._private.private_api import nc_matmul_mx

from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType

from neuronx_distributed.kernels.expert_mlps_mx.utils import _get_lnc_config, get_scale_idx
from neuronx_distributed.kernels.expert_mlps_mx.constants import (
    N_BTYES_TO_NKI_X4_DTYPE_MAP, NUM_QUADRANTS_IN_SBUF, SBUF_QUADRANT_SIZE, SCALE_P_ELEM_PER_QUADRANT, _q_width,
)


def gate_up_projection_mx_lhs_rhs_swap(
    input_sb,
    input_scale_sb,
    weight,
    weight_scale,
    bias,
    clamp_upper_limit=None,
    clamp_lower_limit=None,
    hidden_act_fn=None,
    psum_accumulation_dtype=nl.float32,             # TODO: need to change impl to take advantage of full PSUM w/ bf16
    activation_compute_dtype=nl.bfloat16,           
):
    """
    Function to handle gate or up projection, projection output clamping, and activation function, with lhs/rhs swap. When executed with LNC=2, shards compute on I dimension.
    
    Input shapes:
        input_sb: [16_H * 8_H, H/512, T], 4_H is packed in x4 dtype
        input_scale_sb: [16_H * 8_H, H/512, T], scales located in leading 4P of each SBUF quadrant
        weight: [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], 4_H is packed in x4 dtype
        weight_scale: [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I], scales located in leading 4P of each SBUF quadrant
        bias: [16_I * 8_I, I/512, 4_I]

    Output shape:
        out_sb: [16_I * 8_I, I/512, T, 4_I] 
    """

    # TODO: validate that input_sb, input_scale_sb are in SBUF
    TILE_H, n_H512_tiles, T = input_sb.shape
    TILE_H_, n_H512_tiles_, I = weight.shape  # noqa: E741
    assert TILE_H == TILE_H_, f"Expected same number of partitions in input and weight, got {TILE_H}, {TILE_H_}"
    assert n_H512_tiles == n_H512_tiles_, f"Expected same number of H tiles in input and weight, got {n_H512_tiles}, {n_H512_tiles_}"
    # TODO: remove assertion when support is added for incomplete tiles in I
    assert I % 512 == 0, f"Expected I divisible by 1024, got {I=}."

    # TODO: remove assertion when we have tiling on T dim
    assert T <= 512, "T>512 is not yet supported by all experts MX kernel."

    # LNC config
    n_prgs, prg_id = _get_lnc_config()
    if n_prgs > 1:
        # TODO: add support for I not evenly divisible by 1024
        assert I % 1024 == 0, f"Expected I divisible by 1024 for shard on I kernel with LNC=2, got {I=}."
        I_local = I // 2
    else:
        I_local = I

    # Load weight
    weight_sb_shape = (TILE_H, n_H512_tiles, I_local)
    weight_sb = nl.ndarray(weight_sb_shape, dtype=weight.dtype, buffer=nl.sbuf)
    # Based on our experiments, static DMA demonstrates better performance. We can revert to DGE if we encounter HBM out-of-memory (OOM) issues.
    nisa.dma_copy(
        src=weight[:, :, nl.ds(I_local * prg_id, I_local)], 
        dst=weight_sb[...], 
        dge_mode=nisa.dge_mode.none,
    )
    weight_x4_dtype = N_BTYES_TO_NKI_X4_DTYPE_MAP[np.dtype(weight.dtype).itemsize]
    weight_sb = weight_sb.view(weight_x4_dtype)

    # Load scale into first 4 partitions of each SBUF quadrant
    # NOTE: in the future we can load scale into packed SBUF buffer to save 4x SBUF usage OR 
    #   to do 4x smaller load into 4x more partitions
    weight_scale_sb = nl.ndarray(weight_sb_shape, dtype=weight_scale.dtype, buffer=nl.sbuf)
    for quadrant in nl.affine_range(NUM_QUADRANTS_IN_SBUF):
        # Based on our experiments, static DMA demonstrates better performance. We can revert to DGE if we encounter HBM out-of-memory (OOM) issues.
        nisa.dma_copy(
            src=weight_scale[nl.ds(SCALE_P_ELEM_PER_QUADRANT * quadrant, SCALE_P_ELEM_PER_QUADRANT), :, nl.ds(I_local * prg_id, I_local)], 
            dst=weight_scale_sb[nl.ds(SBUF_QUADRANT_SIZE * quadrant, SCALE_P_ELEM_PER_QUADRANT), :, :], 
            dge_mode=nisa.dge_mode.none,
        )

    # I tiling strategy
    n_I512_tiles, I_4, TILE_I = I_local // 512, 4, 128

    # Load bias
    is_bias = bias is not None
    if is_bias:
        _, n_I512_bias_tiles, _ = bias.shape
        assert n_I512_bias_tiles % 2 == 0, f"Expected n_I512_bias_tiles divisible by 2, got {n_I512_bias_tiles=} with {bias.shape=}."
        bias_sb = nl.ndarray((TILE_I, n_I512_tiles, I_4), dtype=activation_compute_dtype, buffer=nl.sbuf)
        # Based on our experiments, static DMA demonstrates better performance. We can revert to DGE if we encounter HBM out-of-memory (OOM) issues.
        nisa.dma_copy(
            src=bias[:, nl.ds(n_I512_tiles * prg_id, n_I512_tiles)],
            dst=bias_sb[...],
            dge_mode=nisa.dge_mode.none,
        )

    # Tiled MM: compute W_mxfp4/8 (stationary) @ input_mxfp8 (moving)
    # TODO[future] update below to handle tiling on T dim, allocate larger PSUM buffer outside of I loop
    PROJ_OUT_SHAPE = (TILE_I, n_I512_tiles, T, I_4)
    out_sb = nl.ndarray(PROJ_OUT_SHAPE, dtype=activation_compute_dtype, buffer=nl.sbuf)
    for tile_i in nl.sequential_range(n_I512_tiles):
        out_psum = nl.ndarray((TILE_I, I_4, T), psum_accumulation_dtype, buffer=nl.psum)
        for q_width_I_idx in nl.sequential_range(_q_width):
            weight_I_offset = tile_i * 512 + q_width_I_idx * TILE_I
            for tile_h in nl.sequential_range(n_H512_tiles):
                Ws_p, Ws_f = get_scale_idx(TILE_H, TILE_I)
                is_p, is_f = get_scale_idx(TILE_H, T)
                out_psum[:, q_width_I_idx, :] += nc_matmul_mx(
                    stationary=weight_sb[:, tile_h, nl.ds(weight_I_offset, TILE_I)],
                    moving=input_sb[:, tile_h, :],
                    stationary_scale=weight_scale_sb[Ws_p, tile_h, weight_I_offset + Ws_f],
                    moving_scale=input_scale_sb[is_p, tile_h, is_f],
                )

        # Accumulate bias during PSUM eviction
        # TODO[perf] tune engine mix for PSUM eviction/bias add, clamping, act_fn to evenly mix DVE/scalar. Right 
        #   now PSUM spill and act_fn are overloading scalar and DVE is not busy enough with clamping.
        if is_bias:
            for q_width_I_idx in nl.sequential_range(_q_width):
                out_sb[:, tile_i, :, q_width_I_idx] = nisa.tensor_scalar(
                    data=out_psum[:, q_width_I_idx, :],
                    op0=nl.add,
                    operand0=bias_sb[:, tile_i, q_width_I_idx],
                    dtype=activation_compute_dtype,
                )
        else:
            out_sb[:, tile_i, :, :] = nisa.tensor_copy(out_psum[:, :, :], dtype=activation_compute_dtype)

        # Clamp projection output to [clamp_lower_limit, clamp_upper_limit]
        if clamp_upper_limit is not None or clamp_lower_limit is not None:
            out_sb[:, tile_i, :, :] = nisa.tensor_scalar(
                data=out_sb[:, tile_i, :, :],
                op0=nl.minimum if clamp_upper_limit is not None else None,
                operand0=clamp_upper_limit,
                op1=nl.maximum if clamp_lower_limit is not None else None,
                operand1=clamp_lower_limit,
            )
        
        # Compute activation function
        if hidden_act_fn is not None:
            if hidden_act_fn == ActFnType.Swish:
                out_sb[:, tile_i, :, :] = nisa.activation(
                    data=out_sb[:, tile_i, :, :],
                    op=nl.gelu_apprx_sigmoid,
                )
            else:
                raise NotImplementedError(f"{hidden_act_fn=} is not yet supported for all experts MX kernel.")

    return out_sb
