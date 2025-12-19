"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Down projection compute for token-generation with microscaling format (MX) weights.
"""

import math
import numpy as np
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki._private.private_api import nc_matmul_mx

from neuronx_distributed.kernels.expert_mlps_mx.utils import _get_lnc_config, get_scale_idx
from neuronx_distributed.kernels.expert_mlps_mx.constants import (
    N_BTYES_TO_NKI_X4_DTYPE_MAP, NUM_QUADRANTS_IN_SBUF, SBUF_QUADRANT_SIZE, SCALE_P_ELEM_PER_QUADRANT,
)


def down_projection_mx(
    act_sb,
    act_scale_sb,
    weight,
    weight_scale,
    bias,
    psum_accumulation_dtype=nl.float32,             # TODO: need to change impl to take advantage of full PSUM w/ bf16
    activation_compute_dtype=nl.bfloat16,           
):
    """
    Function to handle down projection. When executed with LNC=2, shards compute on I dimension.

    Input shapes:
        act_sb: [16_I * 8_I, I/512, T], 4_I is packed in x4 dtype
        act_scale_sb: [16_I * 8_I, I/512, T], scales located in leading 4P of each SBUF quadrant
        weight: [16_I * 8_I, I/512, H], 4_I is packed in x4 dtype
        weight_scale: [], scales located in leading 4P of each SBUF quadrant
        bias: [1, H] or [H] FIXME: in some cases calling expand_dims() on bias shape [H] throws error
    
    Output shape:
        out_sb: [min(T, 128), ⌈T/128⌉, H]
    """
    
    # TODO: validate that act_sb, act_scale_sb are in SBUF
    # TODO: add shape assertions for stationary & moving matrices
    TILE_I, n_I512_tiles, T = act_sb.shape
    TILE_I_, n_global_I512_tiles, H = weight.shape
    assert TILE_I == TILE_I_, f"Expected same number of partitions in activation and weight, got {TILE_I}, {TILE_I_}"

    # LNC config
    n_prgs, prg_id = _get_lnc_config()
    if n_prgs > 1:
        # TODO: add support for I not evenly divisible by 1024
        assert n_global_I512_tiles % 2 == 0, f"Even number of I tiles is required for shard on I kernel with LNC=2, got {n_I512_tiles} I tiles."
        assert n_global_I512_tiles // 2 == n_I512_tiles, f"Expected number of I tiles in activation equal to number of I tiles in weight divided by 2 with LNC=2, got {n_I512_tiles=}, {n_global_I512_tiles=}"
    else:
       assert n_I512_tiles == n_global_I512_tiles, f"Expected equal number of I tiles in activation and weight, got {n_I512_tiles=}, {n_global_I512_tiles=}"

    # Load weight
    weight_x4_dtype = N_BTYES_TO_NKI_X4_DTYPE_MAP[np.dtype(weight.dtype).itemsize]
    weight_sb_shape = (TILE_I, n_I512_tiles, H)
    weight_sb = nl.ndarray(weight_sb_shape, dtype=weight.dtype, buffer=nl.sbuf)
    # Based on our experiments, static DMA demonstrates better performance. We can revert to DGE if we encounter HBM out-of-memory (OOM) issues.
    nisa.dma_copy(
        src=weight[:, nl.ds(n_I512_tiles * prg_id, n_I512_tiles), :],
        dst=weight_sb[...],
        dge_mode=nisa.dge_mode.none,
    )
    weight_sb = weight_sb.view(weight_x4_dtype)

    # Load scale
    # NOTE: in the future we can load scale into packed SBUF buffer to save 4x SBUF usage OR 
    #   to do 4x smaller load into 4x more partitions
    weight_scale_sb = nl.ndarray(weight_sb_shape, dtype=weight_scale.dtype, buffer=nl.sbuf)
    # Based on our experiments, static DMA demonstrates better performance. We can revert to DGE if we encounter HBM out-of-memory (OOM) issues.
    for quadrant in nl.affine_range(NUM_QUADRANTS_IN_SBUF):
        nisa.dma_copy(
            src=weight_scale[nl.ds(SCALE_P_ELEM_PER_QUADRANT * quadrant, SCALE_P_ELEM_PER_QUADRANT), nl.ds(n_I512_tiles * prg_id, n_I512_tiles), :], 
            dst=weight_scale_sb[nl.ds(SBUF_QUADRANT_SIZE * quadrant, SCALE_P_ELEM_PER_QUADRANT), :, :], 
            dge_mode=nisa.dge_mode.none,
        )

    # Load bias
    is_bias = bias is not None
    TILE_T = min(T, nl.tile_size.pmax)  # T will be partition dim in output
    bias_sb = nl.ndarray((TILE_T, H), dtype=activation_compute_dtype, buffer=nl.sbuf)
    is_bias = (bias is not None) and (prg_id == 0)
    if is_bias:
        # Based on our experiments, static DMA demonstrates better performance. We can revert to DGE if we encounter HBM out-of-memory (OOM) issues.
        nisa.dma_copy(
            src=bias[...],
            dst=bias_sb[0:1, 0:H],
            dge_mode=nisa.dge_mode.none,
        )
        bias_sb[0:TILE_T, 0:H] = nl.broadcast_to(bias_sb[0:1, 0:H], shape=bias_sb.shape)

    # Tiled MM: compute activation_mxfp8 (stationary) @ W_mxfp4/8 (moving)
    # TODO[future] update below to handle tiling on T dim, allocate larger PSUM buffer outside of H loop
    TILE_H = 512  # TODO: make this configurable based on PSUM dtype
    NUM_TILES_IN_T = math.ceil(T / 128)
    NUM_TILES_IN_H = H // TILE_H
    PROJ_OUT_SHAPE = (TILE_T, NUM_TILES_IN_T, H)
    out_sb = nl.ndarray(PROJ_OUT_SHAPE, dtype=activation_compute_dtype, buffer=nl.sbuf)
    for tile_t in nl.sequential_range(NUM_TILES_IN_T):
        act_T_offset = nl.ds(TILE_T * tile_t, TILE_T)
        for tile_h in nl.sequential_range(NUM_TILES_IN_H):
            out_psum = nl.ndarray((TILE_T, TILE_H), psum_accumulation_dtype, buffer=nl.psum)
            weight_H_offset = nl.ds(TILE_H * tile_h, TILE_H)
            for tile_i in nl.sequential_range(n_I512_tiles):
                is_p, is_f = get_scale_idx(TILE_I, TILE_T)
                Ws_p, Ws_f = get_scale_idx(TILE_I, TILE_H)
                out_psum += nc_matmul_mx(
                    stationary=act_sb[:, tile_i, act_T_offset],
                    moving=weight_sb[:, tile_i, weight_H_offset],
                    stationary_scale=act_scale_sb[is_p, tile_i, TILE_T * tile_t + is_f],
                    moving_scale=weight_scale_sb[Ws_p, tile_i, TILE_H * tile_h + Ws_f],
                )

            # Accumulate bias during PSUM eviction
            # TODO[perf] interleave spill on DVE & ScalarE for better spill perf
            output_H_offset = nl.ds(TILE_H * tile_h, TILE_H)
            if is_bias:
                out_sb[:, tile_t, output_H_offset] = nisa.tensor_tensor(out_psum[:, :], bias_sb[:, output_H_offset], op=nl.add)
            else:
                out_sb[:, tile_t, output_H_offset] = nisa.tensor_copy(out_psum[:, :])

    return out_sb
