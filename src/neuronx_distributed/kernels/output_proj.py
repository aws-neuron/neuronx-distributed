from typing import Optional, Tuple
import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from neuronxcc.nki._pre_prod_kernels.stream_shuffle_broadcast import stream_shuffle_broadcast
from neuronxcc.nki._pre_prod_kernels.util.kernel_helpers import get_verified_program_sharding_info


@nki.jit(debug_kernel=True)
def cte_output_proj_dequant_kernel(
    active: nl.ndarray,
    weight: nl.ndarray,
    bias: nl.ndarray = None,
    scale: nl.ndarray = None, # repeated per block_size only on in_channels (partition) dim
    block_size: Optional[Tuple[int, int]] = None
) -> nl.ndarray:

    """
    This output projection kernel implements a naive matrix multiplication (active @ weight) used by output projection in 
    transformer models.

    Dimensions:
        B: Batch size
        N: Number of heads
        S: Sequence length
        H: Hidden dimension size
        D: Head dimension size

    Args:
        active (nl.ndarray): Input tensor on HBM with the shape of [B, N, D, S]
        weight (nl.ndarray): Weight tensor on HBM with the shape of [N * D, H]
        bias   (nl.ndarray): Optional bias tensor on HBM with the shape of [1, H]
        scale  (nl.ndarray): Scale tensor on HBM with the shape of [N * D, H // block_size[1]]
        block_size: Tuple on HBM
    
    Returns: 
        out (nl.ndarray): Output tensor on HBM with the shape of [B, S, H]

    Notes: 
        - This implementation only focuses on the prefill (aka CTE) phase of transformer models:
            (1) S is >= 128 (although S < 128 still work, it may not be performant)
            (2) LNC sharding is performed on the weight tensor hidden dimension
    """

    B, N, D, S = active.shape
    _, H = weight.shape
    _, n_prgs, prg_id = get_verified_program_sharding_info("output_proj_kernel", (0, 1))
    should_dequant = True if scale is not None else False

    # Perform native check
    assert N * D == weight.shape[0], "The contract dimension of input and weight tensors must match"
    assert H <= 32768, "Output projection kernel currently only supports H to be no more than 32768"
    assert H % n_prgs == 0, "Hidden needs to be dividable by LNC size since LNC sharding is on the weight hidden dimension"
    assert D <= 128, "The supported head dimension must be less than or equal to 128"

    # Dequant input check
    if should_dequant:
        assert block_size is not None, "Input `block_size` is required for dequantization"
        assert bias is None, "Dequantization with bias is not supported yet"
        assert weight.shape[0] == scale.shape[0], f"Expect weight and scale matches on dim 0, got {weight.shape[0]} and {scale.shape[0]}"
        assert weight.shape[1] == scale.shape[1] * block_size[1], "Blockwise quantization weight and scale does not match shape on dim 0"

    # Hardware specific parameters
    P = 128
    PSUM_BANK_ELEMS = 512

    # If D < P, try to increase each MM instruction's contraction dimension by folding N into D.
    if D < P:
        # reshape_factor is the largest divisor of N such that (reshape_factor * D) is <= P.
        reshape_factor = N
        while (N % reshape_factor) or (reshape_factor * D) > P:
            reshape_factor -= 1
        assert reshape_factor > 0, 'reshape_factor of 1 should always satisfy the reshape requirement'
        N //= reshape_factor
        D *= reshape_factor
        active = active.reshape((B, N, D, S))

    weight = weight.reshape((N, D, H))

    # Set tiling strategy
    #   - LNC Sharding on the weight tensor's hidden dimension for now
    #   - The following tiling needs to be revisited later for fine-tuning of the perf
    TILE_SIZE = 512
    h_per_shard = H // n_prgs
    h_load_tile_size = h_per_shard
    h_outer_lc = int(np.ceil(h_per_shard / h_load_tile_size)) # 1
    n_lc = N
    b_lc = B
    s_outer_lc = int(np.ceil(S / TILE_SIZE))
    # Allocate (bf16 or fp8) weight sbuf tensor
    weight_sb = nl.ndarray((h_outer_lc, n_lc, nl.par_dim(D), h_load_tile_size), dtype=weight.dtype, buffer=nl.sbuf) # (1, N, D, h_load_tile_size)
    
    if should_dequant:
        # Use the same layout as weight
        scale_h_size = scale.shape[1]
        scale = scale.reshape((N, D, scale_h_size))
        scale_h_size_per_shard = scale_h_size // n_prgs
        # Allocate dequantized weight and scale sbuf tensor
        weight_tile_sb = nl.ndarray((nl.par_dim(D), h_load_tile_size), dtype=active.dtype, buffer=nl.sbuf)
        scale_tile_sb = nl.ndarray((nl.par_dim(D), scale_h_size_per_shard), dtype=nl.float32, buffer=nl.sbuf)
    
    if bias is not None:
        bias_sb = nl.ndarray((nl.par_dim(P), h_outer_lc * h_load_tile_size), dtype=bias.dtype, buffer=nl.sbuf)

    # out tensor
    out = nl.ndarray((B, S, H), dtype=active.dtype, buffer=nl.shared_hbm)

    def process_batch_tile(batch_idx, seq_tile_idx, cur_s_tile_size):
        s_inner_lc = int(np.ceil(cur_s_tile_size / P))
        i_p_load, i_f_load = nl.mgrid[0:D, 0:cur_s_tile_size] # i_p, i_h both in shape (D, cur_s_tile_size)
        active_sb = nl.ndarray((n_lc, nl.par_dim(D), TILE_SIZE), dtype=active.dtype, buffer=nl.sbuf)
        result_sb = nl.ndarray((s_inner_lc, nl.par_dim(P), h_per_shard), dtype=active.dtype, buffer=nl.sbuf)

        for k in nl.affine_range(n_lc):
            active_sb[k, i_p_load, i_f_load] = nl.load(active[batch_idx][k][i_p_load, i_f_load + TILE_SIZE * seq_tile_idx])

        for m in nl.affine_range(s_inner_lc):
            psum_p_size = P
            for k in nl.affine_range(h_outer_lc):
                cur_h_tile = h_load_tile_size
                h_inner_lc = int(np.ceil(cur_h_tile / PSUM_BANK_ELEMS))
                for lc in nl.affine_range(h_inner_lc):
                    psum_f_size = PSUM_BANK_ELEMS
                    res_psum = nl.zeros((nl.par_dim(psum_p_size), psum_f_size), dtype=nl.float32, buffer=nl.psum)
                    for n in nl.affine_range(n_lc):
                        i_p_lhs, i_f_lhs = nl.mgrid[0:D, 0:psum_p_size] # both in shape (D, 128)
                        i_p_rhs, i_f_rhs = nl.mgrid[0:D, 0:psum_f_size] # both in shape (D, 512)

                        res_psum += nisa.nc_matmul(
                            active_sb[n][i_p_lhs, i_f_lhs + m * P],
                            weight_sb[k][n][i_p_rhs, i_f_rhs + lc * psum_f_size]
                        )
                    i_p_psum_copy, i_f_psum_copy = nl.mgrid[0:psum_p_size, 0:psum_f_size]
                    if bias is not None:
                        bias_tile = bias_sb[i_p_psum_copy, k * h_load_tile_size + lc * psum_f_size + i_f_psum_copy]
                        result_sb[m, i_p_psum_copy, k * h_load_tile_size + lc * psum_f_size + i_f_psum_copy] = nisa.tensor_tensor(res_psum, bias_tile, op=nl.add)
                    else:
                        if m % 2 == 0:
                            result_sb[m, i_p_psum_copy, k * h_load_tile_size + lc * psum_f_size + i_f_psum_copy] = nisa.tensor_copy(res_psum, engine=nisa.scalar_engine)
                        else:
                            result_sb[m, i_p_psum_copy, k * h_load_tile_size + lc * psum_f_size + i_f_psum_copy] = nisa.tensor_copy(res_psum, engine=nisa.vector_engine)
        for k in nl.affine_range(s_inner_lc):
            s_p_size = P
            i_s_save, i_h_save = nl.mgrid[0:s_p_size, 0:h_per_shard]
            i_s_index = i_s_save + k * P
            nl.store(out[batch_idx, seq_tile_idx * TILE_SIZE  + i_s_index, i_h_save + h_per_shard * prg_id],
                     result_sb[k, i_s_save, i_h_save],
                     mask=(i_s_index < cur_s_tile_size))


    # Load weight and maybe dequant
    # Reuse weight across the batches and sequence length
    for i in nl.affine_range(h_outer_lc): # 1
        cur_h_tile = h_load_tile_size
        for j in nl.affine_range(n_lc):
            cur_p = D
            # Weight loading
            i_p, i_h = nl.mgrid[0:cur_p, 0:cur_h_tile] # i_p, i_h both in shape (D, cur_h_tile)
            h_index_on_shard = i * h_load_tile_size + i_h
            
            if should_dequant:
                # Scale loading indexing
                i_p_scale, i_h_scale = nl.mgrid[0:cur_p, 0:scale_h_size_per_shard]
                scale_h_index_on_shard = i * scale_h_size_per_shard + i_h_scale

                # dequant tile-by-tile
                weight_tile_sb = nl.load(weight[j][i_p, h_per_shard * prg_id + h_index_on_shard],
                                                mask=(h_index_on_shard < h_per_shard)) # (D, h_load_tile_size)
                scale_tile_sb = nl.load(scale[j][i_p_scale, scale_h_size_per_shard * prg_id + scale_h_index_on_shard],
                                                mask=(scale_h_index_on_shard < scale_h_size_per_shard)) # (D, scale_h_size_per_shard)
                
                # Loop to dequant every block_size on hidden (free) dimension
                if block_size is not None:
                    _, block_h_size = block_size
                for k in nl.affine_range(scale_h_size_per_shard):

                    # tensor_scalar require operand0's free dimension should be 1 (a vector)
                    weight_sb[i][j][:, k*block_h_size:(k+1)*block_h_size] = nisa.tensor_scalar(
                        weight_tile_sb[:, k*block_h_size:(k+1)*block_h_size], # (D, block_h_size)
                        nl.multiply, 
                        scale_tile_sb[:, k], # (D, 1)
                        engine = nisa.vector_engine
                    )
            else:
                weight_sb[i, j, i_p, i_h] = nl.load(weight[j][i_p, h_per_shard * prg_id + h_index_on_shard],
                                                mask=(h_index_on_shard < h_per_shard)) # weight_sb (1, N, D, h_load_tile_size)

    # Load bias and reuse them across the batches and sequence length
    if bias is not None:
        bias_sb_1d = nl.ndarray((nl.par_dim(1), h_outer_lc * h_load_tile_size), dtype=bias.dtype, buffer=nl.sbuf)
        bias_sb_1d[:, :h_per_shard] = nl.load(bias.reshape((1, H))[:, nl.ds(h_per_shard * prg_id, h_per_shard)])
        stream_shuffle_broadcast(bias_sb_1d, bias_sb)

    # Iterate over batch
    for i in nl.affine_range(b_lc):
        for j in nl.affine_range(s_outer_lc-1):
            s_tile_size = TILE_SIZE
            process_batch_tile(i, j, s_tile_size)

        # last iteration on S, with tile_size might be less than TILE_SIZE
        last_s_tile_size = S - TILE_SIZE * (s_outer_lc-1)
        process_batch_tile(i, s_outer_lc-1, last_s_tile_size)

    return out
