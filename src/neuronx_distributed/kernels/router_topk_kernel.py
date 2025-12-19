
import math
import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nt
import neuronxcc.nki.language as nl
from neuronxcc.nki._pre_prod_kernels.topk.topk_core  import topk_core

@nki.jit
def rmsnorm_router_top_k_with_masks_kernel(
    hidden_states: nt.tensor, 
    residual: nt.tensor,
    rmsnorm_weight: nt.tensor,
    router_weight: nt.tensor, 
    top_k: int, 
    router_bias: nt.tensor, 
    act_fn: str = "sigmoid", 
    topk_first: bool = False,
    eps = 1e-6,
    compute_dtype = nl.float32, 
):
    
    """
    NKI kernel for Residual Add + RMSNorm + Router Top-K computation with expert masks for MoE models.
    
    Performs Residual Add then RMSNorm on output hidden states, computes router logits, selects top-k experts,
    and generates expert affinities and masks for mixture-of-experts routing.
    
    Args:
        hidden_states (nt.tensor): Input tensor of shape (T, H)
        residual (nt.tensor): Residual connection tensor of shape (T, H)
        rmsnorm_weight (nt.tensor): RMSNorm scaling weights of shape (H)
        router_weight (nt.tensor): Router linear layer weights of shape (E, H) 
        top_k (int): Number of top experts to select per token
        router_bias (nt.tensor): Optional bias for router layer of shape (1, E)
        act_fn (str): Activation function for expert affinities ("sigmoid" or "softmax")
        topk_first (bool): If True, apply activation only to top-k values before masking;
                          if False, apply activation to all logits then mask
        eps (float): Small epsilon value for numerical stability in RMSNorm
        compute_dtype: Data type for computations
    
    Returns:
        tuple: (router_logits, expert_affinities, expert_index, expert_affinities_masked, expert_mask)
            - residual (nt.tensor): Hidden states for residual add after MoE and hidden states for expert mlps input
            - router_logits (nt.tensor): Raw router scores of shape (T, E)
            - expert_affinities (nt.tensor): Activated affinities of shape (T, top_k) if topk_first
                                            else (T, E)
            - expert_index (nt.tensor): Indices of top-k experts of shape (T, top_k)
            - expert_affinities_masked (nt.tensor): Masked affinities of shape (T, E)
    """
    T, H = hidden_states.shape
    E, H_R = router_weight.shape
    AFF_DIM = top_k if topk_first else E
    
    assert H == H_R, "Hidden dimension mismatch between input and router weights."
    assert E <= 512, "Router Kernel does not support E > 512."
    
    router_logits = nl.ndarray((T, E), dtype=compute_dtype, buffer=nl.shared_hbm)
    expert_affinities = nl.ndarray((T, AFF_DIM), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    expert_index = nl.ndarray((T, top_k), dtype=nl.int32, buffer=nl.shared_hbm)
    expert_affinities_masked = nl.ndarray((T, E), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    # LNC 2 for shard on T
    num_shards = nl.num_programs(0)
    shard_id = nl.program_id(0)

    assert T % num_shards == 0, f"T = {T} dimension is not divisible by number of shards."

    T_PER_SHARD = T // num_shards
    T_OFFSET = T_PER_SHARD * shard_id
    
    TILE_T = min(nl.tile_size.gemm_stationary_fmax, T_PER_SHARD)
    TILE_H = nl.tile_size.pmax
    
    H_ITERS = math.ceil(H/TILE_H)
    T_SHARD_ITERS = math.ceil(T_PER_SHARD/TILE_T)

    # Load bias if provided
    if router_bias is not None:
        sbuf_bias = nl.load(router_bias)

    hidden_tiles = nl.ndarray((nl.par_dim(TILE_H), H_ITERS, T_PER_SHARD), dtype=hidden_states.dtype, buffer=nl.sbuf)
    residual_tiles = nl.ndarray((nl.par_dim(TILE_H), H_ITERS, T_PER_SHARD), dtype=residual.dtype, buffer=nl.sbuf)
    weight_tile = nl.ndarray((nl.par_dim(TILE_H), H_ITERS, E), dtype=compute_dtype, buffer=nl.sbuf)
    rmsnorm_tile = nl.ndarray((nl.par_dim(TILE_H), H_ITERS), dtype=compute_dtype, buffer=nl.sbuf)

    eps_bias = nisa.memset((TILE_H, 1), value=eps, dtype=compute_dtype)
    reduction_vector = nisa.memset((TILE_H, TILE_H), value=1.0, dtype=nl.float32)

    # Load tensors to sbuf
    for h in nl.affine_range(H_ITERS):
        for t in nl.affine_range(num_shards):
            hidden_tiles[:, h, :] = nl.load_transpose2d(hidden_states[T_OFFSET:T_OFFSET+T_PER_SHARD, h*TILE_H:(h+1)*TILE_H])
            residual_tiles[:, h, :] = nl.load_transpose2d(residual[T_OFFSET:T_OFFSET+T_PER_SHARD, h*TILE_H:(h+1)*TILE_H])
        weight_tile[:, h, :] = nl.load_transpose2d(router_weight[:, h*TILE_H:(h+1)*TILE_H])
        rmsnorm_tile[:, h] = nl.load(rmsnorm_weight[h*TILE_H:(h+1)*TILE_H])

    # Residual Add
    hidden_tiles[...] = nisa.tensor_tensor(hidden_tiles, residual_tiles, op=nl.add)
    output_residual = nl.ndarray((T, H), dtype=compute_dtype, buffer=nl.shared_hbm)

    # RMSNorm on Residual Add
    hidden_squared = nisa.activation(op=nl.square, data=hidden_tiles)
    hidden_squared_reduce = nisa.tensor_reduce(data=hidden_squared, axis=1, op=nl.add)
    hidden_squared_reduce_final = nl.ndarray((nl.par_dim(TILE_H), T_PER_SHARD), dtype=compute_dtype, buffer=nl.psum)

    for t in nl.affine_range(T_SHARD_ITERS):
        hidden_squared_reduce_final[:, t*TILE_T:(t+1)*TILE_T] = nisa.nc_matmul(reduction_vector, hidden_squared_reduce[:,t*TILE_T:(t+1)*TILE_T])
    
    rms_scale = nl.ndarray((nl.par_dim(TILE_H), 1, T_PER_SHARD), dtype=compute_dtype, buffer=nl.sbuf)
    rms_scale[:, 0, :] = nisa.activation(op=nl.rsqrt, data=hidden_squared_reduce_final, scale=(1.0/H), bias=eps_bias)

    hidden_weighted = nisa.tensor_tensor(hidden_tiles, rmsnorm_tile, nl.multiply)
    hidden_weighted[...] = nisa.tensor_tensor(hidden_weighted, rms_scale, nl.multiply)

    output_hidden_states = nl.ndarray((T, H), dtype=compute_dtype, buffer=nl.shared_hbm)
    for h in nl.affine_range(H_ITERS):
        for t in nl.affine_range(T_SHARD_ITERS):
            nl.store(output_residual[T_OFFSET + t*TILE_T:T_OFFSET+(t+1)*TILE_T, h*TILE_H:(h+1)*TILE_H], 
                    value=nisa.nc_transpose(hidden_tiles[:, h, t*TILE_T:(t+1)*TILE_T]))
            nl.store(output_hidden_states[T_OFFSET + t*TILE_T:T_OFFSET+(t+1)*TILE_T, h*TILE_H:(h+1)*TILE_H], 
                    value=nisa.nc_transpose(hidden_weighted[:, h, :]))
    
    # Router logits computation
    for t in nl.affine_range(T_SHARD_ITERS):
        psum_logits = nl.zeros((TILE_T, E), nl.float32, buffer=nl.psum)
        for h in nl.affine_range(H_ITERS):
            psum_logits += nisa.nc_matmul(hidden_weighted[:,h,t*TILE_T:(t+1)*TILE_T], weight_tile[:, h, :])
        
        if router_bias is not None:
            sbuf_logits = nl.add(psum_logits, sbuf_bias)
        else:
            sbuf_logits = nisa.activation(op=nl.copy, data=psum_logits, dtype=router_logits.dtype)
        
        # Store router logits
        nl.store(router_logits[T_OFFSET + t*TILE_T:T_OFFSET+(t+1)*TILE_T, :], value=sbuf_logits)

        # Top-k selection
        sbuf_expert_values, sbuf_expert_index = topk_core(sbuf_logits, top_k)

        # Store expert indices
        nl.store(expert_index[T_OFFSET + t*TILE_T:T_OFFSET+(t+1)*TILE_T, :], value=sbuf_expert_index)

        if topk_first:
            # top-k first - apply activation on top-k values and get the masks
            if act_fn == "sigmoid":
                sbuf_affinities = nisa.activation(op=nl.sigmoid, data=sbuf_expert_values)
            else:
                sbuf_affinities = nl.softmax(sbuf_expert_values, axis=1)                

            matches = nl.equal(sbuf_expert_index, nl.arange(E)[None, None, :])   
            mask_tile = nisa.tensor_reduce(data=matches.astype(nl.float32), axis=1, op=nl.add)
            aff_mask = nisa.tensor_tensor(matches.astype(hidden_states.dtype), sbuf_affinities, op=nl.multiply)
            sbuf_affinities_masked = nisa.tensor_reduce(data=aff_mask, axis=1, op=nl.add)
        else:
            # apply activation on router logits and get the masks
            if act_fn == "sigmoid":
                sbuf_affinities = nisa.activation(op=nl.sigmoid, data=sbuf_logits)
            else:
                sbuf_affinities = nl.softmax(sbuf_logits, axis=1)

            matches = nl.equal(sbuf_expert_index, nl.arange(E)[None, None, :])
            mask_tile = nisa.tensor_reduce(data=matches.astype(nl.float32), axis=1, op=nl.add)
            sbuf_affinities_masked = nisa.tensor_tensor(sbuf_affinities, mask_tile, op=nl.multiply)
        
        # Store expert affinities and masks
        nl.store(expert_affinities[T_OFFSET + t*TILE_T:T_OFFSET+(t+1)*TILE_T, :], value=sbuf_affinities)
        nl.store(expert_affinities_masked[T_OFFSET + t*TILE_T:T_OFFSET+(t+1)*TILE_T, :], value=sbuf_affinities_masked)

    return output_residual, output_hidden_states, router_logits, expert_affinities, expert_index, expert_affinities_masked