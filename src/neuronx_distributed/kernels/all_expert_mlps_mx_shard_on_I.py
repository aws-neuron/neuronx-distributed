"""
Copyright (c) 2025, Amazon.com. All Rights Reserved

Expert MLPs NKI kernel for token-generation with microscaling format (MX) weights.
"""

import math
from typing import Optional

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki._private.private_api import sendrecv
from neuronxcc.nki.compiler.backends.neuron.metaclasses import nki_dtype

from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType, ExpertAffinityScaleMode

from neuronx_distributed.kernels.expert_mlps_mx.swizzle_quantize_mx import swizzle_quantize_mx_input, quantize_mx_activation
from neuronx_distributed.kernels.expert_mlps_mx.gate_up_proj import gate_up_projection_mx_lhs_rhs_swap
from neuronx_distributed.kernels.expert_mlps_mx.down_proj import down_projection_mx
from neuronx_distributed.kernels.expert_mlps_mx.utils import _get_lnc_config
from neuronx_distributed.kernels.expert_mlps_mx.constants import (
    SUPPORTED_QMX_INPUT_DTYPES, GATE_FUSED_IDX, UP_FUSED_IDX,
)


@nki.jit(platform="trn3")
def all_expert_mlps_mx_shard_on_I_nki_kernel(
    input: nl.ndarray,
    gate_up_weights: nl.ndarray,
    down_weights: nl.ndarray,
    gate_up_weights_scale: nl.ndarray,
    down_weights_scale: nl.ndarray,
    gate_up_weights_bias: nl.ndarray,
    down_weights_bias: nl.ndarray,
    expert_affinities_masked: nl.ndarray,
    # TODO: decide whether we still need this
    expert_index: nl.ndarray = None,
    # TODO: move additional config to dedicated config object
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE, 
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    hidden_act_fn: ActFnType = ActFnType.Swish,
    # Placeholder variables
    hidden_act_scale_factor: Optional[float] = None,
    hidden_act_bias: Optional[float] = None, 
    input_in_sbuf: bool = False,
    output_in_sbuf: bool = False,
    lhs_rhs_swap: bool = True,
    # TODO: rename these flags and make sure they are propagated throughout kernel ops
    psum_accumulation_dtype: nki_dtype = nl.float32,
    activation_compute_dtype: nki_dtype = nl.bfloat16,
    **kwargs,
):
    """
    Perform expert MLPs on input, using microscaling format (MX) weights. Shards compute on intermediate dimension when run with LNC=2.
    
    Dimensions:
        B: batch size
        S: seqlen length
        T: total number of input tokens (equivalent to B*S)
        H: hidden dimension size of the model
        I: intermediate dimension size of the model after tensor parallelism
        E_L: number of local experts after expert parallelism

    Args:
        input: [32_T * 4_H, T/32, H/512, I/512 * 4_I * 16_I * 8_I]
        gate_up_weights: [E_L, 16_H * 8_H, 2, H/512, I/512 * 4_I * 16_I * 8_I], 4_H is packed in x4 dtype
        down_weights: [E_L, 16_I * 8_I, I/512, H], 4_I is packed in x4 dtype
        gate_up_weights_scale: [E_L, 16_H, 2, H/512, I/512 * 4_I * 16_I * 8_I]
        down_weights_scale: [E_L, 16_I, I/512, H]
        gate_up_weights_bias: [E_L, 16_I * 8_I, 2, I/512, 4_I]
        down_weights_bias: [E_L, H]
        expert_affinities_masked: [min(T, 128), ⌈T/128⌉, E_L]
        expert_index = None: Placeholder variable
        expert_affinities_scaling_mode = ExpertAffinityScaleMode.POST_SCALE: Expert affiniity scaling mode
        gate_clamp_upper_limit = None: Max value to clamp gate projection output to
        gate_clamp_lower_limit = None: Min value to clamp gate projection output to
        up_clamp_upper_limit = None: Max value to clamp up projection output to
        up_clamp_lower_limit = None: Min value to clamp up projection output to
        hidden_act_fn = ActFnType.Swish: Hidden activation function
        hidden_act_scale_factor = None: Placeholder variable
        hidden_act_bias = None: Placeholder variable 
        input_in_sbuf = False: Indicates whether inputs are in SBUF or HBM
        output_in_sbuf = False: Indicates desired output buffer location (SBUF or HBM)
        lhs_rhs_swap = True: Indicates whether to swap LHS and RHS of gate and up projection matmuls
        psum_accumulation_dtype = nl.float32: Accumulation dtype to use for matmul result in PSUM
        activation_compute_dtype = nl.bfloat16: Compute dtype for activations

    Returns:
        output_hbm: [T, H] OR output_sb: [min(T, 128), ⌈T/128⌉, H]
    
    More details on input & weight layout in doc `YFIQAmI1p2nr`
        
    ------------------------------- Remove this section before releasing -------------------------------
        
    Functionality / feature TODO:
    - Add support for H, I/TP not evenly divisible by 512
    - Add support for T>512 (up to arbitrarily large T)
    - Fix accuracy with E_L > 1 (perhaps accumulate sb in fp32)
    - LNC shard on H dim (here or in separate API)?
    - LNC shard on E dim (here or in separate API)?
    
    Perf TODO:
    - Manual allocation of all buffers?
    - Fused swizzle + QMX (right now done in series in step 1.2)
    - Fused activation multiply + QMX (right now done in series in steps 2.3, 2.4)
    - Fused down_proj + expert affinity scaling + expert accumulation (right now done in series in steps 2.5, 2.6)
        - No need for expert accumulation if E_L == 1
    - Fused sendrecv/reduce for shard-on-I (right now done in series in step 3)

    Other TODO:
    - Combine config into config object
    - Combine constants into one constants object that does input validation
    - Add names to all buffers so that profile is more readable
    - Migrate to new NKI FE
    """

    # Step 1: Check shapes and types, prep inputs
    # TODO: add extensive shape and dtype checks here
    assert input.dtype in SUPPORTED_QMX_INPUT_DTYPES, f"Expected input dtype in {SUPPORTED_QMX_INPUT_DTYPES}, got {input.dtype=}."
    assert lhs_rhs_swap, "lhs_rhs_swap=False is not yet supported!"
    
    # TODO: move constant extraction and shape validation to helper func / shape management object
    T32_H4, n_T32_tiles, n_H512_tiles, TILE_H = input.shape
    T = T32_H4 * n_T32_tiles // 4
    H = n_H512_tiles * TILE_H * 4
    TILE_T = min(T, nl.tile_size.pmax)
    NUM_TILES_IN_T = math.ceil(T / TILE_T)

    E_L, *_ = gate_up_weights.shape

    # LNC config
    n_prgs, prg_id = _get_lnc_config()
    PIPE_ID_OUTPUT = 0

    # Step 1.1: Load inputs if inputs are in HBM
    if not input_in_sbuf:
        input_sb = nl.load(input)
        expert_affinities_masked_sb = nl.load(expert_affinities_masked)
    else:
        input_sb = input
        expert_affinities_masked_sb = expert_affinities_masked

    # Step 1.2: Swizzle + QMX Input
    input_quant_sb, input_scale_sb = swizzle_quantize_mx_input(input_sb)

    # Step 2: Compute Expert MLPs sequentially
    OUTPUT_SHAPE = (TILE_T, NUM_TILES_IN_T, H)
    output_sb = nl.zeros(OUTPUT_SHAPE, dtype=activation_compute_dtype, buffer=nl.sbuf)
    
    # Update down bias shape for broadcasting, TODO find a way to move this into down_proj func this w/o hitting 
    #   'NotImplementedError: expand_dims not implemented for base tensor'
    down_weights_bias = down_weights_bias.expand_dims(1)
    
    for expert_idx in nl.sequential_range(E_L):
        # Step 2.1: Compute gate projection, projection output clamping, activation function
        gate_proj_res_sb = gate_up_projection_mx_lhs_rhs_swap(
            input_sb=input_quant_sb[...],
            input_scale_sb=input_scale_sb[...],
            weight=gate_up_weights[expert_idx, :, GATE_FUSED_IDX, :, :],
            weight_scale=gate_up_weights_scale[expert_idx, :, GATE_FUSED_IDX, :, :],
            bias=gate_up_weights_bias[expert_idx, :, GATE_FUSED_IDX, :, :],
            clamp_upper_limit=gate_clamp_upper_limit,
            clamp_lower_limit=gate_clamp_lower_limit,    
            hidden_act_fn=hidden_act_fn, 
            psum_accumulation_dtype=psum_accumulation_dtype,
            activation_compute_dtype=activation_compute_dtype,
        )

        # Step 2.2: Compute up projection, projection output clamping
        up_proj_res_sb = gate_up_projection_mx_lhs_rhs_swap(
            input_sb=input_quant_sb[...],
            input_scale_sb=input_scale_sb[...],
            weight=gate_up_weights[expert_idx, :, UP_FUSED_IDX, :, :],               
            weight_scale=gate_up_weights_scale[expert_idx, :, UP_FUSED_IDX, :, :],         
            bias=gate_up_weights_bias[expert_idx, :, UP_FUSED_IDX, :, :],
            clamp_upper_limit=up_clamp_upper_limit,
            clamp_lower_limit=up_clamp_lower_limit,
            hidden_act_fn=None,
            psum_accumulation_dtype=psum_accumulation_dtype,
            activation_compute_dtype=activation_compute_dtype,
        )

        # Step 2.3: Compute activation = act_fn(clamp(gate)) * clamp(up), reusing buffer for gate
        gate_proj_res_sb[...] = nisa.tensor_tensor(
            data1=gate_proj_res_sb[...],
            op=nl.multiply,
            data2=up_proj_res_sb[...],
        )

        # Step 2.4: QMX Activation
        act_quant_sb, act_scale_sb = quantize_mx_activation(gate_proj_res_sb)

        # Step 2.5: Compute Down Projection
        down_proj_res_sb = down_projection_mx(
            act_sb=act_quant_sb[...],
            act_scale_sb=act_scale_sb[...],
            weight=down_weights[expert_idx, :, :, :],
            weight_scale=down_weights_scale[expert_idx, :, :, :],
            bias=down_weights_bias[expert_idx, :, :],
            psum_accumulation_dtype=psum_accumulation_dtype,
            activation_compute_dtype=activation_compute_dtype,
        )

        # Step 2.6: Expert Affinitiy Scaling
        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
            # Output += down_proj_result[expert] * expert_affinity_mask[expert]
            for tile_t in nl.affine_range(NUM_TILES_IN_T):
                output_sb[:, tile_t, :] += nisa.tensor_scalar(
                    data=down_proj_res_sb[:, tile_t, :], 
                    op0=nl.multiply,
                    operand0=expert_affinities_masked_sb[:, tile_t, expert_idx],
                )
        else:
            raise NotImplementedError(f"Expert affinity scaling mode: {expert_affinities_scaling_mode} is not yet supported!")

    # Step 3: Sendrecv + reduce between NCs when LNC=2
    if n_prgs > 1:
        output_recv = nl.zeros(output_sb.shape, output_sb.dtype, buffer=nl.sbuf)
        sendrecv(send_to_rank=(1 - prg_id), recv_from_rank=(1 - prg_id),
                src=output_sb, dst=output_recv, pipe_id=PIPE_ID_OUTPUT)
        output_sb = nisa.tensor_tensor(output_sb, output_recv, nl.add)

    # Step 4: Return output
    if output_in_sbuf:
        return output_sb
    else:
        # Spill untiled shape by tiling spill of [min(T, 128), ⌈T/128⌉, H] on ⌈T/128⌉ dim
        output_shape = (T, H)
       
        # LNC2: each NC spills half of output to shared HBM
        if n_prgs > 1:
            output_hbm = nl.ndarray(output_shape, output_sb.dtype, buffer=nl.shared_hbm)
            H_offset = H // 2
            for tile_t in nl.affine_range(NUM_TILES_IN_T):
                # Based on our experiments, static DMA demonstrates better performance. We can revert to DGE if we encounter HBM out-of-memory (OOM) issues.
                nisa.dma_copy(
                    src=output_sb[:, tile_t, nl.ds(H_offset * prg_id, H_offset)], 
                    dst=output_hbm[nl.ds(TILE_T * tile_t, TILE_T), nl.ds(H_offset * prg_id, H_offset)], 
                    dge_mode=nisa.dge_mode.unknown,
                )
            return output_hbm
        
        # LNC1: NC spills entire output to HBM
        else:
            output_hbm = nl.ndarray(output_shape, output_sb.dtype, buffer=nl.hbm)
            for tile_t in nl.affine_range(NUM_TILES_IN_T):
                # Based on our experiments, static DMA demonstrates better performance. We can revert to DGE if we encounter HBM out-of-memory (OOM) issues.
                nisa.dma_copy(
                    src=output_sb[:, tile_t, :], 
                    dst=output_hbm[nl.ds(TILE_T * tile_t, TILE_T), :], 
                    dge_mode=nisa.dge_mode.unknown,
                )
            return output_hbm
