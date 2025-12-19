# TODO[release]: move from experimental to testing or utils folder, as this code is not intended to be a prod API
# TODO[release]: determine proper copyright notice for OpenAI attribution

"""
Functionalized version of reference GPT-OSS expert MLPs implementation, in both bf16 and mxfp4.
https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py#L312
"""

import torch

from neuronx_distributed.experimental.quantization.microscaling.swizzle import swizzle_tensor
from neuronx_distributed.experimental.quantization.microscaling.mx_torch import quantize_mxfp8, matmul_mx


def expert_affinity_mask(router_logits, expert_index=None, k=4):
    experts = torch.topk(router_logits, k)
    expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
    expert_indices = experts.indices
    expert_affinities_mask = torch.zeros_like(router_logits)
    expert_affinities_mask.scatter_(dim=1, index=expert_indices, src=expert_weights)

    # Select local experts if expert_index is passed
    if expert_index is not None:
        return expert_affinities_mask[:, expert_index]
    else:
        return expert_affinities_mask


def topk(router_logits, k=4):
    experts = torch.topk(router_logits, k)
    expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
    expert_indices = experts.indices
    return expert_weights, expert_indices


def swiglu(x_glu, x_linear, alpha: float = 1.702, limit: float = 7.0):
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


def all_expert_mlps_bf16(
    norm_out,
    router_logits,
    W_gate,
    W_up,
    W_down,
    bias_gate,
    bias_up,
    bias_down,
):
    """
    All experts implementation of expert MLPs in bf16 for GPT-OSS.

    Inputs:
        norm_out: Output of RMSNorm(hidden)
        weights, biases for gate, up, down projections
    Output:
        result: output of MoE layer WITHOUT residual add
    """

    # Expert Affinity Mask
    expert_affinities_mask = expert_affinity_mask(router_logits)

    # Gate & Up Projections
    gate = torch.einsum("eih,th->tei", W_gate, norm_out) + bias_gate
    up = torch.einsum("eih,th->tei", W_up, norm_out) + bias_up

    # Activation
    act = swiglu(gate, up, alpha=1.702, limit=7.0)

    # Down Projections
    down = torch.einsum("ehi,tei->teh", W_down, act) + bias_down

    # Weighted sum of experts
    result = torch.einsum("teh,te->th", down, expert_affinities_mask)

    return result


def select_expert_mlps_bf16(
    norm_out,
    router_logits,
    W_gate,
    W_up,
    W_down,
    bias_gate,
    bias_up,
    bias_down,
):
    """
    Select experts implementation of expert MLPs in bf16 for GPT-OSS.

    Inputs:
        norm_out: Output of RMSNorm(hidden)
        weights, biases for gate, up, down projections
    Output:
        result: output of MoE layer WITHOUT residual add
    """

    # Select Experts
    expert_weights, expert_indices = topk(router_logits)

    # Gate & Up Projections
    gate = torch.einsum("teih,th->tei", W_gate[expert_indices, ...], norm_out) + bias_gate[expert_indices, ...]
    up = torch.einsum("teih,th->tei", W_up[expert_indices, ...], norm_out) + bias_up[expert_indices, ...]

    # Activation
    act = swiglu(gate, up, alpha=1.702, limit=7.0)

    # Down Projections
    down = torch.einsum("tehi,tei->teh", W_down[expert_indices, ...], act) + bias_down[expert_indices, ...]

    # Weighted sum of experts
    result = torch.einsum("teh,te->th", down, expert_weights)

    return result


def gate_up_projection_mx(input, input_scale, weight, scale, bias, matmul_accumulation_dtype, matmul_output_dtype):
    """
    Helper func used to compute gate or up projection.
    """
    H, T = input.shape
    E, H_, I = weight.shape  # noqa: E741

    assert H == H_, f"Expected equal H dims, got {H}, {H_}"

    proj_res = torch.empty((T, E, I), dtype=matmul_output_dtype)

    # einsum(ht,ehi->tei) + bias
    for expert in range(E):
        proj_res[:, expert, :] = matmul_mx(
            stationaryT_x4=input,
            moving_x4=weight[expert, :, :],
            stationaryT_scale=input_scale,
            moving_scale=scale[expert, :, :],
            accumulation_dtype=matmul_accumulation_dtype,
            output_dtype=matmul_output_dtype,
        )
    proj_res += bias.unsqueeze(0).broadcast_to(proj_res.shape)

    return proj_res


def down_projection_mx(act, act_scale, weight, scale, bias, matmul_accumulation_dtype, matmul_output_dtype):
    """
    Helper func used to compute down projection.
    """
    E, I, T = act.shape  # noqa: E741
    E_, I_, H = weight.shape

    assert I == I_, f"Expected equal H dims, got {I}, {I_}"

    proj_res = torch.empty((T, E, H), dtype=matmul_output_dtype)

    # einsum(eit,eih->teh)
    for expert in range(E):
        proj_res[:, expert, :] = matmul_mx(
            stationaryT_x4=act[expert, :, :],
            moving_x4=weight[expert, :, :],
            stationaryT_scale=act_scale[expert, :, :],
            moving_scale=scale[expert, :, :],
            accumulation_dtype=matmul_accumulation_dtype,
            output_dtype=matmul_output_dtype,
        )
    proj_res += bias.unsqueeze(0).broadcast_to(proj_res.shape)

    return proj_res


def expert_affinity_scale(down, expert_affinities_masked):
    T, E, H = down.shape
    T_, E_ = expert_affinities_masked.shape

    assert T == T_, f"Expected equal T dim in down, expert_affinities_mask; got {T}, {T_}"
    assert E == E_, f"Expected equal E dim in down, expert_affinities_mask; got {E}, {E_}"

    result = torch.einsum("teh,te->th", down, expert_affinities_masked)

    return result


def all_expert_mlps_act_mxfp8_w_mxfp4(
    norm_out,
    W_gate,
    W_up,
    W_down,
    scale_gate,
    scale_up,
    scale_down,
    bias_gate,
    bias_up,
    bias_down,
    router_logits=None,
    expert_index=None,
    expert_affinities_masked=None,
    matmul_accumulation_dtype=torch.float32,
    matmul_output_dtype=torch.bfloat16,
    use_unbiased_scale_qmx_norm=False,
    use_unbiased_scale_qmx_swiglu=False,
    DBG=False,
):
    """
    All experts implementation of expert MLPs in MX for GPT-OSS. Weights are MXFP4 and inputs/activations
    are quantized to MXFP8.

    Inputs:
        norm_out: Output of RMSNorm(hidden)
        weights, scales, biases for gate, up, down projections
        router_logits, expert_index, expert_affinities_masked: optional args to express expert weighting / routing
        matmul_accumulation_dtype = torch.float32: type to accumulate tiled matmul result in
        matmul_output_dtype = torch.bfloat16: type to cast output of matmuls to
    Output:
        result: output of MoE layer WITHOUT residual add
    """

    T, H = norm_out.shape
    E, I, _ = W_gate.shape  # noqa: E741

    # Expert Affinity Mask
    if expert_affinities_masked is not None:
        pass
    else:
        expert_affinities_masked = expert_affinity_mask(router_logits, expert_index=expert_index)

    # QMX(swizzle(norm_out.T))
    # [T, H] -> [H//4, T*4]
    norm_out_formatted = swizzle_tensor(norm_out.T)
    # [H//4, T*4] -> [H//4, T]
    norm_out_quant_x4, norm_out_scale = quantize_mxfp8(norm_out_formatted, use_unbiased_scale=use_unbiased_scale_qmx_norm)

    # Gate & Up Projections
    # [E, I, H//4] -> [E, H//4, I]: move contraction dim to be leading dim of W[e, :, :]
    FORMAT_PERMUTATAION = (0, 2, 1)
    W_gate_formatted = torch.permute(W_gate, FORMAT_PERMUTATAION)
    W_up_formatted = torch.permute(W_up, FORMAT_PERMUTATAION)
    scale_gate_formatted = torch.permute(scale_gate, FORMAT_PERMUTATAION)
    scale_up_formatted = torch.permute(scale_up, FORMAT_PERMUTATAION)

    gate = gate_up_projection_mx(
        input=norm_out_quant_x4,
        input_scale=norm_out_scale,
        weight=W_gate_formatted,
        scale=scale_gate_formatted,
        bias=bias_gate,
        matmul_accumulation_dtype=matmul_accumulation_dtype,
        matmul_output_dtype=matmul_output_dtype,
    )

    up = gate_up_projection_mx(
        input=norm_out_quant_x4,
        input_scale=norm_out_scale,
        weight=W_up_formatted,
        scale=scale_up_formatted,
        bias=bias_up,
        matmul_accumulation_dtype=matmul_accumulation_dtype,
        matmul_output_dtype=matmul_output_dtype,
    )

    # Activation
    act = swiglu(gate, up, alpha=1.702, limit=7.0)

    # QMX(swizzle(act.T))
    act_quant_x4 = torch.empty((E, I // 4, T), dtype=torch.uint32)
    act_scale = torch.empty((E, I // 32, T), dtype=torch.uint8)

    for expert in range(E):
        # [T, I] -> [I//4, T*4]
        act_e_T_swizzled = swizzle_tensor(act[:, expert, :].T)
        # [I//4, T*4] -> [I//4, T]
        quant, scale = quantize_mxfp8(act_e_T_swizzled, use_unbiased_scale=use_unbiased_scale_qmx_swiglu)
        act_quant_x4[expert, :, :], act_scale[expert, :, :] = quant, scale

    # Down Projection
    # [E, H, I//4] -> [E, I//4, H]: move contraction dim to be leading dim of W[e, :, :]
    W_down_formatted = torch.permute(W_down, FORMAT_PERMUTATAION)
    scale_down_formatted = torch.permute(scale_down, FORMAT_PERMUTATAION)

    down = down_projection_mx(
        act=act_quant_x4,
        act_scale=act_scale,
        weight=W_down_formatted,
        scale=scale_down_formatted,
        bias=bias_down,
        matmul_accumulation_dtype=matmul_accumulation_dtype,
        matmul_output_dtype=matmul_output_dtype,
    )

    # Weighted sum of experts
    result = expert_affinity_scale(down, expert_affinities_masked)

    if DBG:
        return norm_out_quant_x4, norm_out_scale, gate, up, act, act_quant_x4, act_scale, down, result
    
    return result
