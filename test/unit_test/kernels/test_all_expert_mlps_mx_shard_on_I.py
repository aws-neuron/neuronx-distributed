# flake8: noqa
import os

import pytest

import numpy as np
import torch
from torch_xla.core import xla_model
import torch_neuronx
import ml_dtypes

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
# TODO[release] switch to nisa before release
import neuronxcc.nki._private.private_api as nki_private

from neuronxcc.nki._pre_prod_kernels.common_types import ActFnType

from neuronx_distributed.experimental.quantization.microscaling.swizzle import swizzle_tensor
from neuronx_distributed.experimental.quantization.microscaling.mx_torch import quantize_mxfp8, dequantize_mx_tensor
from neuronx_distributed.experimental.quantization.microscaling.expert_mlps_mx import gate_up_projection_mx as gate_up_projection_torch, down_projection_mx as down_projection_torch


# NOTE[future]: there is some duplication between this file, the real weights version, and the torch CPU version that could be consolidated

RUN_TRN3_TESTS = "1" in os.environ.get("RUN_TRN3_TESTS", "")

NKI_DTYPE_MAPPING = {
    torch.bfloat16: nl.bfloat16,
    torch.float32: nl.float32,
}

# use values from real weights
GPT_OSS_LAYER_0_CONFIG = (
    128, 2880, 2880,
    ((119, 123), (119, 123), (119, 123)),
    ((-2.0625, 0.5234375), (-1.4375, 0.90625), (-1.6328125, 1.4375)),
    (0.00799560546875, 0.292969, -1.3671875, 1.6484375, 23.0),
    ([54, 11, 23, 58], 16.8750),
)

GPT_OSS_LAYER_0_PADDED_CONFIG = (
    128, 3072, 3072,
    ((119, 123), (119, 123), (119, 123)),
    ((-2.0625, 0.5234375), (-1.4375, 0.90625), (-1.6328125, 1.4375)),
    (0.00799560546875, 0.292969, -1.3671875, 1.6484375, 23.0),
    ([54, 11, 23, 58], 16.8750),
)

GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG = (
    1, 3072, 3072,
    ((119, 123), (119, 123), (119, 123)),
    ((-2.0625, 0.5234375), (-1.4375, 0.90625), (-1.6328125, 1.4375)),
    (0.00799560546875, 0.292969, -1.3671875, 1.6484375, 23.0),
    ([54, 11, 23, 58], 16.8750),
)

GPT_OSS_LAYER_0_ONE_EXPERT_NO_BIAS_PADDED_CONFIG = (
    1, 3072, 3072,
    ((119, 123), (119, 123), (119, 123)),
    ((0, 0), (0, 0), (0, 0)),
    (0.00799560546875, 0.292969, -1.3671875, 1.6484375, 23.0),
    ([54, 11, 23, 58], 16.8750),
)

GPT_OSS_LAYER_0_GATE_UP_OUT_CONFIG = (
    (-0.4238, 0.4707, -8.3750, 5.0625),
    (-0.7070, 0.6797, -5.7500, 6.5000),
)

GPT_OSS_LAYER_0_ACT_OUT_CONFIG = (-0.00958251953125, 0.1650390625, -4.5, 11.0)


def _hidden_act_fn(X, act_fn: ActFnType):
    if act_fn is None:
        return X
    elif act_fn == ActFnType.Swish:
        return X * torch.sigmoid(X * 1.702)
    else:
        raise NotImplementedError
    

class TestAllExpertMLPsKernel:


    def setup_method(self):
        # Set seed
        torch.manual_seed(0)

        # Capture and set env vars
        self.platform_target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE")
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn3"
        self.neuron_cc_flags = os.environ.get("NEURON_CC_FLAGS")
        os.environ["NEURON_CC_FLAGS"] = "--target=trn3"
        self.enable_mla_arch = os.environ.get("ENABLE_MLA_ARCH")
        os.environ["ENABLE_MLA_ARCH"] = "core_v4"

        self.xla_ir_debug = os.environ.get("XLA_IR_DEBUG")
        os.environ["XLA_IR_DEBUG"] = "1"
        self.xla_hlo_debug = os.environ.get("XLA_HLO_DEBUG")
        os.environ["XLA_HLO_DEBUG"] = "1"

        self.ocp_enabled = os.environ.get("NEURON_RT_ENABLE_OCP")
        os.environ["NEURON_RT_ENABLE_OCP"] = "1"
        self.ocp_saturation_enabled = os.environ.get("NEURON_RT_ENABLE_OCP_SATURATION")
        os.environ["NEURON_RT_ENABLE_OCP_SATURATION"] = "1"

        # Uncomment for verbose PJRT/NRT logs during debug
        # self.tf_log_level = os.environ.get("TF_CPP_MIN_LOG_LEVEL")
        # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        # self.tf_vmodule = os.environ.get("TF_CPP_VMODULE")
        # os.environ["TF_CPP_VMODULE"] = "neuronpjrt=1"
        # self.nrt_log_level = os.environ.get("NEURON_RT_LOG_LEVEL")
        # os.environ["NEURON_RT_LOG_LEVEL"] = "DEBUG"


    def teardown_method(self):
        # Reset all env vars
        if self.platform_target is None:
            os.environ.pop("NEURON_PLATFORM_TARGET_OVERRIDE", None)
        else:
            os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = self.platform_target
        if self.neuron_cc_flags is None:
            os.environ.pop("NEURON_CC_FLAGS", None)
        else:
            os.environ["NEURON_CC_FLAGS"] = self.neuron_cc_flags
        if self.enable_mla_arch is None:
            os.environ.pop("ENABLE_MLA_ARCH", None)
        else:
            os.environ["ENABLE_MLA_ARCH"] = self.enable_mla_arch
        if self.xla_ir_debug is None:
            os.environ.pop("XLA_IR_DEBUG", None)
        else:
            os.environ["XLA_IR_DEBUG"] = self.xla_ir_debug
        if self.xla_hlo_debug is None:
            os.environ.pop("XLA_HLO_DEBUG", None)
        else:
            os.environ["XLA_HLO_DEBUG"] = self.xla_hlo_debug
        if self.ocp_enabled is None:
            os.environ.pop("NEURON_RT_ENABLE_OCP", None)
        else:
            os.environ["NEURON_RT_ENABLE_OCP"] = self.ocp_enabled
        if self.ocp_saturation_enabled is None:
            os.environ.pop("NEURON_RT_ENABLE_OCP_SATURATION", None)
        else:
            os.environ["NEURON_RT_ENABLE_OCP_SATURATION"] = self.ocp_saturation_enabled
        
        # Uncomment for verbose PJRT/NRT logs during debug
        # if self.tf_log_level is None:
        #     os.environ.pop("TF_CPP_MIN_LOG_LEVEL", None)
        # else:
        #     os.environ["TF_CPP_MIN_LOG_LEVEL"] = self.tf_log_level
        # if self.tf_vmodule is None:
        #     os.environ.pop("TF_CPP_VMODULE", None)
        # else:
        #     os.environ["TF_CPP_VMODULE"] = self.tf_vmodule
        # if self.nrt_log_level is None:
        #     os.environ.pop("NEURON_RT_LOG_LEVEL", None)
        # else:
        #     os.environ["NEURON_RT_LOG_LEVEL"] = self.nrt_log_level
    

    def generate_random_weights(self, num_experts, hidden_dim, intermediate_dim, scale_ranges, dtype=torch.bfloat16):
        # Generate MXFP4 checkpoint
        W_gate_uint16, W_up_uint16, W_down_uint16 = self.generate_random_uint16_weights(num_experts, hidden_dim, intermediate_dim)
        scale_gate, scale_up, scale_down = self.generate_random_scales(num_experts, hidden_dim, intermediate_dim, scale_ranges)

        # Descale MXFP4 -> bf16
        W_gate_bf16 = dequantize_mx_tensor(W_gate_uint16, scale_gate, dtype=torch.bfloat16)
        W_up_bf16 = dequantize_mx_tensor(W_up_uint16, scale_up, dtype=torch.bfloat16)
        W_down_bf16 = dequantize_mx_tensor(W_down_uint16, scale_down, dtype=torch.bfloat16)
        
        # Only return bf16 for bf16 case
        if dtype == torch.bfloat16:
            return W_gate_bf16, W_up_bf16, W_down_bf16

        # return MXFP4 and bf16 weights for accuracy eval
        else:
            return W_gate_uint16, W_up_uint16, W_down_uint16, scale_gate, scale_up, scale_down, W_gate_bf16, W_up_bf16, W_down_bf16
    

    def generate_random_uint16_weights(self, num_experts, hidden_dim, intermediate_dim):
        UINT16_MIN, UINT16_MAX = 0, 0xFFFF # 2**16
        W_gate_uint16 = torch.randint(UINT16_MIN, UINT16_MAX, (num_experts, intermediate_dim, hidden_dim // 4), dtype=torch.uint16)
        W_up_uint16 = torch.randint(UINT16_MIN, UINT16_MAX, (num_experts, intermediate_dim, hidden_dim // 4), dtype=torch.uint16)
        W_down_uint16 = torch.randint(UINT16_MIN, UINT16_MAX, (num_experts, hidden_dim, intermediate_dim // 4), dtype=torch.uint16)
        return W_gate_uint16, W_up_uint16, W_down_uint16


    def generate_random_scales(self, num_experts, hidden_dim, intermediate_dim, scale_ranges):
        (gate_min, gate_max), (up_min, up_max), (down_min, down_max) = scale_ranges
        scale_gate = torch.randint(gate_min, gate_max, (num_experts, intermediate_dim, hidden_dim // 32), dtype=torch.uint8)
        scale_up = torch.randint(up_min, up_max, (num_experts, intermediate_dim, hidden_dim // 32), dtype=torch.uint8)
        scale_down = torch.randint(down_min, down_max, (num_experts, hidden_dim, intermediate_dim // 32), dtype=torch.uint8)
        return scale_gate, scale_up, scale_down
    

    def generate_random_biases(self, num_experts, hidden_dim, intermediate_dim, bias_ranges, dtype=torch.bfloat16):
        (gate_min, gate_max), (up_min, up_max), (down_min, down_max) = bias_ranges
        bias_gate = torch.randn(num_experts, intermediate_dim, dtype=dtype).clamp(gate_min, gate_max)
        bias_up = torch.randn(num_experts, intermediate_dim, dtype=dtype).clamp(up_min, up_max)
        bias_down = torch.randn(num_experts, hidden_dim, dtype=dtype).clamp(down_min, down_max)
        return bias_gate, bias_up, bias_down
    

    def generate_normalized_inputs(self, shape, mean, std, min, max, per_token_sum, fuzz_scale_factor=0.001, dtype=torch.bfloat16):
        # start from common starting point for all tokens, then fuzz
        tensor = torch.normal(mean, std, (1, shape[1]), dtype=dtype)
        tensor = tensor.broadcast_to(shape)
        noise_scale = fuzz_scale_factor * tensor.abs().mean()
        tensor = tensor + torch.randn_like(tensor) * noise_scale

        tensor = torch.clamp(tensor, min, max)

        # Normalize along the last dimension to have the specified sum
        current_sums = tensor.sum(dim=-1, keepdim=True)
        tensor = tensor * (per_token_sum / current_sums)
        return tensor
    

    def generate_router_logits(self, shape, topk_idx, per_token_sum, dtype=torch.bfloat16):
        router_logits = torch.zeros(shape, dtype=dtype)
        topk_idx = torch.tensor(topk_idx)
        for i in range(len(topk_idx)):
            # Give slightly different values to maintain order
            expert_idx = topk_idx[i]
            router_logits[:, expert_idx] = 10.0 - i

        # Normalize along the last dimension to have the specified sum
        current_sums = router_logits.sum(dim=-1, keepdim=True)
        router_logits = router_logits * (per_token_sum / current_sums)
        return router_logits


    def generate_normal_clamped_outputs(self, shape, mean, std, min, max, dtype=torch.bfloat16):
        tensor = torch.normal(mean, std, shape, dtype=dtype)
        tensor = torch.clamp(tensor, min, max)
        return tensor


    @pytest.mark.parametrize("H", [1536, 3072])
    @pytest.mark.parametrize("T", [32, 64, 128, 256, 512])
    @pytest.mark.trn3
    @pytest.mark.skipif(not RUN_TRN3_TESTS, reason="Skip test when not run on trn3 hardware!")
    def test_swizzle_quantize_mx_input(self, T, H, dtype=torch.bfloat16):
        """
        Validates swizzle + quantize MX NKI function using torch implementation as reference.
        """

        # Needs to be imported after trn3 env vars are set
        from neuronx_distributed.kernels.expert_mlps_mx.swizzle_quantize_mx import swizzle_quantize_mx_input

        @nki.jit(platform="trn3")
        def swizzle_quantize_mx_input_nki_wrapper(input):
            """
            Thin wrapper of NKI swizzle + QMX function, used for testing.
            """
            input_sb = nl.load(input)
            out_quant_sb, out_scale_sb = swizzle_quantize_mx_input(input_sb=input_sb)
            # View output as uint32 so that the x4 dtype can be traced
            out_quant_sb = out_quant_sb.view(nl.uint32)
            out_quant_hbm = nl.ndarray(out_quant_sb.shape, out_quant_sb.dtype, buffer=nl.hbm)
            out_scale_hbm = nl.ndarray(out_scale_sb.shape, out_scale_sb.dtype, buffer=nl.hbm)
            nl.store(out_quant_hbm[...], out_quant_sb[...])
            nl.store(out_scale_hbm[...], out_scale_sb[...])
            return out_quant_hbm, out_scale_hbm

        assert T >= 32
        assert H % 512 == 0
        # [32_T * 4_H, T/32, H/512, 16_H * 8_H]
        # NOTE: for debug can use arange to figure out if indices are getting switched around
        X = torch.randn(size=(T, H), dtype=dtype)
        
        # Use torch as reference output
        golden_quant, golden_scale = quantize_mxfp8(swizzle_tensor(X.T))

        # Prep NKI inputs
        # [T, H] ->
        # [T/32, 32_T, H/512, 16_H, 8_H, 4_H] ->
        # [32_T, 4_H, T/32, H/512, 16_H, 8_H] ->
        # [128, T/32, H/512, 128]
        X_nki = X.clone().reshape(T//32, 32, H//512, 16, 8, 4)
        X_nki = X_nki.permute(1, 5, 0, 2, 3, 4).contiguous()
        X_nki = X_nki.view(128, T//32, H//512, 128)

        # Call NKI kernel
        print(f"Testing with {X_nki.shape=}")
        nki_quant, nki_scale = nki.simulate_kernel(
            swizzle_quantize_mx_input_nki_wrapper,
            X_nki.view(torch.uint16).numpy().view(ml_dtypes.bfloat16),
        )
        nki_quant, nki_scale = torch.from_numpy(nki_quant), torch.from_numpy(nki_scale)

        # Convert NKI outputs -> torch shape
        # [128, H/512, T] -> [H/4, T]
        nki_quant = nki_quant.transpose(0, 1).reshape(H//4, T)
        scale_p_indices = torch.tensor([*range(0, 4), *range(32, 36), *range(64, 68), *range(96, 100)])
        nki_scale = nki_scale[scale_p_indices, ...].transpose(0, 1).reshape(H//32, T)

        # Validate output exactly matches
        assert torch.equal(nki_quant, golden_quant), f"Quantized outputs don't match! \n{golden_quant=}\n{nki_quant=}"
        assert torch.equal(nki_scale, golden_scale), f"Quantization scales don't match! \n{golden_scale=}\n{nki_scale=}"
        print("Test passes!")


    # TODO: add lnc2 unit tests
    @pytest.mark.parametrize(
        "T,E,H,I,scale_ranges,bias_ranges,norm_input_stats,router_input_stats,clamp_upper_limit,clamp_lower_limit,hidden_act_fn,accumulation_dtype,output_dtype,atol,rtol", [
            # fp32 PSUM
            pytest.param(64, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, 7.0, None, ActFnType.Swish, torch.float32, torch.bfloat16, 3.125e-2, 9.765625e-3, id="T64_E1_fp32_bf16_gate"),
            pytest.param(128, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, 7.0, None, ActFnType.Swish, torch.float32, torch.bfloat16, 3.125e-2, 9.765625e-3, id="T128_E1_fp32_bf16_gate"),
            pytest.param(512, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, 7.0, None, ActFnType.Swish, torch.float32, torch.bfloat16, 3.125e-2, 9.765625e-3, id="T512_E1_fp32_bf16_gate"),
            pytest.param(64, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, 8.0, -6.0, None, torch.float32, torch.bfloat16, 1.5625e-2, 3.326416015625e-3, id="T64_E1_fp32_bf16_up"),
            pytest.param(128, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, 8.0, -6.0, None, torch.float32, torch.bfloat16, 1.5625e-2, 3.326416015625e-3, id="T128_E1_fp32_bf16_up"),
            pytest.param(512, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, 8.0, -6.0, None, torch.float32, torch.bfloat16, 1.5625e-2, 3.326416015625e-3, id="T512_E1_fp32_bf16_up"),

            # bf16 PSUM
            pytest.param(64, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, 7.0, None, ActFnType.Swish, torch.bfloat16, torch.bfloat16, 4.6875e-2, 1.46484375e-2, id="T64_E1_bf16_bf16_gate"),
            pytest.param(128, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, 7.0, None, ActFnType.Swish, torch.bfloat16, torch.bfloat16, 4.6875e-2, 1.46484375e-2, id="T128_E1_bf16_bf16_gate"),
            pytest.param(512, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, 7.0, None, ActFnType.Swish, torch.bfloat16, torch.bfloat16, 4.6875e-2, 1.46484375e-2, id="T512_E1_bf16_bf16_gate"),
            pytest.param(64, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, 8.0, -6.0, None, torch.bfloat16, torch.bfloat16, 3.125e-2, 6.65283203125e-3, id="T64_E1_bf16_bf16_up"),
            pytest.param(128, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, 8.0, -6.0, None, torch.bfloat16, torch.bfloat16, 3.125e-2, 6.65283203125e-3, id="T128_E1_bf16_bf16_up"),
            pytest.param(512, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, 8.0, -6.0, None, torch.bfloat16, torch.bfloat16, 3.90625e-2, 8.30078125e-3, id="T512_E1_bf16_bf16_up"),
        ]
    )
    @pytest.mark.trn3
    @pytest.mark.skipif(not RUN_TRN3_TESTS, reason="Skip test when not run on trn3 hardware!")
    def test_gate_up_projection_mx_lhs_rhs_swap_lnc1(self, T, E, H, I, clamp_upper_limit, clamp_lower_limit, hidden_act_fn, scale_ranges, bias_ranges, norm_input_stats, router_input_stats, accumulation_dtype, output_dtype, atol, rtol):
        """
        Validates NKI implementation of gate/up proj in MX, using torch implementation as reference.
        """

        # Needs to be imported after trn3 env vars are set
        from neuronx_distributed.kernels.expert_mlps_mx.gate_up_proj import gate_up_projection_mx_lhs_rhs_swap
        from neuronx_distributed.kernels.expert_mlps_mx.constants import N_BTYES_TO_NKI_X4_DTYPE_MAP

        @nki.jit(platform="trn3")
        def gate_up_projection_lhs_rhs_swap_nki_wrapper(
            input,
            input_scale,
            weight,
            scale,
            bias,
            clamp_upper_limit,
            clamp_lower_limit,
            hidden_act_fn,
            psum_accumulation_dtype=nl.float32,
        ):
            """
            Gate/up projection NKI wrapper that loads quantized inputs and validates gate_up_projection results.
            """

            input_x4_dtype = N_BTYES_TO_NKI_X4_DTYPE_MAP[np.dtype(input.dtype).itemsize]
            input_sb = nl.load(input)
            input_sb = input_sb.view(input_x4_dtype)
            input_scale_sb = nl.ndarray(input_sb.shape, input_scale.dtype, buffer=nl.sbuf)
            for quadrant in nl.affine_range(4):
                input_scale_sb[nl.ds(32*quadrant, 4), :, :] = nl.load(input_scale[nl.ds(4*quadrant, 4), :, :])

            proj_res_sb = gate_up_projection_mx_lhs_rhs_swap(
                input_sb=input_sb,
                input_scale_sb=input_scale_sb,
                weight=weight,
                weight_scale=scale,
                bias=bias,
                clamp_upper_limit=clamp_upper_limit,
                clamp_lower_limit=clamp_lower_limit,
                hidden_act_fn=hidden_act_fn,
                psum_accumulation_dtype=psum_accumulation_dtype,
            )

            proj_res_hbm = nl.ndarray(proj_res_sb.shape, dtype=proj_res_sb.dtype, buffer=nl.shared_hbm)
            nl.store(proj_res_hbm, proj_res_sb)
            return proj_res_hbm

        # Generate test data
        weight, _, _, scale, _, _, _, _, _ = self.generate_random_weights(E, H, I, scale_ranges, dtype=torch.uint16)
        bias, _, _ = self.generate_random_biases(E, H, I, bias_ranges)
        
        # Create simulated norm tensor, transpose, swizzle, QMX
        norm_out = self.generate_normalized_inputs((T, H), *norm_input_stats, dtype=torch.bfloat16)
        input_quant, input_scale = quantize_mxfp8(swizzle_tensor(norm_out.T))

        # Prep torch inputs
        # [E, I, H/4] -> [E, H/4, I]
        W_torch = weight.transpose(1, 2)
        # [E, I, H/32] -> [E, H/32, I]
        scale_torch = scale.transpose(1, 2)
        bias_torch = bias

        # Run torch reference
        golden_result = gate_up_projection_torch(
            input=input_quant,
            input_scale=input_scale,
            weight=W_torch,
            scale=scale_torch,
            bias=bias_torch,
            matmul_accumulation_dtype=accumulation_dtype,
            matmul_output_dtype=output_dtype,
        )

        # Clamp torch reference
        if clamp_lower_limit or clamp_upper_limit:
            golden_result.clamp_(min=clamp_lower_limit, max=clamp_upper_limit)

        # Activation fn on torch reference
        golden_result = _hidden_act_fn(golden_result, hidden_act_fn)

        # Prep NKI inputs
        # TODO: remove check for evenly divisible input shapes once we add support for H, I % 512 != 0
        assert T % 32 == 0
        assert H % 512 == 0
        assert I % 512 == 0

        # [H/4, T] -> [16_H * 8_H, H/512, T]
        input_quant_nki = input_quant\
            .reshape(H//512, 128, T)\
            .permute(1, 0, 2)\
            .reshape(128, H//512, T)
        # [H/32, T] -> [16_H, H/512, T]
        input_scale_nki = input_scale\
            .reshape(H//512, 16, T)\
            .permute(1, 0, 2)\
            .reshape(16, H//512, T)
        # [E, H/4, I] -> [16_H * 8_H, H/512, I/512 * 4_I * 16_I * 8_I]
        W_nki = W_torch.squeeze(0)\
            .reshape(H//512, 128, I//512, 16, 8, 4)\
            .permute(1, 0, 2, 5, 3, 4)\
            .reshape(128, H//512, I)
        # [E, H/32, I] -> [16_H, H/512, I/512 * 4_I * 16_I * 8_I]
        scale_nki = scale_torch.squeeze(0)\
            .reshape(H//512, 16, I//512, 16, 8, 4)\
            .permute(1, 0, 2, 5, 3, 4)\
            .reshape(16, H//512, I)
        # [I] -> [16_I * 8_I, I//512, 4_I]
        bias_nki = bias_torch.squeeze(0)\
            .reshape(I//512, 16, 8, 4)\
            .permute(1, 2, 0, 3)\
            .reshape(128, I//512, 4)
        USE_BIAS = bias_nki.sum().item() != 0

        # Execute NKI kernel
        simulate = False
        if simulate:
            # Sadly this does not work w/ fp8_x4 dtype view inside wrapper kernel: P305560288
            nki_result = nki.simulate_kernel(
                gate_up_projection_lhs_rhs_swap_nki_wrapper,
                input=input_quant_nki.numpy(),
                input_scale=input_scale_nki.numpy(),
                weight=W_nki.numpy(),
                scale=scale_nki.numpy(),
                bias=bias_nki.view(torch.uint16).numpy().view(ml_dtypes.bfloat16) if USE_BIAS else None,
                clamp_upper_limit=clamp_upper_limit,
                clamp_lower_limit=clamp_lower_limit,
                hidden_act_fn=hidden_act_fn,
                psum_accumulation_dtype=NKI_DTYPE_MAPPING[accumulation_dtype],
            )
            nki_result = torch.from_numpy(nki_result.view(np.uint16)).view(torch.uint16)
        else:
            device = xla_model.xla_device()
            nki_result = gate_up_projection_lhs_rhs_swap_nki_wrapper(
                input=input_quant_nki.to(device),
                input_scale=input_scale_nki.to(device),
                weight=W_nki.to(device),
                scale=scale_nki.to(device),
                bias=bias_nki.to(device) if USE_BIAS else None,
                clamp_upper_limit=clamp_upper_limit,
                clamp_lower_limit=clamp_lower_limit,
                hidden_act_fn=hidden_act_fn,
                psum_accumulation_dtype=NKI_DTYPE_MAPPING[accumulation_dtype],
            )
            nki_result = nki_result.cpu()

        # Post-process shapes
        # Squeeze E dim on golden [T, 1, I] -> [T, I]
        golden_result = golden_result.squeeze(1)

        # [16_I * 8_I, I/512, T, 4_I] -> [T, I/512 * 16_I * 8_I * 4_I]
        nki_result = nki_result.permute(2, 1, 0, 3).reshape(T, I)

        # Validate accuracy
        print(f"{golden_result=}")
        print(f"{golden_result.shape=}")
        print(f"{nki_result=}")
        print(f"{nki_result.shape=}")

        torch_neuronx.testing.assert_close(nki_result, golden_result, atol=atol, rtol=rtol)
        print("Test passes!")


    # TODO: add lnc2 unit tests
    @pytest.mark.parametrize(
        "T,E,H,I,scale_ranges,bias_ranges,norm_input_stats,router_input_stats,accumulation_dtype,output_dtype,atol,rtol", [
            # fp32 PSUM
            pytest.param(64, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, torch.float32, torch.bfloat16, 1.5625e-2, 4.33349609375e-3, id="T64_E1_fp32_bf16"),
            pytest.param(128, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, torch.float32, torch.bfloat16, 1.5625e-2, 4.33349609375e-3, id="T128_E1_fp32_bf16"),
            pytest.param(512, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, torch.float32, torch.bfloat16, 1.5625e-2, 4.33349609375e-3, id="T512_E1_fp32_bf16"),

            # bf16 PSUM
            pytest.param(64, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, torch.bfloat16, torch.bfloat16, 3.125e-2, 8.6669921875e-3, id="T64_E1_bf16_bf16"),
            pytest.param(128, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, torch.bfloat16, torch.bfloat16, 3.125e-2, 8.6669921875e-3, id="T128_E1_bf16_bf16"),
            pytest.param(512, *GPT_OSS_LAYER_0_ONE_EXPERT_PADDED_CONFIG, torch.bfloat16, torch.bfloat16, 3.125e-2, 8.6669921875e-3, id="T512_E1_bf16_bf16"),
        ]
    )
    @pytest.mark.parametrize("activation_output_stats", [GPT_OSS_LAYER_0_ACT_OUT_CONFIG])
    @pytest.mark.trn3
    @pytest.mark.skipif(not RUN_TRN3_TESTS, reason="Skip test when not run on trn3 hardware!")
    def test_down_projection_mx_lnc1(self, T, E, H, I, scale_ranges, bias_ranges, norm_input_stats, router_input_stats, activation_output_stats, accumulation_dtype, output_dtype, atol, rtol):
        """
        Validates NKI implementation of down proj in MX, using torch implementation as reference.
        """

        # Needs to be imported after trn3 env vars are set
        from neuronx_distributed.kernels.expert_mlps_mx.down_proj import down_projection_mx
        from neuronx_distributed.kernels.expert_mlps_mx.constants import N_BTYES_TO_NKI_X4_DTYPE_MAP

        @nki.jit(platform="trn3")
        def down_projection_nki_wrapper(
            act,
            act_scale,
            weight,
            scale,
            bias,
            psum_accumulation_dtype=nl.float32,
        ):
            """
            Down projection NKI wrapper that loads quantized activations and validates down_projection results.
            """
            act_x4_dtype = N_BTYES_TO_NKI_X4_DTYPE_MAP[np.dtype(act.dtype).itemsize]
            act_sb = nl.load(act)
            act_sb = act_sb.view(act_x4_dtype)
            act_scale_sb = nl.ndarray(act_sb.shape, act_scale.dtype, buffer=nl.sbuf)
            for quadrant in nl.affine_range(4):
                act_scale_sb[nl.ds(32*quadrant, 4), :, :] = nl.load(act_scale[nl.ds(4*quadrant, 4), :, :])


            proj_res_sb = down_projection_mx(
                act_sb=act_sb,
                act_scale_sb=act_scale_sb,
                weight=weight,
                weight_scale=scale,
                bias=bias,
                psum_accumulation_dtype=psum_accumulation_dtype,
            )

            proj_res_hbm = nl.ndarray(proj_res_sb.shape, dtype=proj_res_sb.dtype, buffer=nl.shared_hbm)
            nl.store(proj_res_hbm, proj_res_sb)
            return proj_res_hbm


        # Generate test data
        _, _, weight, _, _, scale, _, _, _ = self.generate_random_weights(E, H, I, scale_ranges, dtype=torch.uint16)
        _, _, bias = self.generate_random_biases(E, H, I, bias_ranges)
        
        # Create simulated norm tensor, transpose, swizzle, QMX
        # TODO[impl] generalize to use E > 1 and have E dim
        act_out = self.generate_normal_clamped_outputs((T, I), *activation_output_stats, dtype=torch.bfloat16)
        act_quant, act_scale = quantize_mxfp8(swizzle_tensor(act_out.T))
        act_quant = act_quant

        # Prep torch inputs, add E dim
        # [I/4, T] -> [E, I/4, T]
        act_quant_torch = act_quant.unsqueeze(0)
        # [I/32, T] -> [E, I/32, T]
        act_scale_torch = act_scale.unsqueeze(0)
        # [E, H, I/4] -> [E, I/4, H]
        W_torch = weight.transpose(1, 2)
        # [E, H, I/32] -> [E, I/32, H]
        scale_torch = scale.transpose(1, 2)
        bias_torch = bias

        print(f"{act_quant_torch.shape=}")
        print(f"{act_scale_torch.shape=}")
        print(f"{W_torch.shape=}")
        print(f"{scale_torch.shape=}")
        print(f"{bias_torch.shape=}")

        # Run torch reference
        golden_result = down_projection_torch(
            act=act_quant_torch,
            act_scale=act_scale_torch,
            weight=W_torch,
            scale=scale_torch,
            bias=bias_torch,
            matmul_accumulation_dtype=accumulation_dtype,
            matmul_output_dtype=output_dtype,
        )

        # Prep NKI inputs
        # [I/4, T] -> [16_I * 8_I, I/512, T]
        act_quant_nki = act_quant\
            .reshape(I//512, 128, T)\
            .permute(1, 0, 2)
        # [I/32, T] -> [16_I, I/512, T]
        act_scale_nki = act_scale\
            .reshape(I//512, 16, T)\
            .permute(1, 0, 2)
        # [E, I/4, H] -> [16_I * 8_I, I/512, H]
        W_nki = W_torch.squeeze(0)\
            .reshape(I//512, 128, H)\
            .permute(1, 0, 2)
        # [E, I/32, H] -> [16_I, I/512, H]
        scale_nki = scale_torch.squeeze(0)\
            .reshape(I//512, 16, H)\
            .permute(1, 0, 2)
        bias_nki = bias
        USE_BIAS = bias_nki.sum().item() != 0

        # Run NKI kernel
        simulate = False
        if simulate:
            # Sadly this does not work w/ fp8_x4 dtype view inside wrapper kernel: P305560288
            nki_result = nki.simulate_kernel(
                down_projection_nki_wrapper,
                act=act_quant_nki.numpy(),
                act_scale=act_scale_nki.numpy(),
                weight=W_nki.numpy(),
                scale=scale_nki.numpy(),
                bias=bias_nki.view(torch.uint16).numpy().view(ml_dtypes.bfloat16) if USE_BIAS else None,
                psum_accumulation_dtype=NKI_DTYPE_MAPPING[accumulation_dtype],
            )
            nki_result = torch.from_numpy(nki_result.view(np.uint16)).view(torch.bfloat16)
        else:
            device = xla_model.xla_device()
            nki_result = down_projection_nki_wrapper(
                act=act_quant_nki.to(device),
                act_scale=act_scale_nki.to(device),
                weight=W_nki.to(device),
                scale=scale_nki.to(device),
                bias=bias_nki.to(device) if USE_BIAS else None,
                psum_accumulation_dtype=NKI_DTYPE_MAPPING[accumulation_dtype],
            )
            nki_result = nki_result.cpu()

        # Post-process shapes
        # Squeeze E dim on golden [T, 1, H] -> [T, H]
        golden_result = golden_result.squeeze(1)

        # [128_T, T/128, H] -> [T, H]
        nki_result = nki_result.permute(1, 0, 2).reshape(T, I)

        # Validate accuracy
        print(f"{golden_result=}")
        print(f"{golden_result.shape=}")
        print(f"{nki_result=}")
        print(f"{nki_result.shape=}")

        torch_neuronx.testing.assert_close(nki_result, golden_result, atol=atol, rtol=rtol)
        print("Test passes!")


if __name__ == "__main__":
    pytest.main([__file__, '-v'])