import pytest
import torch

from neuronx_distributed.experimental.quantization.microscaling.expert_mlps_mx import (
    all_expert_mlps_bf16,
    select_expert_mlps_bf16,
    all_expert_mlps_act_mxfp8_w_mxfp4,
)
from neuronx_distributed.experimental.quantization.microscaling.mx_torch import dequantize_mx_tensor

# use values from real weights
GPT_OSS_LAYER_0_CONFIG = (
    128, 2880, 2880,
    ((119, 123), (119, 123), (119, 123)),
    ((-2.0625, 0.5234375), (-1.4375, 0.90625), (-1.6328125, 1.4375)),
    (0.00799560546875, 0.292969, -1.3671875, 1.6484375, 23.0),
    ([54, 11, 23, 58], 16.8750),
)

class TestGptOSSExpertMLPs:
    
    def setup_method(self):
        torch.manual_seed(0)
    
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

    @pytest.mark.parametrize(
        "T,atol,rtol", [
            # NOTE: outputs match exactly with real weights (see integration test folder)
            pytest.param(64, 1.6e-2, 1.69e-1),
            pytest.param(512, 1.6e-2, 12.82, marks=pytest.mark.skip(reason="Causes pytest worker crash on trn1.2xl")),
        ]
    )
    @pytest.mark.parametrize(
        "E,H,I,scale_ranges,bias_ranges,norm_input_stats,router_input_stats", [
            pytest.param(*GPT_OSS_LAYER_0_CONFIG),
        ]
    )
    def test_all_experts_vs_select_experts_bf16_random_weights_cpu(self, T, E, H, I, scale_ranges, bias_ranges, norm_input_stats, router_input_stats, atol, rtol, dtype=torch.bfloat16):  # noqa: E741
        # Generate random weights and biases
        W_gate, W_up, W_down = self.generate_random_weights(E, H, I, scale_ranges, dtype=dtype)
        bias_gate, bias_up, bias_down = self.generate_random_biases(E, H, I, bias_ranges, dtype=dtype)
        
        # Generate normalized inputs and router logits
        rmsnorm_out = self.generate_normalized_inputs((T, H), *norm_input_stats, dtype=dtype)
        router_logits = self.generate_router_logits((T, E), *router_input_stats, dtype=dtype)

        # Test both implementations
        result_all_experts = all_expert_mlps_bf16(
            rmsnorm_out,
            router_logits,
            W_gate, 
            W_up, 
            W_down,
            bias_gate,
            bias_up,
            bias_down,
        )
        result_select_experts = select_expert_mlps_bf16(
            rmsnorm_out,
            router_logits,
            W_gate, 
            W_up, 
            W_down,
            bias_gate,
            bias_up,
            bias_down,
        )
        
        torch.testing.assert_close(result_all_experts, result_select_experts, atol=atol, rtol=rtol)
        print("Tensors match!")

    @pytest.mark.parametrize(
        "T,accumulation_dtype,output_dtype,atol,rtol", [
            # NOTE: accuracy is much better with real weights (see integration test folder)
            # NOTE: rtol is higher because we are comparing expert MLPs output without residual add
            pytest.param(64, torch.float32, torch.bfloat16, 1.41e-1, 11264.0),
            pytest.param(64, torch.bfloat16, torch.bfloat16, 1.57e-1, 11328.0),
            pytest.param(512, torch.float32, torch.bfloat16, 1.57e-1, 13632.0),
            pytest.param(512, torch.bfloat16, torch.bfloat16, 1.67e-1, 13824.0),
        ]
    )
    @pytest.mark.parametrize(
        "E,H,I,scale_ranges,bias_ranges,norm_input_stats,router_input_stats", [
            # use values from real weights
            pytest.param(*GPT_OSS_LAYER_0_CONFIG),
        ]
    )
    def test_all_experts_bf16_vs_mxfp4_random_weights_cpu(self, T, E, H, I, scale_ranges, bias_ranges, norm_input_stats, router_input_stats, accumulation_dtype, output_dtype, atol, rtol):  # noqa: E741
        # Generate random weights and biases
        FP4_X4_DTYPE = torch.uint16
        W_gate_uint16, W_up_uint16, W_down_uint16, scale_gate, scale_up, scale_down, W_gate_bf16, W_up_bf16, W_down_bf16 = \
            self.generate_random_weights(E, H, I, scale_ranges, dtype=FP4_X4_DTYPE)
        bias_gate, bias_up, bias_down = self.generate_random_biases(E, H, I, bias_ranges, dtype=torch.bfloat16)
        
        # Generate normalized inputs and router logits
        rmsnorm_out = self.generate_normalized_inputs((T, H), *norm_input_stats, dtype=torch.bfloat16)
        router_logits = self.generate_router_logits((T, E), *router_input_stats, dtype=torch.bfloat16)

        # Test both implementations
        result_all_experts_bf16 = all_expert_mlps_bf16(
            rmsnorm_out,
            router_logits,
            W_gate_bf16, 
            W_up_bf16,
            W_down_bf16,
            bias_gate,
            bias_up,
            bias_down,
        )
        result_all_experts_mx = all_expert_mlps_act_mxfp8_w_mxfp4(
            norm_out=rmsnorm_out,
            router_logits=router_logits,
            W_gate=W_gate_uint16, 
            W_up=W_up_uint16, 
            W_down=W_down_uint16, 
            scale_gate=scale_gate, 
            scale_up=scale_up, 
            scale_down=scale_down,
            bias_gate=bias_gate,
            bias_up=bias_up,
            bias_down=bias_down,
            matmul_accumulation_dtype=accumulation_dtype,
            matmul_output_dtype=output_dtype,
        )
        
        torch.testing.assert_close(result_all_experts_mx, result_all_experts_bf16, atol=atol, rtol=rtol)
        print("Tensors match!")

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
