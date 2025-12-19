import os
import pytest
import torch

from neuronx_distributed.experimental.quantization.microscaling.expert_mlps_mx import (
    all_expert_mlps_bf16,
    select_expert_mlps_bf16,
    all_expert_mlps_act_mxfp8_w_mxfp4,
)
from neuronx_distributed.quantization.microscaling.transform_weights import _pad_tensor

WEIGHTS_DIR_ENV_VAR = "_GPT_OSS_MOE_WEIGHTS_DIR"
GOLDENS_DIR_ENV_VAR = "_GPT_OSS_MOE_GOLDENS_DIR"

WEIGHT_SUBDIR_MAP = {
    torch.bfloat16: "bf16",
    torch.uint16: "mxfp4",
}

GPT_OSS_HEAD_DIM = 2880
GPT_OSS_HEAD_DIM_PADDED = 3072

class TestGptOSSExpertMLPs:
    
    def setup_method(self):
        # Assume that the env vars are set and that weights/goldens are downloaded
        self.weights_dir = os.environ.get(WEIGHTS_DIR_ENV_VAR, None)
        self.goldens_dir = os.environ.get(GOLDENS_DIR_ENV_VAR, None)

        if self.weights_dir is None:
            raise ValueError("Weights directory not found!")

        if self.goldens_dir is None:
            raise ValueError("Goldens directory not found!")

    def load_weights(self, layer, dtype):
        # descaled weights
        if dtype == torch.bfloat16:
            W_gate = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], f"model_layers_{layer}_mlp_experts_gate_proj_weight.pt"))
            W_up = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], f"model_layers_{layer}_mlp_experts_up_proj_weight.pt"))
            W_down = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], f"model_layers_{layer}_mlp_experts_down_proj_weight.pt"))
            return W_gate, W_up, W_down
        
        # MXFP4 weights
        elif dtype == torch.uint16:
            W_gate_uint16 = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], "model_layers_0_mlp_experts_gate_proj_weight.pt"))
            W_up_uint16 = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], "model_layers_0_mlp_experts_up_proj_weight.pt"))
            W_down_uint16 = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], "model_layers_0_mlp_experts_down_proj_weight.pt"))
            scale_gate = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], "model_layers_0_mlp_experts_gate_proj_scales.pt"))
            scale_up = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], "model_layers_0_mlp_experts_up_proj_scales.pt"))
            scale_down = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], "model_layers_0_mlp_experts_down_proj_scales.pt"))
            return W_gate_uint16, W_up_uint16, W_down_uint16, scale_gate, scale_up, scale_down
        
        else:
            raise ValueError(f"Invalid weight dtype: {torch.uint16}")

    def load_biases(self, layer, dtype):
        bias_gate = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], f"model_layers_{layer}_mlp_experts_gate_proj_bias.pt"))
        bias_up = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], f"model_layers_{layer}_mlp_experts_up_proj_bias.pt"))
        bias_down = torch.load(os.path.join(self.weights_dir, WEIGHT_SUBDIR_MAP[dtype], f"model_layers_{layer}_mlp_experts_down_proj_bias.pt"))
        return bias_gate, bias_up, bias_down

    def load_goldens(self, layer, token, T):
        rmsnorm_out = torch.load(os.path.join(self.goldens_dir, f"rmsnorm_out_layer_{layer}_token_{token}.pt"))
        router_logits = torch.load(os.path.join(self.goldens_dir, f"router_logits_layer_{layer}_token_{token}.pt"))
        moe_out = torch.load(os.path.join(self.goldens_dir, f"moe_out_layer_0_token_{token}.pt"))

        return rmsnorm_out[:T, :], router_logits[:T, :], moe_out[:T, :]

    # TODO: add layers 1, 2, 3
    @pytest.mark.parametrize("layer", [0])
    @pytest.mark.parametrize("token", [0])
    @pytest.mark.parametrize("T", [512])
    def test_all_experts_vs_select_experts_bf16_real_weights_cpu(self, layer, token, T):
        # Load weights
        W_gate, W_up, W_down = self.load_weights(layer, dtype=torch.bfloat16)
        bias_gate, bias_up, bias_down = self.load_biases(layer, dtype=torch.bfloat16)

        # Load inputs and router logits
        rmsnorm_out, router_logits, _ = self.load_goldens(layer, token, T)

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
        
        # Results should match exactly with real weights
        assert torch.equal(result_all_experts, result_select_experts)
        print("Tensors match!")

    # TODO: add layers 1, 2, 3
    @pytest.mark.parametrize("layer", [0])
    @pytest.mark.parametrize("token", [0])
    @pytest.mark.parametrize("T", [512])
    @pytest.mark.parametrize("atol,rtol", [(1.29e-1, 17.0)])
    def test_all_experts_bf16_vs_golden_output_real_weights_cpu(self, layer, token, T, atol, rtol):
        """
        Validate matchingness between bf16 and goldens.
        """
        
        # Load weights
        W_gate, W_up, W_down = self.load_weights(layer, dtype=torch.bfloat16)
        bias_gate, bias_up, bias_down = self.load_biases(layer, dtype=torch.bfloat16)

        # Load inputs and router logits
        rmsnorm_out, router_logits, moe_output = self.load_goldens(layer, token, T)

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
        
        torch.testing.assert_close(result_all_experts, moe_output, atol=atol, rtol=rtol)
        print("Tensors match!")

    # TODO: add layers 1, 2, 3
    @pytest.mark.parametrize("layer", [0])
    @pytest.mark.parametrize("token", [0])
    @pytest.mark.parametrize("T", [512])
    @pytest.mark.parametrize("output_dtype", [torch.bfloat16])
    @pytest.mark.parametrize(
        "accumulation_dtype,use_unbiased_scale_qmx_norm,golden_atol,golden_rtol,bf16_atol,bf16_rtol", [
            # NOTE: rtol is higher because we are comparing expert MLPs output without residual add
            # Using biased OCP scaling method for QMX(norm(hidden))
            pytest.param(torch.float32, False, 6.25e-2, 49.25, 1.328125e-1, 190.0),
            pytest.param(torch.bfloat16, False, 5.078125e-2, 52.5, 1.25e-1, 109.5),

            # Using unbiased OCP scaling method for QMX(norm(hidden))
            pytest.param(torch.float32, True, 3.906253e-2, 15.94, 1.25e-1, 152.0),
            pytest.param(torch.bfloat16, True, 6.25e-2, 83.0, 1.25e-1, 100.0),
        ]
    )
    def test_all_experts_mxfp4_vs_golden_and_bf16_output_real_weights_cpu(self, layer, token, T, accumulation_dtype, output_dtype, use_unbiased_scale_qmx_norm, golden_atol, golden_rtol, bf16_atol, bf16_rtol):        
        """
        Validate matchingness between:
        - MXFP4 vs goldens
        - MXFP4 vs bf16

        Combines both comparisons into one test to avoid having to call all_expert_mlps_act_mxfp8_w_mxfp4 2x.
        """
        # Load inputs and router logits
        rmsnorm_out, router_logits, moe_output = self.load_goldens(layer, token, T)
        rmsnorm_out_padded = _pad_tensor(rmsnorm_out, (T, GPT_OSS_HEAD_DIM_PADDED))

        # Load bf16 weights -- descaled straight from HF checkpoint
        W_gate_bf16, W_up_bf16, W_down_bf16 = self.load_weights(layer, dtype=torch.bfloat16)
        bias_gate, bias_up, bias_down = self.load_biases(layer, dtype=torch.bfloat16)
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

        # Load MXFP4 weights (bias loaded from MX dir to account for any padding)
        W_gate_uint16, W_up_uint16, W_down_uint16, scale_gate, scale_up, scale_down = self.load_weights(layer, dtype=torch.uint16)
        bias_gate, bias_up, bias_down = self.load_biases(layer, dtype=torch.uint16)
        result_all_experts_mx = all_expert_mlps_act_mxfp8_w_mxfp4(
            norm_out=rmsnorm_out_padded,
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
            use_unbiased_scale_qmx_norm=use_unbiased_scale_qmx_norm,
        )

        # Remove padding
        result_all_experts_mx = result_all_experts_mx[:, :GPT_OSS_HEAD_DIM]

        # Check MX output against goldens and bf16
        torch.testing.assert_close(result_all_experts_mx, moe_output, atol=golden_atol, rtol=golden_rtol)
        torch.testing.assert_close(result_all_experts_mx, result_all_experts_bf16, atol=bf16_atol, rtol=bf16_rtol)
        print("Tensors match!")

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
