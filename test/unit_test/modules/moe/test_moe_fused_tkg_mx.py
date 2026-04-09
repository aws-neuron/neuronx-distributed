import pytest
from unittest.mock import Mock

import torch
from neuronx_distributed.modules.moe.moe_fused_tkg_mx import MoEFusedTKGMX

@pytest.fixture
def mock_moe_mx_module():
    router = Mock()
    router.weight_T = torch.randn(128, 8)
    router.act_fn = "softmax"
    router.bias = False
    router.apply_act_fn_over_topk = True
    
    expert_mlps = Mock()
    expert_mlps.routed_experts_mlp_config = Mock(
        hidden_size=512,
        top_k=2,
        num_experts=8,
        hidden_act="silu",
        glu_mlp=True,
        normalize_top_k_affinities=False,
        early_expert_affinity_modulation=False,
        glu_type="swiglu",
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
        bias=True
    )
    expert_mlps.moe_expert_model_parallel_group = Mock()
    expert_mlps.moe_expert_model_parallel_group.size.return_value = 1
    expert_mlps.moe_tensor_model_parallel_group = Mock()
    expert_mlps.moe_tensor_model_parallel_group.size.return_value = 1
    expert_mlps.mlp_op = Mock()
    expert_mlps.mlp_op.gate_up_proj = Mock()
    expert_mlps.mlp_op.gate_up_proj.input_size = 512
    expert_mlps.mlp_op.down_proj = Mock()
    expert_mlps.mlp_op.down_proj.input_size_per_partition = 2048
    
    config = Mock()
    config.quantized = False
    config.moe_fused_kernel_enabled = False
    config.router_mm_dtype = torch.float32
    
    module = MoEFusedTKGMX(
        router=router,
        expert_mlps=expert_mlps,
        config=config,
        sequence_dimension=0,
        shared_experts=None,
        post_attention_layernorm=None,
        return_router_logits=False,
        return_expert_index=False
    )
    
    return module

def test_should_use_all_expert_above_threshold(mock_moe_mx_module):
    hidden_states = torch.randn(32, 32, 512)
    result = mock_moe_mx_module._should_use_all_expert(hidden_states)
    assert result

def test_should_use_all_expert_below_threshold(mock_moe_mx_module):
    hidden_states = torch.randn(1, 1, 512)
    result = mock_moe_mx_module._should_use_all_expert(hidden_states)
    assert not result

def test_should_use_all_expert_at_threshold(mock_moe_mx_module):
    mock_moe_mx_module.num_experts_per_tok = 2
    mock_moe_mx_module.num_local_experts = 8
    batch_size = 16
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, 512)
    result = mock_moe_mx_module._should_use_all_expert(hidden_states)
    assert result

def test_can_use_fused_residual_add_above_threshold(mock_moe_mx_module):
    hidden_states = torch.randn(32, 32, 512)
    assert mock_moe_mx_module._can_use_fused_residual_add(hidden_states)

def test_can_use_fused_residual_add_below_batch_threshold(mock_moe_mx_module):
    # batch_x_seq = 1 < 256, fails batch threshold
    hidden_states = torch.randn(1, 1, 512)
    assert not mock_moe_mx_module._can_use_fused_residual_add(hidden_states)

def test_can_use_fused_residual_add_at_batch_boundary(mock_moe_mx_module):
    # batch_x_seq = 16 * 16 = 256, exactly at threshold
    hidden_states = torch.randn(16, 16, 512)
    assert mock_moe_mx_module._can_use_fused_residual_add(hidden_states)

def test_can_use_fused_residual_add_just_below_batch_boundary(mock_moe_mx_module):
    # batch_x_seq = 15 * 17 = 255, just below 256 threshold
    hidden_states = torch.randn(15, 17, 512)
    assert not mock_moe_mx_module._can_use_fused_residual_add(hidden_states)

if __name__ == "__main__":
    pytest.main([__file__, '-v'])