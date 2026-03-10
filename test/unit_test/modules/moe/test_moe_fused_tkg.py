import pytest
from unittest.mock import Mock, patch

import torch
import neuronxcc.nki.language as nl

from neuronx_distributed.modules.moe.moe_fused_tkg import MoEFusedTKG, _convert_torch_dtype_to_nki_dtype

def test_convert_torch_dtype_to_nki_dtype_valid_dtypes():
    assert _convert_torch_dtype_to_nki_dtype(torch.float16) == nl.float16
    assert _convert_torch_dtype_to_nki_dtype(torch.bfloat16) == nl.bfloat16
    assert _convert_torch_dtype_to_nki_dtype(torch.float32) == nl.float32

def test_convert_torch_dtype_to_nki_dtype_invalid_dtype():
    with pytest.raises(AssertionError, match="expected dtype in"):
        _convert_torch_dtype_to_nki_dtype(torch.int32)

@pytest.fixture
def mock_moe_module():
    router = Mock()
    router.weight_T = torch.randn(128, 8)
    router.act_fn = "softmax"
    router.bias = False
    router.apply_act_fn_over_topk = True
    router.return_value = (
        torch.randn(16, 8),
        torch.randn(16, 8),
        torch.randint(0, 8, (16, 2))
    )
    
    expert_mlps = Mock()
    expert_mlps.routed_experts_mlp_config = Mock(
        hidden_size=128,
        top_k=2,
        num_experts=8,
        hidden_act="silu",
        glu_mlp=True,
        normalize_top_k_affinities=False,
        early_expert_affinity_modulation=False
    )
    expert_mlps.moe_expert_model_parallel_group = Mock()
    expert_mlps.moe_expert_model_parallel_group.size.return_value = 1
    expert_mlps.moe_tensor_model_parallel_group = Mock()
    expert_mlps.moe_tensor_model_parallel_group.size.return_value = 1
    expert_mlps.return_value = torch.randn(16, 128)
    
    config = Mock()
    config.quantized = False
    config.moe_fused_kernel_enabled = False
    
    module = MoEFusedTKG(
        router=router,
        expert_mlps=expert_mlps,
        config=config,
        sequence_dimension=0,
        shared_experts=None,
        post_attention_layernorm=None,
        return_router_logits=False,
        return_expert_index=False
    )
    
    module._router_topk = Mock(return_value=(
        torch.randn(16, 8),
        torch.randn(16, 8),
        torch.randint(0, 8, (16, 2))
    ))
    module._expert_mlp = Mock(return_value=torch.randn(4, 4, 128))
    
    return module

@patch('neuronx_distributed.parallel_layers.parallel_state.get_world_group')
@patch('neuronx_distributed.parallel_layers.mappings.reduce_from_tensor_model_parallel_region')
@patch('neuronx_distributed.parallel_layers.mappings.copy_to_tensor_model_parallel_region')
def test_forward_residual_add_without_fused(mock_copy, mock_reduce, mock_world_group, mock_moe_module):
    mock_world_group.return_value = Mock()
    mock_reduce.side_effect = lambda x, **kwargs: x
    mock_copy.side_effect = lambda x: x
    
    hidden_states = torch.randn(4, 4, 128)
    residual = torch.randn(4, 4, 128)
    
    mock_moe_module._can_use_fused_residual_add = Mock(return_value=False)
    mock_moe_module._can_use_nki_kernel = Mock(return_value=False)
    
    result = mock_moe_module.forward(hidden_states, residual=residual)
    
    assert len(result) == 2
    output, returned_residual = result
    assert output.shape == hidden_states.shape
    assert returned_residual.shape == residual.shape
    assert not torch.allclose(returned_residual, residual)

@patch('neuronx_distributed.parallel_layers.parallel_state.get_world_group')
@patch('neuronx_distributed.parallel_layers.mappings.reduce_from_tensor_model_parallel_region')
@patch('neuronx_distributed.parallel_layers.mappings.copy_to_tensor_model_parallel_region')
def test_forward_residual_none(mock_copy, mock_reduce, mock_world_group, mock_moe_module):
    mock_world_group.return_value = Mock()
    mock_reduce.side_effect = lambda x, **kwargs: x
    mock_copy.side_effect = lambda x: x
    
    hidden_states = torch.randn(4, 4, 128)
    
    mock_moe_module._can_use_fused_residual_add = Mock(return_value=False)
    mock_moe_module._can_use_nki_kernel = Mock(return_value=False)
    
    result = mock_moe_module.forward(hidden_states, residual=None)
    
    assert len(result) == 1
    output = result[0]
    assert output.shape == hidden_states.shape

@patch('neuronx_distributed.parallel_layers.parallel_state.get_world_group')
@patch('neuronx_distributed.parallel_layers.mappings.reduce_from_tensor_model_parallel_region')
@patch('neuronx_distributed.parallel_layers.mappings.copy_to_tensor_model_parallel_region')
def test_forward_residual_add_values(mock_copy, mock_reduce, mock_world_group, mock_moe_module):
    mock_world_group.return_value = Mock()
    mock_reduce.side_effect = lambda x, **kwargs: x
    mock_copy.side_effect = lambda x: x
    
    hidden_states = torch.ones(4, 4, 128)
    residual = torch.ones(4, 4, 128) * 2
    
    mock_moe_module._can_use_fused_residual_add = Mock(return_value=False)
    mock_moe_module._can_use_nki_kernel = Mock(return_value=False)
    mock_moe_module._expert_mlp = Mock(return_value=torch.zeros(4, 4, 128))
    
    result = mock_moe_module.forward(hidden_states, residual=residual)
    
    _, returned_residual = result
    expected_residual = hidden_states + residual
    assert torch.allclose(returned_residual, expected_residual)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])