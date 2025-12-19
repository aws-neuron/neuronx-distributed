import unittest
from unittest.mock import Mock, patch

from neuronx_distributed.modules.moe.expert_mlps_v2 import can_use_find_index_kernel
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2


class TestCanUseFindIndexKernel(unittest.TestCase):
    def test_can_use_find_index_kernel_configs(self):
        """Test can_use_find_index_kernel with various configurations"""
        
        # Format: (T, block_size, E_local, logical_nc_config, tp_size, ep_size, expected_result)
        test_configs = [
            # Valid cases - should return True
            (1024, 64, 16, 2, 2, 2, True),
            (2048, 32, 32, 2, 1, 4, True),
            (512, 16, 24, 2, 3, 2, True),
            (1536, 128, 64, 2, 4, 1, True),
            
            # Invalid: T not divisible by block_size
            (1000, 64, 16, 2, 2, 2, False),
            (1023, 32, 32, 2, 1, 4, False),
            (100, 16, 24, 2, 3, 2, False),
            
            # Invalid: logical_nc_config != 2
            (1024, 64, 16, 4, 2, 2, False),
            (2048, 32, 32, 1, 1, 4, False),
            (512, 16, 24, 3, 3, 2, False),
            
            # Invalid: E_local not divisible by logical_nc_config
            (1024, 64, 15, 2, 2, 2, False),  # 15 % 2 != 0
            (2048, 32, 31, 2, 1, 4, False),  # 31 % 2 != 0
            (512, 16, 23, 2, 3, 2, False),   # 23 % 2 != 0
            
            # Invalid: tp_size == 8 and ep_size == 8
            (1024, 64, 16, 2, 8, 8, False),
            
            # Multiple invalid conditions
            (1000, 64, 15, 4, 2, 2, False),  # T not divisible, E_local not divisible, logical_nc_config != 2
        ]
        
        for i, (T, block_size, E_local, logical_nc_config, tp_size, ep_size, expected) in enumerate(test_configs):
            with self.subTest(config=i):
                result = can_use_find_index_kernel(
                    T=T,
                    block_size=block_size,
                    E_local=E_local,
                    logical_nc_config=logical_nc_config,
                    tp_size=tp_size,
                    ep_size=ep_size
                )
                self.assertEqual(
                    result, 
                    expected,
                    f"Config {i}: T={T}, block_size={block_size}, E_local={E_local}, "
                    f"logical_nc_config={logical_nc_config}, tp_size={tp_size}, ep_size={ep_size}"
                )


class TestUseIndexCalcKernel(unittest.TestCase):
    
    def setUp(self):
        """Set up a mock instance with all required attributes"""
        self.mock_instance = Mock()
        
        # Set up nested mock objects
        self.mock_instance.routed_experts_mlp_config = Mock()
        self.mock_instance.blockwise_matmul_config = Mock()
        self.mock_instance.moe_expert_model_parallel_group = Mock()
        self.mock_instance.moe_tensor_model_parallel_group = Mock()
        
        # Set up MLP operation mocks
        mock_mlp_op = Mock()
        mock_mlp_op.gate_up_proj._n_local_experts = 16
        mock_mlp_op.down_proj._n_local_experts = 16
        self.mock_instance.get_mlp_op.return_value = mock_mlp_op
    
    def test_use_index_calc_kernel_configs(self):
        """Test use_index_calc_kernel with various configurations"""
        
        # Format: (training, is_prefill, use_index_calc_kernel_config, ep_size, 
        #          block_size, logical_nc_config, tp_size, total_tokens, 
        #          can_use_find_index_kernel_return, expected_result)
        test_configs = [
            # Early returns - should return False
            (True, True, True, 2, 64, 2, 2, 1024, True, False),   # training=True
            (False, False, True, 2, 64, 2, 2, 1024, True, False), # is_prefill=False
            (False, True, False, 2, 64, 2, 2, 1024, True, False), # use_index_calc_kernel=False
            
            # Valid cases that depend on can_use_find_index_kernel
            (False, True, True, 1, 64, 2, 2, 1024, True, True),   # All conditions met, kernel returns True
            (False, True, True, 2, 64, 2, 2, 1024, True, True),   # All conditions met, kernel returns True
            (False, True, True, 4, 32, 2, 1, 2048, False, False), # All conditions met, kernel returns False
        ]
        
        with patch('neuronx_distributed.modules.moe.expert_mlps_v2.can_use_find_index_kernel') as mock_can_use_index_calc:
            for i, (training, is_prefill, use_kernel_config, ep_size, 
                   block_size, logical_nc_config, tp_size, total_tokens, 
                   kernel_return, expected) in enumerate(test_configs):
                
                with self.subTest(config=i):
                    # Set up mock attributes for this test case
                    self.mock_instance.training = training
                    self.mock_instance.is_prefill = is_prefill
                    self.mock_instance.routed_experts_mlp_config.use_index_calc_kernel = use_kernel_config
                    self.mock_instance.moe_expert_model_parallel_group.size.return_value = ep_size
                    self.mock_instance.blockwise_matmul_config.block_size = block_size
                    self.mock_instance.blockwise_matmul_config.logical_nc_config = logical_nc_config
                    self.mock_instance.moe_tensor_model_parallel_group.size.return_value = tp_size
                    
                    # Set up the mock return value for can_use_find_index_kernel
                    mock_can_use_index_calc.return_value = kernel_return
                    
                    # Call the method
                    result = ExpertMLPsV2.use_index_calc_kernel(self.mock_instance, total_tokens)
                    
                    # Assert the result
                    self.assertEqual(result, expected, 
                        f"Config {i}: training={training}, is_prefill={is_prefill}, "
                        f"use_kernel_config={use_kernel_config}, ep_size={ep_size}")
                    
                    # Verify can_use_find_index_kernel was called with correct args (only for non-early-return cases)
                    if not any([training, not is_prefill, not use_kernel_config, ep_size <= 1]):
                        mock_can_use_index_calc.assert_called_with(
                            T=total_tokens,
                            block_size=block_size,
                            E_local=16,  # from our mock setup
                            logical_nc_config=logical_nc_config,
                            tp_size=tp_size,
                            ep_size=ep_size,
                        )
                    
                    mock_can_use_index_calc.reset_mock()


if __name__ == '__main__':
    unittest.main()
