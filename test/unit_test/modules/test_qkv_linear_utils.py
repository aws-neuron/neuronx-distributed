import unittest
from unittest.mock import MagicMock, patch
import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.modules.qkv_linear_utils import (
    _qkvlinear_autograd_base_setup_fwd,
    _qkvlinear_autograd_base_setup_bwd,
    _qkvlinear_autograd_bwd_grad_reduce,
    _qkvlinear_autograd_bwd_no_weight_grad,
    _qkvlinear_autograd_bwd_input_grad
)

class TestCustomFunction(unittest.TestCase):
    def test_ctx_attributes_base_setup_fwd(self):
        # Mock the context object
        mock_ctx = MagicMock()
        
        input = torch.tensor([1.0], requires_grad=True)
        weight_q = torch.tensor([2.0], requires_grad=True)
        weight_k = torch.tensor([2.0], requires_grad=True)
        weight_v = torch.tensor([2.0], requires_grad=True)
        bias_q = torch.tensor([3.0], requires_grad=True)
        bias_k = torch.tensor([3.0], requires_grad=True)
        bias_v = torch.tensor([3.0], requires_grad=True)
        async_grad_allreduce = False
        sequence_parallel_enabled = False
        kv_size_multiplier = 2
        weight_qkv = None
        bias_qkv = None
        fuse_qkv = False
        output_size_q = None
        output_size_kv = None
        reduce_dtype = torch.bfloat16
        # Call the forward method with the mocked ctx
        result = _qkvlinear_autograd_base_setup_fwd(
            mock_ctx, 
            input, 
            weight_q, 
            weight_k,
            weight_v,
            bias_q, 
            bias_k,
            bias_v,
            async_grad_allreduce, 
            sequence_parallel_enabled,
            kv_size_multiplier,
            weight_qkv,
            bias_qkv, 
            fuse_qkv, 
            output_size_q,
            output_size_kv,
            reduce_dtype
        )

        # Check if the constant attribute is correctly set
        self.assertEqual(mock_ctx.use_bias, True)
        self.assertEqual(mock_ctx.async_grad_allreduce, async_grad_allreduce)
        self.assertEqual(mock_ctx.sequence_parallel_enabled, sequence_parallel_enabled)
        self.assertEqual(mock_ctx.compute_weight_gradient, True)
        self.assertEqual(mock_ctx.kv_size_multiplier, kv_size_multiplier)
        self.assertEqual(mock_ctx.fuse_qkv, fuse_qkv)
        self.assertEqual(mock_ctx.reduce_dtype, torch.bfloat16)
        # The number 29 is because the mock_ctx object has 29 attributes, some of them are set by us i.e. 6, but other attributes are
        # there because of the mock object, putting an assert over the total number will ensure if someone adds/removes anything, the test will fail
        self.assertEqual(len(mock_ctx.__dict__), 29)
        # Verify save_for_backward was called with the correct arguments
        mock_ctx.save_for_backward.assert_called_once_with(input, weight_q, weight_k, weight_v)

        self.assertTrue(torch.equal(result, input))
            
        # Testing the case where sequence_parallel_enabled is True
        sequence_parallel_enabled = True
        mock_ctx = MagicMock()
        with (
            patch('neuronx_distributed.modules.qkv_linear_utils.xm.all_gather', return_value = input) as mock_all_gather,
            patch('neuronx_distributed.modules.qkv_linear_utils.get_tensor_model_parallel_replica_groups',return_value = [0]) as mock_parallel_replica_groups,
        ):
            result = _qkvlinear_autograd_base_setup_fwd(mock_ctx, 
                input, 
                weight_q, 
                weight_k,
                weight_v,
                bias_q, 
                bias_k,
                bias_v,
                async_grad_allreduce, 
                sequence_parallel_enabled,
                kv_size_multiplier,
                weight_qkv,
                bias_qkv, 
                fuse_qkv, 
                output_size_q,
                output_size_kv,
                reduce_dtype
            )
            self.assertEqual(len(mock_ctx.__dict__), 29)
            self.assertEqual(mock_ctx.sequence_parallel_enabled, sequence_parallel_enabled)
            mock_all_gather.assert_called_once_with(input, groups=mock_parallel_replica_groups(), pin_layout=False)
            self.assertTrue(torch.equal(result, input))

    def test_ctx_attributes_base_setup_bwd(self):
        mock_ctx = MagicMock()

        mock_ctx.compute_weight_gradient = True
        mock_ctx.fuse_qkv = False
        mock_ctx.sequence_parallel_enabled = False
        mock_ctx.kv_size_multiplier = 2
        mock_ctx.reduce_dtype = torch.bfloat16
        mock_ctx.sequence_parallel_enabled = True
        mock_ctx.saved_tensors = (torch.tensor([1.]),torch.tensor([1.]),torch.tensor([1.]),torch.tensor([1.]))
        grad_output_q = torch.tensor([[1.0]], requires_grad=True)
        grad_output_k = torch.tensor([[1.0]], requires_grad=True)
        grad_output_v = torch.tensor([[1.0]], requires_grad=True)
        input = torch.tensor([1.])
        

        with (
            patch('neuronx_distributed.modules.qkv_linear_utils.xm.all_reduce', return_value = [grad_output_k, grad_output_v]) as mock_all_reduce,
            patch('neuronx_distributed.modules.qkv_linear_utils.parallel_state.get_kv_shared_replica_groups', return_value = [0]) as mock_kv_shared_replica_groups,
            patch('neuronx_distributed.modules.qkv_linear_utils.xm.all_gather', return_value = input) as mock_all_gather,
            patch('neuronx_distributed.modules.qkv_linear_utils.get_tensor_model_parallel_replica_groups',return_value = [0]) as mock_parallel_replica_groups
        ):
            total_input, weight_qkv, weight_q,weight_k, weight_v, result_grad_output_k, result_grad_output_v  = _qkvlinear_autograd_base_setup_bwd(
                                                                                            mock_ctx,
                                                                                            grad_output_q,
                                                                                            grad_output_k,
                                                                                            grad_output_v
                                                                                            )

            self.assertTrue(torch.equal(total_input, torch.tensor([1.])))
            self.assertTrue(torch.equal(weight_q, torch.tensor([1.])))
            self.assertTrue(torch.equal(weight_k, torch.tensor([1.])))
            self.assertTrue(torch.equal(weight_v, torch.tensor([1.])))
            self.assertTrue(weight_qkv is None)
            self.assertTrue(torch.equal(result_grad_output_k, grad_output_k))
            self.assertTrue(torch.equal(result_grad_output_v, grad_output_v))
            mock_all_gather.assert_called_once_with(input, groups=mock_parallel_replica_groups(), pin_layout=False)
            mock_all_reduce.assert_called_once_with(xm.REDUCE_SUM, [grad_output_k, grad_output_v], scale = 1.0, groups = mock_kv_shared_replica_groups(), pin_layout = False)
            self.assertTrue(torch.equal(total_input, input))

            # Testing the case where compute_weight_gradient is False
            mock_ctx.compute_weight_gradient = False
            total_input, weight_qkv, weight_q,weight_k, weight_v, result_grad_output_k, result_grad_output_v  = _qkvlinear_autograd_base_setup_bwd(
                                                                                            mock_ctx,
                                                                                            grad_output_q,
                                                                                            grad_output_k,
                                                                                            grad_output_v
                                                                                            )

            self.assertTrue(total_input is None)
            self.assertTrue(torch.equal(weight_q, torch.tensor([1.])))
            self.assertTrue(torch.equal(weight_k, torch.tensor([1.])))
            self.assertTrue(torch.equal(weight_v, torch.tensor([1.])))

    def test_qkvlinear_autograd_bwd_grad_reduce(self):
        mock_ctx = MagicMock()
        grad_input = torch.tensor([1.])

        # Checking the case where async_grad_allreduce is True
        mock_ctx.async_grad_allreduce = True
        mock_ctx.reduce_dtype = torch.bfloat16

        with (
            patch('torch.distributed.all_reduce') as mock_all_reduce,
            patch('neuronx_distributed.modules.qkv_linear_utils.get_tensor_model_parallel_group', return_value = [0]) as mock_tensor_model_parallel_group,
        ):
            mock_all_reduce.return_value=grad_input
            result_grad_input = _qkvlinear_autograd_bwd_grad_reduce(mock_ctx, grad_input, grad_input.dtype)
            mock_all_reduce.assert_called_once_with(grad_input, group = mock_tensor_model_parallel_group())
            self.assertTrue(torch.equal(result_grad_input, grad_input))

            # Checking the case where async_grad_allreduce is False
            mock_ctx.async_grad_allreduce = False
            result_grad_input = _qkvlinear_autograd_bwd_grad_reduce(mock_ctx, grad_input, grad_input.dtype)
            self.assertTrue(torch.equal(result_grad_input, grad_input))


    def test_qkvlinear_autograd_bwd_no_weight_grad(self):
        mock_ctx = MagicMock()

        grad_input = torch.tensor([1.])
        original_dtype = torch.bfloat16
        mock_ctx.async_grad_allreduce = False
        mock_ctx.reduce_dtype = torch.bfloat16

        
        
        with (
            patch('neuronx_distributed.modules.qkv_linear_utils.get_tensor_model_parallel_size', return_value = 1),
            patch('neuronx_distributed.modules.qkv_linear_utils.xm.reduce_scatter', return_value = grad_input) as mock_reduce_scatter,
            patch('neuronx_distributed.modules.qkv_linear_utils.get_tensor_model_parallel_replica_groups',return_value = [0]) as mock_parallel_replica_groups,
            patch('torch.empty', return_value = torch.zeros(grad_input.shape)) as mock_torch_empty,
        ):
            shape = list(grad_input.shape)
            shape[0] //= 1
            sub_grad_input = mock_torch_empty(
                torch.Size(shape),
                dtype=torch.bfloat16,
                device=grad_input.device,
                requires_grad=False,
            )

            result=_qkvlinear_autograd_bwd_no_weight_grad(mock_ctx, grad_input, original_dtype)
            mock_reduce_scatter.assert_called_once_with(
                xm.REDUCE_SUM,
                grad_input,
                output = sub_grad_input,
                groups = mock_parallel_replica_groups(),
                shard_count = 1,
                scatter_dim = 0,
                scale = 1,
                pin_layout = False,
            )

            self.assertTrue(torch.equal(result, sub_grad_input))
            self.assertTrue(result.dtype == original_dtype)

    def test_qkvlinear_autograd_bwd_input_grad(self):
        mock_ctx = MagicMock()

        grad_input = torch.tensor([1.])
        mock_ctx.async_grad_allreduce = False
        mock_ctx.reduce_dtype = torch.bfloat16

        with (
            patch('neuronx_distributed.modules.qkv_linear_utils.get_tensor_model_parallel_size', return_value = 1) as mock_tensor_parallel_size,
            patch('neuronx_distributed.modules.qkv_linear_utils.xm.reduce_scatter', return_value = grad_input) as mock_reduce_scatter,
            patch('neuronx_distributed.modules.qkv_linear_utils.get_tensor_model_parallel_replica_groups',return_value = [0]) as mock_parallel_replica_groups,
            patch('torch.empty', return_value = torch.zeros(grad_input.shape)) as mock_torch_empty,
        ):
            sub_grad_input = mock_torch_empty(
                torch.Size(grad_input.shape),
                dtype=torch.bfloat16,
                device=grad_input.device,
                requires_grad=False,
            )

            result = _qkvlinear_autograd_bwd_input_grad(mock_ctx, grad_input, grad_input.dtype)
            
            mock_reduce_scatter.assert_called_once_with(
                xm.REDUCE_SUM,
                grad_input,
                output = sub_grad_input,
                groups = mock_parallel_replica_groups(),
                shard_count = mock_tensor_parallel_size(),
                scatter_dim = 0,
                scale = 1,
                pin_layout = False,
            )

            self.assertTrue(torch.equal(result, sub_grad_input))


if __name__ == "__main__":
    unittest.main()