import unittest
from unittest.mock import MagicMock, patch
import torch

from neuronx_distributed.parallel_layers.layers_utils import (
    _linear_autograd_base_setup_fwd,
    _linear_autograd_base_setup_bwd,
    _linear_autograd_bwd_grad_reduce,
    _linear_autograd_bwd_no_weight_grad,
    _linear_autograd_bwd_input_grad
)

class TestCustomFunction(unittest.TestCase):
    def test_ctx_attributes_base_setup_fwd(self):
        # Mock the context object
        mock_ctx = MagicMock()
        
        input = torch.tensor([1.0], requires_grad=True)
        weight = torch.tensor([2.0], requires_grad=True)
        bias = torch.tensor([3.0], requires_grad=True)
        async_grad_allreduce = False
        sequence_parallel_enabled = False
        sequence_dimension = 0
        process_group = [0]
        save_for_backward = True
        reduce_dtype = torch.bfloat16
        # Call the forward method with the mocked ctx
        result = _linear_autograd_base_setup_fwd(
            mock_ctx, 
            input, weight, 
            bias, 
            async_grad_allreduce, 
            sequence_parallel_enabled,
            sequence_dimension,
            save_for_backward, 
            process_group, 
            reduce_dtype
        )

        # Check if the constant attribute is correctly set
        self.assertEqual(mock_ctx.async_grad_allreduce, async_grad_allreduce)
        self.assertEqual(mock_ctx.sequence_parallel_enabled, sequence_parallel_enabled)
        self.assertEqual(mock_ctx.sequence_dimension, sequence_dimension)
        self.assertEqual(mock_ctx.compute_weight_gradient, True)
        self.assertEqual(mock_ctx.process_group, [0])
        self.assertEqual(mock_ctx.reduce_dtype, torch.bfloat16)
        # The number 28 is because the mock_ctx object has 28 attributes, some of them are set by us i.e. 6, but other attributes are
        # there because of the mock object, putting an assert over the total number will ensure if someone adds/removes anything, the test will fail
        self.assertEqual(len(mock_ctx.__dict__), 29)
        # Verify save_for_backward was called with the correct arguments
        mock_ctx.save_for_backward.assert_called_once_with(input,weight)

        self.assertTrue(torch.equal(result, input))
            
        # Testing the case where sequence_parallel_enabled is True
        sequence_parallel_enabled = True
        mock_ctx = MagicMock()
        with patch('neuronx_distributed.parallel_layers.layers_utils._gather_along_dim') as mock_gather_along_dim:
            mock_gather_along_dim.return_value=input
            result = _linear_autograd_base_setup_fwd(mock_ctx, input, weight, bias, async_grad_allreduce, sequence_parallel_enabled,sequence_dimension,save_for_backward, process_group)
            self.assertEqual(mock_ctx.sequence_parallel_enabled, sequence_parallel_enabled)
            mock_ctx.save_for_backward.assert_called_once_with(input,weight)
            mock_gather_along_dim.assert_called_once_with(input, sequence_dimension,process_group= process_group)
            self.assertTrue(torch.equal(result, input))

    
    def test_ctx_attributes_base_setup_bwd(self):
        mock_ctx = MagicMock()

        mock_ctx.compute_weight_gradient = True
        mock_ctx.sequence_parallel_enabled = False
        mock_ctx.sequence_dimension = 0
        mock_ctx.process_group = [0]
        mock_ctx.saved_tensors = (torch.tensor([1.]),torch.tensor([2.]))
        grad_outputs = torch.tensor([[3.0]], requires_grad=True)
        input = torch.tensor([1.])
        total_input, weight, grad_output = _linear_autograd_base_setup_bwd(mock_ctx, grad_outputs)

        self.assertTrue(torch.equal(total_input, torch.tensor([1.])))
        self.assertTrue(torch.equal(weight, torch.tensor([2.])))
        self.assertTrue(torch.equal(grad_output, torch.tensor([3.])))

        # Testing the case where sequence_parallel_enabled is True
        mock_ctx.sequence_parallel_enabled = True

        with patch('neuronx_distributed.parallel_layers.layers_utils._gather_along_dim') as mock_gather_along_dim:
            mock_gather_along_dim.return_value = input
            total_input, weight, grad_output = _linear_autograd_base_setup_bwd(mock_ctx, grad_outputs)
            mock_gather_along_dim.assert_called_once_with(input, mock_ctx.sequence_dimension, process_group=mock_ctx.process_group)
            self.assertTrue(torch.equal(total_input, torch.tensor([1.])))
            self.assertTrue(torch.equal(weight, torch.tensor([2.])))
            self.assertTrue(torch.equal(grad_output, torch.tensor([3.])))

            # Testing the case where compute_weight_gradient is False
            mock_ctx.compute_weight_gradient = False
            total_input, weight, grad_output = _linear_autograd_base_setup_bwd(mock_ctx, grad_outputs)
            self.assertTrue(total_input is None)
            self.assertTrue(torch.equal(weight, torch.tensor([1.])))
            self.assertTrue(torch.equal(grad_output, torch.tensor([3.])))


    def test_linear_autograd_bwd_grad_reduce(self):
        mock_ctx = MagicMock()

        grad_input = torch.tensor([1.])
        mock_ctx.async_grad_allreduce = False
        mock_ctx.reduce_dtype = torch.bfloat16
        handle = _linear_autograd_bwd_grad_reduce(mock_ctx, grad_input, grad_input.dtype)

        self.assertTrue(handle is None)

        # Testing the case where async_grad_allreduce is True

        mock_ctx.async_grad_allreduce = True
        mock_ctx.process_group = [0]
        with patch('torch.distributed.all_reduce') as mock_all_reduce:
            mock_all_reduce.return_value=grad_input
            handle = _linear_autograd_bwd_grad_reduce(mock_ctx, grad_input, grad_input.dtype)
            mock_all_reduce.assert_called_once_with(grad_input, group = mock_ctx.process_group, async_op= True)
            self.assertTrue(torch.equal(handle, grad_input))
            
    def test_linear_autograd_bwd_no_weight_grad(self):
        mock_ctx = MagicMock()

        grad_input = torch.tensor([1.])
        original_dtype = torch.bfloat16
        mock_ctx.async_grad_allreduce = False
        mock_ctx.sequence_dimension = 0
        mock_ctx.process_group = [0]
        mock_ctx.reduce_dtype = torch.bfloat16

        with patch('neuronx_distributed.parallel_layers.layers_utils._reduce_scatter_along_dim') as mock_reduce_scatter:
            mock_reduce_scatter.return_value = grad_input
            result=_linear_autograd_bwd_no_weight_grad(mock_ctx, grad_input, original_dtype)
            mock_reduce_scatter.assert_called_once_with(grad_input, mock_ctx.sequence_dimension, process_group=mock_ctx.process_group)
            self.assertTrue(torch.equal(result, grad_input))
            self.assertTrue(result.dtype == original_dtype)

    def test_linear_autograd_bwd_input_grad(self):
        mock_ctx = MagicMock()
        mock_handle = MagicMock()
        grad_input = torch.tensor([1.])
        original_dtype = torch.bfloat16

        mock_ctx.async_grad_allreduce = False
        mock_ctx.sequence_parallel_enabled = True
        mock_ctx.sequence_dimension = 0
        mock_ctx.process_group = [0]
        mock_ctx.reduce_dtype = torch.bfloat16

        with patch('neuronx_distributed.parallel_layers.layers_utils._reduce_scatter_along_dim') as mock_reduce_scatter:
            mock_reduce_scatter.return_value = grad_input
            result_subgrad_input, result_grad_input = _linear_autograd_bwd_input_grad(mock_ctx, grad_input, mock_handle, original_dtype)
            mock_reduce_scatter.assert_called_once_with(grad_input, mock_ctx.sequence_dimension, process_group=mock_ctx.process_group)
            self.assertTrue(torch.equal(result_subgrad_input, grad_input))
            self.assertTrue(result_subgrad_input.dtype == original_dtype)

            # Testing the case where async_grad_allreduce is True and sequence_parallel_enabled is False
            mock_ctx.async_grad_allreduce = True
            mock_ctx.sequence_parallel_enabled = False
            result_subgrad_input, result_grad_input = _linear_autograd_bwd_input_grad(mock_ctx, grad_input, mock_handle, original_dtype)
            self.assertTrue(torch.equal(result_grad_input, grad_input))
            self.assertTrue(result_grad_input.dtype==original_dtype)

if __name__ == "__main__":
    unittest.main()