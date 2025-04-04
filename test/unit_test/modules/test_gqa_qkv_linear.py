import unittest
from unittest.mock import MagicMock, patch
import torch

from neuronx_distributed.modules.qkv_linear_utils import (
    _qkvlinear_autograd_base_setup_fwd,
    _qkvlinear_autograd_base_setup_bwd,
    _qkvlinear_autograd_bwd_grad_reduce,
    _qkvlinear_autograd_bwd_no_weight_grad,
    _qkvlinear_autograd_bwd_input_grad
)

from neuronx_distributed.modules.qkv_linear import GQAQKVLinearWithAsyncCommunication

class TestCustomFunction(unittest.TestCase):
    def test_fwd_qkvlinear_asyncommunication(self):
        '''This test checks the forward of GQAQKVLinearWithAsyncCommunication
        We are checking that the functions inside GQAQKVLinearWithAsyncCommunication are called
        with the correct arguments and also asserting the CC ops are not called explicitly.'''

        mock_ctx = MagicMock()
        mock_ctx.compute_weight_gradient = True
        mock_ctx.fuse_qkv = False

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
        expected_output = torch.tensor([[3.0, 3.0],[3.0, 3.0]])

        with (
            patch('neuronx_distributed.modules.qkv_linear._linear_forward', return_value = expected_output),
            patch('neuronx_distributed.modules.qkv_linear_utils.xm.all_gather', return_value = input) as mock_all_gather,
            patch('neuronx_distributed.modules.qkv_linear_utils.xm.reduce_scatter' ) as mock_reduce_scatter,
            patch('neuronx_distributed.modules.qkv_linear_utils.get_tensor_model_parallel_replica_groups',return_value = [0]),
            patch('neuronx_distributed.modules.qkv_linear._qkvlinear_autograd_base_setup_fwd',return_value = input) as mock_qkvlinear_autograd_base_setup_fwd
        ):

            output_q, output_k, output_v = GQAQKVLinearWithAsyncCommunication.forward(
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
            
            self.assertTrue(torch.equal(output_q, expected_output))
            self.assertTrue(torch.equal(output_k, expected_output))
            self.assertTrue(torch.equal(output_v, expected_output))
            mock_reduce_scatter.assert_not_called()
            mock_all_gather.assert_not_called()
            mock_qkvlinear_autograd_base_setup_fwd.assert_called_once_with(mock_ctx,
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

    def test_bwd_qkvlinear_asycommunication(self):
        '''This test checks the backward of GQAQKVLinearWithAsyncCommunication
        We are checking that the functions inside GQAQKVLinearWithAsyncCommunication are called
        with the correct arguments and called only once and also asserting the CC ops are not called 
        explicitly.'''

        #Checking the scenarios where compute_weight_gradient is True, sequence_parallel_enabled is False
        mock_ctx = MagicMock()
        mock_ctx.compute_weight_gradient = True
        mock_ctx.fuse_qkv = False
        mock_ctx.sequence_parallel_enabled = True
        mock_ctx.async_grad_allreduce = False

        input = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)
        weight_q = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)
        weight_k = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)
        weight_v = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)
        bias_q = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)
        bias_k = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)
        bias_v = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)
        grad_output_q = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)
        grad_output_k = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)
        grad_output_v = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)

        mock_ctx.saved_tensors = (input, weight_q, weight_k, weight_v)

        mock_ctx.input = input
        mock_ctx.weight_q = weight_q
        mock_base_setup_bwd_return_value = (input, None, weight_q, weight_k, weight_v, grad_output_k, grad_output_v)
        with (
            patch('neuronx_distributed.modules.qkv_linear.xm.all_reduce', return_value = [grad_output_k, grad_output_v]) as mock_all_reduce,
            patch('neuronx_distributed.modules.qkv_linear._qkvlinear_autograd_base_setup_bwd', return_value = mock_base_setup_bwd_return_value) as mock_qkvlinear_autograd_base_setup_bwd,
            patch('neuronx_distributed.modules.qkv_linear.xm.all_gather', return_value = input) as mock_all_gather,
            patch('neuronx_distributed.modules.qkv_linear._qkvlinear_autograd_bwd_grad_reduce',return_value = input) as mock_qkvlinear_autograd_bwd_grad_reduce,
            patch('neuronx_distributed.modules.qkv_linear._qkvlinear_autograd_bwd_input_grad', return_value = input) as mock_qkvlinear_autograd_bwd_input_grad,
            patch('neuronx_distributed.modules.qkv_linear._qkvlinear_autograd_bwd_no_weight_grad', return_value = input) as mock_qkvlinear_autograd_bwd_no_weight_grad,
            patch('neuronx_distributed.modules.qkv_linear.xm.reduce_scatter', return_value = input) as mock_reduce_scatter,
            patch('neuronx_distributed.modules.qkv_linear._compute_gradients', return_value = (weight_q, bias_q)),
            patch('torch.empty', return_value = torch.zeros(input.shape)),
        ):
            
            (result_subgrad_input, result_grad_weight_q, \
            result_grad_weight_k, result_grad_weight_v, \
            grad_bias_q, grad_bias_k, grad_bias_v, result_none, \
            _, _, _, _, _, _, _, _, _ ) =  GQAQKVLinearWithAsyncCommunication.backward(mock_ctx, grad_output_q, grad_output_k, grad_output_v )

            mock_qkvlinear_autograd_bwd_grad_reduce.assert_called_once()
            mock_qkvlinear_autograd_bwd_input_grad.assert_called_once()
            mock_qkvlinear_autograd_base_setup_bwd.assert_called_once_with(mock_ctx, grad_output_q, grad_output_k, grad_output_v)
            self.assertTrue(torch.equal(result_subgrad_input, input))
            self.assertTrue(torch.equal(result_grad_weight_q, result_grad_weight_q))
            self.assertTrue(torch.equal(result_grad_weight_k, result_grad_weight_k))
            self.assertTrue(torch.equal(result_grad_weight_v, result_grad_weight_v))
            self.assertTrue(torch.equal(result_grad_weight_v, result_grad_weight_v))
            self.assertTrue(torch.equal(grad_bias_q, bias_q))
            self.assertTrue(torch.equal(grad_bias_k, bias_k))
            self.assertTrue(torch.equal(grad_bias_v, bias_v))
            self.assertTrue(result_none is None)

            #Checking the scenarios where compute_weight_gradient is False, sequence_parallel_enabled is True
            mock_ctx.compute_weight_gradient = False
            mock_ctx.sequence_parallel_enabled = True

            (result_subgrad_input, result_none, \
            _, _, _, _, _, _, _, _, _, _, _, _, _, _ )=  GQAQKVLinearWithAsyncCommunication.backward(mock_ctx, grad_output_q, grad_output_k, grad_output_v )
            
            mock_reduce_scatter.assert_not_called()
            mock_all_reduce.assert_not_called()
            mock_all_gather.assert_not_called()
            mock_qkvlinear_autograd_bwd_no_weight_grad.assert_called_once()
            self.assertTrue(result_none is None)
            self.assertTrue(torch.equal(result_subgrad_input, input))

if __name__ == "__main__":
    unittest.main()