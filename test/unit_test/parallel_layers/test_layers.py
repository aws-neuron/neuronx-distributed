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

from neuronx_distributed.parallel_layers.layers import LinearWithAsyncCommunication, ColumnParallelLinear, RowParallelLinear, BaseParallelLinear

class TestCustomFunction(unittest.TestCase):
    def test_fwd_linear_asycommunication(self):
        '''This test checks the forward of LinearAsyncCommunication'''

        mock_ctx = MagicMock()
        mock_ctx.compute_weight_gradient = True

        input = torch.tensor([[1.0, 1.0],[1.0, 1.0]], requires_grad=True)
        weight = torch.tensor([[1.0, 1.0],[1.0, 1.0]], requires_grad=True)
        bias = torch.tensor([[1.0, 1.0],[1.0, 1.0]], requires_grad=True)
        expected_output = torch.tensor([[3.0, 3.0],[3.0, 3.0]])
        async_grad_allreduce = False
        sequence_parallel_enabled = True
        sequence_dimension = 0
        save_for_backward = True
        process_group = [0]
        reduce_dtype = torch.float32

        with (
            patch('neuronx_distributed.parallel_layers.layers._linear_autograd_base_setup_fwd') as mock_linear_autograd_base_setup_fwd,
            patch('neuronx_distributed.parallel_layers.layers._reduce_scatter_along_dim') as mock_reduce_scatter,
            patch('neuronx_distributed.parallel_layers.layers._gather_along_dim', return_value = None ) as mock_gather_along_dim
        ):
            mock_linear_autograd_base_setup_fwd.return_value = input

            output = LinearWithAsyncCommunication.apply(
                                                        input,
                                                        weight,
                                                        bias,
                                                        async_grad_allreduce,
                                                        sequence_parallel_enabled,
                                                        sequence_dimension,
                                                        save_for_backward,
                                                        process_group,
                                                        reduce_dtype
                                                    )
            
            self.assertTrue(torch.equal(output, expected_output))
            mock_reduce_scatter.assert_not_called()
            mock_gather_along_dim.assert_not_called()
            mock_linear_autograd_base_setup_fwd.assert_called_once()
        
    def test_bwd_linear_asycommunication(self):
        '''This test checks the backward of LinearAsyncCommunication'''
        mock_ctx = MagicMock()
        mock_ctx.compute_weight_gradient = True

        input = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)
        weight = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)
        grad_output = torch.tensor([[[1.0, 1.0],[1.0, 1.0]]], requires_grad=True)

        grad_input = torch.tensor([[[2.0, 2.0],[2.0, 2.0]]], requires_grad=True)

        mock_ctx.input = input
        mock_ctx.weight = weight

        with (
            patch('neuronx_distributed.parallel_layers.layers._linear_autograd_base_setup_bwd', return_value = (input, weight, grad_output)) as mock_linear_autograd_base_setup_bwd,
            patch('neuronx_distributed.parallel_layers.layers._linear_autograd_bwd_grad_reduce', return_value = mock_ctx) as mock_linear_linear_autograd_bwd_grad_reduce,
            patch('neuronx_distributed.parallel_layers.layers._linear_autograd_bwd_no_weight_grad', return_value = input) as mock_linear_autograd_bwd_no_weight_grad,
            patch('neuronx_distributed.parallel_layers.layers._linear_autograd_bwd_input_grad', return_value = (grad_output, grad_output)) as mock_linear_autograd_bwd_input_grad,
            patch('neuronx_distributed.parallel_layers.layers._reduce_scatter_along_dim') as mock_reduce_scatter,
            patch('neuronx_distributed.parallel_layers.layers._gather_along_dim', return_value = None ) as mock_gather_along_dim,

        ):
            
            LinearWithAsyncCommunication.backward(mock_ctx, grad_output)
            mock_reduce_scatter.assert_not_called()
            mock_gather_along_dim.assert_not_called()
            mock_linear_autograd_base_setup_bwd.assert_called_once()
            mock_linear_linear_autograd_bwd_grad_reduce.assert_called_once()
            mock_linear_autograd_bwd_no_weight_grad.assert_not_called()
            mock_linear_autograd_bwd_input_grad.assert_called_once()

            mock_ctx.compute_weight_gradient = False
            mock_ctx.sequence_parallel_enabled = True

            LinearWithAsyncCommunication.backward(mock_ctx, grad_output)
            mock_reduce_scatter.assert_not_called()
            mock_gather_along_dim.assert_not_called()
            mock_linear_autograd_bwd_no_weight_grad.assert_called_once()

            mock_ctx.compute_weight_gradient = False
            mock_ctx.sequence_parallel_enabled = False
            mock_ctx.async_grad_allreduce = True

            result_grad_input, _, _, _, _, _, _, _, _ = LinearWithAsyncCommunication.backward(mock_ctx, grad_output)
            self.assertTrue(torch.equal(result_grad_input, grad_input))


    def test_check_pad_false_for_training(self):
        
        model= BaseParallelLinear()
        
        with self.assertRaises(RuntimeError) as context:
            model.pad = True
            model.training = True
            
            model._check_pad_false_for_training()
            self.assertEqual(str(context.exception), "`pad=True` is only supported for inference. Set model.eval()")

        try:
            model.pad = False
            model.training = True
            model._check_pad_false_for_training()

            model.training = False
            model._check_pad_false_for_training()

            model.pad = True
            model._check_pad_false_for_training()
        except Exception as e:
            self.fail(f"_check_pad_false_for_training() raised ExceptionType {e}  unexpectedly!")


    def test_cpl_maybe_gather_output(self):
        
        output_parallel = torch.tensor([1., 1., 1.])

        with (
            patch('neuronx_distributed.parallel_layers.layers.copy_to_tensor_model_parallel_region'),
            patch("neuronx_distributed.parallel_layers.layers.divide") as mock_divide,
            patch("neuronx_distributed.parallel_layers.layers.ColumnParallelLinear.initialize_weight_and_bias") as mock_init,
            patch('neuronx_distributed.parallel_layers.layers.gather_from_tensor_model_parallel_region') as mock_gather,
            patch("torch.distributed.get_world_size") as mock_world_size,
            patch("torch.narrow") as mock_torch_narrow
        ):
            mock_divide.return_value = 1
            mock_world_size.return_value = 1
            mock_init.return_value = None
            mock_gather.return_value = output_parallel
            mock_torch_narrow.return_value = output_parallel

            model= ColumnParallelLinear(
                input_size = 10,
                output_size = 10,
                bias = False,
                tensor_model_parallel_group= torch.tensor([0])
            )
            model.sequence_parallel_enabled = False
            model.tensor_parallel_group = [0]
            model.gather_output = False

            result_output = model._cpl_maybe_gather_output(output_parallel)
            self.assertTrue(torch.equal(result_output, output_parallel))
            mock_gather.assert_not_called()

            model.gather_output = True
            model.pad = True
            model.pad_size = 0
            model.output_size = 4
            result_output = model._cpl_maybe_gather_output(output_parallel)
            self.assertTrue(torch.equal(result_output, output_parallel))
            mock_torch_narrow.assert_not_called()
            mock_gather.assert_called_once()

            model.pad = False
            model.pad_size = 1
            model.output_size = 4
            result_output = model._cpl_maybe_gather_output(output_parallel)
            self.assertTrue(torch.equal(result_output, output_parallel))
            mock_torch_narrow.assert_not_called()

            model.pad = True
            result_output = model._cpl_maybe_gather_output(output_parallel)
            self.assertTrue(torch.equal(result_output, output_parallel))
            mock_torch_narrow.assert_called_once_with(output_parallel, -1, 0, model.output_size - model.pad_size)
            
            with self.assertRaises(AssertionError) as context:
                model.sequence_parallel_enabled = True
                model._cpl_maybe_gather_output(output_parallel)
                self.assertEqual(str(context.exception), "")

    def test_cpl_maybe_input_copy_to_tp_region(self):

        input = torch.tensor([1., 1., 1.])

        with (
            patch('neuronx_distributed.parallel_layers.layers.copy_to_tensor_model_parallel_region') as mock_copy_to_tp_region,
            patch("neuronx_distributed.parallel_layers.layers.divide") as mock_divide,
            patch("neuronx_distributed.parallel_layers.layers.ColumnParallelLinear.initialize_weight_and_bias") as mock_init,
            patch("torch.distributed.get_world_size") as mock_world_size,
        ):
            mock_divide.return_value = 1
            mock_world_size.return_value = 1
            mock_init.return_value = None

            model= ColumnParallelLinear(
                input_size = 10,
                output_size = 10,
                bias = False,
                tensor_model_parallel_group= torch.tensor([0])
            )

            model.tensor_parallel_group = [0]
            model.async_tensor_model_parallel_allreduce = False
            model.sequence_parallel_enabled = True
            
            result_input = model._cpl_maybe_input_copy_to_tp_region(input)
            self.assertTrue(torch.equal(result_input, input))
            mock_copy_to_tp_region.assert_not_called()

            model.sequence_parallel_enabled = False
            model.async_tensor_model_parallel_allreduce = True

            result_input = model._cpl_maybe_input_copy_to_tp_region(input)
            self.assertTrue(torch.equal(result_input, input))
            mock_copy_to_tp_region.assert_not_called()

            model.async_tensor_model_parallel_allreduce = False
            mock_copy_to_tp_region.return_value = input
            result_input = model._cpl_maybe_input_copy_to_tp_region(input)
            self.assertTrue(torch.equal(result_input, input))
            mock_copy_to_tp_region.assert_called_once_with(input, process_group=model.tensor_parallel_group)

    def test_CPL_forward(self):
        
        hidden_states = torch.tensor([1., 1.])
        with (
            patch("torch.distributed.get_world_size") as mock_world_size,
            patch("neuronx_distributed.parallel_layers.layers.ColumnParallelLinear.initialize_weight_and_bias") as mock_init,
            patch("neuronx_distributed.parallel_layers.layers.BaseParallelLinear._check_pad_false_for_training"),
            patch("neuronx_distributed.parallel_layers.layers.ColumnParallelLinear._cpl_maybe_input_copy_to_tp_region") as mock_input_copy,
            patch("neuronx_distributed.parallel_layers.layers.ColumnParallelLinear._cpl_maybe_gather_output") as mock_gather,

        ):
            mock_world_size.return_value = 1
            mock_init.return_value = None
            mock_gather.return_value = hidden_states 
            mock_input_copy.return_value = hidden_states
            mock_forward = MagicMock()
            mock_forward.return_value = hidden_states

            model= ColumnParallelLinear(
                input_size = 10,
                output_size = 10,
                bias = False,
                tensor_model_parallel_group= torch.tensor([0])
            )

            
            model._forward_impl = mock_forward
            model.bias = None
            model.weight = torch.tensor([1., 1.])

            result_output = model.forward(hidden_states)
            self.assertTrue(torch.equal(result_output, hidden_states))
            mock_forward.assert_called_once_with(
                input=hidden_states,
                weight=model.weight,
                bias=None,
                async_grad_allreduce=model.async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=model.sequence_parallel_enabled,
                sequence_dimension=model.sequence_dimension,
                autograd_func_class=model.autograd_func_class,
                process_group=model.tensor_parallel_group,
                reduce_dtype = model.reduce_dtype,
            )

            mock_input_copy.assert_called_once_with(hidden_states)

            mock_gather.assert_called_once_with(hidden_states)

    def test_rpl_maybe_reduce_output(self):
        
        output = torch.tensor([1., 1., 1.])

        with (
            patch("neuronx_distributed.parallel_layers.layers.RowParallelLinear.initialize_weight_and_bias") as mock_init,
            patch("neuronx_distributed.parallel_layers.layers.divide") as mock_divide,
            patch('neuronx_distributed.parallel_layers.layers.reduce_scatter_to_sequence_parallel_region') as mock_reduce_scatter,
            patch('neuronx_distributed.parallel_layers.layers.reduce_from_tensor_model_parallel_region') as mock_reduce,

        ):
            mock_reduce_scatter.return_value = output
            mock_reduce.return_value = output
            mock_init.return_value = None
            mock_divide.return_value = 1

            model= RowParallelLinear(
                input_size = 10,
                output_size = 10,
                bias = False,
                tensor_model_parallel_group= torch.tensor([0])
            )

            model.tensor_parallel_group = [0]
            model.sequence_dimension = 0
            model.reduce_output = False

            result_output = model._rpl_maybe_reduce_output(output)
            self.assertTrue(torch.equal(result_output, output))
            mock_reduce_scatter.assert_not_called()
            mock_reduce.assert_not_called()

            model.reduce_output = True
            model.reduce_dtype = torch.float32
            model.sequence_parallel_enabled = True
            result_output = model._rpl_maybe_reduce_output(output)
            self.assertTrue(torch.equal(result_output, output))
            mock_reduce_scatter.assert_called_once_with(
                output, 
                model.sequence_dimension, 
                process_group=model.tensor_parallel_group,
                dtype=model.reduce_dtype,
            )
            mock_reduce.assert_not_called()


            model.sequence_parallel_enabled = False
            result_output = model._rpl_maybe_reduce_output(output)
            self.assertTrue(torch.equal(result_output, output))
            mock_reduce.assert_called_once_with(
                output, 
                process_group=model.tensor_parallel_group,
            )
    
    def test_rpl_maybe_scatter_input(self):
        
        input = torch.tensor([1., 1., 1.])

        with (
            patch("neuronx_distributed.parallel_layers.layers.RowParallelLinear.initialize_weight_and_bias") as mock_init,
            patch("neuronx_distributed.parallel_layers.layers.divide") as mock_divide,
            patch('neuronx_distributed.parallel_layers.layers.scatter_to_tensor_model_parallel_region') as mock_scatter,
            patch('torch.nn.functional.pad') as mock_torch_pad,

        ):
            mock_scatter.return_value = input
            mock_torch_pad.return_value = input
            mock_init.return_value = None
            mock_divide.return_value = 1

            model= RowParallelLinear(
                input_size = 10,
                output_size = 10,
                bias = False,
                tensor_model_parallel_group= torch.tensor([0])
            )

            model.input_is_parallel = True
            model.tensor_parallel_group = [0]
            
            result_input = model._rpl_maybe_scatter_input(input)
            self.assertTrue(torch.equal(result_input, input))
            mock_scatter.assert_not_called()
            
            model.sequence_parallel_enabled = False
            model.input_is_parallel = False
            model.pad = False
            model.pad_size = 1
            
            result_input = model._rpl_maybe_scatter_input(input)
            self.assertTrue(torch.equal(result_input, input))
            mock_torch_pad.assert_not_called()
            mock_scatter.assert_called_once_with(input, process_group=model.tensor_parallel_group)

            model.pad = True
            model.pad_size = 0
            result_input = model._rpl_maybe_scatter_input(input)
            self.assertTrue(torch.equal(result_input, input))
            mock_torch_pad.assert_not_called()

            model.pad = True
            model.pad_size = 1
            result_input = model._rpl_maybe_scatter_input(input)
            self.assertTrue(torch.equal(result_input, input))
            mock_torch_pad.assert_called_once_with(input, (0, model.pad_size))

            
            with self.assertRaises(AssertionError) as context:
                model.sequence_parallel_enabled = True
                model._rpl_maybe_scatter_input(input)
                self.assertEqual(str(context.exception), "")
            
    def test_RPL_forward(self):
        
        hidden_states = torch.tensor([1., 1.])
        with (
            patch("neuronx_distributed.parallel_layers.layers.RowParallelLinear.initialize_weight_and_bias") as mock_init,
            patch("neuronx_distributed.parallel_layers.layers.BaseParallelLinear._check_pad_false_for_training") as mock_assertions,
            patch("neuronx_distributed.parallel_layers.layers.RowParallelLinear._rpl_maybe_scatter_input") as mock_scatter,
            patch("neuronx_distributed.parallel_layers.layers.RowParallelLinear._rpl_maybe_reduce_output") as mock_reduce,
            patch("neuronx_distributed.parallel_layers.layers.divide") as mock_divide,

        ):

            mock_init.return_value = None
            mock_divide.return_value = 1
            mock_reduce.return_value = hidden_states 
            mock_scatter.return_value = hidden_states
            mock_forward = MagicMock()
            mock_forward.return_value = hidden_states

            model= RowParallelLinear(
                input_size = 10,
                output_size = 10,
                bias = False,
                tensor_model_parallel_group= torch.tensor([0])
            )

            
            model._forward_impl = mock_forward
            model.bias = None
            model.weight = torch.tensor([1., 1.])

            result_output = model.forward(hidden_states)
            self.assertTrue(torch.equal(result_output, hidden_states))
            mock_forward.assert_called_once_with(
                input=hidden_states,
                weight=model.weight,
                bias=None,
                async_grad_allreduce=False,
                sequence_parallel_enabled=False,
                sequence_dimension=model.sequence_dimension,
                autograd_func_class=model.autograd_func_class,
                process_group=model.tensor_parallel_group,
                reduce_dtype = model.reduce_dtype,
            )
            mock_reduce.assert_called_once_with(hidden_states)
            mock_scatter.assert_called_once_with(hidden_states)
            mock_assertions.assert_called_once()

    def test_lm_head_padding(self):
        batch_size = 2
        input_size = 16
        output_size = 1500
        world_size = 64
        padded_output_size_per_rank = 24
        padded_output_size = world_size * padded_output_size_per_rank
        pad_size = padded_output_size - output_size

        with (
            patch("neuronx_distributed.parallel_layers.layers.ColumnParallelLinear.initialize_weight_and_bias") as mock_init,
            patch("torch.distributed.get_world_size") as mock_world_size,
            patch("neuronx_distributed.parallel_layers.layers.linear_with_async_allreduce") as mock_linear,
            patch("neuronx_distributed.parallel_layers.layers.gather_from_tensor_model_parallel_region") as mock_output_gather,
        ):
            mock_world_size.return_value = world_size
            mock_init.return_value = None
            mock_linear.return_value = torch.rand(batch_size, padded_output_size_per_rank)
            mock_output_gather.return_value = torch.ones(batch_size, padded_output_size)

            model= ColumnParallelLinear(
                input_size=input_size,
                output_size=output_size,
                pad=True,
                bias=True,
                pad_alignment_size_per_rank=padded_output_size_per_rank, 
	            keep_padded_output=True,                
                tensor_model_parallel_group=torch.arange(world_size)
            )

            model.weight = torch.rand(24, 16)
            model.bias = torch.zeros((1536,))
            model.eval()

            # Validate padding shapes
            model.set_weight_and_bias_config()
            assert(model.weight_shape == (24, 16)) 
            assert(model.bias_shape == (1536,))

            # Validate sharding and padding
            model_state_dict = {
                'layer.weight': torch.ones(output_size, input_size), 
                'layer.bias': torch.ones(output_size,)
            }
            orig_weight = model_state_dict['layer.weight'].clone()
            orig_bias = model_state_dict['layer.bias'].clone()
            model.preshard_hook(model_state_dict, 'layer.weight')
            model.preshard_hook(model_state_dict, 'layer.bias')
            new_weight = model_state_dict['layer.weight']
            new_bias = model_state_dict['layer.bias']
            assert(new_weight.shape == (padded_output_size, input_size))
            assert(torch.equal(new_weight, torch.nn.functional.pad(orig_weight, (0, 0, 0, pad_size))))
            assert(new_bias.shape == (padded_output_size,))
            min_value = torch.finfo(orig_bias.dtype).min
            assert(torch.equal(new_bias, torch.nn.functional.pad(orig_bias, (0, pad_size), 'constant', min_value)))

            # Validate the output is kept padded
            final_output = model(torch.ones(2, 16))
            assert(final_output.shape == (2, 1536))

if __name__ == "__main__":
    unittest.main()