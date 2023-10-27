# Standard Library
import unittest
from unittest.mock import patch, MagicMock

import neuronx_distributed.pipeline.partition as partition
from .test_base import get_traced_model_gpt
from .. import update_result


class TestSharedWeights(unittest.TestCase):
    
    @patch('neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size', MagicMock(return_value=1))
    @patch('neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank', MagicMock(return_value=1))
    @patch('neuronx_distributed.pipeline.partition.get_pipeline_model_parallel_rank', MagicMock(return_value=0))
    @patch('neuronx_distributed.parallel_layers.layers._initialize_affine_weight_cpu', MagicMock(return_value=None))
    @patch('neuronx_distributed.parallel_layers.layers._initialize_affine_weight_neuron', MagicMock(return_value=None))
    @patch('neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized', MagicMock(return_value=True))
    @patch('neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size', MagicMock(return_value=8))
    @patch('neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank', MagicMock(return_value=1))
    @patch('neuronx_distributed.pipeline.model.parallel_state')
    @patch('torch.distributed.get_rank') 
    def test_analyze_shared_weights_across_stages(self, rank_mock, state_mock):
        try:
            traced_model = get_traced_model_gpt()
            split_mod = partition.partition_traced_model(traced_model)
            partitions = []
            for _, module in split_mod.named_children():
                partitions.append(module)
            shared_weights = partition.analyze_shared_weights_across_stages(traced_model, partitions)
            assert shared_weights == [[('transformer_wte.weight', 0), ('lm_head.weight', 7)]]
        except:
            update_result({"inference_success": 0})
            raise

if __name__ == "__main__":
    unittest.main()