import unittest
from unittest.mock import MagicMock, patch
import torch
from transformers import PretrainedConfig
import sys
sys.path.append('/home/ubuntu/ktest/NeuronxDistributed/examples/training/llama/')
from training_utils import get_batch_on_this_context_parallel_rank
from neuronx_distributed.parallel_layers import (
    parallel_state,
)


class TestCpBatch(unittest.TestCase):
    @patch("neuronx_distributed.pipeline.model.parallel_state.get_context_parallel_rank", MagicMock(return_value=0))
    @patch("neuronx_distributed.pipeline.model.parallel_state", MagicMock(return_value=None))
    @patch("neuronx_distributed.pipeline.model.parallel_state.get_context_model_parallel_size")
    def test_batch_slicing(self, mock_get_context_model_parallel_size):
        mbs = 1
        seq_len = 4096
        for cp_degree in [1, 2, 4, 8, 16]:
            with self.subTest(cp_degree=cp_degree):
                cp_chunk_size = seq_len // cp_degree
                batch = {}
                mock_get_context_model_parallel_size.return_value = cp_degree
                for key in ['input_ids', 'attention_mask', 'labels']:
                    batch[key] = torch.ones(mbs, seq_len)
                cp_batch = get_batch_on_this_context_parallel_rank(batch, parallel_state)
                    
                for k, v in cp_batch.items():
                    if cp_degree > 1:
                        expected_size = [mbs, cp_chunk_size]
                    else:
                        expected_size = [mbs, seq_len]
                    self.assertEqual(list(v.size()), expected_size, 
                                     f"Test failed for cp_degree: {cp_degree}")
                    print(f"Test passed for cp_degree: {cp_degree} and cp_chunk_size: {cp_chunk_size}")
                    

if __name__ == "__main__":
    unittest.main()