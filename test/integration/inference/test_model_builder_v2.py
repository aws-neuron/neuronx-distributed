import unittest
import torch
import os
import shutil
import tempfile

from neuronx_distributed.trace.model_builder import ModelBuilderV2

class TestModelBuilderV2(unittest.TestCase):
    def setUp(self):
        # Create a simple model for testing
        self.model = torch.nn.Linear(10, 5)
        self.world_size = 2
        self.builder = ModelBuilderV2(self.model, self.world_size)

    def test_trace(self):
        # Test that trace method raises NotImplementedError
        example_inputs = torch.randn(1, 10)
        with self.assertRaises(NotImplementedError):
            self.builder.trace(example_inputs)

    def test_compile(self):
        # Test that compile method raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.builder.compile()

if __name__ == '__main__':
    unittest.main()
