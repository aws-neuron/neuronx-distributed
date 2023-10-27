# Standard Library
import unittest

# Third Party
import torch

from neuronx_distributed.parallel_layers.utils import cast_if_autocast_enabled, verify_casted_dtype

class TestCasting(unittest.TestCase):
    # Test the casting utils work as expected
    def test_casting(self):
        inputs = (torch.tensor([0.0], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32), 12)
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type='cuda'):
            casted_inputs = cast_if_autocast_enabled(*inputs)
        for item in casted_inputs:
            if isinstance(item, torch.Tensor):
                assert item.dtype == torch.bfloat16
    
    # Test there won't be casting if autocast if False
    def test_without_casting(self):
        inputs = (torch.tensor([0.0], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32), 12)
        with torch.autocast(enabled=False, device_type='cuda'):
            casted_inputs = cast_if_autocast_enabled(*inputs)
        for item in casted_inputs:
            if isinstance(item, torch.Tensor):
                assert item.dtype == torch.float32
    
    def test_verify_casted_dtype(self):
        inputs = (torch.tensor([0.0], dtype=torch.float32), torch.tensor([0.0], dtype=torch.float32), 12)
        # Rule out false negative
        verify_casted_dtype(inputs)
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type='cuda'):
            casted_inputs = cast_if_autocast_enabled(*inputs)
            # Rule out false negative
            verify_casted_dtype(casted_inputs)
            with self.assertRaises(Exception) as context:
                verify_casted_dtype(inputs)
            self.assertTrue("Datatype of tensor is expected to be torch.bfloat16, got torch.float32 instead" in str(context.exception))

if __name__ == "__main__":
    unittest.main()