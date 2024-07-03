import unittest

import torch

from neuronx_distributed.quantization.dequantize import dequantize


class TestDequantize(unittest.TestCase):
    def test_dequantize(self):
        tensor = torch.tensor([-10, 30], dtype=torch.int8)
        scale = torch.tensor(10.0)
        input = torch.tensor([-89.9, 84.8], dtype=torch.bfloat16)

        dequantized_tensor = dequantize(tensor=tensor, scale=scale, upcast_dtype=input.dtype)

        assert dequantized_tensor.dtype == torch.bfloat16
        torch.testing.assert_close(dequantized_tensor, torch.tensor([-100.0, 300.0], dtype=torch.bfloat16))
