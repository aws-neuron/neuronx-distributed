import random
import unittest

import torch

from neuronx_distributed.quantization.microscaling.transform_weights import get_mxfp4_tensor, get_mxfp4_tensor_from_uint16, pack_fp4_x4_uint16
from neuronx_distributed.quantization.dequantize import blockwise_scale_dequantize, direct_cast_dequantize, scale_dequantize, get_broadcastable_shapes_for_blockwise_scale_dequantize
from neuronx_distributed.quantization.quantization_config import QuantizedDtype, ScaleDtype, DtypeBound


class TestDequantize(unittest.TestCase):
    def test_direct_cast_dequantize(self):
        tensor = torch.tensor([-10, 30], dtype=torch.int8)
        dequantized_tensor = direct_cast_dequantize(tensor=tensor, upcast_dtype=torch.bfloat16)

        assert dequantized_tensor.dtype == torch.bfloat16
        torch.testing.assert_close(dequantized_tensor, torch.tensor([-10.0, 30.0], dtype=torch.bfloat16))

    def test_scale_dequantize(self):
        tensor = torch.tensor([-10, 30], dtype=torch.int8)
        scale = torch.tensor(10.0)
        input = torch.tensor([-89.9, 84.8], dtype=torch.bfloat16)

        dequantized_tensor = scale_dequantize(tensor=tensor, scale=scale, upcast_dtype=input.dtype)

        assert dequantized_tensor.dtype == torch.bfloat16
        torch.testing.assert_close(dequantized_tensor, torch.tensor([-100.0, 300.0], dtype=torch.bfloat16))


class TestBlockwiseScaleDequantize(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)

    def simulate_float8_e4m3(self, tensor):
        # Clamp values to the range of float8_e4m3fn
        return torch.clamp(tensor, min=-240, max=240).to(torch.float8_e4m3fn)

    def test_2d_cases(self):
        # Test cases for 2D weights
        test_cases_2d = [
            ((7168, 18432), (128, 128)),
            ((96, 64), (128, 128)),
            ((96, 18432), (128, 128)),
            ((18432, 96), (128, 128)),
            ((3072, 3072), (1, 32)),
            ((3072, 3072), (4, 8)),
        ]

        for (M, N), (block_size_dim1, block_size_dim2) in test_cases_2d:
            with self.subTest(f"2D case: {M}x{N}"):
                # Create input tensor and simulate float8_e4m3fn
                tensor = torch.randn(M, N)
                tensor = self.simulate_float8_e4m3(tensor)

                # Calculate block dimensions
                block_M = min(block_size_dim1, M)
                block_N = min(block_size_dim2, N)
                scale_M, scale_N = M // block_M, N // block_N

                # Create random scale tensor
                scale = torch.rand(scale_M, scale_N) * 10  # Random values between 0 to 10

                # Perform dequantization
                result = blockwise_scale_dequantize(tensor, scale, torch.float32)

                # Verify shape and dtype
                self.assertEqual(result.shape, tensor.shape)
                self.assertEqual(result.dtype, torch.float32)

                # Verify first block scaling
                block_00 = result[:block_M, :block_N]
                scale_00 = scale[0, 0]
                expected_block = tensor[:block_M, :block_N].to(torch.float32) * scale_00
                self.assertTrue(torch.allclose(block_00, expected_block, atol=1e-5))

    def test_3d_cases(self):
        # Test cases for 3D weights
        test_cases_3d = [
            ((5, 7168, 18432), (128, 128)),
            ((5, 96, 64), (128, 128)),
            ((5, 96, 18432), (128, 128)),
            ((5, 18432, 96), (128, 128)),
            ((5, 3072, 3072), (1, 32)),
            ((5, 3072, 3072), (4, 8)),
        ]

        for (C, M, N), (block_size_dim1, block_size_dim2) in test_cases_3d:
            with self.subTest(f"3D case: {C}x{M}x{N}"):
                # Create input tensor and simulate float8_e4m3
                tensor = torch.randn(C, M, N)
                tensor = self.simulate_float8_e4m3(tensor)

                # Calculate block dimensions
                block_M = min(block_size_dim1, M)
                block_N = min(block_size_dim2, N)
                scale_M, scale_N = M // block_M, N // block_N

                # Create random scale tensor
                scale = torch.rand(C, scale_M, scale_N) * 10   # Random values between 0 and 10

                # Perform dequantization
                result = blockwise_scale_dequantize(tensor, scale, torch.float32)

                # Verify shape
                self.assertEqual(result.shape, tensor.shape)
                self.assertEqual(result.dtype, torch.float32)

                # Verify first block scaling
                block_000 = result[0, :block_M, :block_N]
                scale_000 = scale[0, 0, 0]
                expected_block = tensor[0, :block_M, :block_N].to(torch.float32) * scale_000
                self.assertTrue(torch.allclose(block_000, expected_block, atol=1e-5))

    def test_cases_mxfp4_x4(self):
        test_cases_mxfp4 = [
            [(2880, 90, 16), (2880, 90)],
            [(5, 2880, 90, 16), (5, 2880, 90)]
        ]

        for tensor_shape, scale_shape, in test_cases_mxfp4:
            with self.subTest(f"MXFP4 case: {tensor_shape}"):
                # Create input tensor and simulate fp4
                tensor = torch.randint(0, 255, tensor_shape, dtype=torch.uint8)
                tensor = pack_fp4_x4_uint16(tensor)
                scale = torch.randint(low=127-5, high=127+5, size=scale_shape, dtype=torch.uint8)
                expected = get_mxfp4_tensor_from_uint16(tensor, scale, dtype=torch.float32)

                # Perform dequantization
                tensor = tensor.view(QuantizedDtype.F4E2M1FN_X4.value)
                scale = scale.view(ScaleDtype.F8E8M0.value)
                result = blockwise_scale_dequantize(tensor, scale, torch.float32).reshape(expected.shape)

                # Verify
                torch.testing.assert_close(result, expected)

class TestGetShapesForBlockwiseScaleDequantize(unittest.TestCase):
    
    def test_negative(self):
        negative_cases = [
            # tensor     # scale
            [(128, 128), (128, 128)],
            [(128, 128), (256, 256)],
            [(128, 128), (64, 64, 64)],
            [(128, 128), (49, 49)],
        ]
        
        for tensor_shape, scale_shape in negative_cases:
            with self.subTest(f"Negative case: {tensor_shape} and {scale_shape}"):
                with self.assertRaises(AssertionError):
                    get_broadcastable_shapes_for_blockwise_scale_dequantize(tensor_shape, scale_shape)
    
    def test_positive(self):
        positive_cases = [
            # tensor     # scale
            [[(128, 128), (128, 1)], [(128, 128), (128, 1)]],
            [[(128, 128), (1, 64)],  [(128, 64, 2), (1, 64, 1)]],
            [[(128, 128), (64,)], [(64, 2, 128), (64, 1, 1)]],
            [[(128, 128), (64, 64)], [(64, 2, 64, 2), (64, 1, 64, 1)]],
            [[(128, 128, 128), (128, 64, 128)], [(128, 64, 2, 128), (128, 64, 1, 128)]],
        ]
        for (tensor_shape, scale_shape), (expected_tensor_shape, expected_scale_shape) in positive_cases:
            with self.subTest(f"Positive case: {tensor_shape} and {scale_shape}"):
                new_tensor_shape, new_scale_shape = get_broadcastable_shapes_for_blockwise_scale_dequantize(tensor_shape, scale_shape)
                self.assertEqual(new_tensor_shape, expected_tensor_shape)
                self.assertEqual(new_scale_shape, expected_scale_shape)


if __name__ == "__main__":
    unittest.main()