import unittest

import pytest
import torch
import torch.ao.quantization

from neuronx_distributed.quantization.quantization_utils import (
    extract_q_scale,
    quantize_per_channel_symmetric,
    quantize_per_tensor_symmetric,
    quantize_fp8_per_channel
)


class TestExtractQscale(unittest.TestCase):
    def test_extract_q_scale_per_tensor(self):
        q_tensor = torch.quantize_per_tensor(torch.randn(4, 4), 0.1, 2, torch.quint8)
        assert extract_q_scale(q_tensor) == 0.1

        q_tensor = torch.quantize_per_channel(
            torch.randn(4, 4),
            torch.tensor([0.1, 0.2, 0.3, 0.4]),
            torch.tensor([0, 1, 2, 3]),
            0,
            torch.qint8,
        )
        assert torch.allclose(extract_q_scale(q_tensor), torch.Tensor([[0.1], [0.2], [0.3], [0.4]]))


@pytest.mark.skip("Not testing convert_qint8_to_int8_state_dict as its a temporary solution before refactoring")
class TestConvertQint8ToInt8StateDict(unittest.TestCase):
    def test_convert_qint8_to_int8_state_dict(self):
        pass


@pytest.mark.skip("Customer facing optional code. Test later")
class TestQuantizePytorchModelPerChannelSymmetric(unittest.TestCase):
    def test_quantize_pytorch_model_per_channel_symmetric(self):
        pass


@pytest.mark.skip("Customer facing optional code. Test later")
class TestQuantizePytorchModelPerTensorSymmetric(unittest.TestCase):
    def test_quantize_pytorch_model_per_tensor_symmetric(self):
        pass


class TestQuantizePerTensorSymmetric(unittest.TestCase):
    # Tests quantize_per_tensor_symmetric function
    def test_quantize_per_tensor_symmetric(self):
        tensor = torch.Tensor([[1.0, -190, -950, 900, 100]])
        expected_scale = 950 / (float(127 - (-128)) / 2)
        expected_scale = torch.tensor(expected_scale)

        # We just verify the scale. Once that is fine, _quantize method is already pytorch tested
        quantized_tensor = quantize_per_tensor_symmetric(tensor)
        assert torch.allclose(torch.tensor(quantized_tensor.q_scale()), expected_scale)


class TestQuantizePerChannelSymmetric(unittest.TestCase):
    # Tests quantize_per_channel_symmetric function
    def test_quantize_per_channel_symmetric(self):
        tensor = torch.Tensor(
            [
                [1.0, -190, -950, 900, 100],
                [2.0, 255, -900, 80, -80],
            ]
        )

        # channel axis = 1
        expected_scale = torch.Tensor([2.0, 255, 950, 900, 100]) / 127.0
        quantized_tensor = quantize_per_channel_symmetric(tensor, 1)
        assert torch.allclose(quantized_tensor.q_per_channel_scales().to(torch.float32), expected_scale)

        # channel axis = 0
        expected_scale = torch.Tensor([950, 900]) / 127
        quantized_tensor = quantize_per_channel_symmetric(tensor, 0)
        assert torch.allclose(quantized_tensor.q_per_channel_scales().to(torch.float32), expected_scale)


class TestQuantizeFP8PerChannel(unittest.TestCase):
    def test_quantize_fp8_per_channel(self):
        tensor = torch.tensor(data=
            [
                [1.0, -150, 2.5, 300],
                [1.0, 50, -50, 2.5],
            ], 
            dtype=torch.bfloat16
        )
        expected_quantized_tensor = torch.tensor(
            [
                [240.0, -240.0, 12.0, 240.0],
                [240.0, 80.0, -240.0, 3.0],
            ],
            dtype=torch.bfloat16
        )

        expected_quantized_tensor_without_clamp_bound = torch.tensor(
            [
                [240.0, -240.0, 12.0, 240.0],
                [240.0, 80.0, -240.0, 2.0],
            ],
            dtype=torch.bfloat16
        )
        expected_scale = torch.tensor([1, 150, 50, 200])/240.0
        expected_scale_without_clamp_bound = torch.tensor([1, 150, 50, 300])/240.0

        actual_quantized_tensor, actual_scale = quantize_fp8_per_channel(tensor=tensor, dtype=torch.float8_e4m3fn, channel_axis=1, clamp_bound=200)
        actual_quantized_tensor_without_clamp_bound, actual_scale_without_clamp_bound = quantize_fp8_per_channel(tensor=tensor, dtype=torch.float8_e4m3fn, channel_axis=1)
        actual_quantized_tensor = actual_quantized_tensor.to(torch.bfloat16)
        actual_quantized_tensor_without_clamp_bound = actual_quantized_tensor_without_clamp_bound.to(torch.bfloat16)
        assert torch.allclose(actual_quantized_tensor, expected_quantized_tensor)
        assert torch.allclose(actual_scale, expected_scale)
        assert torch.allclose(actual_quantized_tensor_without_clamp_bound , expected_quantized_tensor_without_clamp_bound)
        assert torch.allclose(actual_scale_without_clamp_bound , expected_scale_without_clamp_bound)
