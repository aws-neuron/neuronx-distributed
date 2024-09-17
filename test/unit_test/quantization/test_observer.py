import unittest

import torch

from neuronx_distributed.quantization.observer import PerChannelAbsMaxObserver


class TestPerChannelAbsMaxObserver(unittest.TestCase):
    def test_forward(self):
        # Channel axis = 0
        tensor_observer = PerChannelAbsMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0
        )()
        tensor = torch.Tensor(
            [
                [1.0, -190, -950, 900, 100],
                [2.0, 255, -900, 80, -80],
            ]
        )

        tensor_observer(tensor)

        expected_max_val = torch.Tensor([[950.0], [900.0]])
        expected_scale = expected_max_val / 127.0
        assert torch.allclose(tensor_observer.max_val, expected_max_val)
        assert torch.allclose(tensor_observer.calculate_qparams()[0], expected_scale.squeeze(1))

        # Channel axis = 1
        tensor_observer = PerChannelAbsMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=1
        )()
        tensor = torch.Tensor(
            [
                [1.0, -190, -950, 900, 100],
                [2.0, 255, -900, 80, -80],
            ]
        )

        tensor_observer(tensor)
        expected_max_val = torch.Tensor([[2.0], [255.0], [950.0], [900.0], [100.0]])
        expected_scale = expected_max_val / 127.0
        assert torch.allclose(tensor_observer.max_val, expected_max_val)
        assert torch.allclose(tensor_observer.calculate_qparams()[0], expected_scale.squeeze(1))
