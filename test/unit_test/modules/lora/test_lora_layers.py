# Standard Library
import unittest
from unittest.mock import MagicMock, patch

import torch
from neuronx_distributed.parallel_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from neuronx_distributed.modules.lora import LoraConfig
from neuronx_distributed.modules.lora.layer import (
    LoraLinear,
    LoraConv2d,
    LoraEmbedding,
)

from neuronx_distributed.modules.lora.tp_layer import (
    LoraParallelLinear,
)


def get_lora_config(use_rslora=False, init_lora_weights="default"):
    return LoraConfig(
        enable_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        use_rslora=use_rslora,
        lora_verbose=False,
        init_lora_weights=init_lora_weights,
    )


class TestLoraLayers(unittest.TestCase):
    def test_torch_linear_layer(self):
        layer = torch.nn.Linear(32, 32)
        rslora_modes = [False, True]
        init_weights_modes = ["default", "gaussian"]

        for rslora in rslora_modes:
            for init_mode in init_weights_modes:
                lora_config = get_lora_config(use_rslora=rslora, init_lora_weights=init_mode)
                lora_layer = LoraLinear(layer, lora_config)
                layer_str = str(lora_layer)
                assert "lora" in layer_str

    def test_torch_conv2d_layer(self):
        layer = torch.nn.Conv2d(32, 32, 2)
        rslora_modes = [False, True]
        init_weights_modes = ["default", "gaussian"]

        for rslora in rslora_modes:
            for init_mode in init_weights_modes:
                lora_config = get_lora_config(use_rslora=rslora, init_lora_weights=init_mode)
                lora_layer = LoraConv2d(layer, lora_config)
                layer_str = str(lora_layer)
                assert "lora" in layer_str

    def test_torch_embedding_layer(self):
        layer = torch.nn.Embedding(32, 32)
        rslora_modes = [False, True]
        init_weights_modes = ["default", "gaussian"]

        for rslora in rslora_modes:
            for init_mode in init_weights_modes:
                lora_config = get_lora_config(use_rslora=rslora, init_lora_weights=init_mode)
                lora_layer = LoraEmbedding(layer, lora_config)
                layer_str = str(lora_layer)
                assert "lora" in layer_str

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=True))
    @patch("neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True))
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=8))
    def test_tp_linear_layers(self):
        layers = [ColumnParallelLinear(32, 32), RowParallelLinear(32, 32)]
        rslora_modes = [False, True]
        init_weights_modes = ["default", "gaussian"]

        for layer in layers:
            for rslora in rslora_modes:
                for init_mode in init_weights_modes:
                    lora_config = get_lora_config(use_rslora=rslora, init_lora_weights=init_mode)
                    lora_layer = LoraParallelLinear(layer, lora_config)
                    layer_str = str(lora_layer)
                    assert "lora" in layer_str


if __name__ == "__main__":
    unittest.main()
