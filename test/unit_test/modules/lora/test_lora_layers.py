# Standard Library
import unittest
from unittest.mock import MagicMock, patch

import torch
from neuronx_distributed.parallel_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear

from neuronx_distributed.modules.lora import LoraConfig
from neuronx_distributed.modules.lora.layer import (
    LoraLinear,
    LoraConv2d,
    LoraEmbedding,
)

from neuronx_distributed.modules.lora.tp_layer import (
    LoraParallelLinear,
    LoraGQAQKVParallelLinear,
)
from . import MockGroup


def get_lora_config(use_rslora=False, init_lora_weights="default"):
    return LoraConfig(
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
        layers = [ColumnParallelLinear(32, 32, tensor_model_parallel_group=MockGroup()), RowParallelLinear(32, 32, tensor_model_parallel_group=MockGroup())]
        rslora_modes = [False, True]
        init_weights_modes = ["default", "gaussian"]

        for layer in layers:
            for rslora in rslora_modes:
                for init_mode in init_weights_modes:
                    lora_config = get_lora_config(use_rslora=rslora, init_lora_weights=init_mode)
                    lora_layer = LoraParallelLinear(layer, lora_config)
                    layer_str = str(lora_layer)
                    assert "lora" in layer_str

    @patch("neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_kv_group", MagicMock(return_value=True))
    @patch("neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=True))
    @patch("neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True))
    @patch("neuronx_distributed.modules.qkv_linear.GQAQKVColumnParallelLinear.initialize_weight_biases", MagicMock(return_value=True))
    def test_gqaqkv_linear_layers(self):
        hidden_size = 32
        num_heads = 32
        input_size = hidden_size

        kv_size_multipliers = [1, 4]
        fuse_qkvs = [True, False]
        init_weights_modes = ["default", "gaussian"]
        rslora_modes = [False, True]

        for kv_size_multiplier, fuse_qkv in zip(kv_size_multipliers, fuse_qkvs):
            num_heads_kv_group = num_heads // kv_size_multiplier
            output_sizes = [num_heads*hidden_size, num_heads_kv_group*hidden_size]
            layer = GQAQKVColumnParallelLinear(input_size, output_sizes, kv_size_multiplier, fuse_qkv=fuse_qkv)
            for rslora, init_mode in zip(rslora_modes, init_weights_modes):
                lora_config = get_lora_config(use_rslora=rslora, init_lora_weights=init_mode)
                lora_layer = LoraGQAQKVParallelLinear(layer, lora_config)
                layer_str = str(lora_layer)
                assert "lora" in layer_str
                lora_B_layer = lora_layer.lora_B
                q, k, v = lora_layer.get_qkv(lora_layer.lora_B)
                assert q.shape == (lora_B_layer.q_output_size_per_partition, lora_layer.lora_rank)
                assert k.shape == (lora_B_layer.kv_output_size_per_partition, lora_layer.lora_rank)
                assert v.shape == (lora_B_layer.kv_output_size_per_partition, lora_layer.lora_rank)


if __name__ == "__main__":
    unittest.main()
