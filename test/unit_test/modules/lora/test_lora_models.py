# Standard Library
import unittest
from unittest.mock import MagicMock, patch

# Third Party
import torch
from neuronx_distributed.parallel_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear

from neuronx_distributed.modules.lora import LoraConfig, LoraModel
from . import MockGroup


class NxDModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rpl = ColumnParallelLinear(32, 32, tensor_model_parallel_group=MockGroup())
        self.cpl = RowParallelLinear(32, 32, tensor_model_parallel_group=MockGroup())
        self.qkv = GQAQKVColumnParallelLinear(32, [4 * 32, 1 * 32])
        self.linear = torch.nn.Linear(32, 32)


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
        self.conv2d = torch.nn.Conv2d(32, 32, 4)
        self.embedding = torch.nn.Embedding(32, 32)


def get_nxd_lora_config(bias="none"):
    return LoraConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_verbose=False,
        bias=bias,
        target_modules=["rpl", "cpl", "qkv"],
    )


def get_lora_config(bias="none"):
    return LoraConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias=bias,
        lora_verbose=True,
        target_modules=["linear", "conv2d", "embedding"],
    )


class TestLoraModels(unittest.TestCase):
    def test_torch_model(self):
        bias_modes = ["none", "all", "lora_only"]
        for mode in bias_modes:
            model = Module()
            lora_config=get_lora_config(bias=mode)
            model = LoraModel(model, lora_config)
            assert isinstance(model, LoraModel)
            model_str = str(model)
            assert "LoraModel" in model_str

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=True))
    @patch("neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True))
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.modules.qkv_linear.GQAQKVColumnParallelLinear.initialize_weight_biases", MagicMock(return_value=True))
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_kv_group", MagicMock(return_value=True))
    def test_nxd_model(self):
        bias_modes = ["none", "all", "lora_only"]
        for mode in bias_modes:
            model = NxDModule()
            lora_config=get_nxd_lora_config(bias=mode)
            model = LoraModel(model, lora_config)
            assert isinstance(model, LoraModel)
            model_str = str(model)
            assert "LoraModel" in model_str


if __name__ == "__main__":
    unittest.main()
