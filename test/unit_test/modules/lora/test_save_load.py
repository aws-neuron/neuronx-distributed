# Standard Library
import unittest
from unittest.mock import MagicMock, patch

import torch
from neuronx_distributed.parallel_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.modules.lora import LoraConfig, LoraModel, get_lora_model


class NxDModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rpl = ColumnParallelLinear(32, 32)
        self.cpl = RowParallelLinear(32, 32)
        self.linear = torch.nn.Linear(32, 32)



class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
        self.conv2d = torch.nn.Conv2d(32, 32, 4)
        self.embedding = torch.nn.Embedding(32, 32)



def get_nxd_lora_config(save_lora_base, merge_lora, load_lora_from_ckpt=False):
    return LoraConfig(
        enable_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_verbose=True,
        target_modules=["rpl", "cpl"],
        save_lora_config_adapter=False,
        save_lora_base=save_lora_base,
        merge_lora=merge_lora,
        load_lora_from_ckpt=load_lora_from_ckpt,
    )

def get_lora_config(save_lora_base, merge_lora, load_lora_from_ckpt=False):
    return LoraConfig(
        enable_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_verbose=True,
        target_modules=["linear", "conv2d", "embedding"],
        save_lora_config_adapter=False,
        save_lora_base=save_lora_base,
        merge_lora=merge_lora,
        load_lora_from_ckpt=load_lora_from_ckpt,
    )



class TestLoraSaveLoad(unittest.TestCase):
    def test_save_load_no_base_single_device(self):
        model = Module()
        base_model_state_dict = model.state_dict()
        lora_config = get_lora_config(save_lora_base=False, merge_lora=False)
        model = LoraModel(model, lora_config)
        state_dict = model.state_dict()
        for key in state_dict:
            assert "lora_" in key

        model.lora_config = get_lora_config(save_lora_base=False, merge_lora=False, load_lora_from_ckpt=True)
        model.lora_ckpt = state_dict
        model.is_checkpoint_loaded = True

        load_result = model.load_state_dict(base_model_state_dict)
        assert load_result is not None
        assert len(load_result.unexpected_keys) == 0


    def test_save_load_with_base_single_device(self):
        model = Module()
        lora_config = get_lora_config(save_lora_base=True, merge_lora=False)
        model = LoraModel(model, lora_config)
        state_dict = model.state_dict()

        model.lora_config = get_lora_config(save_lora_base=True, merge_lora=False, load_lora_from_ckpt=True)
        model.lora_ckpt = state_dict
        model.is_checkpoint_loaded = True

        load_result = model.load_state_dict()
        assert load_result is not None
        assert len(load_result.missing_keys) == 0
        assert len(load_result.unexpected_keys) == 0

    def test_save_load_with_base_merged_single_device(self):
        model = Module()
        base_model_state_dict = model.state_dict()
        base_model_keys = base_model_state_dict.keys()
        lora_config = get_lora_config(save_lora_base=True, merge_lora=True)
        model = LoraModel(model, lora_config)
        state_dict = model.state_dict()
        keys = state_dict.keys()

        for key in keys:
            assert key in base_model_keys

        for key in base_model_keys:
            assert key in keys

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    def test_save_load_no_base(self):
        model = NxDModule()
        base_model_state_dict = model.state_dict()

        lora_config = get_nxd_lora_config(save_lora_base=False, merge_lora=False)
        model = get_lora_model(model, lora_config)

        state_dict = model.state_dict()
        for key in state_dict:
            assert "lora_" in key

        model.lora_config = get_nxd_lora_config(save_lora_base=False, merge_lora=False, load_lora_from_ckpt=True)
        model.lora_ckpt = state_dict
        model.is_checkpoint_loaded = True

        load_result = model.load_state_dict(base_model_state_dict)
        assert load_result is not None
        assert len(load_result.unexpected_keys) == 0

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    def test_save_load_with_base(self):
        model = NxDModule()
        lora_config = get_nxd_lora_config(save_lora_base=True, merge_lora=False)
        model = get_lora_model(model, lora_config)
        state_dict = model.state_dict()

        model.lora_config = get_nxd_lora_config(save_lora_base=True, merge_lora=False, load_lora_from_ckpt=True)
        model.lora_ckpt = state_dict
        model.is_checkpoint_loaded = True

        load_result = model.load_state_dict(None)
        assert load_result is not None
        assert len(load_result.missing_keys) == 0
        assert len(load_result.unexpected_keys) == 0

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    def test_save_load_with_base_merged(self):
        model = NxDModule()
        base_model_state_dict = model.state_dict()
        base_model_keys = base_model_state_dict.keys()

        lora_config = get_nxd_lora_config(save_lora_base=True, merge_lora=True)
        model = get_lora_model(model, lora_config)
        state_dict = model.state_dict()
        keys = state_dict.keys()

        for key in keys:
            assert key in base_model_keys

        for key in base_model_keys:
            assert key in keys


if __name__ == "__main__":
    unittest.main()
