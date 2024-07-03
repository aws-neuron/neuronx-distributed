# Standard Library
import unittest
from unittest.mock import MagicMock, patch

# Third Party
import torch
from transformers import AutoModelForCausalLM, GPT2Config

import neuronx_distributed as nxd
from neuronx_distributed.modules.lora import LoraConfig, LoraModel, get_lora_model

from ... import update_result


def get_model():
    seq_len = 512
    model_config = GPT2Config(
        vocab_size=50257,
        n_positions=seq_len,
        n_embd=768,
        n_layer=8,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-05,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.0,
        use_cache=False,
        bos_token_id=50256,
        eos_token_id=50256,
        return_dict=False,
    )
    model = AutoModelForCausalLM.from_config(model_config)
    return model

    
def get_lora_config(save_lora_base, merge_lora, load_lora_from_ckpt=False):
    lora_config = LoraConfig(
        enable_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        lora_verbose=False,
        target_modules=["c_attn"],
        save_lora_config_adapter=False,
        save_lora_base=save_lora_base,
        merge_lora=merge_lora,
        load_lora_from_ckpt=load_lora_from_ckpt,
    )
    return lora_config


class TestModelWrapper(unittest.TestCase):
    def test_save_load_no_base_single_device(self):
        try:
            model = get_model()
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
        except:
            update_result({"inference_success": 0})
            raise
        
        
    
    def test_save_load_with_base_single_device(self):
        try:
            model = get_model()
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
        except:
            update_result({"inference_success": 0})
            raise
        
    
    def test_save_load_with_base_merged_single_device(self):
        try:
            model = get_model()
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
        except:
            update_result({"inference_success": 0})
            raise
        
        
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("torch.distributed.get_rank")
    def test_save_load_no_base(self, rank_mock):
        try:
            lora_config = get_lora_config(save_lora_base=False, merge_lora=False)
            nxd_config = nxd.neuronx_distributed_config(
                tensor_parallel_size=8,
                optimizer_config={
                    "zero_one_enabled": True,
                    "grad_clipping": True,
                    "max_grad_norm": 1.0,
                },
                sequence_parallel=True,
                activation_checkpoint_config="full",
            )
            model = nxd.initialize_parallel_model(nxd_config, get_model)
            base_model_state_dict = model.state_dict()
            model = get_lora_model(model, lora_config)
            state_dict = model.state_dict()
            for key in state_dict:
                assert "lora_" in key
            
            model.module.lora_config = get_lora_config(save_lora_base=False, merge_lora=False, load_lora_from_ckpt=True)
            model.module.lora_ckpt = state_dict
            model.module.is_checkpoint_loaded = True
            
            load_result = model.load_state_dict(base_model_state_dict)
            assert load_result is not None
            assert len(load_result.unexpected_keys) == 0
        except:
            update_result({"inference_success": 0})
            raise
        
        
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("torch.distributed.get_rank")
    def test_save_load_with_base(self, rank_mock):
        try:
            lora_config = get_lora_config(save_lora_base=True, merge_lora=False)
            nxd_config = nxd.neuronx_distributed_config(
                tensor_parallel_size=8,
                optimizer_config={
                    "zero_one_enabled": True,
                    "grad_clipping": True,
                    "max_grad_norm": 1.0,
                },
                sequence_parallel=True,
                activation_checkpoint_config="full",
                lora_config=lora_config,
            )
            model = nxd.initialize_parallel_model(nxd_config, get_model)
            state_dict = model.state_dict()
            
            model.module.lora_config = get_lora_config(save_lora_base=True, merge_lora=False, load_lora_from_ckpt=True)
            model.module.lora_ckpt = state_dict
            model.module.is_checkpoint_loaded = True
            
            load_result = model.load_state_dict(None)
            assert load_result is not None
            assert len(load_result.missing_keys) == 0
            assert len(load_result.unexpected_keys) == 0
        except:
            update_result({"inference_success": 0})
            raise
        
    
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("torch.distributed.get_rank")
    def test_save_load_with_base_merged(self, rank_mock):
        try:
            lora_config = get_lora_config(save_lora_base=True, merge_lora=True)
            nxd_config = nxd.neuronx_distributed_config(
                tensor_parallel_size=8,
                optimizer_config={
                    "zero_one_enabled": True,
                    "grad_clipping": True,
                    "max_grad_norm": 1.0,
                },
                sequence_parallel=True,
                activation_checkpoint_config="full",
            )
            model = nxd.initialize_parallel_model(nxd_config, get_model)
            base_model_state_dict = model.state_dict()
            base_model_keys = base_model_state_dict.keys()
            model = get_lora_model(model, lora_config)
            state_dict = model.state_dict()
            keys = state_dict.keys()
            
            for key in keys:
                assert key in base_model_keys
                
            for key in base_model_keys:
                assert key in keys
        except:
            update_result({"inference_success": 0})
            raise
    

if __name__ == "__main__":
    unittest.main()
