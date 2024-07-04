# Standard Library
import unittest
from unittest.mock import MagicMock, patch

# Third Party
import torch
from transformers import AutoModelForCausalLM, BertConfig, GPT2Config, LlamaConfig

import neuronx_distributed as nxd
from neuronx_distributed.modules.lora import LoraConfig, LoraModel

from ... import update_result


def get_gpt2_model():
    seq_len = 512
    model_config = GPT2Config(
        vocab_size=50257,
        n_positions=seq_len,
        n_embd=768,
        n_layer=4,
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


def get_bert_model():
    model_config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=4,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
    )
    model = AutoModelForCausalLM.from_config(model_config)
    return model


def get_llama_model():
    model_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=4,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
    )
    model = AutoModelForCausalLM.from_config(model_config)
    return model


def get_lora_config(target_modules):
    lora_config = LoraConfig(
        enable_lora=True,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        lora_verbose=True,
        target_modules=target_modules,
    )
    return lora_config


class TestModelWrapper(unittest.TestCase):
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("torch.distributed.get_rank")
    def test_gpt2_model_wrapper(self, rank_mock):
        try:
            nxd_config = nxd.neuronx_distributed_config(
                tensor_parallel_size=8,
                optimizer_config={
                    "zero_one_enabled": True,
                    "grad_clipping": True,
                    "max_grad_norm": 1.0,
                },
                sequence_parallel=True,
                activation_checkpoint_config="full",
                lora_config=get_lora_config(target_modules=["c_attn"]),
            )
            model = nxd.initialize_parallel_model(nxd_config, get_gpt2_model)

            assert isinstance(model, nxd.trainer.model.NxDModel)
            assert model.nxd_config == nxd_config
            assert not model.pp_enabled
            assert model.dtype == torch.float32
            assert isinstance(model.module, LoraModel)
            model_str = str(model)
            assert "NxDPPModel" not in model_str
            assert "NxDCheckpointWrapper" in model_str
            assert "LoraModel" in model_str

        except:
            update_result({"inference_success": 0})
            raise

    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("torch.distributed.get_rank")
    def test_bert_model_wrapper(self, rank_mock):
        try:
            nxd_config = nxd.neuronx_distributed_config(
                tensor_parallel_size=8,
                optimizer_config={
                    "zero_one_enabled": True,
                    "grad_clipping": True,
                    "max_grad_norm": 1.0,
                },
                sequence_parallel=True,
                lora_config=get_lora_config(target_modules=["query", "value"]),
            )
            model = nxd.initialize_parallel_model(nxd_config, get_bert_model)

            assert isinstance(model, nxd.trainer.model.NxDModel)
            assert model.nxd_config == nxd_config
            assert not model.pp_enabled
            assert model.dtype == torch.float32
            assert isinstance(model.module, LoraModel)
            model_str = str(model)
            assert "NxDPPModel" not in model_str
            assert "LoraModel" in model_str

        except:
            update_result({"inference_success": 0})
            raise

    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("torch.distributed.get_rank")
    def test_llama_model_wrapper(self, rank_mock):
        try:
            nxd_config = nxd.neuronx_distributed_config(
                tensor_parallel_size=8,
                optimizer_config={
                    "zero_one_enabled": True,
                    "grad_clipping": True,
                    "max_grad_norm": 1.0,
                },
                sequence_parallel=True,
                activation_checkpoint_config="full",
                lora_config=get_lora_config(target_modules=["q_proj", "v_proj"]),
            )
            model = nxd.initialize_parallel_model(nxd_config, get_llama_model)

            assert isinstance(model, nxd.trainer.model.NxDModel)
            assert model.nxd_config == nxd_config
            assert not model.pp_enabled
            assert model.dtype == torch.float32
            assert isinstance(model.module, LoraModel)
            model_str = str(model)
            assert "NxDPPModel" not in model_str
            assert "NxDCheckpointWrapper" in model_str
            assert "LoraModel" in model_str

        except:
            update_result({"inference_success": 0})
            raise


if __name__ == "__main__":
    unittest.main()
