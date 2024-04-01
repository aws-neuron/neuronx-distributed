# Standard Library
import unittest
from unittest.mock import MagicMock, patch

# Third Party
import torch
from transformers import AutoModelForCausalLM, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import neuronx_distributed as nxd

from .. import update_result


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


class TestModelWrapper(unittest.TestCase):
    @patch("neuronx_distributed.pipeline.model.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=8)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_tensor_model_parallel_size", MagicMock(return_value=8)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_tensor_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("neuronx_distributed.pipeline.partition.get_pipeline_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.model.NxDPPModel._create_pg_with_ranks", MagicMock(return_value=None))
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("torch.distributed.get_rank")
    def test_model_wrapper(self, rank_mock):
        try:
            pipeline_cuts = [
                "transformer.h.1",
                "transformer.h.2",
                "transformer.h.3",
                "transformer.h.4",
                "transformer.h.5",
                "transformer.h.6",
                "transformer.h.7",
            ]
            nxd_config = nxd.neuronx_distributed_config(
                tensor_parallel_size=8,
                pipeline_parallel_size=8,
                pipeline_config={
                    "transformer_layer_cls": GPT2Block,
                    "tracer_cls": "hf",
                    "num_microbatches": 1,
                    "output_loss_value_spec": True,
                    "input_names": ["input_ids", "attention_mask", "labels"],
                    "pipeline_cuts": pipeline_cuts,
                    "param_init_fn": None,
                    "leaf_module_cls": ["GPT2Block"],
                    "use_zero1_optimizer": True,
                    "use_optimizer_wrapper": True,
                },
                optimizer_config={
                    "zero_one_enabled": True,
                    "grad_clipping": True,
                    "max_grad_norm": 1.0,
                },
                sequence_parallel=True,
                activation_checkpoint_config="full",
            )
            model = nxd.initialize_parallel_model(nxd_config, get_model)

            assert isinstance(model, nxd.trainer.model.NxDModel)
            assert model.nxd_config == nxd_config
            assert model.pp_enabled
            assert model.dtype == torch.float32
            model_str = str(model)
            assert "NxDPPModel" in model_str
            assert "NxDCheckpointWrapper" in model_str

        except:
            update_result({"inference_success": 0})
            raise

    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("torch.distributed.get_rank")
    def test_model_wrapper_no_pp(self, rank_mock):
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
            )
            model = nxd.initialize_parallel_model(nxd_config, get_model)

            assert isinstance(model, nxd.trainer.model.NxDModel)
            assert model.nxd_config == nxd_config
            assert not model.pp_enabled
            assert model.dtype == torch.float32
            model_str = str(model)
            assert "NxDPPModel" not in model_str
            assert "NxDCheckpointWrapper" in model_str

        except:
            update_result({"inference_success": 0})
            raise


if __name__ == "__main__":
    unittest.main()
