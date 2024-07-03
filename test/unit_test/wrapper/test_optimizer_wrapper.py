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


class TestOptimizerWrapper(unittest.TestCase):
    @patch(
        "neuronx_distributed.optimizer.zero_redundancy_optimizer.NeuronZero1Optimizer.step",
        MagicMock(return_value=None),
    )
    @patch(
        "neuronx_distributed.optimizer.zero_redundancy_optimizer.NeuronZero1Optimizer.zero_grad",
        MagicMock(return_value=None),
    )
    @patch(
        "neuronx_distributed.optimizer.zero_redundancy_optimizer.NeuronZero1Optimizer.state_dict",
        MagicMock(return_value=None),
    )
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
    @patch("neuronx_distributed.pipeline.partition.get_pipeline_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.pipeline.model.NxDPPModel._create_pg_with_ranks", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.get_data_parallel_group",
        MagicMock(return_value=[list(range(32))]),
    )
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_group",
        MagicMock(return_value=None),
    )
    @patch(
        "neuronx_distributed.optimizer.zero_redundancy_optimizer.model_parallel_is_initialized",
        MagicMock(return_value=True),
    )
    @patch(
        "neuronx_distributed.optimizer.zero_redundancy_optimizer.get_data_parallel_group",
        MagicMock(return_value=[list(range(32))]),
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("torch.distributed.get_rank")
    def test_optimizer_wrapper(self, rank_mock):
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
            optimizer = nxd.initialize_parallel_optimizer(nxd_config, torch.optim.AdamW, model.parameters(), lr=1e-3)

            assert optimizer.nxd_config == nxd_config
            assert isinstance(optimizer, nxd.trainer.optimizer.NxDOptimizer)
            assert isinstance(optimizer.optimizer, nxd.optimizer.NeuronZero1Optimizer)
            assert isinstance(optimizer.optimizer.base_optimizer, torch.optim.AdamW)
            assert len(list(model.parameters())) == len(optimizer.params)
            assert optimizer.grad_norm is None

            for method in ["step", "zero_grad", "state_dict"]:
                getattr(optimizer, method)()
                assert getattr(nxd.optimizer.zero_redundancy_optimizer.NeuronZero1Optimizer, method).called

        except:
            update_result({"inference_success": 0})
            raise

    @patch("torch.optim.AdamW.step", MagicMock(return_value=None))
    @patch("torch.optim.AdamW.zero_grad", MagicMock(return_value=None))
    @patch("torch.optim.AdamW.state_dict", MagicMock(return_value=None))
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
    @patch("neuronx_distributed.pipeline.partition.get_pipeline_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.pipeline.model.NxDPPModel._create_pg_with_ranks", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.get_data_parallel_group",
        MagicMock(return_value=[list(range(32))]),
    )
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_group",
        MagicMock(return_value=None),
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("neuronx_distributed.trainer.optimizer.grads.clip_grad_norm", MagicMock(return_value=None))
    @patch("torch.distributed.get_rank")
    def test_optimizer_wrapper_no_zero1(self, rank_mock):
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
                    "zero_one_enabled": False,
                    "grad_clipping": True,
                    "max_grad_norm": 1.0,
                },
                sequence_parallel=True,
                activation_checkpoint_config="full",
            )
            model = nxd.initialize_parallel_model(nxd_config, get_model)
            optimizer = nxd.initialize_parallel_optimizer(nxd_config, torch.optim.AdamW, model.parameters(), lr=1e-3)

            assert optimizer.nxd_config == nxd_config
            assert isinstance(optimizer, nxd.trainer.optimizer.NxDOptimizer)
            assert isinstance(optimizer.optimizer, torch.optim.AdamW)
            assert len(list(model.parameters())) == len(optimizer.params)
            assert optimizer.grad_norm is None

            for method in ["step", "zero_grad", "state_dict"]:
                getattr(optimizer, method)()
                assert getattr(torch.optim.AdamW, method).called, method

        except:
            update_result({"inference_success": 0})
            raise


if __name__ == "__main__":
    unittest.main()
