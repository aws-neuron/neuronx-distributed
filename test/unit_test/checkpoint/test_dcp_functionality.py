import torch
import pytest
from packaging import version

if version.parse(torch.__version__) != version.parse("2.1"):
    pytest.skip("skip this test", allow_module_level=True)

# Standard Library
import os
import unittest
from copy import deepcopy
from unittest.mock import MagicMock, patch

# Third Party
import torch_xla.core.xla_model as xm
import torch_xla.utils.serialization as xser
from transformers import AutoModelForCausalLM, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import neuronx_distributed as nxd
from neuronx_distributed.optimizer.zero_dcp_utils import (
    _get_optim_pid_to_params,
    _get_optim_pid_to_param_names,
    _get_param_to_param_names,
    _wrap_optim_state_dict,
    _unwrap_optim_state_dict,
    get_dcp_aux_infos,
)


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


class DCPFunctionalityTest(unittest.TestCase):
    @patch("torch.distributed.is_initialized", MagicMock(return_value=True))
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
        "neuronx_distributed.parallel_layers.parallel_state.get_data_parallel_replica_groups",
        MagicMock(return_value=[[i] for i in range(64)]),
    )
    @patch(
        "neuronx_distributed.trainer.checkpoint.get_data_parallel_replica_groups",
        MagicMock(return_value=[[i] for i in range(64)]),
    )
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_replica_groups",
        MagicMock(return_value=[[i] for i in range(64)]),
    )
    @patch(
        "neuronx_distributed.optimizer.zero_redundancy_optimizer.model_parallel_is_initialized",
        MagicMock(return_value=True),
    )
    @patch(
        "neuronx_distributed.optimizer.zero_redundancy_optimizer.get_data_parallel_replica_groups",
        MagicMock(return_value=[[i] for i in range(64)]),
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("neuronx_distributed.trainer.checkpoint.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.trainer.checkpoint.get_pipeline_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.trainer.checkpoint.get_local_world_size", MagicMock(return_value=32))
    @patch("neuronx_distributed.trainer.trainer.get_expert_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.trainer.checkpoint.get_expert_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.trainer.checkpoint.get_expert_model_parallel_rank", MagicMock(return_value=0))
    @patch("neuronx_distributed.trainer.checkpoint.get_expert_data_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.trainer.checkpoint.get_expert_data_parallel_rank", MagicMock(return_value=0))
    @patch(
        "neuronx_distributed.trainer.checkpoint.get_expert_model_parallel_replica_groups",
        MagicMock(return_value=[[i] for i in range(64)]),
    )
    @patch(
        "neuronx_distributed.trainer.checkpoint.get_expert_data_parallel_replica_groups",
        MagicMock(return_value=[[i] for i in range(64)]),
    )
    @patch(
        "neuronx_distributed.trainer.checkpoint.model_parallel_is_initialized",
        MagicMock(return_value=True),
    )
    @patch("neuronx_distributed.optimizer.zero_dcp_utils.get_pipeline_model_parallel_rank", MagicMock(return_value=1))
    @patch(
        "neuronx_distributed.optimizer.zero_dcp_utils.get_tensor_model_parallel_replica_groups",
        MagicMock(return_value=[[i] for i in range(8)]),
    )
    @patch(
        "neuronx_distributed.optimizer.zero_dcp_utils.get_pipeline_model_parallel_replica_groups",
        MagicMock(return_value=[[i] for i in range(8)]),
    )
    @patch(
        "neuronx_distributed.optimizer.zero_dcp_utils.get_data_parallel_replica_groups",
        MagicMock(return_value=[[i] for i in range(64)]),
    )
    @patch("torch.distributed.get_rank", MagicMock(return_value=0))
    @patch("torch.distributed.get_world_size", MagicMock(return_value=1))
    def test_checkpoint_dcp(self):
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

        for k, v in _get_optim_pid_to_params(optimizer).items():
            assert isinstance(k, int)
            assert isinstance(v, torch.nn.Parameter)
        for k, v in _get_param_to_param_names(model).items():
            assert isinstance(k, torch.nn.Parameter)
            assert isinstance(v, str) and v.startswith("transformer")
        for k, v in _get_optim_pid_to_param_names(model, optimizer).items():
            assert isinstance(k, int)
            assert isinstance(v, str) and v.startswith("transformer")

        aux_infos = get_dcp_aux_infos(model, optimizer)
        wrapped_state_dict = _wrap_optim_state_dict(optimizer.state_dict(), aux_infos)
        state_dict = _unwrap_optim_state_dict(wrapped_state_dict, aux_infos)
        torch.testing.assert_close(state_dict, optimizer.state_dict(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
