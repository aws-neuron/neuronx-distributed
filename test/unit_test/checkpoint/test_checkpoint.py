# Standard Library
import os
import time
import pytest
import unittest
from copy import deepcopy
from packaging import version
from unittest.mock import MagicMock, patch

# Third Party
import torch
import torch_xla.core.xla_model as xm
import torch_xla.utils.serialization as xser
from transformers import AutoModelForCausalLM, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import neuronx_distributed as nxd


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

def load_state_dict(self, state_dict):
    """ Mock func to load Z1 optim w/ sharded master weight 
        Omits operations to restore param from master weight, thus model params will be invalid
    """
    sharding_scheme = self._get_sharding_scheme({})
    state_dict = deepcopy(state_dict)
    base_state = state_dict.pop('base_state')

    # re-init base optimizer to make sure we have right shards
    self.init_zero()

    tmp = self.base_optimizer.state_dict()
    tmp['state'] = base_state
    self.base_optimizer.load_state_dict(tmp)

    xm.mark_step()
    # add mark_step around allgather to avoid large number of compilation
    self.allgather_weights_and_update_full_parameter(sharding_scheme)
    xm.mark_step()

def mock_get_data_parallel_group(as_list=False):
    if(as_list):
        return [[i] for i in range(64)]
    mock = MagicMock(spec=torch.distributed.ProcessGroup)
    mock.size.return_value = 1
    return mock

class TestCheckpoint(unittest.TestCase):
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
    @patch("neuronx_distributed.pipeline.model.parallel_state.get_data_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.partition.get_pipeline_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.partition.get_pipeline_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.pipeline.model.NxDPPModel._create_pg_with_ranks", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.get_data_parallel_group",
        MagicMock(side_effect=mock_get_data_parallel_group)
    )
    @patch(
        "neuronx_distributed.trainer.trainer.parallel_state.get_data_parallel_replica_groups",
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
    @patch("neuronx_distributed.parallel_layers.parallel_state.get_expert_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.trainer.trainer.parallel_state.get_expert_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.trainer.checkpoint.get_expert_model_parallel_rank", MagicMock(return_value=0))
    @patch("neuronx_distributed.trainer.checkpoint.get_expert_data_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.trainer.checkpoint.get_expert_data_parallel_rank", MagicMock(return_value=0))
    @patch("neuronx_distributed.trainer.checkpoint.get_expert_model_parallel_replica_groups", MagicMock(return_value=[]))
    @patch("neuronx_distributed.parallel_layers.parallel_state.get_expert_model_parallel_replica_groups", MagicMock(return_value=[]))
    @patch("neuronx_distributed.parallel_layers.parallel_state.get_expert_model_parallel_group", MagicMock(side_effect=mock_get_data_parallel_group))
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
    @patch("torch.distributed.get_rank", MagicMock(return_value=0))
    def test_checkpoint(self):
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

        model_state = deepcopy(model.state_dict())
        opt_state = deepcopy(optimizer.state_dict())

        nxd.save_checkpoint(
            "ckpts",
            "unittest",
            model=model,
            optimizer=optimizer,
            num_workers=8,
            use_xser=True,
        )

        nxd.load_checkpoint(
            "ckpts",
            "unittest",
            model=model,
            optimizer=optimizer,
            num_workers=8,
        )

        # test save load functionality
        torch.testing.assert_close(model.state_dict(), model_state, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), opt_state, rtol=0, atol=0)

        # test able to be loaded by xla
        xmodel_state = xser.load("ckpts/unittest/model/dp_rank_00_tp_rank_01_pp_rank_01.pt")
        xmodel_state = xm.send_cpu_data_to_device(xmodel_state, xm.xla_device())
        torch.testing.assert_close(xmodel_state, model_state, rtol=0, atol=0)
        xopt_state = xser.load("ckpts/unittest/optim/dp_rank_00_tp_rank_01_pp_rank_01.pt")
        xopt_state = xm.send_cpu_data_to_device(xopt_state, xm.xla_device())
        torch.testing.assert_close(xopt_state, opt_state, rtol=0, atol=0)

        # check format
        assert os.path.exists("ckpts/unittest/done") and os.path.isfile("ckpts/unittest/done")
        assert os.path.exists("ckpts/unittest/checkpoint") and os.path.isfile("ckpts/unittest/checkpoint")
        assert os.path.isfile("ckpts/unittest/model/dp_rank_00_tp_rank_01_pp_rank_01.pt")
        assert os.path.isfile("ckpts/unittest/optim/dp_rank_00_tp_rank_01_pp_rank_01.pt")
        assert os.path.isdir("ckpts/unittest/model/dp_rank_00_tp_rank_01_pp_rank_01.pt.tensors")
        assert os.path.isdir("ckpts/unittest/optim/dp_rank_00_tp_rank_01_pp_rank_01.pt.tensors")

        # test auto resume
        nxd.load_checkpoint(
            "ckpts",
            tag=None,
            model=model,
            optimizer=optimizer,
            num_workers=8,
        )
        torch.testing.assert_close(model.state_dict(), model_state, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), opt_state, rtol=0, atol=0)

    @pytest.mark.skipif(
        version.parse(torch.__version__) != version.parse("2.1"), reason="skip this test if no DCP support"
    )
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

        model_state = deepcopy(model.state_dict())
        opt_state = deepcopy(optimizer.state_dict())

        nxd.save_checkpoint(
            "ckpts",
            "unittest_dcp",
            model=model,
            optimizer=optimizer,
            num_workers=8,
            use_xser=True,
            async_save=True,
            use_zero1_dcp=True,
        )

        time.sleep(5)

        nxd.load_checkpoint(
            "ckpts",
            "unittest_dcp",
            model=model,
            optimizer=optimizer,
            num_workers=8,
            use_zero1_dcp=True,
        )

        # test save load functionality
        torch.testing.assert_close(model.state_dict(), model_state, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state_dict(), opt_state, rtol=0, atol=0)

        # check format
        assert os.path.exists("ckpts/unittest_dcp/done") and os.path.isfile("ckpts/unittest_dcp/done")
        assert os.path.exists("ckpts/unittest_dcp/checkpoint") and os.path.isfile("ckpts/unittest_dcp/checkpoint")
        assert os.path.isfile("ckpts/unittest_dcp/model/dp_rank_00_tp_rank_01_pp_rank_01.pt")
        assert os.path.isdir("ckpts/unittest_dcp/model/dp_rank_00_tp_rank_01_pp_rank_01.pt.tensors")
        assert os.path.isfile("ckpts/unittest_dcp/optim/.metadata")
        assert os.path.isfile("ckpts/unittest_dcp/optim/__0_0.distcp")

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
    @patch("neuronx_distributed.pipeline.model.parallel_state.get_data_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.partition.get_pipeline_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.partition.get_pipeline_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.pipeline.model.NxDPPModel._create_pg_with_ranks", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.get_data_parallel_group",
        MagicMock(side_effect=mock_get_data_parallel_group)
    )
    @patch(
        "neuronx_distributed.trainer.trainer.parallel_state.get_data_parallel_replica_groups",
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
    @patch("neuronx_distributed.parallel_layers.parallel_state.get_expert_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.trainer.trainer.parallel_state.get_expert_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.trainer.checkpoint.get_expert_model_parallel_rank", MagicMock(return_value=0))
    @patch("neuronx_distributed.trainer.checkpoint.get_expert_data_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.trainer.checkpoint.get_expert_data_parallel_rank", MagicMock(return_value=0))
    @patch("neuronx_distributed.trainer.checkpoint.get_expert_model_parallel_replica_groups", MagicMock(return_value=[]))
    @patch("neuronx_distributed.parallel_layers.parallel_state.get_expert_model_parallel_replica_groups", MagicMock(return_value=[]))
    @patch("neuronx_distributed.parallel_layers.parallel_state.get_expert_model_parallel_group", MagicMock(side_effect=mock_get_data_parallel_group))
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
    @patch("torch_xla.distributed.zero_redundancy_optimizer.ZeroRedundancyOptimizer.load_state_dict", load_state_dict, create=True)
    @patch("torch.distributed.get_rank", MagicMock(return_value=0))
    def test_checkpoint_z1_avoid_saving_master_weights(self):
        pipeline_cuts = [
            "transformer.h.1",
            "transformer.h.2",
            "transformer.h.3",
            "transformer.h.4",
            "transformer.h.5",
            "transformer.h.6",
            "transformer.h.7",
        ]
        zero1_enabled = True
        mixed_precision_config = {
            "use_master_weights": zero1_enabled,
            "use_fp32_grad_acc": zero1_enabled,
            "use_master_weights_in_ckpt": True,
        }
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
                "use_zero1_optimizer": zero1_enabled,
                "use_optimizer_wrapper": True,
            },
            optimizer_config={
                "zero_one_enabled": zero1_enabled,
                "grad_clipping": True,
                "max_grad_norm": 1.0,
            },
            sequence_parallel=True,
            activation_checkpoint_config="full",
            mixed_precision_config=mixed_precision_config,
        )
        model = nxd.initialize_parallel_model(nxd_config, get_model)
        optimizer = nxd.initialize_parallel_optimizer(nxd_config, torch.optim.AdamW, model.parameters(), lr=1e-3)

        opt_state = deepcopy(optimizer.state_dict())

        tag = "unittest-z1-dist"
        nxd.save_checkpoint(
            "ckpts",
            tag,
            model=model,
            optimizer=optimizer,
            num_workers=8,
            use_xser=True,
            avoid_saving_lower_precision_weights=True,
        )

        nxd.load_checkpoint(
            "ckpts",
            tag,
            model=model,
            optimizer=optimizer,
            num_workers=8,
        )

        # only test optim states here as model weights are not saved
        # test save load functionality
        torch.testing.assert_close(optimizer.state_dict(), opt_state, rtol=0, atol=0)

        # test able to be loaded by xla
        xopt_state = xser.load(f"ckpts/{tag}/optim/dp_rank_00_tp_rank_01_pp_rank_01.pt")
        xopt_state = xm.send_cpu_data_to_device(xopt_state, xm.xla_device())
        torch.testing.assert_close(xopt_state, opt_state, rtol=0, atol=0)

        # check format
        assert os.path.exists(f"ckpts/{tag}/done") and os.path.isfile(f"ckpts/{tag}/done")
        assert os.path.exists(f"ckpts/{tag}/checkpoint") and os.path.isfile(f"ckpts/{tag}/checkpoint")
        assert os.path.isfile(f"ckpts/{tag}/model/dp_rank_00_tp_rank_01_pp_rank_01.pt")
        assert os.path.isfile(f"ckpts/{tag}/optim/dp_rank_00_tp_rank_01_pp_rank_01.pt")
        assert os.path.isdir(f"ckpts/{tag}/model/dp_rank_00_tp_rank_01_pp_rank_01.pt.tensors")
        assert os.path.isdir(f"ckpts/{tag}/optim/dp_rank_00_tp_rank_01_pp_rank_01.pt.tensors")

        # check to see that no tensors were saved in model tensor dir
        assert not os.listdir(f"ckpts/{tag}/model/dp_rank_00_tp_rank_01_pp_rank_01.pt.tensors")


if __name__ == "__main__":
    unittest.main()
