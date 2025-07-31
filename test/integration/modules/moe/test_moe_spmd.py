import copy
import os
import unittest
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch_neuronx
import torch_xla.core.xla_model as xm

# Imports from MoE unit tests (for this import to succeed, test/unit_test/modules/moe must be added to PYTHONPATH)
import utils_testing as ut
from device_correctness_test_configs import get_model_config

from neuronx_distributed import parallel_layers
from neuronx_distributed.modules.moe.shared_experts import SharedExperts
from neuronx_distributed.modules.moe import MoE, ExpertMLPs, RouterTopK
from neuronx_distributed.modules.moe.moe_configs import MoEFusedTKGConfig
from neuronx_distributed.modules.moe.moe_fused_tkg import MoEFusedTKG
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.mappings import (
    _reduce_scatter_along_dim,
    gather_from_sequence_parallel_region,
)
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from neuronx_distributed.utils.model_utils import get_platform_lnc, LogicalNCConfig


def set_tp_degree():
    return 64 if get_platform_lnc() == 2 else 32

def init_parallel_cpu_golden():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group(backend="xla", init_method="env://")
    parallel_layers.parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

torch.manual_seed(42)

@dataclass
class ExptCfg:
    batch_size: int
    seq_len: int
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    dtype: torch.dtype
    hidden_act: str = "silu"
    num_shared_experts: int = 0
    early_expert_affinity_modulation: bool = False
    router_sequence_parallel_enabled: bool = False
    input_is_sequence_parallel: bool = False
    shared_experts_sequence_parallel_enabled: bool = False
    moe_fused_tkg_enabled: bool = False
    moe_fused_tkg_kernel_enabled: Optional[bool] = None
    transpose_weights: bool = False
    rms_norm_eps: int = 1e-5

def init_moe(cfg: ExptCfg):

    expert_mlps_args = dict(
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        hidden_act=cfg.hidden_act,
        capacity_factor=None,
        init_method=torch.nn.init.kaiming_uniform_,
        output_layer_init_method=torch.nn.init.kaiming_uniform_,
        glu_mlp=True,
        dtype=cfg.dtype,
        use_torch_block_wise=True,
        device=torch.device("cpu"),
        logical_nc_config=get_platform_lnc(),
        parallelize_token_to_block_mapping=True,
        skip_dma_token=False,
        skip_dma_weight=False,
        use_block_parallel=False,
        early_expert_affinity_modulation=False,
        block_sharding_strategy=None,
        enable_spmd_rank=False,
    )

    expert_mlps = ExpertMLPs(**expert_mlps_args)
    expert_mlps.normalize_top_k_affinities = True
    router_args = dict(
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        hidden_size=cfg.hidden_size,
        sequence_parallel_enabled=cfg.router_sequence_parallel_enabled,
        sequence_dimension=1,
        dtype=cfg.dtype,
        act_fn="sigmoid",
        device=torch.device("cpu"),
    )
    if cfg.num_shared_experts > 0:
        shared_experts = SharedExperts(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_shared_experts=cfg.num_shared_experts,
            fused_gate_up_projection=False,
            hidden_act="silu",
            dtype=cfg.dtype,
            sequence_parallel_enabled=cfg.shared_experts_sequence_parallel_enabled,
            #tensor_model_parallel_group=create_process_group(),
            transpose_weights=cfg.transpose_weights,
        )
    else:
        shared_experts = None
    router = RouterTopK(**router_args)
    neuron_model = MoE(
        router=router,
        expert_mlps=expert_mlps,
        return_router_logits=False,
        sequence_parallel_enabled=cfg.input_is_sequence_parallel,
        sequence_dimension=1,
        shared_experts=shared_experts,
    )
    neuron_model.training = False
    if cfg.moe_fused_tkg_enabled is True:
        assert not cfg.router_sequence_parallel_enabled and not cfg.input_is_sequence_parallel
        moe = neuron_model
        post_attention_layernorm = ut.LlamaRMSNormV2(hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps)
        moe_fused_tkg_config = MoEFusedTKGConfig(
            quantized=False,
            moe_fused_kernel_enabled=cfg.moe_fused_tkg_kernel_enabled,
        )
        neuron_model = MoEFusedTKG(
            config=moe_fused_tkg_config,
            post_attention_layernorm=post_attention_layernorm,
            moe=moe,
            return_router_logits=False,
        )

    return neuron_model.eval()

class SequenceParallelWrapper(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._tensor_model_parallel_group = parallel_state.get_world_group()
        self.moe = init_moe(cfg).eval()

    def forward(self, hidden_states):
        hidden_states = _reduce_scatter_along_dim(
            hidden_states,
            1,
            xm.REDUCE_MAX,
            process_group=self._tensor_model_parallel_group,
        )

        output = self.moe(hidden_states)[0]

        output = gather_from_sequence_parallel_region(
            output,
            sequence_dimension=1,
            process_group=self._tensor_model_parallel_group,
        )
        return output

def _add_compiler_args():
    cc_flags = [
        "--model-type=transformer",
        "--enable-saturate-infinity",  # clip matmul transpose input to [-MAX, MAX] to avoid nans (0*INF)
        "--auto-cast=none"
    ]
    return " ".join(cc_flags)

def _load_module_moe(cfg: ExptCfg):
    return init_moe(cfg).eval()

def _load_module_moe_sp(cfg: ExptCfg):
    return SequenceParallelWrapper(cfg).eval()

def compile_neuron_moe_model(
        sample_inputs,
        checkpoint,
        load_module,
        test_config,
        tp_degree=set_tp_degree(),
):
    checkpoint = copy.copy(checkpoint)

    builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=lambda: checkpoint,
        logical_nc_config=get_platform_lnc(),
        debug=False,
    )

    builder.add(
        key="main",
        model_instance=BaseModelInstance(
            module_cls=partial(load_module, test_config),
            input_output_aliases={},
        ),
        example_inputs=[(sample_inputs,)],
        compiler_args=_add_compiler_args(),
    )

    neuron_model = builder.trace(initialize_model_weights=True)
    return neuron_model

def prepend_to_keys(dictionary, prepend_string):
    return {f"{prepend_string}{key}": value for key, value in dictionary.items()}

def _generate_test_configs():
    test_configs = [
        # Mixtral
        ExptCfg(
            batch_size=1,
            seq_len=128,
            dtype=torch.float32,
            **get_model_config("mixtral"),
        ),
        # llama4-100b
        ExptCfg(
            batch_size=1,
            seq_len=128,
            dtype=torch.float32,
            **get_model_config("llama4-100b"),
        ),
        # token gen
        ExptCfg(
            batch_size=1,
            seq_len=1,
            dtype=torch.float32,
            **get_model_config("llama4-100b"),
        ),
        # llama4-400b
        ExptCfg(
            batch_size=1,
            seq_len=128,
            dtype=torch.bfloat16,
            **get_model_config("llama4-400b"),
        ),
        ExptCfg(
            batch_size=1,
            seq_len=128,
            dtype=torch.float32,
            num_experts=128,
            top_k=8,
            hidden_size=4096,
            intermediate_size=1024,
            num_shared_experts=2,
        ),
    ]

    # MoE TKG kernel only supported on Trn2
    if get_platform_lnc() == LogicalNCConfig.LNC_2:
        test_configs.extend(
            [
                # ExptCfg(
                #     batch_size=1,
                #     seq_len=1,
                #     dtype=torch.bfloat16,
                #     early_expert_affinity_modulation=True,
                #     moe_fused_tkg_enabled=True,
                #     **get_model_config("llama4-100b"),
                # ),
                # This is currently broken, might be due to compiler regression
                # ExptCfg(
                #     batch_size=1,
                #     seq_len=1,
                #     dtype=torch.bfloat16,
                #     early_expert_affinity_modulation=True,
                #     moe_fused_tkg_enabled=True,
                #     moe_fused_tkg_kernel_enabled=False,
                #     **get_model_config("llama4-100b"),
                # ),
                ExptCfg(
                    batch_size=4, # Note batch_size=4 is not working with the alpha compiler wheel yet
                    seq_len=1,
                    dtype=torch.bfloat16,
                    early_expert_affinity_modulation=True,
                    moe_fused_tkg_enabled=True,
                    **get_model_config("llama4-100b"),
                ),
                ExptCfg(
                    batch_size=1,
                    seq_len=1,
                    dtype=torch.bfloat16,
                    early_expert_affinity_modulation=True,
                    moe_fused_tkg_enabled=True,
                    **get_model_config("llama4-100b"),
                ),
            ]
        )
    return test_configs


class TestMoERoutedAndSharedExperts(unittest.TestCase):
    def test_moe_against_cpu_golden_parallel(self):
        for test_config in _generate_test_configs():
            seq_len = test_config.seq_len
            hidden_size = test_config.hidden_size
            dtype = test_config.dtype
            batch_size = test_config.batch_size
            sample_inputs = torch.rand(batch_size, seq_len, hidden_size, dtype=dtype)
            # CPU golden
            init_parallel_cpu_golden()
            cpu_golden = _load_module_moe(test_config)
            expected_output = cpu_golden(sample_inputs)[0]
            hf_checkpoint = cpu_golden.state_dict()

            moe_fused_tkg_enabled = test_config.moe_fused_tkg_enabled
            if moe_fused_tkg_enabled:
                input_is_sequence_parallel_list = [False]
            else:
                input_is_sequence_parallel_list = [True, False]
            # Testing with both input in sequence in parallel or not
            for input_is_sequence_parallel in input_is_sequence_parallel_list:
                if input_is_sequence_parallel and seq_len == 1:
                    continue
                neuron_checkpoint = copy.deepcopy(hf_checkpoint)
                if moe_fused_tkg_enabled:
                    neuron_checkpoint["moe.shared_experts.spmd_rank.rank"] = torch.arange(
                        0, set_tp_degree(), dtype=torch.int32
                    )
                    test_config.transpose_weights = True
                else:
                    neuron_checkpoint["shared_experts.spmd_rank.rank"] = torch.arange(
                        0, set_tp_degree(), dtype=torch.int32
                    )
                test_config.input_is_sequence_parallel = input_is_sequence_parallel
                if input_is_sequence_parallel:
                    load_module = _load_module_moe_sp
                    neuron_checkpoint = prepend_to_keys(neuron_checkpoint, "moe.")
                else:
                    load_module = _load_module_moe
                # Testing with both running shared expert in sequence parallel or tensor parallel
                run_sequence_parallel_shared_expert_list = [True, False]
                for run_sequence_parallel_shared_expert in run_sequence_parallel_shared_expert_list:
                    test_config.shared_experts_sequence_parallel_enabled = run_sequence_parallel_shared_expert
                    if seq_len > 1 and (not input_is_sequence_parallel) and run_sequence_parallel_shared_expert:
                        continue
                    print(f"Running test config:  {test_config}")
                    neuron_model = compile_neuron_moe_model(
                        sample_inputs,
                        neuron_checkpoint,
                        load_module,
                        test_config=test_config,
                    )
                    neuron_output = neuron_model(sample_inputs)[0]
                    torch_neuronx.testing.assert_close(neuron_output, expected_output)
                    print("Test results ok")
                del neuron_checkpoint

if __name__ == "__main__":
    unittest.main(verbosity=2)
