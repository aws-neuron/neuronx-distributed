import copy
from dataclasses import dataclass
import pytest

import torch
from torch import nn
from typing import Any, Dict
from functools import partial
import torch_neuronx
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from neuronx_distributed.modules.moe import ExpertMLPs, RouterTopK

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.mappings import (
    reduce_from_tensor_model_parallel_region,
)
from neuronx_distributed_inference.models.config import get_platform_lnc, get_platform_target
from neuronx_distributed.modules.moe.moe_process_group import (
    init_tensor_expert_parallel_moe_process_groups,
    get_moe_tp_ep_group,
    get_moe_ep_group,
)

from .utils import init_parallel_cpu_golden, CPUExpert, fuse_experts_weights

torch.manual_seed(42)

@dataclass
class TestConfig:
    tp_degree: int
    ep_degree: int
    world_size: int
    hidden_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    intermediate_size: int
    batch_size: int
    seq_len: int
    norm_topk_prob: bool
    torch_dtype: torch.dtype = torch.bfloat16
    hidden_act: str = 'silu'
    glu_type: str = "glu"
    hidden_act_scaling_factor: float = 1.
    hidden_act_bias: float = 0.
    apply_act_fn_over_topk: bool = False
    router_bias: bool = False
    experts_bias: bool = False

class CPUMoE(nn.Module):
    def __init__(
            self,
            n_routed_experts: int,
            topk: int,
            hidden_size: int,
            moe_inter_dim: int,
            glu_type: str,
            hidden_act: str,
            hidden_act_scaling_factor: float = 1.,
            hidden_act_bias: float = 0.,
            experts_bias: bool = False,
            router_bias: bool = False,
            apply_act_fn_over_topk: bool = False,
            normalize_top_k_affinities: bool = False,
            dtype = torch.bfloat16,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.topk = topk
        self.apply_act_fn_over_topk = apply_act_fn_over_topk
        self.normalize_top_k_affinities = normalize_top_k_affinities
        self.router = nn.Linear(hidden_size, n_routed_experts, bias=router_bias, dtype=torch.float32)
        self.experts = nn.ModuleList([
            CPUExpert(
                hidden_size,
                moe_inter_dim,
                glu_type=glu_type,
                hidden_act=hidden_act,
                hidden_act_scaling_factor=hidden_act_scaling_factor,
                hidden_act_bias=hidden_act_bias,
                bias=experts_bias,
                dtype=dtype
            ) for i in range(self.n_routed_experts)
        ])

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states = hidden_states.to(torch.float32)
        shape = hidden_states.size()
        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits = self.router(hidden_states)
        if self.apply_act_fn_over_topk:
            experts = torch.topk(router_logits, k=self.topk, dim=-1, sorted=True)
            topk_weight = torch.nn.functional.softmax(experts.values, dim=1)
            indices = experts.indices
        else:
            weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float64)
            experts = torch.topk(weights, k=self.topk, dim=-1, sorted=True)
            indices = experts.indices
            topk_weight = weights.gather(1, indices)

        if self.normalize_top_k_affinities:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        hidden_states = hidden_states.to(torch.bfloat16)

        expert_outputs = torch.zeros(hidden_states.shape[0], self.topk, self.hidden_size,
                                    dtype=hidden_states.dtype, device=hidden_states.device)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(0, self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            expert_outputs[idx, top] = expert(hidden_states[idx])

        output = torch.einsum('beh,be->bh', expert_outputs, topk_weight.to(hidden_states.dtype))
        return output.detach().view(shape)

class MoEWrapper(torch.nn.Module):
    def __init__(self, config):
        self.config = config

        super().__init__()
        if config.ep_degree > 1:
            init_tensor_expert_parallel_moe_process_groups(config.tp_degree, config.ep_degree, config.tp_degree, config.ep_degree)
            self.cte_tensor_model_parallel_group=get_moe_tp_ep_group(prefill = True)
            self.cte_expert_model_parallel_group=get_moe_ep_group(prefill = True)
            self.tkg_tensor_model_parallel_group=get_moe_tp_ep_group(prefill = False)
            self.tkg_expert_model_parallel_group=get_moe_ep_group(prefill = False)
        else:
            self.cte_tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group()
            self.cte_expert_model_parallel_group=parallel_state.get_expert_model_parallel_group()
            self.tkg_tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group()
            self.tkg_expert_model_parallel_group=parallel_state.get_expert_model_parallel_group()

        self.router = RouterTopK(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            sequence_dimension=0,
            act_fn="softmax",
            dtype=torch.float32,
            bias=config.router_bias,
            apply_act_fn_over_topk=config.apply_act_fn_over_topk,
        )
        self.expert_mlps = ExpertMLPs(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            normalize_top_k_affinities=config.norm_topk_prob,
            hidden_act=config.hidden_act,
            bias=config.experts_bias,
            glu_mlp=True,
            glu_type=config.glu_type,
            hidden_act_scaling_factor=config.hidden_act_scaling_factor,
            hidden_act_bias=config.hidden_act_bias,
            capacity_factor=None,
            dtype=config.torch_dtype,
            logical_nc_config=get_platform_lnc(),
            enable_spmd_rank=True,
            use_torch_block_wise=True,  # use torch blockwise until we have support for bias in BWMM kernel
            cte_tensor_model_parallel_group=self.cte_tensor_model_parallel_group,
            cte_expert_model_parallel_group=self.cte_expert_model_parallel_group,
            tkg_tensor_model_parallel_group=self.tkg_tensor_model_parallel_group,
            tkg_expert_model_parallel_group=self.tkg_expert_model_parallel_group,
        )

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        router_logits, expert_affinities, expert_index = self.router(hidden_states)

        hidden_states = hidden_states.to(torch.bfloat16)
        output_experts = self.expert_mlps(
            hidden_states.view(-1, hidden_size),
            expert_affinities,
            expert_index,
            seq_len,
        )
        output_experts = output_experts.view(batch_size, -1, hidden_size)
        output_experts = self._reduce_output(output_experts)
        return output_experts

    def _reduce_output(self, output: torch.Tensor) -> torch.Tensor:
        original_dtype = output.dtype
        output = output.to(torch.float32)
        if self.expert_mlps.moe_expert_model_parallel_group.size() > 1:
            output = reduce_from_tensor_model_parallel_region(
                output,
                process_group=parallel_state.get_world_group(),
            )
        else:
            output = reduce_from_tensor_model_parallel_region(
                output,
                process_group=parallel_state.get_tensor_model_parallel_group(as_list=False),
            )
        output = output.to(original_dtype)
        return output

    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        model_state_dict["router.linear_router.weight"] = model_state_dict["router.weight"]
        if self.config.router_bias:
            model_state_dict["router.linear_router.bias"] = model_state_dict["router.bias"]
            del model_state_dict["router.bias"]
        del model_state_dict["router.weight"]
        fuse_experts_weights(
            model_state_dict,
            self.config.n_routed_experts,
            self.expert_mlps.moe_tensor_model_parallel_group.size(),
            self.config.experts_bias,
        )
        create_spmd_ranks(
            model_state_dict=model_state_dict,
            prefix=prefix,
            world_size=self.config.world_size,
            n_routed_experts=self.config.n_routed_experts,
            expert_model_parallel_group=self.expert_mlps.moe_expert_model_parallel_group,
        )

def create_spmd_ranks(
    model_state_dict: Dict[str, Any],
    prefix: str,
    world_size,
    n_routed_experts: int,
    expert_model_parallel_group,
):
    # add weight for spmd rank
    model_state_dict["expert_mlps.spmd_rank.rank"] = torch.arange(
        0, world_size, dtype=torch.int32
    )
    if expert_model_parallel_group.size() > 1:
        expert_indices = []
        for rank in range(world_size):
            curr_expert_rank = parallel_state.get_expert_parallel_rank_from_global_rank(
                rank=rank, expert_parallel_group=expert_model_parallel_group
            )
            curr_expert_indices = parallel_state.get_experts_for_expert_parallel_rank(
                curr_expert_rank,
                total_number_of_experts=n_routed_experts,
                expert_model_parallel_size=expert_model_parallel_group.size(),
            )
            expert_indices.append(curr_expert_indices)

        model_state_dict["expert_mlps.spmd_rank.local_expert_indices"] = torch.tensor(
            expert_indices, dtype=torch.int32
        )


def compile_neuron_moe_model(sample_inputs, checkpoint, load_module, inference_config):
    checkpoint = copy.copy(checkpoint)

    builder = ModelBuilder(
        router=None,
        tp_degree=64 if get_platform_target() == "trn2" else 32,
        ep_degree=1,
        checkpoint_loader=lambda: checkpoint,
        logical_nc_config=get_platform_lnc(),
    )

    builder.add(
        key="main",
        model_instance=BaseModelInstance(
            module_cls=partial(load_module, inference_config),
            input_output_aliases={},
        ),
        example_inputs=[sample_inputs],
        compiler_args=_add_compiler_args(),
    )

    neuron_model = builder.trace(initialize_model_weights=True)
    return neuron_model

def _add_compiler_args():
    """
    Over-ride function from base class for better control over compiler flags
    """
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
    compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    # add dma optimzation flag
    compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
    compiler_args += " --auto-cast=none"
    return compiler_args


def _load_module_moe(config):
    return MoEWrapper(config=config).eval()


def _load_sample_inputs(batch_size, seq_len, hidden_size, dtype):
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
    return (hidden_states, )

def generate_test_config(model_configs, tp_degree, ep_degree, router_bias, experts_bias, glu_type, hidden_act, norm_topk_prob):
    test_config = TestConfig(
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        world_size=tp_degree * ep_degree,
        router_bias=router_bias,
        experts_bias=experts_bias,
        glu_type=glu_type,
        hidden_act=hidden_act,
        norm_topk_prob=norm_topk_prob,
        **model_configs,
    )

    return test_config

class TestMoESPMDAgainstHFCPUGolden:
    def _create_models(
            self,
            sample_inputs,
            test_config,
        ):
        cpu_model = CPUMoE(
            n_routed_experts=test_config.n_routed_experts,
            topk=test_config.num_experts_per_tok,
            hidden_size=test_config.hidden_size,
            moe_inter_dim=test_config.intermediate_size,
            glu_type=test_config.glu_type,
            hidden_act=test_config.hidden_act,
            hidden_act_scaling_factor=test_config.hidden_act_scaling_factor,
            hidden_act_bias=test_config.hidden_act_bias,
            experts_bias=test_config.experts_bias,
            router_bias=test_config.router_bias,
            normalize_top_k_affinities=test_config.norm_topk_prob,
            dtype=test_config.torch_dtype,
        ).eval()

        hf_checkpoint = cpu_model.state_dict()
        neuron_checkpoint = copy.deepcopy(hf_checkpoint)

        neuron_model = compile_neuron_moe_model(
            sample_inputs,
            neuron_checkpoint,
            _load_module_moe,
            test_config,
        )

        return cpu_model, neuron_model

    @pytest.mark.parametrize("model_configs", [
        {"hidden_size": 7168, "intermediate_size": 1024, "n_routed_experts": 64, "seq_len": 1024, "num_experts_per_tok": 4, "batch_size": 1},
        # prefill (forward blockwise EP)
        {"hidden_size": 3072, "intermediate_size": 3072, "n_routed_experts": 128, "seq_len": 2048, "num_experts_per_tok": 4, "batch_size": 1},
        # decode (forward all experts EP)
        {"hidden_size": 3072, "intermediate_size": 3072, "n_routed_experts": 128, "seq_len": 1, "num_experts_per_tok": 4, "batch_size": 32},
    ])
    @pytest.mark.parametrize("tp_degree,ep_degree", [(64, 1), (16, 4), (4, 16), (1, 64)])
    @pytest.mark.parametrize("router_bias", [False, True])
    @pytest.mark.parametrize("experts_bias", [False, True])
    @pytest.mark.parametrize("glu_type,hidden_act", [("glu", "silu"), ("swiglu", "sigmoid")])
    @pytest.mark.parametrize("norm_topk_prob", [False, True])
    @pytest.mark.skipif(get_platform_target() != 'trn2', reason="Test only runs on trn2 platform")
    def test_moe_spmd(self, model_configs, tp_degree, ep_degree, router_bias, experts_bias, glu_type, hidden_act, norm_topk_prob):
        test_config = generate_test_config(model_configs, tp_degree, ep_degree, router_bias, experts_bias, glu_type, hidden_act, norm_topk_prob)

        print(f"Running test config:  {test_config}")
        sample_inputs = _load_sample_inputs(
            test_config.batch_size,
            test_config.seq_len,
            test_config.hidden_size,
            dtype=torch.bfloat16,
        )

        cpu_model, neuron_model = self._create_models(
            sample_inputs=sample_inputs,
            test_config=test_config,
        )

        init_parallel_cpu_golden()
        cpu_output = cpu_model(*sample_inputs)
        neuron_output = neuron_model(*sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output, cpu_output)
        del cpu_model, neuron_model
