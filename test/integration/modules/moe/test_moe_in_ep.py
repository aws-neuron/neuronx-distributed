import copy
from dataclasses import dataclass
import pytest
from typing import Any, Dict
import torch
from torch import nn
from functools import partial

from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from torch_neuronx.utils import get_platform_target
import torch_neuronx
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.models.config import MoENeuronConfig, InferenceConfig
from neuronx_distributed.utils.model_utils import get_platform_lnc
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

from .utils import init_parallel_cpu_golden, CPUExpert, CPURMSNorm, fuse_experts_weights

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
    rms_norm_eps: float = 1e-5
    n_shared_experts: int = 0
    torch_dtype: torch.dtype = torch.bfloat16
    rpl_reduce_dtype: torch.dtype = torch.float32
    hidden_act: str = 'silu'
    norm_topk_prob: bool = False
    glu_type: str = "glu"
    hidden_act_scaling_factor: float = 1.
    hidden_act_bias: float = 0.
    apply_act_fn_over_topk: bool = False
    router_bias: bool = False
    experts_bias: bool = False
    init_tkg_module: bool = False

class CPUMoE(nn.Module):
    def __init__(
            self,
            n_routed_experts: int,
            topk: int,
            hidden_size: int,
            moe_inter_dim: int,
            glu_type: str,
            hidden_act: str,
            rms_norm_eps: float,
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
        self.rmsnorm = CPURMSNorm(hidden_size, rms_norm_eps)
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
                dtype=dtype,
            ) for i in range(self.n_routed_experts)
        ])

    def forward(self, hidden_states) -> torch.Tensor:
        shape = hidden_states.size()
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # upcast hidden_states in order to run router in fp32
        hidden_states = hidden_states.to(torch.float32)
        hidden_states = self.rmsnorm(hidden_states)
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

        # downcast hidden_states back to original dtype
        hidden_states = hidden_states.to(original_dtype)

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
        return output.detach().view(shape), router_logits.detach().to(original_dtype)

class MoEWrapper(torch.nn.Module):
    def __init__(self, config, init_tkg_module, router_bias, experts_bias):
        self.config = config
        self.router_bias = router_bias
        self.experts_bias = experts_bias
        super().__init__()
        rmsnorm = CustomRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.moe = initialize_moe_module(
            self.config,
            rmsnorm=rmsnorm,
            init_tkg_module=init_tkg_module,
            router_bias=router_bias,
            experts_bias=experts_bias,
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(torch.bfloat16)
        output = self.moe(hidden_states)
        return output

    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        prefix = prefix.removesuffix("weight")

        model_state_dict["moe.rmsnorm.weight"] = model_state_dict["rmsnorm.weight"]
        del model_state_dict["rmsnorm.weight"]
        model_state_dict["moe.router.linear_router.weight"] = model_state_dict["router.weight"]
        model_state_dict["moe.router.weight_T"] = model_state_dict["router.weight"].detach().transpose(0, 1).clone()
        del model_state_dict["router.weight"]
        if self.router_bias:
            model_state_dict["moe.router.linear_router.bias"] = model_state_dict["router.bias"]
            del model_state_dict["router.bias"]
        fuse_experts_weights(
            model_state_dict,
            n_routed_experts=self.config.n_routed_experts,
            tp_degree=self.moe.expert_mlps.moe_tensor_model_parallel_group.size(),
            experts_bias=self.experts_bias,
            prefix=f"{prefix}moe.",
        )
        create_spmd_ranks(
            model_state_dict=model_state_dict,
            prefix=prefix,
            world_size=parallel_state.get_world_group().size(),
            n_routed_experts=self.config.n_routed_experts,
            expert_model_parallel_group=self.moe.expert_mlps.moe_expert_model_parallel_group,
        )

def create_spmd_ranks(
    model_state_dict: Dict[str, Any],
    prefix: str,
    world_size,
    n_routed_experts,
    expert_model_parallel_group,
):
    # add weight for spmd rank
    model_state_dict["moe.expert_mlps.spmd_rank.rank"] = torch.arange(
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

        model_state_dict["moe.expert_mlps.spmd_rank.local_expert_indices"] = torch.tensor(
            expert_indices, dtype=torch.int32
        )

def generate_test_config(model_configs, tp_degree, ep_degree, router_bias, experts_bias, glu_type, hidden_act, init_tkg_module, rms_norm_eps=1e-5):
    test_config = TestConfig(
        tp_degree=tp_degree,
        ep_degree=ep_degree,
        world_size=tp_degree * ep_degree,
        router_bias=router_bias,
        experts_bias=experts_bias,
        glu_type=glu_type,
        hidden_act=hidden_act,
        init_tkg_module=init_tkg_module,
        rms_norm_eps=rms_norm_eps,
        **model_configs,
    )

    return test_config

def generate_inference_config(config, seq_len):
    inference_config = {
        "hidden_size": config.hidden_size,
        "hidden_act": config.hidden_act,
        "num_local_experts": config.n_routed_experts,
        "n_routed_experts": config.n_routed_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "intermediate_size": config.intermediate_size,
        "n_shared_experts": config.n_shared_experts,
        "dtype": config.torch_dtype,
        "rms_norm_eps": config.rms_norm_eps,
    }

    neuron_config = MoENeuronConfig(
        torch_dtype=config.torch_dtype,
        rpl_reduce_dtype=config.rpl_reduce_dtype,
        tp_degree=64 if get_platform_target() == "trn2" else 32,
        ep_degree=1,
        moe_tp_degree=config.tp_degree,
        moe_ep_degree=config.ep_degree,
        seq_len=seq_len,
        disable_normalize_top_k_affinities=not config.norm_topk_prob,
        router_config={"dtype": torch.float32, "act_fn":"softmax"},
        blockwise_matmul_config={
            "block_size": 256,
            "use_block_parallel": False,
            "block_sharding_strategy":"HI_LO",
            "skip_dma_token": True,
            "skip_dma_weight": True,
            "parallelize_token_to_block_mapping": True,
        },
    )

    inference_config = InferenceConfig(
        neuron_config=neuron_config,
        **inference_config,
    )
    return inference_config

def compile_neuron_moe_model(sample_inputs, checkpoint, load_module, test_config, init_tkg_module, router_bias, experts_bias):
    seq_len = sample_inputs.shape[1]
    inference_config = generate_inference_config(test_config, seq_len)
    checkpoint = copy.copy(checkpoint)

    builder = ModelBuilder(
        router=None,
        tp_degree=test_config.tp_degree,
        ep_degree=test_config.ep_degree,
        checkpoint_loader=lambda: checkpoint,
        logical_nc_config=get_platform_lnc(),
    )

    builder.add(
        key="main",
        model_instance=BaseModelInstance(
            module_cls=partial(load_module, inference_config, init_tkg_module, router_bias, experts_bias),
            input_output_aliases={},
        ),
        example_inputs=[(sample_inputs,)],
        compiler_args=_add_compiler_args(),
    )

    neuron_model = builder.trace(initialize_model_weights=True)
    return neuron_model

def _add_compiler_args():
    """
    Over-ride function from base class for better control over compiler flags
    """
    compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
    compiler_args += (
        " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
    )
    # add dma optimzation flag
    compiler_args += " --tensorizer-options='--vectorize-strided-dma'"
    compiler_args += " --auto-cast=none"
    return compiler_args


def _load_module_moe(config, init_tkg_module, router_bias, experts_bias):
    return MoEWrapper(config=config, init_tkg_module=init_tkg_module, router_bias=router_bias, experts_bias=experts_bias).eval()

def _load_sample_inputs(batch_size, seq_len, hidden_size, dtype):
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
    return hidden_states

class TestMoEEPAgainstHFCPUGolden:
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
            rms_norm_eps=test_config.rms_norm_eps,
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
            test_config.init_tkg_module,
            test_config.router_bias,
            test_config.experts_bias,
        )
        return cpu_model, neuron_model

    @pytest.mark.parametrize("model_configs", [
        {"hidden_size": 3072, "intermediate_size": 3072, "n_routed_experts": 128, "seq_len": 1, "num_experts_per_tok": 4, "batch_size": 32},
    ])
    @pytest.mark.parametrize("tp_degree,ep_degree", [(4, 16)])
    @pytest.mark.parametrize("router_bias", [False])
    @pytest.mark.parametrize("experts_bias", [False])
    @pytest.mark.parametrize("glu_type,hidden_act", [("glu", "silu")])
    @pytest.mark.parametrize("init_tkg_module", [False])
    @pytest.mark.skipif(get_platform_target() != 'trn2', reason="Test only runs on trn2 platform")
    def test_moe_spmd(self, model_configs, tp_degree, ep_degree, router_bias, experts_bias, glu_type, hidden_act, init_tkg_module):
        test_config = generate_test_config(model_configs, tp_degree, ep_degree, router_bias, experts_bias, glu_type, hidden_act, init_tkg_module)

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
        cpu_output = cpu_model(sample_inputs)[0]
        neuron_output = neuron_model(sample_inputs)[0]

        torch_neuronx.testing.assert_close(neuron_output, cpu_output)
        del cpu_model, neuron_model
