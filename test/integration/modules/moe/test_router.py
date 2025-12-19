import copy
import unittest

import torch
from functools import partial
import torch_neuronx
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder
from neuronx_distributed.modules.moe import RouterTopK

from torch import nn
from neuronx_distributed_inference.models.config import get_platform_lnc

from .utils import init_parallel_cpu_golden

torch.manual_seed(42)

class TestConfig:
    def __init__(
        self,
        torch_dtype,
        rpl_reduce_dtype,
        hidden_size: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
    ):
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.torch_dtype = torch_dtype
        self.rpl_reduce_dtype = rpl_reduce_dtype

class CPUTopKRouter(nn.Module):
    def __init__(
            self,
            n_routed_experts: int,
            topk: int,
            hidden_size: int,
            router_bias: bool = False,
            apply_act_fn_over_topk: bool = False,
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_routed_experts = n_routed_experts
        self.topk = topk
        self.apply_act_fn_over_topk = apply_act_fn_over_topk
        self.router = nn.Linear(hidden_size, n_routed_experts, bias=router_bias, dtype=torch.float32)

    def forward(self, hidden_states) -> torch.Tensor:
        router_logits = self.router(hidden_states)
        if self.apply_act_fn_over_topk:
            experts = torch.topk(router_logits, k=self.topk, dim=-1, sorted=True)
            weights = torch.nn.functional.softmax(experts.values, dim=1)
            indices = experts.indices
        else:
            router_logits = self.router(hidden_states)
            weights = torch.nn.functional.softmax(router_logits, dim=1, dtype=torch.float64)
            experts = torch.topk(weights, k=self.topk, dim=-1, sorted=True)
            indices = experts.indices
        return router_logits, weights.to(hidden_states.dtype), indices

def compile_neuron_moe_model(sample_inputs, checkpoint, load_module, router_bias=False, apply_act_fn_over_topk=False, torch_dtype=torch.bfloat16, **inference_config):
    inference_config = TestConfig(
        torch_dtype= torch_dtype,
        rpl_reduce_dtype=torch.float32,
        **inference_config,
    )

    checkpoint = copy.copy(checkpoint)

    builder = ModelBuilder(
        router=None,
        tp_degree=1,
        checkpoint_loader=lambda: checkpoint,
        logical_nc_config=get_platform_lnc(),
    )

    builder.add(
        key="main",
        model_instance=BaseModelInstance(
            module_cls=partial(load_module, inference_config, router_bias, apply_act_fn_over_topk),
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


def _load_module_moe(config, router_bias: bool = False, apply_act_fn_over_topk: bool = False,):
    return RouterTopK(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            sequence_dimension=0,
            act_fn="softmax",
            dtype=torch.float32,
            bias=router_bias,
            apply_act_fn_over_topk=apply_act_fn_over_topk,
        ).eval()


def _load_sample_inputs(seq_len, hidden_size, dtype):
    hidden_states = torch.randn(seq_len, hidden_size, dtype=dtype)
    return (hidden_states, )

class TestExpertMLP(unittest.TestCase):
    def setUp(self):
        # Common test parameters
        self.seq_len = 1024
        self.hidden_size = 7168
        self.intermediate_size = 1024
        self.n_routed_experts = 64
        self.num_experts_per_tok = 4

    def _get_common_model_params(self):
        return {
            "hidden_size": self.hidden_size,
            "n_routed_experts": self.n_routed_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
        }

    def _create_models(self, dtype, sample_inputs, router_bias=False, apply_act_fn_over_topk=False):
        cpu_model = CPUTopKRouter(
            n_routed_experts=self.n_routed_experts,
            topk=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            router_bias=router_bias,
            apply_act_fn_over_topk=apply_act_fn_over_topk,
        ).eval()

        hf_checkpoint = cpu_model.state_dict()
        neuron_checkpoint = copy.deepcopy(hf_checkpoint)
        neuron_checkpoint["linear_router.weight"] = neuron_checkpoint["router.weight"]
        del neuron_checkpoint["router.weight"]

        common_params = self._get_common_model_params()
        neuron_model = compile_neuron_moe_model(
            sample_inputs,
            neuron_checkpoint,
            _load_module_moe,
            router_bias=router_bias,
            apply_act_fn_over_topk=apply_act_fn_over_topk,
            torch_dtype=dtype,
            **common_params
        )

        return cpu_model, neuron_model

    def test_topk_router(self):
        sample_inputs = _load_sample_inputs(
            self.seq_len,
            self.hidden_size,
            dtype=torch.float32,
        )

        cpu_model, neuron_model = self._create_models(
            dtype=torch.bfloat16,
            sample_inputs=sample_inputs,
        )

        init_parallel_cpu_golden()
        cpu_output = cpu_model(*sample_inputs)
        neuron_output = neuron_model(*sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output, cpu_output)

    def test_topk_router_apply_act_fn_over_topk(self):
        sample_inputs = _load_sample_inputs(
            self.seq_len,
            self.hidden_size,
            dtype=torch.float32,
        )

        cpu_model, neuron_model = self._create_models(
            dtype=torch.bfloat16,
            sample_inputs=sample_inputs,
            apply_act_fn_over_topk=True,
        )

        init_parallel_cpu_golden()
        cpu_output = cpu_model(*sample_inputs)
        neuron_output = neuron_model(*sample_inputs)

        torch_neuronx.testing.assert_close(neuron_output, cpu_output)
