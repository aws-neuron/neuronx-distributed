import os
import torch
from torch import nn
from typing import Dict, Any

from neuronx_distributed import parallel_layers
from neuronx_distributed.modules.moe import ACT2FN
from neuronx_distributed_inference.models.config import get_platform_lnc

def set_tp_degree():
    return 64 if get_platform_lnc() == 2 else 32

def set_world_size():
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

def fuse_experts_weights(
    model_state_dict: Dict[str, Any],
    n_routed_experts: int,
    tp_degree: int,
    experts_bias: bool = False,
    prefix: str = "",
) -> None:
    down_proj_weights_list = []
    gate_up_proj_weights_list = []

    if experts_bias:
        down_proj_biases_list = []
        gate_up_proj_biases_list = []

    for i in range(n_routed_experts):
        down_proj_weight = (
            model_state_dict[f"experts.{i}.down_proj.weight"].transpose(0, 1).contiguous()
        )
        down_proj_weights_list.append(down_proj_weight)
        del model_state_dict[f"experts.{i}.down_proj.weight"]

        up_proj_weight = model_state_dict[f"experts.{i}.up_proj.weight"]
        gate_proj_weight = model_state_dict[f"experts.{i}.gate_proj.weight"]
        gate_up_proj_weights_list.append(
            torch.cat((gate_proj_weight, up_proj_weight), dim=0).transpose(0, 1).contiguous()
        )

        del model_state_dict[f"experts.{i}.up_proj.weight"]
        del model_state_dict[f"experts.{i}.gate_proj.weight"]

        if experts_bias:
            # we do all-reduce at the end of MoE so we divide bias by TP here to avoid needing
            # to separately add down_proj bias after all-reduce
            down_proj_bias = model_state_dict[f"experts.{i}.down_proj.bias"] / tp_degree
            down_proj_biases_list.append(down_proj_bias)
            del model_state_dict[f"experts.{i}.down_proj.bias"]

            up_proj_bias = model_state_dict[f"experts.{i}.up_proj.bias"]
            gate_proj_bias = model_state_dict[f"experts.{i}.gate_proj.bias"]
            gate_up_proj_biases_list.append(
                torch.cat((gate_proj_bias, up_proj_bias), dim=0)
            )

            del model_state_dict[f"experts.{i}.up_proj.bias"]
            del model_state_dict[f"experts.{i}.gate_proj.bias"]

    down_proj_weights = torch.stack(down_proj_weights_list)
    model_state_dict[f"{prefix}expert_mlps.mlp_op.down_proj.weight"] = down_proj_weights

    gate_up_proj_weights = torch.stack(gate_up_proj_weights_list)
    model_state_dict[f"{prefix}expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj_weights

    if experts_bias:
        down_proj_biases = torch.stack(down_proj_biases_list)
        model_state_dict[f"{prefix}expert_mlps.mlp_op.down_proj.bias"] = down_proj_biases

        gate_up_proj_biases = torch.stack(gate_up_proj_biases_list)
        model_state_dict[f"{prefix}expert_mlps.mlp_op.gate_up_proj.bias"] = gate_up_proj_biases

class CPUExpert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, glu_type: str, hidden_act: str, hidden_act_scaling_factor: float = 1., hidden_act_bias: float = 0., bias: bool = False, dtype=torch.bfloat16):
        super().__init__()
        if glu_type not in ["glu", "swiglu"]:
            raise ValueError(f"glu_type='{glu_type}' not supported")
        self.glu_type = glu_type
        self.activation_fn = ACT2FN[hidden_act]
        self.hidden_act_scaling_factor = hidden_act_scaling_factor
        self.hidden_act_bias = hidden_act_bias

        self.gate_proj = nn.Linear(dim, inter_dim, bias=bias, dtype=dtype)
        self.down_proj = nn.Linear(inter_dim, dim, bias=bias, dtype=dtype)
        self.up_proj = nn.Linear(dim, inter_dim, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.glu_type == "glu":
            return self.down_proj(self.activation_fn(self.gate_proj(x) * self.hidden_act_scaling_factor) * (self.up_proj(x) + self.hidden_act_bias))
        elif self.glu_type == "swiglu":
            gate = self.gate_proj(x)
            gate = gate * self.activation_fn(gate * self.hidden_act_scaling_factor)
            up = self.up_proj(x)
            gate = gate * (up + self.hidden_act_bias)
            return self.down_proj(gate)
        else:
            raise NotImplementedError(f"glu_type='{self.glu_type}' not supported")

class CPURMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Critical difference with LlamaRMSNorm: We multiply in full precision and then convert
        # to the target data type instead of converting hidden_states to the target data type and
        # then multiplying in full precision.
        output = self.weight * hidden_states
        return output.to(input_dtype)
