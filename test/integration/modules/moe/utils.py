"""
Utilities Module
---------------
This module contains common utilities, constants, and helper functions used across
the device correctness test framework.
"""

import logging
from typing import Dict, Any, List, Optional

import torch
from torch import nn

from neuronx_distributed import parallel_layers
from neuronx_distributed.modules.moe import ACT2FN
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from utils_testing import ExptCfg

import os

# ===== CONSTANTS =====

# Device types
XLA_DEVICE = "xla"
CPU_DEVICE = "cpu"

# Test modes
TRAINING_MODE = "training"
INFERENCE_MODE = "inference"

# Dimensions
BATCH_DIM_TRAINING = 1
BATCH_DIM_INFERENCE = 0
# For gathering router logits and expert indices from sequence parallel region,
# always use dimension 0 because 0th dimension is T = S*B (tokens = sequence length * batch size)
TOKEN_DIM = 0

# Optimizer settings
GRAD_CLIPPING_ENABLED = True

# Data types
PRECISION_SENSITIVE_DTYPE = torch.bfloat16

# ===== LOGGING SETUP =====

# Configure logger
logger = logging.getLogger(__name__)


def print_rank0(message: str) -> None:
    """
    Print a message only from rank 0 process.

    Args:
        message: Message to print
    """
    if xr.global_ordinal() == 0:
        print(message)


def reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    """
    Reduce loss across expert data parallel and expert model parallel groups.

    Args:
        loss: Loss tensor

    Returns:
        Reduced loss tensor
    """
    from neuronx_distributed.parallel_layers import parallel_state

    # Get parallel groups
    edp_groups = parallel_state.get_expert_data_parallel_replica_groups()
    emp_groups = parallel_state.get_expert_model_parallel_replica_groups()
    dp_size = parallel_state.get_data_parallel_size()

    # Scale loss by data parallel size
    loss /= dp_size

    # Reduce across expert data parallel groups
    xm.all_reduce("sum", [loss], groups=edp_groups)

    # Reduce across expert model parallel groups
    xm.all_reduce("sum", [loss], groups=emp_groups)

    return loss


def split_inputs_into_chunks(
    inputs: torch.Tensor,
    dp_size: int,
    is_cpu: bool,
    test_mode: str
) -> List[torch.Tensor]:
    """
    Split input tensor into chunks for data parallel processing.

    Args:
        inputs: Input tensor
        dp_size: Data parallel size
        is_cpu: Whether running on CPU
        test_mode: Test mode (training or inference)

    Returns:
        List of input chunks
    """
    # For inference or XLA device, inputs are already sharded
    if test_mode == INFERENCE_MODE or not is_cpu:
        return [inputs]

    # For training on CPU, split inputs by data parallel size
    batch_dim = BATCH_DIM_TRAINING if test_mode == TRAINING_MODE else BATCH_DIM_INFERENCE
    split_tensor = torch.tensor_split(inputs, dp_size, dim=batch_dim)

    # Make contiguous for better performance
    return [t.contiguous() for t in split_tensor]


def shard_batch(
    tensor: torch.Tensor,
    cfg: ExptCfg,
    dp_size: int,
    dp_rank: int,
    test_mode: str
) -> torch.Tensor:
    """
    Shard a tensor for data parallel processing.

    Args:
        tensor: Input tensor
        cfg: Test configuration
        dp_size: Data parallel size
        dp_rank: Data parallel rank
        test_mode: Test mode (training or inference)

    Returns:
        Sharded tensor
    """
    # Validate tensor dimensions
    assert tensor.dim() < 4 and tensor.dim() > 0, f"Tensor must have 1-3 dimensions, got {tensor.dim()}"

    # Get original shape
    shape = list(tensor.shape)

    if test_mode == TRAINING_MODE:
        # For training mode, reshape to (seq_len, dp_size * batch_size, hidden_size)
        # and narrow to get the shard for this rank
        tensor = tensor.reshape(cfg.seq_len, dp_size * cfg.batch_size, -1)
        tensor = tensor.narrow(1, dp_rank * cfg.batch_size, cfg.batch_size)

        # Update shape to account for sharding
        if len(shape) > 2:
            shape[1] //= dp_size
        else:
            shape[0] //= dp_size

        # Reshape back to original dimensions but with sharded batch
        return tensor.reshape(*shape)
    else:
        # For inference mode, return tensor as is (already sharded)
        return tensor


def get_appropriate_grad_context(cfg: ExptCfg):
    """
    Get the appropriate gradient context manager based on test mode.

    Args:
        cfg: Test configuration

    Returns:
        Context manager for gradients
    """
    if cfg.test_mode == TRAINING_MODE:
        return torch.enable_grad
    else:
        return torch.no_grad


def should_transpose_shared_experts_weights(cfg: ExptCfg) -> bool:
    """
    Determine if shared experts weights should be transposed based on configuration.

    Args:
        cfg: Configuration object with kernel settings

    Returns:
        bool: True if weights should be transposed, False otherwise
    """
    # Check if MoE fused TKG is enabled
    if cfg.moe_fused_tkg_enabled:
        # Case 1: Explicitly enabled kernel
        if cfg.moe_fused_tkg_kernel_enabled:
            return True

        # Case 2: Kernel setting is None but device is XLA
        if cfg.moe_fused_tkg_kernel_enabled is None and cfg.device == XLA_DEVICE:
            return True

        # Case 3: Kernel disabled but shared MLP kernel is None and device is XLA
        if (cfg.moe_fused_tkg_kernel_enabled is False and
            cfg.shared_mlp_kernel_enabled is None and
            cfg.device == XLA_DEVICE):
            return True

    # Default case: don't transpose
    return False

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
    hidden_act_bias: float = 0.,
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
        # Add hidden_act_bias to the up_proj portion (second half) of the concatenated bias
        intermediate_size = gate_up_proj_biases.shape[-1] // 2
        gate_up_proj_biases[:, intermediate_size:] += hidden_act_bias
        model_state_dict[f"{prefix}expert_mlps.mlp_op.gate_up_proj.bias"] = gate_up_proj_biases

class CPUExpert(nn.Module):
    def __init__(
            self,
            dim: int,
            inter_dim: int,
            glu_type: str,
            hidden_act: str,
            hidden_act_scaling_factor: float = 1.,
            gate_clamp_upper_limit: Optional[float] = None,
            gate_clamp_lower_limit: Optional[float] = None,
            up_clamp_upper_limit: Optional[float] = None,
            up_clamp_lower_limit: Optional[float] = None,
            hidden_act_bias: float = 0.,
            bias: bool = False,
            dtype=torch.bfloat16
        ):
        super().__init__()
        if glu_type not in ["glu", "swiglu"]:
            raise ValueError(f"glu_type='{glu_type}' not supported")
        self.glu_type = glu_type
        self.activation_fn = ACT2FN[hidden_act]
        self.hidden_act_scaling_factor = hidden_act_scaling_factor
        self.hidden_act_bias = hidden_act_bias
        self.gate_clamp_upper_limit = gate_clamp_upper_limit
        self.gate_clamp_lower_limit = gate_clamp_lower_limit
        self.up_clamp_upper_limit = up_clamp_upper_limit
        self.up_clamp_lower_limit = up_clamp_lower_limit

        self.gate_proj = nn.Linear(dim, inter_dim, bias=bias, dtype=dtype)
        self.down_proj = nn.Linear(inter_dim, dim, bias=bias, dtype=dtype)
        self.up_proj = nn.Linear(dim, inter_dim, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.glu_type == "glu":
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            if self.gate_clamp_lower_limit or self.gate_clamp_upper_limit:
                gate = gate.clamp(min=self.gate_clamp_lower_limit, max=self.gate_clamp_upper_limit)
            if self.up_clamp_lower_limit or self.up_clamp_upper_limit:
                up = up.clamp(min=self.up_clamp_lower_limit, max=self.up_clamp_upper_limit)
            return self.down_proj(self.activation_fn(gate * self.hidden_act_scaling_factor) * (up + self.hidden_act_bias))
        elif self.glu_type == "swiglu":
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            if self.gate_clamp_lower_limit or self.gate_clamp_upper_limit:
                gate = gate.clamp(min=self.gate_clamp_lower_limit, max=self.gate_clamp_upper_limit)
            if self.up_clamp_lower_limit or self.up_clamp_upper_limit:
                up = up.clamp(min=self.up_clamp_lower_limit, max=self.up_clamp_upper_limit)
            gate = gate * self.activation_fn(gate * self.hidden_act_scaling_factor)
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
