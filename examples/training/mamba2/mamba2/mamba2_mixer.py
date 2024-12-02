# coding=utf-8
# Copyright 2024 state-spaces/mamba2 org and HuggingFace Inc. team.
# Modifications Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional

import torch
from torch import nn
from torch.types import Device
from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
from transformers.activations import ACT2FN
from einops import rearrange
import neuronx_distributed.parallel_layers.parallel_state as ps
from neuronx_distributed.parallel_layers import RowParallelLinear, ColumnParallelLinear

from .configuration_mamba2 import Mamba2Config

from .mamba2_kernel import mamba2_kernel
from .conv1d_grouped import ConvNKI


def softplus(x, threshold=10):
    return torch.where(x < threshold, torch.log(1 + torch.exp(x)), x)


# Note: this implementation is different from the same module in `transformers` when n_groups>1
#       we normalize each channel group independently, while the original normalizes all channels in the same device
#       regardless of their group. Our version ensures that the checkpoint will behave the same when used with
#       a different tp degrees than during training (note however that using a large n_groups with few channels may
#       introduce training instabilities).
class MambaRMSNormGated(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None, n_groups: int = 1, rmsnorm_within_groups=True):
        """Gated Root Mean Square Layer Normalization with support for groups

        Paper: https://arxiv.org/abs/1910.07467

        Mamba Official: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/layernorm_gated.py#L18
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))
        self.n_groups = n_groups
        self.rmsnorm_within_groups = rmsnorm_within_groups
        self.parallel_split()

    def parallel_split(self):
        # Split weights across cores based on the current tensor parallelism rank.
        tp_rank = ps.get_tensor_model_parallel_rank()
        tp_size = ps.get_tensor_model_parallel_size()
        dim = self.weight.shape[0]
        assert dim % tp_size == 0
        assert self.n_groups % tp_size == 0
        self.n_groups = self.n_groups // tp_size
        chunk = slice(dim // tp_size * tp_rank, dim // tp_size * (tp_rank + 1))
        self.weight.data = self.weight.data[chunk].detach().clone()
        return self

    def forward(self, hidden_states, gate=None):
        hidden_states = hidden_states.to(torch.float32)

        if self.rmsnorm_within_groups:
            hidden_states = rearrange(hidden_states, "... (g d) -> ... g d", g=self.n_groups)
            gate = rearrange(gate, "... (g d) -> ... g d", g=self.n_groups)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.rmsnorm_within_groups:
            res = self.weight * rearrange(hidden_states, "... g d -> ... (g d)", g=self.n_groups)
        else:
            res = self.weight * hidden_states

        return res


class Mamba2Mixer(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, config: Mamba2Config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm
        self.rmsnorm_within_groups = config.rmsnorm_within_groups

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        assert self.intermediate_size % self.head_dim == 0
        assert self.intermediate_size // self.head_dim == self.num_heads

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        # This is a custom replacement of a grouped conv1d written as a NKI kernel for better efficiency.
        # Note: the SiLU non-linearity is already applied inside the kernel.
        self.conv1d = ConvNKI(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
            activation='silu',
        )

        # projection of the input hidden states
        self.in_proj_z = ColumnParallelLinear(self.hidden_size, self.intermediate_size, bias=config.use_bias, gather_output=False)
        self.in_proj_xBC = ColumnParallelLinear(self.hidden_size, self.conv_dim, bias=config.use_bias, gather_output=False)
        self.in_proj_dt = ColumnParallelLinear(self.hidden_size, self.num_heads, bias=config.use_bias, gather_output=False)

        # time step projection (discretization)
        dt = torch.exp(
            torch.rand(config.num_heads)
            * (math.log(config.time_step_max) - math.log(config.time_step_min))
            + math.log(config.time_step_min)
        ).clamp(min=config.time_step_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_reinit = True

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon, n_groups=self.n_groups, rmsnorm_within_groups=self.rmsnorm_within_groups)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = RowParallelLinear(self.intermediate_size, self.hidden_size, bias=config.use_bias, input_is_parallel=True)
        self.use_bias = config.use_bias
        self.parallel_split()

    def parallel_split(self):
        # Split weights across cores based on the current tensor parallelism rank.
        tp_rank = ps.get_tensor_model_parallel_rank()
        tp_size = ps.get_tensor_model_parallel_size()
        assert self.intermediate_size % tp_size == 0
        assert self.n_groups % tp_size == 0
        self.intermediate_size_tp = self.intermediate_size // tp_size
        self.n_groups_tp = self.n_groups // tp_size
        self.num_heads_tp = self.num_heads // tp_size
        self.conv_dim_tp = self.conv_dim // tp_size
        head_chunk = slice(self.num_heads_tp * tp_rank, self.num_heads_tp * (tp_rank + 1))
        # note: we have to use .clone(), otherwise the result would be a view and the original would remain in memory
        self.D.data = self.D.data[head_chunk].detach().clone()
        self.A_log.data = self.A_log.data[head_chunk].detach().clone()
        self.dt_bias.data = self.dt_bias.data[head_chunk].detach().clone()
        return self

    def nki_kernels_forward(
            self,
            hidden_states: torch.Tensor,
            cache_params: Optional[Mamba2Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        # set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size_tp = self.n_groups_tp * self.ssm_state_size

        assert cache_params is None, "cache not supported yet"
        assert self.training, "only training supported right now"
        assert attention_mask is None, "attention mask not supported yet"
        assert self.time_step_limit[0] == 0.0 and self.time_step_limit[1] == float("inf"), "dt limit not supported yet"
        assert self.activation in ["silu", "swish"]

        A = -torch.exp(self.A_log.float())  # (num_heads) or (intermediate_size, state_size)

        gate = self.in_proj_z(hidden_states)
        hidden_states_B_C = self.in_proj_xBC(hidden_states)
        time_step = self.in_proj_dt(hidden_states)

        # 1D Convolution (SiLU non-linearity is fused inside)
        hidden_states_B_C = self.conv1d(input=hidden_states_B_C.transpose(1, 2)).transpose(1, 2)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size_tp, groups_time_state_size_tp, groups_time_state_size_tp],
            dim=-1,
        )

        time_step = softplus(time_step + self.dt_bias)

        scan_output = mamba2_kernel(time_step,
                                    A,
                                    hidden_states.view(batch_size, seq_len, self.num_heads_tp, -1),
                                    B.view(batch_size, seq_len, self.n_groups_tp, -1),
                                    C.view(batch_size, seq_len, self.n_groups_tp, -1),
                                    self.D)

        scan_output = scan_output.view(batch_size, seq_len, -1)
        # Multiply "gate" branch and apply extra normalization layer
        scan_output = self.norm(scan_output, gate)
        out = self.out_proj(scan_output)

        return out

    def forward(
            self,
            hidden_states,
            cache_params: Optional[Mamba2Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        assert "xla" in self.in_proj_xBC.weight.device.type, "This model only supports forward on an XLA device"
        return self.nki_kernels_forward(hidden_states, cache_params, cache_position, attention_mask)
