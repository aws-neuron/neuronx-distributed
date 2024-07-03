# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch Mixtral model."""
import math
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch_xla.core.xla_model as xm
from packaging import version
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.models.llama.modeling_llama import LlamaMLP as LlamaMLPHF
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    MIXTRAL_INPUTS_DOCSTRING,
    MIXTRAL_START_DOCSTRING,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention as MixtralAttentionHF,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer as MixtralDecoderLayerHF,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM as MixtralForCausalLMHF,
)
from transformers.models.mixtral.modeling_mixtral import MixtralModel as MixtralModelHF
from transformers.models.mixtral.modeling_mixtral import MixtralPreTrainedModel
from transformers.models.mixtral.modeling_mixtral import (
    MixtralRMSNorm as MixtralRMSNormHF,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils
from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPsCapacityFactor
from neuronx_distributed.modules.moe.loss_function import load_balancing_loss_func
from neuronx_distributed.modules.moe.model import MoE, MoESequenceParallelMode
from neuronx_distributed.modules.moe.moe_parallel_layers import InputParallelLinear
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_size,
)
from neuronx_distributed.utils.model_utils import move_model_to_device


def _init_normal(std, w):
    return nn.init.normal_(w, mean=0.0, std=std)


if version.parse(torch.__version__) >= version.parse("2.1"):
    from torch_xla.utils.checkpoint import checkpoint

    checkpoint_method = checkpoint
else:
    checkpoint_method = torch.utils.checkpoint.checkpoint


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MixtralConfig"


# TODO: Add generalized modules for RMSNorm, Attention and MLP for use across the different example scripts
class MixtralRMSNorm(MixtralRMSNormHF):
    """Neuron implementation of MixtralRMSNorm which upcasts hidden_states to torch.double for improved numeric accuracy."""

    def __init__(self, hidden_size, eps=1e-6, sequence_parallel_enabled=False):
        """
        MixtralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size, eps=eps)
        setattr(self.weight, "sequence_parallel_enabled", sequence_parallel_enabled)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # Upcast hidden_states to float64
        hidden_states = hidden_states.to(torch.double)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class CoreAttention(nn.Module):
    def __init__(self, sliding_window=None):
        super().__init__()
        self.sliding_window = sliding_window

    def forward(self, query_states, key_states, value_states):
        device = query_states.device

        bsz, num_heads, q_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[-2]
        assert q_len == kv_seq_len, "KV-Cache flow is not fully supported"

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        causal_mask = torch.triu(torch.ones((1, 1, q_len, kv_seq_len), device=device), diagonal=1).bool()

        if self.sliding_window is not None:
            # Mask out tokens more than sliding_window steps in the past
            past_mask = torch.tril(torch.ones((1, 1, q_len, kv_seq_len), device=device), diagonal=-self.sliding_window)
            causal_mask = causal_mask.logical_or(past_mask)

        attn_weights = attn_weights.masked_fill_(causal_mask, -10000.0)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.double).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output


class MixtralAttention(MixtralAttentionHF):
    """Neuron Implementation of MixtralAttention which enabled Neuron Tensor Parallel."""

    def __init__(self, config: MixtralConfig):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if not hasattr(config, "kv_shared_group_size"):
            config.kv_shared_group_size = 1

        if not hasattr(config, "qkv_linear"):
            config.qkv_linear = False

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        init_method = partial(_init_normal, config.initializer_range)
        if self.num_heads == self.num_key_value_heads:
            self.qkv_proj = ColumnParallelLinear(
                self.hidden_size,
                3 * self.num_heads * self.head_dim,
                stride=3,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
            )
            self.split_size = self.num_heads * self.head_dim // get_tensor_model_parallel_size()
        elif self.config.qkv_linear:
            self.qkv_proj = GQAQKVColumnParallelLinear(
                self.hidden_size,
                [self.num_heads * self.head_dim, self.num_key_value_heads * self.head_dim],
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                kv_size_multiplier=self.config.kv_shared_group_size,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
            )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
        )
        self.num_heads = neuronx_dist_utils.divide(config.num_attention_heads, get_tensor_model_parallel_size())
        self.num_key_value_heads = neuronx_dist_utils.divide(
            config.num_key_value_heads * self.config.kv_shared_group_size, get_tensor_model_parallel_size()
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.core_attn = CoreAttention(sliding_window=config.sliding_window)

        self.rotary_emb = MixtralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        if config.move_model_to_device:
            move_model_to_device(self, xm.xla_device())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert use_cache is False, "KV-Cache flow is not fully supported"
        assert past_key_value is None, "KV-Cache flow is not fully supported"
        assert attention_mask is None, "Attention mask is handled in CoreAttention"

        bsz, q_len, _ = hidden_states.size()

        if self.config.sequence_parallel_enabled:
            q_len, bsz, _ = hidden_states.size()
            q_len = q_len * get_tensor_model_parallel_size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            if self.num_heads == self.num_key_value_heads and self.config.kv_shared_group_size == 1:
                qkv_states = self.qkv_proj(hidden_states)
                query_states, key_states, value_states = qkv_states.split(self.split_size, dim=2)
            elif self.config.qkv_linear:
                query_states, key_states, value_states = self.qkv_proj(hidden_states)
            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

        if self.config.sequence_parallel_enabled:
            query_states = query_states.view(q_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            key_states = key_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
            value_states = value_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = self.core_attn(query_states, key_states, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        if self.config.sequence_parallel_enabled:
            attn_output = attn_output.permute(2, 0, 1, 3)
            attn_output = attn_output.reshape(q_len, bsz, self.hidden_size // get_tensor_model_parallel_size())
        else:
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size // get_tensor_model_parallel_size())

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def initialize_mixtral_moe_layer(config):
    if config.sequence_parallel_enabled:
        assert (
            config.moe_sequence_parallel_mode != MoESequenceParallelMode.NO_SP
        ), "sequence_parallel_enabled=true, but moe_sequence_parallel_mode set to NO_SP"
        sequence_parallel_mode = MoESequenceParallelMode[config.moe_sequence_parallel_mode]
    else:
        sequence_parallel_mode = MoESequenceParallelMode.NO_SP

    # Default to RouterTopK (without Sinkhorn)
    router = RouterTopK(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        sequence_parallel_mode=sequence_parallel_mode,
    )

    init_method = partial(_init_normal, config.initializer_range)

    # TODO: Potentially add activation checkpointing in the ExpertMLPs, depending on profile/performance needs
    expert_mlps = ExpertMLPsCapacityFactor(
        num_experts=config.num_local_experts,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        capacity_factor=config.capacity_factor,
        init_method=init_method,
        output_layer_init_method=init_method,
        glu_mlp=True,
        sequence_parallel_mode=sequence_parallel_mode,
        permute_strategy=config.expert_mlps_permute_strategy,
        top_k=config.num_experts_per_tok,
        normalize_top_k_affinities=True,
    )

    moe_layer = MoE(
        router=router, expert_mlps=expert_mlps, return_router_logits=True, sequence_parallel_mode=sequence_parallel_mode
    )

    return moe_layer


class LlamaMLP(LlamaMLPHF):
    """Neuron implementation of Llama MLP layer which enables Tensor Parallelism for linear layers."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.pretraining_tp = config.pretraining_tp
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        init_method = partial(_init_normal, config.initializer_range)
        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            2 * self.intermediate_size,
            stride=2,
            bias=False,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled,
        )
        self.split_size = self.intermediate_size // get_tensor_model_parallel_size()
        if config.move_model_to_device:
            move_model_to_device(self, xm.xla_device())

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)]
            down_proj = sum(down_proj)
        else:
            gate_proj, up_proj = self.gate_up_proj(x).split(self.split_size, dim=2)

            def activation_mlp(gate_proj, up_proj):
                activation_output = self.act_fn(gate_proj)
                return activation_output * up_proj

            # We checkpoint the MLP compute too, since we see extra data movement which is more
            # expensive than the recompute in this case.
            if self.config.selective_checkpoint_enabled:
                intermediate_states = checkpoint_method(activation_mlp, gate_proj, up_proj)
            else:
                intermediate_states = self.act_fn(gate_proj) * up_proj
            down_proj = self.down_proj(intermediate_states)

        return down_proj


class MixtralDecoderLayer(MixtralDecoderLayerHF):
    """
    Neuron Implementation of MixtralDecoderLayer. Comparing to HuggingFace version, the following changes are included:
        1. Use Neuron version of MoE layer implementation.
        2. Support moe_frequency > 1: Insert a MoE layer every {moe_frequency} decoder layers, othereise use normal MLP layer.
    """

    def __init__(self, config: MixtralConfig, layer_index):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = MixtralAttention(config=config)
        if layer_index % config.moe_frequency == 0:
            self.mlp = initialize_mixtral_moe_layer(config)
        else:
            self.mlp = LlamaMLP(config)
        self.input_layernorm = MixtralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=config.sequence_parallel_enabled
        )
        self.post_attention_layernorm = MixtralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=config.sequence_parallel_enabled
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_router_logits: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            past_router_logits(`torch.FloatTensor`): logits of all previous routers
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, LlamaMLP):
            hidden_states = self.mlp(hidden_states)
        elif isinstance(self.mlp, MoE):
            hidden_states, router_logits = self.mlp(hidden_states)
        else:
            raise Exception(f"MLP Layer type must be either LlamaMLP or MoE, got {type(self.mlp).__name__}.")
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            # Concatenate the router logits with previous router logits
            if past_router_logits is not None:
                if isinstance(self.mlp, MoE):
                    router_logits = torch.cat((past_router_logits, router_logits), dim=0)
                else:
                    router_logits = past_router_logits
            outputs += (router_logits,)

        # TODO: Return a tuple here to workaround a NxD PP issue, can revert once the issue is fixed.
        return (outputs,)


@add_start_docstrings(
    "The bare Mixtral Model outputting raw hidden-states without any specific head on top.",
    MIXTRAL_START_DOCSTRING,
)
class MixtralModel(MixtralModelHF):
    """
    Neuron implementation of MixtralModel. Comparing to HuggingFace version, the following changes are included:
        1. Support Neuron Sequence Parallel.
        2. Refactor rouger logits accumulation to support Neuron Pipeline Parallel.

    Args:
        config: MixtralConfig
    """

    def __init__(self, config: MixtralConfig):
        MixtralPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        init_method = partial(_init_normal, config.initializer_range)
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, init_method=init_method
        )
        self.layers = nn.ModuleList([MixtralDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = MixtralRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=config.sequence_parallel_enabled
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert past_key_values is None, "KV-Cache flow is not fully supported"
        assert not use_cache, "KV-Cache flow is not fully supported"

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_router_logits = None

        if self.config.sequence_parallel_enabled:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            hidden_states = mappings.scatter_to_sequence_parallel_region(hidden_states)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # TODO: Check why modeling_llama_nxd defines create_custom_forward here to pass in extra arguments
                layer_outputs = checkpoint_method(
                    decoder_layer.__call__,
                    hidden_states,
                    all_router_logits,
                    attention_mask,
                    position_ids,
                    None,
                    output_attentions,
                    output_router_logits,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    all_router_logits,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                )
            layer_outputs = layer_outputs[0]

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits = layer_outputs[-1]

        hidden_states = self.norm(hidden_states)

        if self.config.sequence_parallel_enabled:
            hidden_states = mappings.gather_from_sequence_parallel_region(hidden_states, to_model_parallel=False)
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


class MixtralForCausalLM(MixtralForCausalLMHF):
    """
    Neuron implementation of MixtralForCausalLM. Comparing to HuggingFace version, the following changes are included:
        1. Use custom router aux loss function that is compatible with Neuron devices.
        2. Switch to loss function implementations that support Neuron Tensor Parallel.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        MixtralPreTrainedModel.__init__(self, config)
        self.model = MixtralModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size

        init_method = partial(_init_normal, config.initializer_range)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=False,
            init_method=init_method,
        )
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(MIXTRAL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MixtralForCausalLM

        >>> model = MixtralForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert past_key_values is None, "KV-Cache flow is not fully supported"
        assert not use_cache, "KV-Cache flow is not fully supported"

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)

        logits = logits.double()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = parallel_cross_entropy
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))

            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            loss = torch.mean(loss)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1], self.num_experts, self.num_experts_per_tok
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


def init_weights(module):
    """
    Re-init weights after partition
    Referred from HF transformers https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/mixtral/modeling_mixtral.py#L849
    """
    if isinstance(module, MixtralRMSNorm):
        module.weight.data.fill_(1.0)
    elif isinstance(module, (ParallelEmbedding, RowParallelLinear, ColumnParallelLinear, InputParallelLinear)):
        module.init_weight_cpu()
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, GQAQKVColumnParallelLinear):
        module.initialize_weight_biases()
