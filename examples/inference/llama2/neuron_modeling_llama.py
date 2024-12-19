# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch LLaMA model for NXD inference."""
import gc
from typing import Optional, Tuple, Type, Union

import torch
import torch.distributed
from modules.attention.attention_base import NeuronAttentionBase
from modules.attention.flashdecode_attention import NeuronFDAttentionBase
from modules.attention.utils import RotaryEmbedding
from modules.custom_calls import CustomRMSNorm
from torch import nn
from transformers import LlamaPreTrainedModel
from transformers.activations import ACT2FN
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
)

from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

from modules.model_base import NeuronBaseModel, NeuronBaseForCausalLM  # noqa: E402
from modules.config import NeuronConfig  # noqa: E402

from transformers import LlamaForCausalLM  # noqa: E402

from neuronx_distributed.parallel_layers import parallel_state, utils  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402
    ColumnParallelLinear,  # noqa: E402
    ParallelEmbedding,  # noqa: E402
    RowParallelLinear,  # noqa: E402
)  # noqa: E402

_LLAMA_MODULE_MAP = {}


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return CustomRMSNorm if parallel_state.model_parallel_is_initialized() else LlamaRMSNorm


def _register_module(key: str, cls: Type[nn.Module]):
    _LLAMA_MODULE_MAP[key] = cls


def register_module(key: str):
    """
    Register a module for use in NeuronLlama.

    Arguments:
        key: String used to identify the module

    Example:
        @register_module("NeuronLlamaAttention")
        class NeuronLlamaAttention(nn.Module):
            ...
    """

    def inner(cls: Type[nn.Module]):
        _register_module(key, cls)
        return cls

    return inner

def convert_state_dict_to_fused_qkv(llama_state_dict, cfg):
    """
    This function concats the qkv weights to a Wqkv weight for fusedqkv, and deletes the qkv weights.
    """
    for l in range(cfg.hf_config.num_hidden_layers):  # noqa: E741
        llama_state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = torch.cat([
                llama_state_dict[f"layers.{l}.self_attn.q_proj.weight"],
                llama_state_dict[f"layers.{l}.self_attn.k_proj.weight"],
                llama_state_dict[f"layers.{l}.self_attn.v_proj.weight"],
            ],
        )
        del llama_state_dict[f"layers.{l}.self_attn.q_proj.weight"]
        del llama_state_dict[f"layers.{l}.self_attn.k_proj.weight"]
        del llama_state_dict[f"layers.{l}.self_attn.v_proj.weight"]

    gc.collect()

    return llama_state_dict


class NeuronLlamaMLP(nn.Module):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, neuron_config: NeuronConfig):
        super().__init__()
        self.neuron_config = neuron_config
        self.tp_degree = neuron_config.tp_degree
        self.hidden_size = neuron_config.hf_config.hidden_size
        self.intermediate_size = neuron_config.hf_config.intermediate_size
        self.act_fn = ACT2FN[neuron_config.hf_config.hidden_act]

        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=neuron_config.hf_config.torch_dtype,
                pad=True,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=neuron_config.hf_config.torch_dtype,
                pad=True,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=neuron_config.hf_config.torch_dtype,
                pad=True,
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

@register_module("NeuronLlamaFDAttention")
class NeuronLlamaFlashDecodeAttention(NeuronFDAttentionBase):
    """
    Flash decoding works with sequence-sharded KV cache, and introduces additional cc ops:
    1. all-gather all the query heads that associated within one KV group
    2. all-gather the max value in the softmax outputs within one KV group
    3. reduce-scatter the attention outputs before merging all the heads
    """

    def __init__(self, neuron_config: NeuronConfig):
        super().__init__()

        self.neuron_config = neuron_config
        self.hidden_size = neuron_config.hf_config.hidden_size
        self.num_attention_heads = neuron_config.hf_config.num_attention_heads
        self.num_key_value_heads = neuron_config.hf_config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = neuron_config.hf_config.max_position_embeddings
        self.rope_theta = neuron_config.hf_config.rope_theta
        self.padding_side = neuron_config.padding_side
        self.torch_dtype = neuron_config.hf_config.torch_dtype
        self.is_medusa = neuron_config.is_medusa

        self.num_cores_per_group = neuron_config.num_cores_per_group
        self.tp_degree = neuron_config.tp_degree

        self.fused_qkv = False
        self.clip_qkv = None

        self.init_gqa_properties()

        self.init_rope()

    def init_rope(self):
        if getattr(self.neuron_config.hf_config, "rope_scaling", None) is None:
            # TODO(yihsian): Check if we can just use our own implementation
            rotary_emb_cls = LlamaRotaryEmbedding if self.is_medusa else RotaryEmbedding
            self.rotary_emb = rotary_emb_cls(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.neuron_config.hf_config.rope_scaling["type"]
            assert scaling_type in ["linear", "dynamic"]
            rotary_emb_cls = LlamaLinearScalingRotaryEmbedding if scaling_type == "linear" else (
                LlamaDynamicNTKScalingRotaryEmbedding)
            scaling_factor = self.neuron_config.hf_config.rope_scaling["factor"]
            self.rotary_emb = rotary_emb_cls(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )


@register_module("NeuronLlamaAttention")
class NeuronLlamaAttention(NeuronAttentionBase):
    """
    Compared with LlamaAttention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, neuron_config: NeuronConfig):
        super().__init__()

        self.neuron_config = neuron_config
        self.hidden_size = neuron_config.hf_config.hidden_size
        self.num_attention_heads = neuron_config.hf_config.num_attention_heads
        self.num_key_value_heads = neuron_config.hf_config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = neuron_config.hf_config.max_position_embeddings
        self.rope_theta = neuron_config.hf_config.rope_theta
        self.padding_side = neuron_config.padding_side
        self.torch_dtype = neuron_config.hf_config.torch_dtype
        self.is_medusa = neuron_config.is_medusa

        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        else:
            self.tp_degree = 1
        self.fused_qkv = getattr(neuron_config,'fused_qkv',False)
        self.clip_qkv = None

        self.init_gqa_properties()

        self.init_rope()

    def init_rope(self):
        if not hasattr(self.neuron_config.hf_config, "rope_scaling") or self.neuron_config.hf_config.rope_scaling is None:
            # TODO(yihsian): Check if we can just use our own implementation
            if self.is_medusa:
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
        else:
            scaling_type = self.neuron_config.hf_config.rope_scaling["type"]
            scaling_factor = self.neuron_config.hf_config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")


class NeuronLlamaDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, neuron_config: NeuronConfig):
        super().__init__()
        self.hidden_size = neuron_config.hf_config.hidden_size
        self.self_attn = _LLAMA_MODULE_MAP[neuron_config.attn_cls](neuron_config=neuron_config)
        self.mlp = NeuronLlamaMLP(neuron_config)
        self.input_layernorm = get_rmsnorm_cls()(
            neuron_config.hf_config.hidden_size,
            eps=neuron_config.hf_config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            neuron_config.hf_config.hidden_size,
            eps=neuron_config.hf_config.rms_norm_eps,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value)

        return outputs


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class NeuronLlamaModel(NeuronBaseModel, LlamaPreTrainedModel):
    """
    The neuron version of the LlamaModel
    """
    def setup_attr_for_model(self, neuron_config: NeuronConfig):
        # Needed for init_inference_optimization()
        self.on_device_sampling = neuron_config.on_device_sampling
        self.tp_degree = neuron_config.tp_degree
        self.hidden_size = neuron_config.hf_config.hidden_size
        self.num_attention_heads = neuron_config.hf_config.num_attention_heads
        self.num_key_value_heads = neuron_config.hf_config.num_key_value_heads
        self.max_batch_size = neuron_config.max_batch_size
        self.buckets = neuron_config.buckets

    def init_model(self, neuron_config: NeuronConfig):

        self.padding_idx = neuron_config.hf_config.pad_token_id
        self.vocab_size = neuron_config.hf_config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                neuron_config.hf_config.vocab_size,
                neuron_config.hf_config.hidden_size,
                self.padding_idx,
                dtype=neuron_config.hf_config.torch_dtype,
                shard_across_embedding=True,
                # We choose to shard across embedding dimension because this stops XLA from introducing
                # rank specific constant parameters into the HLO. We could shard across vocab, but that
                # would require us to use non SPMD parallel_model_trace.
                pad=True,
            )
            self.lm_head = ColumnParallelLinear(neuron_config.hf_config.hidden_size,
                neuron_config.hf_config.vocab_size,
                bias=False,
                pad=True,
            )
        else:
            self.embed_tokens = nn.Embedding(
                neuron_config.hf_config.vocab_size,
                neuron_config.hf_config.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(
                neuron_config.hf_config.hidden_size,
                neuron_config.hf_config.vocab_size,
                bias=False,
            )

        self.layers = nn.ModuleList(
            [NeuronLlamaDecoderLayer(neuron_config) for _ in range(neuron_config.hf_config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(neuron_config.hf_config.hidden_size, eps=neuron_config.hf_config.rms_norm_eps)

        self.is_medusa = neuron_config.is_medusa
        self.num_medusa_heads = neuron_config.num_medusa_heads
        self.medusa_speculation_length = neuron_config.medusa_speculation_length

        if self.is_medusa:
            if parallel_state.model_parallel_is_initialized():
                medusa_head_cls = ColumnParallelLinear
            else:
                medusa_head_cls = nn.Linear
            for i in range(self.num_medusa_heads):
                medusa_head = nn.Sequential(
                    *([ResBlock(neuron_config.hf_config.hidden_size)] * 1),
                    medusa_head_cls(neuron_config.hf_config.hidden_size, neuron_config.hf_config.vocab_size, bias=False),
                )
                setattr(self, f"medusa_head_{i}", medusa_head)


class NeuronLlamaForCausalLM(NeuronBaseForCausalLM, LlamaPreTrainedModel):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NeuronLlamaModel

    @staticmethod
    def load_hf_model(model_path):
        return LlamaForCausalLM.from_pretrained(model_path)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, neuron_config: NeuronConfig) -> dict:
        """ This function should be over-ridden in child classes as needed """
        if getattr(neuron_config,'fused_qkv', False):
            state_dict = convert_state_dict_to_fused_qkv(state_dict, neuron_config)

        # to facilitate rank usage in attention
        num_layers = neuron_config.hf_config.num_hidden_layers
        for i in range(num_layers):
            state_dict[f'layers.{i}.self_attn.rank_util.rank'] = torch.arange(0, neuron_config.tp_degree,
                                                                              dtype=torch.int32)

        # to facilitate rank usage in base model
        state_dict['rank_util.rank'] = torch.arange(0, neuron_config.tp_degree, dtype=torch.int32)

        return state_dict
