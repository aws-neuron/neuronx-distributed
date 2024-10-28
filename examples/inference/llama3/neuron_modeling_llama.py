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
from typing import Optional, Tuple, Type, Union

import torch

from modules.attention.attention_base import NeuronAttentionBase
from modules.attention.utils import RotaryEmbedding
from modules.custom_calls import CustomRMSNorm
from torch import nn
from transformers import LlamaPreTrainedModel
from transformers.activations import ACT2FN
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaRMSNorm
)
import math
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

from modules.autobucketing import slice_lhs, slice_rhs  # noqa: E402
from modules.gqa import (  # noqa: E402
    BaseGroupQueryAttention,  # noqa: E402
    determine_sharding_strategy,  # noqa: E402
    get_shardable_head_counts,  # noqa: E402
)  # noqa: E402
from modules.model_base import NeuronBaseModel, NeuronBaseForCausalLM  # noqa: E402
from modules.config import NeuronInferenceConfig  # noqa: E402

from transformers import LlamaForCausalLM  # noqa: E402

from neuronx_distributed.parallel_layers import parallel_state, utils  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402
    ColumnParallelLinear,  # noqa: E402
    ParallelEmbedding,  # noqa: E402
    RowParallelLinear,  # noqa: E402
)  # noqa: E402
from neuronx_distributed.utils.sampling import Sampler  # noqa: E402

_LLAMA_MODULE_MAP = {}


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return CustomRMSNorm if parallel_state.model_parallel_is_initialized() else LlamaRMSNorm


def preshard_hook_fn(module: torch.nn.Module, model_state_dict: dict, prefix: str) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)

    return False


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


class NeuronLlamaConfig(NeuronInferenceConfig, LlamaConfig):
    def __init__(
            self, max_batch_size=1, tp_degree=1, n_positions=128, padding_side="right", speculation_length=0, **kwargs
    ):
        self.attn_cls = "NeuronLlamaAttention"
        self.generation_config = {}

        super().__init__(
            tp_degree=tp_degree,
            seq_len=n_positions,
            padding_side=padding_side,
            speculation_length=speculation_length,
            max_batch_size=max_batch_size,
            **kwargs,
        )


class NeuronLlamaMLP(nn.Module):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.tp_degree = config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.torch_dtype,
                pad=True,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=config.torch_dtype,
                pad=True,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=config.torch_dtype,
                pad=True,
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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

    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.padding_side = config.padding_side
        self.torch_dtype = config.torch_dtype
        self.is_medusa = config.is_medusa

        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        else:
            self.tp_degree = 1
        self.fused_qkv = False
        self.clip_qkv = None

        self.init_gqa_properties()

        self.init_rope()

    def init_rope(self):
        if not hasattr(self.config, "rope_scaling") or self.config.rope_scaling is None:
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
            rope_type = self.config.rope_scaling.get(
                "rope_type", self.config.rope_scaling.get("type", None)
            )
            if rope_type == "llama3":
                self.rotary_emb = Llama3RotaryEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                    factor=self.config.rope_scaling["factor"],
                    low_freq_factor=self.config.rope_scaling["low_freq_factor"],
                    high_freq_factor=self.config.rope_scaling["high_freq_factor"],
                    original_max_position_embeddings=self.config.rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            else:
                # LlamaRotaryEmbedding automatically chooses the correct scaling type from config.
                self.rotary_emb = LlamaRotaryEmbedding(self.config)

class Llama3RotaryEmbedding(nn.Module):
    """
    Adapted from Llama 4.43 impl
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/llama/modeling_llama.py#L78
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/modeling_rope_utils.py#L345

    This implementation ensures inv_freq is calculated and stored in fp32.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=131072,
        base=500000.0,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = original_max_position_embeddings
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )

            low_freq_wavelen = self.old_context_len / self.low_freq_factor
            high_freq_wavelen = self.old_context_len / self.high_freq_factor
            new_freqs = []
            for freq in inv_freq:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / self.factor)
                else:
                    assert low_freq_wavelen != high_freq_wavelen
                    smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                        self.high_freq_factor - self.low_freq_factor
                    )
                    new_freqs.append((1 - smooth) * freq / self.factor + smooth * freq)
            self.inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class NeuronLlamaDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = _LLAMA_MODULE_MAP[config.attn_cls](config=config)
        self.mlp = NeuronLlamaMLP(config)
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

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
    def setup_attr_for_model(self, config: NeuronLlamaConfig):
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.on_device_sampling
        self.tp_degree = config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.max_batch_size
        self.buckets = config.buckets

    def init_model(self, config: NeuronLlamaConfig):
        self.padding_idx = None
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.torch_dtype,
                shard_across_embedding=True,
                # We choose to shard across embedding dimension because this stops XLA from introducing
                # rank specific constant parameters into the HLO. We could shard across vocab, but that
                # would require us to use non SPMD parallel_model_trace.
                pad=True,
            )
            self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False, pad=True)
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.layers = nn.ModuleList([NeuronLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        self.is_medusa = config.is_medusa
        self.num_medusa_heads = config.num_medusa_heads
        self.medusa_speculation_length = config.medusa_speculation_length

        if self.is_medusa:
            if parallel_state.model_parallel_is_initialized():
                medusa_head_cls = ColumnParallelLinear
            else:
                medusa_head_cls = nn.Linear
            for i in range(self.num_medusa_heads):
                medusa_head = nn.Sequential(
                    *([ResBlock(config.hidden_size)] * 1),
                    medusa_head_cls(config.hidden_size, config.vocab_size, bias=False),
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
