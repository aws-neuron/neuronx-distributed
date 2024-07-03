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
""" PyTorch LLaMA model for NXD inference."""
import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from modules.custom_calls import CustomRMSNorm
from torch import nn
from transformers import LlamaPreTrainedModel
from transformers.activations import ACT2FN
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

from dataclasses import dataclass

from modules.autobucketing import slice_lhs, slice_rhs
from modules.gqa import (
    BaseGroupQueryAttention,
    GroupQueryAttention_O,
    GroupQueryAttention_QKV,
    determine_sharding_strategy,
    get_shardable_head_counts,
)
from modules.model_base import NeuronBaseForCausalLM
from modules.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    SPECULATION_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    ModelWrapper,
)
from neuronxcc.nki.kernels.attention import attention_isa_kernel
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import LlamaForCausalLM

from neuronx_distributed.parallel_layers import parallel_state, utils
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.utils.sampling import Sampler

_flash_fwd_call = nki_jit()(attention_isa_kernel)

_LLAMA_MODULE_MAP = {}


def get_rmsnorm_cls():
    # Intialize to the approperiate implementation of RMSNorm
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


class NeuronLlamaConfig(LlamaConfig):
    def __init__(
        self, max_batch_size=1, tp_degree=1, n_positions=128, padding_side="right", speculation_length=0, **kwargs
    ):
        self.max_batch_size = max_batch_size
        self.tp_degree = tp_degree
        self.attn_cls = "NeuronLlamaAttention"
        self.n_positions = n_positions
        self.padding_side = padding_side
        self.speculation_length = speculation_length

        self.trace_tokengen_model = True

        self.ctx_batch_size = kwargs.pop("ctx_batch_size", max_batch_size)
        self.tkg_batch_size = kwargs.pop("tkg_batch_size", max_batch_size)

        # decoder specific params
        self.batch_size = max_batch_size
        self.n_active_tokens = n_positions

        # bucketing specific params
        self.enable_context_encoding_bucketing = False
        self.enable_token_generation_bucketing = False
        self.buckets = [n_positions]
        self.bucket_n_active_tokens = self.enable_context_encoding_bucketing

        self.is_continuous_batching = kwargs.pop("is_continuous_batching", False)
        self.on_device_sampling = kwargs.pop("on_device_sampling", False)

        # Quantization specific params
        self.quantized = kwargs.get("quantized", False)
        self.quantized_checkpoints_path = kwargs.get("quantized_checkpoints_path", None)
        # TODO: Add validation for quantized_checkpoints_path after the design discussions

        super().__init__(**kwargs)


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
class NeuronLlamaAttention(nn.Module):
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
        self.tp_degree = config.tp_degree
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.gqa_qkv = GroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=config.hidden_size // config.num_attention_heads,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=config.torch_dtype,
            gather_output=False,
        )

        self.o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=config.hidden_size // config.num_attention_heads,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=config.torch_dtype,
            desired_sharding_strategy=self.gqa_qkv.get_sharding_strategy(),
            input_is_parallel=True,
        )

        self.num_heads = utils.divide(self.gqa_qkv.get_num_attention_heads(), self.tp_degree)
        self.num_key_value_heads = utils.divide(self.gqa_qkv.get_num_key_value_heads(), self.tp_degree)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
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

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors."""

        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        Q, K, V = self.gqa_qkv(hidden_states=hidden_states)

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        Q = Q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Rotate(Q)
        # Rotate(K)
        kv_seq_len = K.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(V, seq_len=kv_seq_len)
        Q, K = self._apply_rotary_pos_emb(Q, K, cos, sin, position_ids)

        if past_key_value is None:
            # Context encoding
            K_active = self._repeat_kv(K, self.num_key_value_groups)
            V_active = self._repeat_kv(V, self.num_key_value_groups)

            # use flash attention if (i) sequence length is large enough to get best performance,
            # (ii) Q, K, and V have the same shape. Conditions can be changed in future.

            if q_len >= 4096 and Q.shape == K_active.shape == V_active.shape:
                # original shape of q, k, v is BHSD, and expected output is also BHSD.
                logging.debug(f"Using flash_fwd for Q.shape={Q.shape}")
                # make sure to cast inputs to self.config.torch_dtype (this is needed because the downcast to bf16 might happen
                # after the kernel hlo creation step). Also convert shapes as expected by the kernel.
                Q = Q.permute(0, 1, 3, 2).reshape((bsz*self.num_heads, self.head_dim, q_len)).to(self.config.torch_dtype)
                Q = Q / math.sqrt(self.head_dim)
                K_active = K_active.permute(0, 1, 3, 2).reshape((bsz*self.num_heads, self.head_dim, q_len)).to(self.config.torch_dtype)
                V_active = V_active.reshape((bsz*self.num_heads, q_len, self.head_dim)).to(self.config.torch_dtype)
                attn_output = torch.zeros(bsz*self.num_heads, q_len, self.head_dim, dtype=Q.dtype, device=Q.device)
                _flash_fwd_call(Q, K_active, V_active, 1.0, attn_output, kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap")
                attn_output = attn_output.reshape((bsz, self.num_heads, q_len, self.head_dim))
            else:
                logging.debug(f"Not using flash_fwd for Q.shape={Q.shape}")

                # (Q.K'/√dkv) + mask
                active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)
                active_scores = torch.where(attention_mask, active_scores, torch.finfo(active_scores.dtype).min)

                # Softmax
                active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(Q.dtype)
                attn_output = torch.matmul(active_scores, V_active)
        else:
            is_speculation = position_ids.shape[-1] > 1

            # Decomposed attention for token generation
            K_prior = past_key_value[0]
            V_prior = past_key_value[1]

            # Replicate KV for GQA/MQA
            K_prior = self._repeat_kv(K_prior, self.num_key_value_groups)
            V_prior = self._repeat_kv(V_prior, self.num_key_value_groups)
            K_active = self._repeat_kv(K, self.num_key_value_groups)
            V_active = self._repeat_kv(V, self.num_key_value_groups)

            # (Q.K'/√dkv) + mask
            prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) / math.sqrt(self.head_dim)

            prior_scores = torch.where(attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min)

            active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)

            # Mask active scores for speculation
            if is_speculation:
                active_scores = torch.where(active_mask, active_scores, torch.finfo(active_scores.dtype).min)

            # Softmax across prior and active scores
            prior_scores = prior_scores.to(torch.float32)
            active_scores = active_scores.to(torch.float32)

            max_score = torch.max(prior_scores, dim=-1, keepdim=True)[0]
            if is_speculation:
                max_active_score = torch.max(active_scores, dim=-1, keepdim=True)[0]
                max_score = torch.maximum(max_score, max_active_score)
            else:
                max_score = torch.maximum(max_score, active_scores)

            prior_scores = prior_scores - max_score
            active_scores = active_scores - max_score

            prior_scores = torch.exp(prior_scores)
            active_scores = torch.exp(active_scores)

            divisor = prior_scores.sum(dim=-1, keepdim=True)
            if is_speculation:
                divisor += active_scores.sum(dim=-1, keepdim=True)
            else:
                divisor += active_scores

            softmax_prior = prior_scores / divisor
            softmax_active = active_scores / divisor

            softmax_prior = softmax_prior.to(Q.dtype)
            softmax_active = softmax_active.to(Q.dtype)

            attn_prior = torch.matmul(softmax_prior, V_prior)
            attn_active = torch.matmul(softmax_active, V_active)

            attn_output = attn_prior + attn_active

        # transpose BHSD -> BSHD
        attn_output = attn_output.transpose(1, 2).contiguous()

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.o_proj(attn_output)

        past_key_value = (K, V)

        return attn_output, past_key_value


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


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: NeuronLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        tp_degree = config.tp_degree

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.torch_dtype,
                shard_across_embedding=True,
                # We choose to shard across embedding dimesion because this stops XLA from introducing
                # rank specific constant parameters into the HLO. We could shard across vocab, but that
                # would require us to use non SPMD parallel_model_trace.
                pad=True,
            )
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList([NeuronLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
    ):
        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.embed_tokens(input_ids)

        # NeuronLlamaModel class manages the KV cache. So the attention_mask will be generated and passed
        # through to LlamaModel. We override the HF's code that generates attention mask because HF does
        # not support left aligned RHS padding. This enables Neuron to achieve higher performance and
        # extensibility.
        #
        # 4d mask is passed through the layers
        # attention_mask = _prepare_4d_causal_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        # )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = ()

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
            )

            hidden_states = layer_outputs[0]

            next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        return (hidden_states, next_decoder_cache)


class NeuronLlamaModel(LlamaModel):
    """
    NeuronLlamaModel extends the LlamaModel to be traceable.
    The forward function of this class is traced.
    """

    def __init__(self, config: NeuronLlamaConfig):
        super().__init__(config)
        tp_degree = config.tp_degree
        self.batch_size = config.batch_size
        self.n_positions = config.n_positions
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.speculation_length = config.speculation_length
        self.padding_side = config.padding_side
        self.on_device_sampling = config.on_device_sampling
        if config.on_device_sampling:
            self.sampler = Sampler(config)
        self.hidden_dim_per_head = config.hidden_size // config.num_attention_heads

        gqa_sharding_strategy = determine_sharding_strategy(tp_degree, config.num_key_value_heads)
        _, num_key_value_heads = get_shardable_head_counts(
            tp_degree, config.num_attention_heads, config.num_key_value_heads, gqa_sharding_strategy
        )

        self.num_kv_heads_per_partition = num_key_value_heads

        if parallel_state.model_parallel_is_initialized():
            world_size = parallel_state.get_tensor_model_parallel_size()  # Same as tp_degree
            self.num_kv_heads_per_partition = utils.divide(num_key_value_heads, world_size)
            self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False, pad=True)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.kv_shape = (
            self.config.max_batch_size,
            self.num_kv_heads_per_partition,
            self.config.buckets[-1],
            # self.n_positions,
            self.hidden_dim_per_head,
        )
        self.past_key_values = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.kv_shape, dtype=config.torch_dtype), requires_grad=False)
                for _ in range(config.num_hidden_layers * 2)
            ]
        )

    def _bucket_slice_kv_cacheline(self, idx):
        dim = 2
        if self.padding_side == "right":
            return slice_lhs(self.past_key_values[idx], self.n_positions, dim)
        else:
            max_idx = self.past_key_values[idx].shape[dim]
            return slice_rhs(self.past_key_values[idx], self.n_positions, max_idx, dim)

    def _gather_bucket_slice_into_kv_cacheline(self, idx, bucket_slice):
        dim = 2
        max_idx = self.past_key_values[idx].shape[dim]
        if self.padding_side == "right":
            remaining = slice_rhs(self.past_key_values[idx], max_idx - self.n_positions, max_idx, dim)
            return torch.cat([bucket_slice, remaining], dim=2)
        else:
            remaining = slice_lhs(self.past_key_values[idx], max_idx - self.n_positions, dim)
            return torch.cat([remaining, bucket_slice], dim=2)

    def create_attn_mask(self, attention_mask, is_for_context_encoding, is_for_speculation, position_ids):
        if is_for_context_encoding:
            mask = torch.full((self.n_positions, self.n_positions), True, device=attention_mask.device).tril(diagonal=0)
            mask = mask[None, None, :, :].expand(self.batch_size, 1, self.n_positions, self.n_positions)

            if self.padding_side == "right":
                return mask
            else:
                expanded_mask = (
                    attention_mask[:, None, None, :]
                    .expand(self.batch_size, 1, self.n_positions, self.n_positions)
                    .to(torch.bool)
                )
                return torch.logical_and(mask, expanded_mask)
        elif is_for_speculation:
            return (
                attention_mask[:, None, None, :]
                .expand(self.batch_size, 1, self.speculation_length, self.n_positions)
                .to(torch.bool)
            )
        else:
            return attention_mask[:, None, None, :].expand(self.batch_size, 1, 1, self.n_positions).to(torch.bool)

    def forward(self, input_ids, attention_mask, position_ids, seq_ids):
        is_for_context_encoding = input_ids.shape[-1] > 1 and self.speculation_length != input_ids.shape[-1]
        is_for_speculation = input_ids.shape[-1] == self.speculation_length
        # It is either for context encoding or for token generation
        if is_for_context_encoding:
            past_key_values = None
        else:
            past_key_values = []
            for key_layer_idx in range(0, len(self.past_key_values), 2):
                key_state = self._bucket_slice_kv_cacheline(key_layer_idx)
                value_state = self._bucket_slice_kv_cacheline(key_layer_idx + 1)
                past_key_values.append([key_state, value_state])

        # Prepare attention mask(s)
        attention_mask = self.create_attn_mask(
            attention_mask, is_for_context_encoding, is_for_speculation, position_ids
        )
        active_mask = None
        if is_for_speculation:
            active_mask = torch.full(
                (self.speculation_length, self.speculation_length), True, device=attention_mask.device
            ).tril(diagonal=0)
            active_mask = active_mask[None, None, :, :].expand(
                self.batch_size, 1, self.speculation_length, self.speculation_length
            )

        hidden_states, past_key_values = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
        )

        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(past_key_values):
            k_cache = self._bucket_slice_kv_cacheline(idx * 2)
            v_cache = self._bucket_slice_kv_cacheline(idx * 2 + 1)

            if is_for_context_encoding:
                if self.config.is_continuous_batching:
                    # scatter back to the desired seq_ids
                    seq_id_index_shape = seq_ids.shape[:1] + k_cache.shape[1:]
                    seq_id_index = seq_ids.view(-1, 1, 1, 1).expand(seq_id_index_shape)
                    k_cache = torch.scatter(k_cache, 0, seq_id_index, kv_per_layer[0])
                    v_cache = torch.scatter(v_cache, 0, seq_id_index, kv_per_layer[1])
                else:
                    # assign back to full kv_cacheline
                    k_cache = kv_per_layer[0]
                    v_cache = kv_per_layer[1]
            else:
                if self.padding_side == "left":
                    # TODO: fix it with scatter after right padding
                    k_cache = k_cache[:, :, 1:, :]
                    v_cache = v_cache[:, :, 1:, :]
                    k_cache = torch.cat([k_cache, kv_per_layer[0]], dim=2)
                    v_cache = torch.cat([v_cache, kv_per_layer[1]], dim=2)
                else:
                    scatter_index = position_ids.view(-1, 1, position_ids.shape[-1], 1).expand_as(kv_per_layer[0])
                    k_cache = torch.scatter(k_cache, 2, scatter_index, kv_per_layer[0])
                    v_cache = torch.scatter(v_cache, 2, scatter_index, kv_per_layer[1])

            k_cache = self._gather_bucket_slice_into_kv_cacheline(idx * 2, k_cache)
            v_cache = self._gather_bucket_slice_into_kv_cacheline(idx * 2 + 1, v_cache)

            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.config.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            # simple token generation
            if position_ids.shape[-1] != self.speculation_length:
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(self.batch_size, 1, self.config.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)
            # speculative decoding case; only batch_size=1
            # will need to extend the logic to support multi-batch later
            # maybe just use position_ids for index?
            else:
                index = torch.min(position_ids)
                index = torch.arange(index, index + self.speculation_length, device=hidden_states.device)
                index = index[None, :, None].expand(self.batch_size, self.speculation_length, self.config.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        logits_or_next_tokens = logits
        if self.on_device_sampling:
            # perform sampling on Neuron to get tokens
            logits_or_next_tokens = self.sampler.sample(logits[:, -1, :])

        return [logits_or_next_tokens] + updated_kv_cache


class NeuronLlamaForCausalLM(NeuronBaseForCausalLM, LlamaPreTrainedModel):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    def __init__(self, model_path: str, config: NeuronLlamaConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_side = config.padding_side
        self.kv_cache_populated = False

        self.sampler = None

        self.models = []
        self.enable_context_encoding()
        if config.trace_tokengen_model:
            self.enable_token_generation()
        if config.speculation_length > 0:
            self.enable_speculation()
        self.model_path = model_path

    @staticmethod
    def load_hf_model(model_path):
        return LlamaForCausalLM.from_pretrained(model_path)

    def enable_context_encoding(self):
        new_config = copy.deepcopy(self.config)
        new_config.batch_size = self.config.ctx_batch_size
        new_config.n_active_tokens = self.config.n_positions

        if not new_config.enable_context_encoding_bucketing:
            new_config.buckets = [new_config.buckets[-1]]

        self.context_encoding_model = ModelWrapper(new_config, NeuronLlamaModel, tag=CONTEXT_ENCODING_MODEL_TAG)

        self.models.append(self.context_encoding_model)

    def enable_token_generation(self):
        new_config = copy.deepcopy(self.config)
        new_config.batch_size = self.config.tkg_batch_size
        new_config.n_active_tokens = 1
        new_config.bucket_n_active_tokens = False

        if not new_config.enable_token_generation_bucketing:
            new_config.buckets = [new_config.buckets[-1]]

        self.token_generation_model = ModelWrapper(new_config, NeuronLlamaModel, tag=TOKEN_GENERATION_MODEL_TAG)

        self.models.append(self.token_generation_model)

    def enable_speculation(self):
        new_config = copy.deepcopy(self.config)
        new_config.batch_size = self.config.spec_batch_size
        new_config.n_active_tokens = self.config.speculation_length
        self.speculation_model = ModelWrapper(new_config, NeuronLlamaModel, tag=SPECULATION_MODEL_TAG)

        self.models.append(self.speculation_model)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # infer attention_mask from position_ids if not provided
        if attention_mask is None:
            assert position_ids is not None, "need to call forward with position_ids if attention_mask is not provided"
            batch_size, seq_len = position_ids.shape
            if position_ids.shape[-1] == 1:
                seq_len = self.config.n_positions
                position_ids_to_compare = position_ids.expand(batch_size, seq_len) - 1
            else:
                seq_len = position_ids.shape[-1]
                position_ids_to_compare = position_ids
            mask = torch.arange(seq_len).view(1, -1).expand(batch_size, seq_len)
            attention_mask = (position_ids_to_compare >= mask).to(dtype=position_ids.dtype)

        logging.debug("---input---")
        logging.debug("input_ids shape = %s type=%s", input_ids.shape, input_ids.type())
        logging.debug("attention_mask shape = %s type=%s", attention_mask.shape, attention_mask.type())
        logging.debug("position_ids shape = %s type=%s", position_ids.shape, position_ids.type())
        logging.debug("input_ids =%s", input_ids)
        logging.debug("attention_mask =%s", attention_mask)
        logging.debug("position_ids =%s", position_ids)
        logging.debug(f"seq_ids: {seq_ids}")

        if self.config.trace_tokengen_model and not self.token_generation_model.is_neuron():
            logging.debug(f"first layer kv_cache: {self.token_generation_model.model.past_key_values[0][:, 0, :, 0]}")

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        if input_ids.shape[-1] > 1 and input_ids.shape[-1] != self.config.speculation_length:
            outputs = self.context_encoding_model(input_ids, attention_mask, position_ids, seq_ids)

            if self.context_encoding_model.is_neuron():
                # Copy the KV cache from the context_encoding_model to token generation model
                if self.config.trace_tokengen_model:
                    for encoder_model, token_gen_model in zip(
                        self.context_encoding_model.model.models, self.token_generation_model.model.models
                    ):
                        encoder_kv_cache_line = encoder_model.states
                        token_gen_kv_cache_line = token_gen_model.states
                        for name, _ in token_gen_kv_cache_line._parameters.items():
                            token_gen_kv_cache_line._parameters[name] = encoder_kv_cache_line._parameters[name]
                # Also need to copy to the speculation model for speculation
                if self.config.speculation_length > 0:
                    for encoder_model, speculation_model in zip(
                        self.context_encoding_model.model.models, self.speculation_model.model.models
                    ):
                        encoder_kv_cache_line = encoder_model.states
                        speculation_kv_cache_line = speculation_model.states
                        for name, _ in speculation_kv_cache_line._parameters.items():
                            speculation_kv_cache_line._parameters[name] = encoder_kv_cache_line._parameters[name]
            self.kv_cache_populated = True
        elif input_ids.shape[-1] == self.config.speculation_length:
            outputs = self.speculation_model(input_ids, attention_mask, position_ids, seq_ids)
        else:
            outputs = self.token_generation_model(input_ids, attention_mask, position_ids, seq_ids)

        if self.config.trace_tokengen_model and not self.token_generation_model.is_neuron():
            # When traced the output kv tensors are aliased to the kv parameter list.
            # The code below mimicks that on CPU.
            new_past_key_values = outputs[1:]
            for i, new_past_key_value in enumerate(new_past_key_values):
                self.token_generation_model.model.past_key_values[i].data = new_past_key_value
                self.context_encoding_model.model.past_key_values[i].data = new_past_key_value

        logits_or_next_tokens, *_ = outputs

        logging.debug("---output---")
        logging.debug(f"{'tokens' if self.config.on_device_sampling else 'logits'} = %s, ", logits_or_next_tokens)

        next_tokens = logits_or_next_tokens

        OutputParams = CausalLMOutputWithPast(
            loss=0,
            logits=None if self.config.on_device_sampling else logits_or_next_tokens,
            past_key_values=[],
            hidden_states=logits_or_next_tokens,
            attentions=None,
        )
        OutputParams.tokens = next_tokens
        return OutputParams

    # We override this function because we want to change the way attention_mask
    # is updated each iteration.
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_for_token_generation: Optional[bool] = False,
        is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values

        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if is_for_token_generation:
                if self.padding_side == "left":
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
                    attention_mask = attention_mask[:, 1:]
                else:
                    attention_mask = torch.cat(
                        [attention_mask.new_ones((attention_mask.shape[0], 1)), attention_mask], dim=-1
                    )
                    attention_mask = attention_mask[:, :-1]
            model_kwargs["attention_mask"] = attention_mask
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if self.kv_cache_populated:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if self.kv_cache_populated:
                position_ids = torch.amax(position_ids, 1, keepdim=True)
                position_ids = position_ids + 1

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def reset(self):
        # We need to reset the KV cache flag for a new batch of inference.
        # When the flag is reset, the subsequent run will invoke the
        # context encoding model.
        self.kv_cache_populated = False

    def reset_kv_cache(self):
        # Zero out kv cache for debug.
        # For new batch inference, use reset() instead
        if not self.context_encoding_model.is_neuron():
            for i, kv_tensor in enumerate(self.context_encoding_model.model.past_key_values):
                self.context_encoding_model.model.past_key_values[i] = torch.zeros_like(kv_tensor)

        if not self.token_generation_model.is_neuron():
            for i, kv_tensor in enumerate(self.token_generation_model.model.past_key_values):
                self.token_generation_model.model.past_key_values[i] = torch.zeros_like(kv_tensor)

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        We override the GenerationMixin sample function to add support for right side padding.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False
        # auto-regressive generation
        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            is_for_token_generation = self.kv_cache_populated

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
            )

            if not self.config.on_device_sampling:
                if self.sampler is None:
                    self.config.do_sample = True
                    self.sampler = Sampler(self.config)
                next_tokens = self.sampler.sample(outputs.logits[:, -1, :])
            else:
                next_tokens = outputs.tokens

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                is_for_token_generation=is_for_token_generation,
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, None):
                this_peer_finished = True

            if this_peer_finished:
                break

        return input_ids
