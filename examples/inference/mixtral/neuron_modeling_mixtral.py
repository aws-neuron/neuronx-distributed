# coding=utf-8
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
""" PyTorch Mixtral model for NXD inference."""
import copy
import gc
import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from modules.autobucketing import slice_lhs, slice_rhs
from modules.gqa import (
    GQA,
    BaseGroupQueryAttention,
    GroupQueryAttention_O,
    GroupQueryAttention_QKV,
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
from torch import nn
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import MixtralForCausalLM, MixtralPreTrainedModel
from transformers.cache_utils import Cache
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    ModelOutput,
    MoeModelOutputWithPast,
)
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralRMSNorm,
    MixtralRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPsCapacityFactor
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.model_utils import MoESequenceParallelMode
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.parallel_layers import parallel_state, utils
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)

_flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE


def convert_mixtral_to_neuron_state_dict(neuron_state_dict, cfg):
    """
    Helper function which returns the model weights from the mixtral model in a state dictionary compatible with the stucture of the neuron MoE model.
    """
    assert cfg.glu_mlp == True, f"Only GLU MLP is supported for Mixtral Top-K model"

    for l in range(cfg.num_hidden_layers):
        # Copy router weights
        neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
            neuron_state_dict[f"layers.{l}.block_sparse_moe.gate.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.block_sparse_moe.gate.weight"]

        intermediate_size, hidden_size = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.0.w1.weight"].shape
        device = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.0.w1.weight"].device
        dtype = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.0.w1.weight"].dtype

        # copy the MLP parameters
        gate_up_proj = torch.empty(
            cfg.num_local_experts, hidden_size, 2 * intermediate_size, dtype=dtype, device=device
        )
        for e in range(cfg.num_local_experts):
            # Copy gate_proj and up_proj after concatenation
            gate_proj_weights = (
                neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w1.weight"].T.detach().clone()
            )
            up_proj_weights = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w3.weight"].T.detach().clone()

            gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
            gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
            gate_proj_slice.copy_(gate_proj_weights)
            up_proj_slice = torch.narrow(gate_up_proj_slice, 2, intermediate_size, intermediate_size)
            up_proj_slice.copy_(up_proj_weights)

            del neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w1.weight"]
            del neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w3.weight"]
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.gate_up_proj.weight"] = gate_up_proj

        down_proj = torch.empty(cfg.num_local_experts, intermediate_size, hidden_size, dtype=dtype, device=device)
        for e in range(cfg.num_local_experts):
            # Copy down_proj
            down_proj_weights = (
                neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w2.weight"].T.detach().clone()
            )
            down_proj_slice = torch.narrow(down_proj, 0, e, 1)
            down_proj_slice.copy_(down_proj_weights)
            del neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w2.weight"]
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.down_proj.weight"] = down_proj

        gc.collect()

    return neuron_state_dict


def preshard_hook_fn(module: torch.nn.Module, model_state_dict: dict, prefix: str) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)
    return False


class NeuronMixtralConfig(MixtralConfig):
    def __init__(
        self,
        batch_size: int = 1,
        tp_degree: int = 1,
        max_context_length: int = 128,
        max_new_tokens: int = 128,
        permute_strategy: str = "matmul",
        moe_sequence_parallel_mode: MoESequenceParallelMode = MoESequenceParallelMode.NO_SP,
        capacity_factor: float = 4.0,
        glu_mlp: bool = True,
        padding_side: str = "right",
        speculation_length: int = 0,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.tp_degree = tp_degree
        self.max_new_tokens = max_new_tokens
        self.max_context_length = max_context_length
        self.max_length = max_new_tokens + max_context_length

        self.permute_strategy = permute_strategy
        self.moe_sequence_parallel_mode = moe_sequence_parallel_mode
        self.capacity_factor = float(capacity_factor)
        self.glu_mlp = glu_mlp

        self.padding_side = padding_side
        self.speculation_length = speculation_length
        self.trace_tokengen_model = True
        self.n_positions = self.max_length
        self.n_active_tokens = self.max_length
        self.max_batch_size = batch_size
        self.ctx_batch_size = kwargs.pop("ctx_batch_size", self.max_batch_size)
        self.tkg_batch_size = kwargs.pop("tkg_batch_size", self.max_batch_size)

        # bucketing specific params
        self.enable_context_encoding_bucketing = False
        self.enable_token_generation_bucketing = False
        self.buckets = [self.n_positions]
        self.bucket_n_active_tokens = self.enable_context_encoding_bucketing

        self.is_continuous_batching = kwargs.pop("is_continuous_batching", False)

        super().__init__(**kwargs)


class NeuronMixtralAttention(nn.Module):
    """
    Compared with MixtralAttention, this class just
    1. replaces the linear layers in attention with NxD parallel linear layers
    2. updates attention heads and KV heads to work with given TP degree
    3. uses decomposed attention during token generation for lower latency
    """

    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logging.warning(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will lead"
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                f"NeuronMixtralAttention has to be initialized in a distributed env. Please use neuronx_distributed"
                f" module to initialize a distributed env."
            )
        self.world_size = parallel_state.get_tensor_model_parallel_size()

        self.qkv_proj = GroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.world_size,
            dtype=config.torch_dtype,
            gather_output=False,
            desired_sharding_strategy=GQA_SHARDING_STRATEGY,
        )
        self.o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.world_size,
            dtype=config.torch_dtype,
            input_is_parallel=True,
            desired_sharding_strategy=GQA_SHARDING_STRATEGY,
        )
        self.num_heads = utils.divide(self.qkv_proj.get_num_attention_heads(), self.world_size)
        self.num_key_value_heads = utils.divide(self.qkv_proj.get_num_key_value_heads(), self.world_size)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.rotary_emb = MixtralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        active_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        1. replace the q_proj, k_proj, v_proj with qkv_proj
        2. replace the `attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)` with
        `attn_output = attn_output.reshape(bsz, q_len, self.hidden_size // self.world_size)`
        """
        bsz, q_len, _ = hidden_states.size()

        Q, K, V = self.qkv_proj(hidden_states)

        Q = Q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        V = V.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = K.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(V, seq_len=kv_seq_len)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids)

        if past_key_value is None:
            # Context encoding
            K_active = repeat_kv(K, self.num_key_value_groups)
            V_active = repeat_kv(V, self.num_key_value_groups)

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
            K_prior = repeat_kv(K_prior, self.num_key_value_groups)
            V_prior = repeat_kv(V_prior, self.num_key_value_groups)
            K_active = repeat_kv(K, self.num_key_value_groups)
            V_active = repeat_kv(V, self.num_key_value_groups)

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

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        past_key_value = (K, V)

        return attn_output, past_key_value


class NeuronMixtralDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: NeuronMixtralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronMixtralAttention(config=config, layer_idx=layer_idx)

        router = RouterTopK(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            sequence_parallel_mode=config.moe_sequence_parallel_mode,
        )
        expert_mlps = ExpertMLPsCapacityFactor(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            capacity_factor=config.capacity_factor,
            glu_mlp=config.glu_mlp,
            sequence_parallel_mode=config.moe_sequence_parallel_mode,
            permute_strategy=config.permute_strategy,
            normalize_top_k_affinities=True,
        )
        self.mlp = MoE(
            router=router,
            expert_mlps=expert_mlps,
            sequence_parallel_mode=config.moe_sequence_parallel_mode,
        )
        self.mlp.eval()  # Set MoE module in eval mode

        self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.FloatTensor`, *optional*):
                position ids of size `(batch_size, sequence_length)`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

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

        # MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value)

        return outputs


class NeuronMixtralModel(MixtralPreTrainedModel):
    """
    NeuronMixtralModel extends the MixtralModel to be traceable.
    The forward function of this class is traced.
    """

    def __init__(self, config: NeuronMixtralConfig):
        # Initialization to ensure proper processing from Mixtral model
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.padding_side = config.padding_side

        self.speculation_length = config.speculation_length
        self.n_positions = config.n_positions

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                f"NeuronMixtralAttention has to be initialized in a distributed env. Please use neuronx_distributed"
                f" to initialize a distributed env."
            )

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [NeuronMixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialization to ensure proper KV cache management
        world_size = parallel_state.get_tensor_model_parallel_size()
        _, num_key_value_heads = get_shardable_head_counts(
            world_size, config.num_attention_heads, config.num_key_value_heads, GQA_SHARDING_STRATEGY
        )
        num_kv_heads_per_partition = utils.divide(num_key_value_heads, world_size)
        head_dim = config.hidden_size // config.num_attention_heads
        kv_shape = (config.max_batch_size, num_kv_heads_per_partition, config.buckets[-1], head_dim)
        self.past_key_values = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(kv_shape, dtype=config.torch_dtype), requires_grad=False)
                for _ in range(config.num_hidden_layers * 2)
            ]
        )

        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

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

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
        """
        This function is to maintain the KV cache during inference
        """
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

        hidden_states, past_key_values = self._forward(
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
                # scatter back to the desired seq_ids
                seq_id_index_shape = seq_ids.shape[:1] + k_cache.shape[1:]
                seq_id_index = seq_ids.view(-1, 1, 1, 1).expand(seq_id_index_shape)
                k_cache = torch.scatter(k_cache, 0, seq_id_index, kv_per_layer[0])
                v_cache = torch.scatter(v_cache, 0, seq_id_index, kv_per_layer[1])
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

        return [logits] + updated_kv_cache

    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        """
        This function is similar to the forward function of Huggingface MixtralModel
        """
        _, seq_length = input_ids.shape

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        inputs_embeds = self.embed_tokens(input_ids)

        # NeuronMixtralModel class manages the KV cache. So the attention_mask will be generated and passed
        # through to NeuronMixtralModel. We override the HF's code that generates attention mask because HF does
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


class NeuronMixtralForCausalLM(NeuronBaseForCausalLM, MixtralPreTrainedModel):
    """
    This class can be used as MixtralForCausalLM
    """

    def __init__(self, model_path: str, config: NeuronMixtralConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_side = config.padding_side
        self.kv_cache_populated = False

        self.models = []
        self.enable_context_encoding()
        if config.trace_tokengen_model:
            self.enable_token_generation()
        if config.speculation_length > 0:
            self.enable_speculation()
        self.model_path = model_path

    @staticmethod
    def load_hf_model(model_path):
        return MixtralForCausalLM.from_pretrained(model_path)

    @classmethod
    def get_state_dict(cls, model_path: str, config: MixtralConfig) -> dict:
        # TODO: Move to hook for state conversion
        model_sd = super().get_state_dict(model_path, config)
        model_sd = convert_mixtral_to_neuron_state_dict(model_sd, config)
        return model_sd

    def get_compiler_args(self):
        if self.config.torch_dtype == torch.bfloat16:
            return "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        else:
            return "--enable-saturate-infinity --enable-mixed-precision-accumulation --auto-cast=none --model-type transformer -O1"

    def enable_context_encoding(self):
        new_config = copy.deepcopy(self.config)
        new_config.batch_size = self.config.ctx_batch_size
        new_config.n_active_tokens = self.config.n_positions

        if not new_config.enable_context_encoding_bucketing:
            new_config.buckets = [new_config.buckets[-1]]

        self.context_encoding_model = ModelWrapper(
            config=new_config,
            model_cls=NeuronMixtralModel,
            tag=CONTEXT_ENCODING_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
        )

        self.models.append(self.context_encoding_model)

    def enable_token_generation(self):
        new_config = copy.deepcopy(self.config)
        new_config.batch_size = self.config.tkg_batch_size
        new_config.n_active_tokens = 1
        new_config.bucket_n_active_tokens = False

        if not new_config.enable_token_generation_bucketing:
            new_config.buckets = [new_config.buckets[-1]]

        self.token_generation_model = ModelWrapper(
            config=new_config,
            model_cls=NeuronMixtralModel,
            tag=TOKEN_GENERATION_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
        )

        self.models.append(self.token_generation_model)

    def enable_speculation(self):
        new_config = copy.deepcopy(self.config)
        new_config.batch_size = self.config.spec_batch_size
        new_config.n_active_tokens = self.config.speculation_length
        self.speculation_model = ModelWrapper(
            config=new_config,
            model_cls=NeuronMixtralModel,
            tag=SPECULATION_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
        )

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        logits = outputs[0]

        if self.config.trace_tokengen_model and not self.token_generation_model.is_neuron():
            # When traced the output kv tensors are aliased to the kv parameter list.
            # The code below mimicks that on CPU.
            new_past_key_values = outputs[1:]
            for i, new_past_key_value in enumerate(new_past_key_values):
                self.token_generation_model.model.past_key_values[i].data = new_past_key_value
                self.context_encoding_model.model.past_key_values[i].data = new_past_key_value

        logging.debug("---output---")
        logging.debug("logits = %s", logits)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=[],
            hidden_states=logits,
            attentions=None,
        )

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

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

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
