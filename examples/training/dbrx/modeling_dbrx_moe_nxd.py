# coding=utf-8
# Copyright 2024 Databricks Mosaic Research and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch DBRX model."""

import math
import warnings
from functools import partial
from typing import Any, Optional, Tuple, Union

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
from transformers.models.dbrx.configuration_dbrx import DbrxConfig
from transformers.models.dbrx.modeling_dbrx import (
    DBRX_INPUTS_DOCSTRING,
    DBRX_START_DOCSTRING,
)
from transformers.models.dbrx.modeling_dbrx import (
    DbrxAttention as DbrxAttentionHF,
    DbrxNormAttentionNorm as DbrxNormAttentionNormHF,
    DbrxBlock as DbrxBlockHF,
    DbrxModel as DbrxModelHF,
    DbrxForCausalLM as DbrxForCausalLMHF,
)

from transformers.models.dbrx.modeling_dbrx import DbrxPreTrainedModel

from transformers.models.dbrx.modeling_dbrx import (
    DbrxRotaryEmbedding as DbrxRotaryEmbeddingHF,
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

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.cache_utils import Cache, DynamicCache, StaticCache

import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils
from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPs
from neuronx_distributed.modules.moe.loss_function import load_balancing_loss_func
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
from neuronx_distributed.parallel_layers.layer_norm import LayerNorm
from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_size,
    get_tensor_model_parallel_rank,
)
from neuronx_distributed.parallel_layers.pad import generate_padding_mask
from neuronx_distributed.utils.model_utils import move_model_to_device
from neuronx_distributed.utils.utils import hardware
from torch_neuronx.utils import get_platform_target

def _init_normal(std, w):
    return nn.init.normal_(w, mean=0.0, std=std)

if version.parse(torch.__version__) >= version.parse("2.1"):
    from torch_xla.utils.checkpoint import checkpoint

    checkpoint_method = checkpoint
else:
    checkpoint_method = torch.utils.checkpoint.checkpoint

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DbrxConfig"

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

        attn_weights = attn_weights.masked_fill_(causal_mask, -10000.0)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.double).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output
    

class DbrxRotaryEmbedding(DbrxRotaryEmbeddingHF):
    """
    Wrapper for HF Dbrx Rotary Embedding, similar to the wrapper for HF Mixtral Rotary Embedding.
    The forward function is overriden to use `double()` instead of `float()` for numerical precision,
    because NxD is using downcast. See https://github.com/huggingface/transformers/pull/29285.
    """

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :]
        freqs = (inv_freq_expanded.double() @ position_ids_expanded.double()).transpose(
            1, 2
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DbrxAttention(DbrxAttentionHF):
    """Multi-head self attention."""

    def __init__(self, config: DbrxConfig, block_idx: Optional[int] = None):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_seq_len
        self.block_idx = block_idx
        if block_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `block_idx` is not recommended and will "
                + "lead to errors during the forward call if caching is used. Please make sure to provide a `block_idx` "
                + "when creating this class."
            )

        attn_config = config.attn_config
        self.attn_pdrop = attn_config.attn_pdrop
        self.clip_qkv = attn_config.clip_qkv
        self.num_key_value_heads = attn_config.kv_n_heads
        self.rope_theta = attn_config.rope_theta
        self.is_causal = True
        self.use_flash_attention = config.attn_config.use_flash_attention

        if not hasattr(config, "kv_shared_group_size"):
            config.kv_shared_group_size = 1

        # Pad the number of attention heads to be divisble by the TP degree.
        tp_degree = get_tensor_model_parallel_size()
        num_heads_to_pad = 0 if self.num_heads % tp_degree == 0 else tp_degree - self.num_heads % tp_degree
        self.num_heads_with_pad = self.num_heads + num_heads_to_pad

        init_method = partial(_init_normal, config.initializer_range)
        self.Wqkv = GQAQKVColumnParallelLinear(
                self.hidden_size,
                [self.num_heads_with_pad * self.head_dim, self.num_key_value_heads * self.head_dim],
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=self.config.sequence_parallel_enabled,
                kv_size_multiplier=self.config.kv_shared_group_size,
                fuse_qkv=False
            )
        self.o_proj = RowParallelLinear(
            self.num_heads_with_pad * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.config.sequence_parallel_enabled
        )

        self.core_attn = CoreAttention()

        # Obtain padding mask for the current TP rank.
        self.attn_head_padding_mask = generate_padding_mask(
            num_heads=self.num_heads,
            num_heads_with_pad=self.num_heads_with_pad,
            num_kv_heads=self.num_key_value_heads,
            tp_degree=get_tensor_model_parallel_size(),
            tp_rank=get_tensor_model_parallel_rank(),
            hardware_type=hardware(get_platform_target()),
        )

        # Calculate number of heads / KV heads per head.
        self.num_heads_with_pad = neuronx_dist_utils.divide(self.num_heads_with_pad, tp_degree)
        self.num_key_value_heads = neuronx_dist_utils.divide(
            self.num_key_value_heads * self.config.kv_shared_group_size, tp_degree
        )
        self.num_key_value_groups = self.num_heads_with_pad // self.num_key_value_heads

        self.rotary_emb = DbrxRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        if config.move_model_to_device:
            move_model_to_device(self, xm.xla_device())

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert use_cache is False, "KV-Cache flow is not fully supported"
        assert past_key_value is None, "KV-Cache flow is not fully supported"
        assert attention_mask is None, "Attention mask is handled in CoreAttention"

        bsz, q_len, _ = hidden_states.size()

        if self.config.sequence_parallel_enabled:
            q_len, bsz, _ = hidden_states.size()
            q_len = q_len * get_tensor_model_parallel_size()

        min_val = -self.clip_qkv if self.clip_qkv is not None else None
        max_val = self.clip_qkv

        query_states, key_states, value_states = self.Wqkv(hidden_states)

        query_states = query_states.clamp(min=min_val, max=max_val)
        key_states = key_states.clamp(min=min_val, max=max_val)
        value_states = value_states.clamp(min=min_val, max=max_val)

        if self.config.sequence_parallel_enabled:
            # (bsz, number of heads, q_len, head_dim)
            query_states = query_states.view(q_len, bsz, self.num_heads_with_pad, self.head_dim).permute(1, 2, 0, 3)
            key_states = key_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
            value_states = value_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads_with_pad, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = (
            nki_flash_attn_func(query_states, key_states, value_states)
            if self.use_flash_attention
            else self.core_attn(query_states, key_states, value_states)
        )

        if attn_output.size() != (bsz, self.num_heads_with_pad, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads_with_pad, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # Zero the output of padded heads.
        attn_output = self.attn_head_padding_mask.view(1,-1,1,1) * attn_output

        if self.config.sequence_parallel_enabled:
            attn_output = attn_output.permute(2, 0, 1, 3)
            attn_output = attn_output.reshape(q_len, bsz, self.num_heads_with_pad * self.head_dim)
        else:
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads_with_pad * self.head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class DbrxNormAttentionNorm(DbrxNormAttentionNormHF):
    def __init__(self, config: DbrxConfig, block_idx: Optional[int] = None):
        nn.Module.__init__(self)
        self.block_idx = block_idx
        self.resid_pdrop = config.resid_pdrop
        self.norm_1 = LayerNorm(config.d_model,
                                sequence_parallel_enabled=config.sequence_parallel_enabled,
                                bias=False)
        self.attn = DbrxAttention(
            config=config,
            block_idx=block_idx,
        )
        self.norm_2 = LayerNorm(config.d_model,
                                sequence_parallel_enabled=config.sequence_parallel_enabled,
                                bias=False)
    
def initialize_dbrx_moe_layer(config: DbrxConfig):
    ffn_config = config.ffn_config

    # Default to RouterTopK (without Sinkhorn)
    router = RouterTopK(
        num_experts=ffn_config.moe_num_experts,
        top_k=ffn_config.moe_top_k,
        hidden_size=config.d_model,
        jitter_eps=ffn_config.moe_jitter_eps
    )

    init_method = partial(_init_normal, config.initializer_range)

    expert_mlps = ExpertMLPs(
        num_experts=ffn_config.moe_num_experts,
        top_k=ffn_config.moe_top_k,
        hidden_size=config.d_model,
        intermediate_size=ffn_config.ffn_hidden_size_padded,
        hidden_act=ffn_config.ffn_act_fn['name'],
        glu_mlp=True,
        capacity_factor=config.capacity_factor,
        normalize_top_k_affinities=True,
        init_method=init_method,
        output_layer_init_method=init_method
    )

    moe_layer = MoE(
        router=router,
        expert_mlps=expert_mlps,
        return_router_logits=True,
        sequence_parallel_enabled=config.sequence_parallel_enabled,
    )

    return moe_layer

class DbrxBlock(DbrxBlockHF):
    def __init__(self, config: DbrxConfig, block_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.d_model
        self.resid_pdrop = config.resid_pdrop
        self.block_idx = block_idx
        self.norm_attn_norm = DbrxNormAttentionNorm(
            config=config,
            block_idx=block_idx,
        )

        ## Compute padding needed for ffn intermediate size (I) to make NKI blockwise run efficiently
        I_size= config.ffn_config.ffn_hidden_size
        tp_degree = get_tensor_model_parallel_size()
        assert I_size% tp_degree == 0, f"FFN intermediate size({I_size}) must be divisible by the TP degree ({tp_degree})"
        # Compute padding needed per TP degree
        I_tp = I_size// tp_degree
        I_tp_round_factor = config.i_tp_round_factor if hasattr(config, "i_tp_round_factor") else 0
        padding_tp = 0 if I_tp % I_tp_round_factor == 0 else I_tp_round_factor - I_tp % I_tp_round_factor
        self.gate_up_padding_mask = torch.cat([torch.ones(I_tp), torch.zeros(padding_tp), torch.ones(I_tp), torch.zeros(padding_tp)])
        self.down_padding_mask = torch.cat([torch.ones(I_tp), torch.zeros(padding_tp)])
        # Set ffn intermediate size with padding needed
        config.ffn_config.ffn_hidden_size_padded = config.ffn_config.ffn_hidden_size + padding_tp * tp_degree

        self.ffn = initialize_dbrx_moe_layer(config=config)

    def zero_padding_weights(self):
        with torch.no_grad():
            # Only applying mask to weight b/c bias is not used.
            self.ffn.expert_mlps.mlp_op.gate_up_proj.weight.mul_(self.gate_up_padding_mask.view(1,1,-1))
            self.ffn.expert_mlps.mlp_op.down_proj.weight.mul_(self.down_padding_mask.view(1,-1,1))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_router_logits: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: torch.LongTensor = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Union[
        Tuple[torch.Tensor],
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[Cache]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[Cache], Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[torch.Tensor]],
    ]:
        """Forward function for DbrxBlock.

        Args:
            hidden_states (`torch.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            position_ids (`torch.LongTensor`): position ids of shape `(batch, seq_len)`
            attention_mask (`torch.Tensor`, *optional*): attention mask of size (batch_size, sequence_length)
                if flash attention is used or (batch_size, 1, query_sequence_length, key_sequence_length)
                if default attention is used.
            past_key_value (`Tuple(torch.Tensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*): Whether or not to return the attentions tensors of all
                attention layers. See `attentions` under returned tensors for more detail.
            output_router_logits (`bool`, *optional*): Whether or not to return the router logits.
            use_cache (`bool`, *optional*): If set to `True`, `past_key_values` key value states are
                returned and can be used to speed up decoding (see `past_key_values`).
            cache_position (`torch.LongTensor`, *optional*): position ids of the cache
        """

        # Norm + Attention + Norm
        resid_states, hidden_states, self_attn_weights, present_key_value = self.norm_attn_norm(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        # Fully Connected
        hidden_states, router_logits = self.ffn(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.resid_pdrop, training=self.training)
        hidden_states = resid_states + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            # Concatenate the router logits with previous router logits
            if past_router_logits is not None:
                router_logits = torch.cat((past_router_logits, router_logits), dim=0)
            outputs += (router_logits,)

        return outputs

class DbrxModel(DbrxModelHF):
    """Transformer decoder consisting of *config.num_hidden_layers*. Each layer is a [`DbrxBlock`] layer.

    Args:
        config ([`DbrxConfig`]): Model configuration class with all parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    def __init__(self, config: DbrxConfig):
        DbrxPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.emb_pdrop = config.emb_pdrop

        init_method = partial(_init_normal, config.initializer_range)
        self.wte = ParallelEmbedding(
            config.vocab_size, config.d_model, self.padding_idx, init_method=init_method,
            sequence_parallel_enabled=config.sequence_parallel_enabled
        )
        self.blocks = nn.ModuleList([DbrxBlock(config, block_idx) for block_idx in range(config.n_layers)])
        self.norm_f = LayerNorm(config.d_model,
                                sequence_parallel_enabled=config.sequence_parallel_enabled, bias=False)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def zero_padding_weights(self):
        for block in self.blocks:
            block.zero_padding_weights()

    @add_start_docstrings_to_model_forward(DBRX_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
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
            inputs_embeds = self.wte(input_ids)

        inputs_embeds = nn.functional.dropout(inputs_embeds, p=self.emb_pdrop, training=self.training)

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_router_logits = None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                block_outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    all_router_logits,
                    attention_mask,
                    position_ids,
                    None,
                    output_attentions,
                    output_router_logits,
                )
            else:
                block_outputs = block(
                    hidden_states,
                    all_router_logits,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = block_outputs[0]

            if use_cache:
                next_decoder_cache += (block_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (block_outputs[1],)

            if output_router_logits:
                all_router_logits = block_outputs[-1]

        hidden_states = self.norm_f(hidden_states)

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


@add_start_docstrings("The DBRX Model transformer for causal language modeling.", DBRX_START_DOCSTRING)
class DbrxForCausalLM(DbrxForCausalLMHF):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DbrxConfig):
        DbrxPreTrainedModel.__init__(self, config)
        self.config = config
        self.transformer = DbrxModel(config)
        self.vocab_size = config.vocab_size

        init_method = partial(_init_normal, config.initializer_range)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=False,
            sequence_parallel_enabled=config.sequence_parallel_enabled,
            init_method=init_method
        )
        self.moe_loss_weight = config.ffn_config.moe_loss_weight
        self.num_experts = config.ffn_config.moe_num_experts
        self.num_experts_per_tok = config.ffn_config.moe_top_k

        self.post_init()

    def zero_padding_weights(self):
        self.transformer.zero_padding_weights()

    @add_start_docstrings_to_model_forward(DBRX_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""Forward function for causal language modeling.

        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >> from transformers import AutoTokenizer, DbrxForCausalLM

        >> model = DbrxForCausalLM.from_pretrained("databricks/dbrx-instruct")
        >> tokenizer = AutoTokenizer.from_pretrained("databricks/dbrx-instruct")

        >> prompt = "Hey, are you conscious? Can you talk to me?"
        >> inputs = tokenizer(prompt, return_tensors="pt")

        >> # Generate
        >> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert past_key_values is None, "KV-Cache flow is not fully supported"
        assert not use_cache, "KV-Cache flow is not fully supported"

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
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
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        if self.config.sequence_parallel_enabled:
            # Swap transpose to einsum to work with FSDP
            logits = torch.einsum('ijk->jik', logits).contiguous()

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
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok
            )
            if labels is not None and loss is not None:
                loss += self.moe_loss_weight * aux_loss.to(loss.device)  # make sure to reside in the same device

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

def init_weights(module, device):
    """
    Re-init weights after partition
    """
    if isinstance(module, (ParallelEmbedding, RowParallelLinear, ColumnParallelLinear)):
        module.init_weight_cpu()
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, (LayerNorm, torch.nn.Linear)):
        module.reset_parameters()
    elif isinstance(module, GQAQKVColumnParallelLinear):
        module.initialize_weight_biases()
