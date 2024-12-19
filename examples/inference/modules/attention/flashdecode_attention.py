import logging
import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from modules.attention.utils import apply_rotary_pos_emb, repeat_kv, distributed_softmax, move_heads_front, RotaryEmbedding
from modules.config import NeuronConfig
from modules.gqa import (  # noqa: E402
    GroupQueryAttention_O,  # noqa: E402
    GroupQueryAttention_QKV,  # noqa: E402
)  # noqa: E402
from neuronx_distributed.parallel_layers import utils  # noqa: E402
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.parallel_state import get_kv_shared_group, get_world_group

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel  # noqa: E402
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel  # noqa: E402
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402

import torch_xla.core.xla_model as xm

from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
)

_flash_fwd_call = nki_jit()(attention_isa_kernel)


class NeuronFDAttentionBase(nn.Module):
    """
    This base attention class implements the core Neuron related adaptation including
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self):
        super().__init__()
        self.is_causal = True
        self.num_key_value_groups = None
        self.num_key_value_heads = None
        self.num_heads = None
        self.rotary_emb = None
        self.o_proj = None
        self.qkv_proj = None
        self.bias = False
        self.num_cores_per_group = None

        self.o_proj_layer_name = "o_proj"

        self.rank_util = SPMDRank(world_size=get_world_group().size())

    def init_gqa_properties(self):
        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
            )
        self.qkv_proj = GroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.bias,
            gather_output=False,
            fused_qkv=self.fused_qkv,
            clip_qkv=self.clip_qkv
        )
        self.o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.bias,
            input_is_parallel=True,
            layer_name=self.o_proj_layer_name
        )
        self.num_heads = utils.divide(self.qkv_proj.get_num_attention_heads(), self.tp_degree)
        self.num_key_value_heads = utils.divide(self.qkv_proj.get_num_key_value_heads(), self.tp_degree)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    def scaled_qk(self, Q, K, attention_mask):
        QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            QK = torch.where(attention_mask, QK, torch.finfo(QK.dtype).min)
        return QK

    def prep_qkv_tensors(self, position_ids, hidden_states):
        """ take care of the shape, layout, group query, custom position encoding, etc. """
        Q, K, V = self.qkv_proj(hidden_states=hidden_states)

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()
        Q = move_heads_front(Q, bsz, q_len, self.num_heads, self.head_dim)
        K = move_heads_front(K, bsz, q_len, self.num_key_value_heads, self.head_dim)
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Rotate Q and K
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(V, position_ids)
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        return Q, K, V

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """  attention computation at prefilling (context encoding) phase """
        n_repeat = self.num_key_value_groups
        assert n_repeat == Q.shape[1], (f"Repeat number is {n_repeat}, but it should be the same as number of Q heads "
                                        f"(H dim): {Q.shape[1]}")
        K_active = repeat_kv(K, n_repeat)
        V_active = repeat_kv(V, n_repeat)

        # use flash attention if (i) sequence length is large enough to get the best performance,
        # (ii) Q, K, and V have the same shape. Conditions can be changed in the future.
        flash_attention_eligible = q_len >= 4096 and Q.shape == K_active.shape == V_active.shape

        if flash_attention_eligible:
            # if we are using left padding, then the bzs needs be 1 (otherwise we get wrong result
            # because flash attention does not use attention_mask). In practice, we use right
            # padding so this is unlikely to cause issues
            assert self.padding_side == "right" or bsz == 1

            # original shape of q, k, v is BHSD, and expected output is also BHSD.
            logging.debug(f"Using flash_fwd for Q.shape={Q.shape}")
            # make sure to cast inputs to torch_dtype (this is needed because the downcast to bf16
            # might happen after the kernel hlo creation step). Also convert shapes as expected by the kernel.
            Q = (
                Q.permute(0, 1, 3, 2)
                .reshape((bsz * self.num_heads, self.head_dim, q_len))
                .to(self.neuron_config.hf_config.torch_dtype)
            )
            Q = Q / math.sqrt(self.head_dim)
            K_active = (
                K_active.permute(0, 1, 3, 2)
                .reshape((bsz * self.num_heads, self.head_dim, q_len))
                .to(self.neuron_config.hf_config.torch_dtype)
            )
            V_active = V_active.reshape((bsz * self.num_heads, q_len, self.head_dim)).to(
                self.neuron_config.hf_config.torch_dtype)
            attn_output = torch.zeros(bsz * self.num_heads, q_len, self.head_dim, dtype=Q.dtype, device=Q.device)
            _flash_fwd_call(
                Q, K_active, V_active, 1.0, attn_output, kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap"
            )
            attn_output = attn_output.reshape((bsz, self.num_heads, q_len, self.head_dim))
        else:
            logging.debug(f"Not using flash_fwd for Q.shape={Q.shape}")
            active_scores = self.scaled_qk(Q, K_active, attention_mask)
            active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(Q.dtype)
            attn_output = torch.matmul(active_scores, V_active)
        return attn_output

    def compute_for_token_gen(self, Q, K, V, past_key_value, attention_mask, active_mask) -> Tensor:

        # active attention
        n_repeat = Q.shape[1]
        K_active = repeat_kv(K, n_repeat)  # 1,1,52,4
        V_active = repeat_kv(V, n_repeat)
        active_scores = (torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)).to(torch.float32)
        active_scores = torch.where(active_mask, active_scores, torch.finfo(active_scores.dtype).min)

        # prior attention
        K_prior = repeat_kv(past_key_value[0], n_repeat)
        V_prior = repeat_kv(past_key_value[1], n_repeat)
        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) / math.sqrt(self.head_dim)
        prior_scores = torch.where(attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min)
        prior_scores = prior_scores.to(torch.float32)

        # attention scores
        softmax_prior, softmax_active = distributed_softmax(prior_scores, active_scores)
        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active

        return attn_output

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            active_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:

        """ Implements each layer's forward pass for the attention block. """
        bsz, q_len, _ = hidden_states.size()
        Q, K, V = self.prep_qkv_tensors(position_ids, hidden_states)
        rank_id = self.rank_util.get_rank()
        rank_id_in_kv_group = torch.remainder(rank_id, self.num_cores_per_group).to(torch.int32)

        if past_key_value is None:
            attn_output = self.perform_prefill(Q, K, V, q_len, bsz, attention_mask)
            # shard KV by seq len and pick the values based on rank
            assert q_len == Q.shape[2], f"Q shape is {Q.shape}"
            # selecting positions (on S dim) that belongs to the current rank
            selected_seq_pos = torch.arange(rank_id_in_kv_group.item(), q_len, self.num_cores_per_group,
                                           dtype=torch.int64, device=Q.device)
            K = torch.index_select(input=K, dim=2, index=selected_seq_pos)
            V = torch.index_select(input=V, dim=2, index=selected_seq_pos)
        else:
            # assert active_mask is not None, "Flash decoding requires active mask is not None!"
            # gather Q from all cores in its KV group
            groups = get_kv_shared_group(as_list=True)
            Q = xm.all_gather(Q, dim=1, groups=groups, pin_layout=False)

            attn_output = self.compute_for_token_gen(Q, K, V, past_key_value, attention_mask, active_mask)
            attn_output = xm.reduce_scatter(xm.REDUCE_SUM, attn_output, scale=1, scatter_dim=1,
                                            shard_count=len(groups[0]), groups=groups, pin_layout=False)

        # transpose BHSD -> BSHD
        attn_output = attn_output.transpose(1, 2).contiguous()

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.o_proj(attn_output)

        past_key_value: Tuple[Tensor, Tensor] = (K, V)
        return attn_output, past_key_value


class FlashDecodingAttention(NeuronFDAttentionBase):
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
        if not hasattr(self.neuron_config.hf_config,
                       "rope_scaling") or self.neuron_config.hf_config.rope_scaling is None:
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
