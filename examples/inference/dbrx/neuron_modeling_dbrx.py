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
""" PyTorch Dbrx model for NXD inference."""
import logging
import warnings
import gc
from typing import Optional, Tuple, Union

import torch
from modules.gqa import (
    GQA,
    BaseGroupQueryAttention,
)
from modules.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from torch import nn

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import DbrxForCausalLM, DbrxPreTrainedModel
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from modules.attention.attention_base import NeuronAttentionBase
from modules.attention.utils import RotaryEmbedding
from modules.config import NeuronInferenceConfig
from transformers.models.dbrx.configuration_dbrx import DbrxConfig


from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPs
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.parallel_layers import parallel_state, utils
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils.sampling import Sampler

_flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE

logger = logging.getLogger(__name__)


def convert_dbrx_to_neuron_state_dict(dbrx_state_dict, cfg):
    """
    Helper function which returns the model weights from the dbrx model in a state dictionary compatible with the stucture of the neuron MoE model.
    """

    assert cfg.glu_mlp is True, "Only GLU MLP is supported for Dbrx Top-K model"
    neuron_state_dict = {}
    neuron_state_dict["embed_tokens.weight"] = dbrx_state_dict["wte.weight"].clone().detach()
    neuron_state_dict["norm.weight"] = dbrx_state_dict["norm_f.weight"].clone().detach()
    neuron_state_dict["lm_head.weight"] = dbrx_state_dict["lm_head.weight"].clone().detach()

    for l in range(cfg.n_layers):  # noqa: E741
        # Copy router weights
        neuron_state_dict[f"layers.{l}.ffn.router.linear_router.weight"] = (
            dbrx_state_dict[f"blocks.{l}.ffn.router.layer.weight"].clone().detach()
        )

        num_experts = cfg.ffn_config.moe_num_experts
        intermediate_size, hidden_size = cfg.ffn_config.ffn_hidden_size, cfg.d_model

        # Copy gate_proj and up_proj after concatenation
        # [num_experts, hidden_size, 2 * intermediate_size]
        gate_proj_weights = dbrx_state_dict[f"blocks.{l}.ffn.experts.mlp.w1"].view(num_experts, intermediate_size, hidden_size)
        up_proj_weights = dbrx_state_dict[f"blocks.{l}.ffn.experts.mlp.v1"].view(num_experts, intermediate_size, hidden_size)
        gate_up_proj = torch.cat([gate_proj_weights, up_proj_weights], dim=1).transpose(1, 2)
        neuron_state_dict[f"layers.{l}.ffn.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        # Copy down_proj
        # [num_experts, intermediate_size, hidden_size]
        down_proj = dbrx_state_dict[f"blocks.{l}.ffn.experts.mlp.w2"].view(num_experts, intermediate_size, hidden_size)
        neuron_state_dict[f"layers.{l}.ffn.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        neuron_state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = (
            dbrx_state_dict[f"blocks.{l}.norm_attn_norm.attn.Wqkv.weight"].clone().detach()
        )
        neuron_state_dict[f"layers.{l}.self_attn.o_proj.weight"] = (
            dbrx_state_dict[f"blocks.{l}.norm_attn_norm.attn.out_proj.weight"].clone().detach()
        )
        neuron_state_dict[f"layers.{l}.input_layernorm.weight"] = (
            dbrx_state_dict[f"blocks.{l}.norm_attn_norm.norm_1.weight"].clone().detach()
        )
        neuron_state_dict[f"layers.{l}.post_attention_layernorm.weight"] = (
            dbrx_state_dict[f"blocks.{l}.norm_attn_norm.norm_2.weight"].clone().detach()
        )

    dbrx_state_dict.clear()
    gc.collect()

    return neuron_state_dict


def preshard_hook_fn(module: torch.nn.Module, model_state_dict: dict, prefix: str) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)
    return False


class NeuronDbrxConfig(NeuronInferenceConfig, DbrxConfig):
    def __init__(
            self,
            batch_size: int = 1,
            tp_degree: int = 1,
            max_context_length: int = 128,
            max_new_tokens: int = 128,
            capacity_factor: float = None,
            glu_mlp: bool = True,
            padding_side: str = "right",
            speculation_length: int = 0,
            **kwargs,
    ):
        self.max_new_tokens = max_new_tokens
        self.max_context_length = max_context_length
        self.max_length = max_new_tokens + max_context_length
        self.fused_qkv = True

        # capacity_factor = None corresponds to full capacity (no token dropping)
        self.capacity_factor = float(capacity_factor) if capacity_factor is not None else None
        self.glu_mlp = glu_mlp

        super().__init__(
            tp_degree=tp_degree,
            batch_size=batch_size,
            seq_len=max_context_length+max_new_tokens,
            padding_side=padding_side,
            max_context_length=max_context_length,
            speculation_length=speculation_length,
            **kwargs,
        )


class NeuronDbrxAttention(NeuronAttentionBase):

    def __init__(self, config: DbrxConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.num_attention_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_seq_len
        self.torch_dtype = config.torch_dtype
        self.padding_side = config.padding_side
        self.num_key_value_heads = config.attn_config.kv_n_heads
        self.rope_theta = config.attn_config.rope_theta

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronDbrxAttention has to be initialized in a distributed env. Please use neuronx_distributed"
                " module to initialize a distributed env."
            )
        self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        self.fused_qkv = config.fused_qkv
        self.clip_qkv = config.attn_config.clip_qkv

        self.init_gqa_properties()

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


class NeuronDbrxBlock(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: NeuronDbrxConfig, block_idx: int):
        super().__init__()
        self.hidden_size = config.d_model
        self.resid_pdrop = config.resid_pdrop
        self.block_idx = block_idx
        self.self_attn = NeuronDbrxAttention(config=config)

        ffn_config = config.ffn_config
        router = RouterTopK(
            num_experts=ffn_config.moe_num_experts,
            top_k=ffn_config.moe_top_k,
            hidden_size=config.d_model,
        )
        expert_mlps = ExpertMLPs(
            num_experts=ffn_config.moe_num_experts,
            top_k=ffn_config.moe_top_k,
            hidden_size=config.d_model,
            intermediate_size=ffn_config.ffn_hidden_size,
            hidden_act=ffn_config.ffn_act_fn['name'],
            capacity_factor=config.capacity_factor,
            glu_mlp=config.glu_mlp,
            normalize_top_k_affinities=True,
        )
        self.ffn = MoE(
            router=router,
            expert_mlps=expert_mlps,
        )
        self.ffn.eval()  # Set MoE module in eval mode

        self.input_layernorm = nn.LayerNorm(config.d_model, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.d_model, bias=False)

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
        hidden_states = self.input_layernorm(hidden_states).to(dtype=hidden_states.dtype)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states).to(dtype=hidden_states.dtype)

        # FFN
        hidden_states = self.ffn(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value)

        return outputs


class NeuronDbrxModel(NeuronBaseModel, DbrxPreTrainedModel):
    """Transformer decoder consisting of *config.num_hidden_layers*. Each layer is a [`DbrxBlock`] layer.

    Args:
        config ([`DbrxConfig`]): Model configuration class with all parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

    _model_cls = DbrxPreTrainedModel

    def setup_attr_for_model(self, config: NeuronDbrxConfig):
        self.emb_pdrop = config.emb_pdrop

        # Needed for init_inference_optimization()
        self.on_device_sampling = config.on_device_sampling
        self.tp_degree = config.tp_degree
        self.hidden_size = config.d_model
        self.num_attention_heads = config.n_heads
        self.num_key_value_heads = config.attn_config.kv_n_heads
        self.max_batch_size = config.max_batch_size
        self.buckets = config.buckets

    def init_model(self, config: NeuronDbrxConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.d_model,
            self.padding_idx,
            dtype=config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList([NeuronDbrxBlock(config, block_idx) for block_idx in range(config.n_layers)])
        self.norm = nn.LayerNorm(config.d_model, bias=False)
        self.lm_head = ColumnParallelLinear(config.d_model, config.vocab_size, bias=False)



class NeuronDbrxForCausalLM(NeuronBaseForCausalLM, DbrxPreTrainedModel):
    """
    This class can be used as DbrxForCausalLM
    """
    _STATE_DICT_MODEL_PREFIX = "transformer."

    _model_cls = NeuronDbrxModel

    def __init__(self, model_path: str, config: NeuronDbrxConfig):
        super().__init__(model_path, config)
        self.sampler = Sampler(self.config)

    @staticmethod
    def load_hf_model(model_path, config):
        return DbrxForCausalLM.from_pretrained(model_path, torch_dtype=config.torch_dtype)

    @classmethod
    def get_state_dict(cls, model_path: str, config: DbrxConfig) -> dict:
        model_sd = super().get_state_dict(model_path, config)
        model_sd = convert_dbrx_to_neuron_state_dict(model_sd, config)
        return model_sd

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        # Add flags for cc-overlap
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        # Prevent auto-downcasting when running with fp32
        if self.config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        # TODO: Remove this flag after compiler fix is merged (NCC-2677)
        compiler_args += " --internal-hlo2tensorizer-options=--expand-batch-norm-training"
        return compiler_args
