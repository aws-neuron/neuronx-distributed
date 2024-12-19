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
import copy
import gc
from typing import Optional, Tuple, Union

import torch
from modules.autobucketing import generate_buckets
from modules.gqa import (
    GQA,
)
from modules.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from modules.model_wrapper import TOKEN_GENERATION_MODEL_TAG

from torch import nn

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import DbrxForCausalLM, DbrxPreTrainedModel, PretrainedConfig
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from modules.attention.attention_base import NeuronAttentionBase
from modules.attention.utils import RotaryEmbedding
from modules.config import MoENeuronConfig


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


def convert_dbrx_to_neuron_state_dict(dbrx_state_dict, neuron_config):
    """
    Helper function which returns the model weights from the dbrx model in a state dictionary compatible with the stucture of the neuron MoE model.
    """

    assert neuron_config.glu_mlp is True, "Only GLU MLP is supported for Dbrx Top-K model"
    neuron_state_dict = {}
    neuron_state_dict["embed_tokens.weight"] = dbrx_state_dict["wte.weight"].clone().detach()
    neuron_state_dict["norm.weight"] = dbrx_state_dict["norm_f.weight"].clone().detach()
    neuron_state_dict["lm_head.weight"] = dbrx_state_dict["lm_head.weight"].clone().detach()

    for l in range(neuron_config.hf_config.n_layers):  # noqa: E741
        # Copy router weights
        neuron_state_dict[f"layers.{l}.ffn.router.linear_router.weight"] = (
            dbrx_state_dict[f"blocks.{l}.ffn.router.layer.weight"].clone().detach()
        )

        num_experts = neuron_config.hf_config.ffn_config.moe_num_experts
        intermediate_size, hidden_size = neuron_config.hf_config.ffn_config.ffn_hidden_size, neuron_config.hf_config.d_model

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


class NeuronDbrxConfig(MoENeuronConfig):
    def __init__(self, hf_config: PretrainedConfig = None, **kwargs):
        super().__init__(hf_config, **kwargs)
        self.fused_qkv = True


class NeuronDbrxAttention(NeuronAttentionBase):

    def __init__(self, neuron_config: NeuronDbrxConfig):
        super().__init__()
        self.neuron_config = neuron_config
        self.hidden_size = neuron_config.hf_config.d_model
        self.num_attention_heads = neuron_config.hf_config.n_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = neuron_config.hf_config.max_seq_len
        self.torch_dtype = neuron_config.hf_config.torch_dtype
        self.padding_side = neuron_config.padding_side
        self.num_key_value_heads = neuron_config.hf_config.attn_config.kv_n_heads
        self.rope_theta = neuron_config.hf_config.attn_config.rope_theta

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronDbrxAttention has to be initialized in a distributed env. Please use neuronx_distributed"
                " module to initialize a distributed env."
            )
        self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        self.fused_qkv = neuron_config.fused_qkv
        self.clip_qkv = neuron_config.hf_config.attn_config.clip_qkv

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

    def __init__(self, neuron_config: NeuronDbrxConfig, block_idx: int):
        super().__init__()
        self.hidden_size = neuron_config.hf_config.d_model
        self.resid_pdrop = neuron_config.hf_config.resid_pdrop
        self.block_idx = block_idx
        self.self_attn = NeuronDbrxAttention(neuron_config=neuron_config)

        ffn_config = neuron_config.hf_config.ffn_config
        router = RouterTopK(
            num_experts=ffn_config.moe_num_experts,
            top_k=ffn_config.moe_top_k,
            hidden_size=neuron_config.hf_config.d_model,
            sequence_parallel_enabled=False,
            sequence_dimension=1,
        )
        expert_mlps = ExpertMLPs(
            num_experts=ffn_config.moe_num_experts,
            top_k=ffn_config.moe_top_k,
            hidden_size=neuron_config.hf_config.d_model,
            intermediate_size=ffn_config.ffn_hidden_size,
            hidden_act=ffn_config.ffn_act_fn['name'],
            capacity_factor=neuron_config.capacity_factor,
            glu_mlp=neuron_config.glu_mlp,
            normalize_top_k_affinities=True,
        )
        self.ffn = MoE(
            router=router,
            expert_mlps=expert_mlps,
            sequence_parallel_enabled=False,
            sequence_dimension=1,
        )
        self.ffn.eval()  # Set MoE module in eval mode

        self.input_layernorm = nn.LayerNorm(neuron_config.hf_config.d_model, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(neuron_config.hf_config.d_model, bias=False)

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

    def setup_attr_for_model(self, neuron_config: NeuronDbrxConfig):
        self.emb_pdrop = neuron_config.hf_config.emb_pdrop

        # Needed for init_inference_optimization()
        self.on_device_sampling = neuron_config.on_device_sampling
        self.tp_degree = neuron_config.tp_degree
        self.hidden_size = neuron_config.hf_config.d_model
        self.num_attention_heads = neuron_config.hf_config.n_heads
        self.num_key_value_heads = neuron_config.hf_config.attn_config.kv_n_heads
        self.max_batch_size = neuron_config.max_batch_size
        self.buckets = neuron_config.buckets

    def init_model(self, neuron_config: NeuronDbrxConfig):
        self.padding_idx = neuron_config.hf_config.pad_token_id
        self.vocab_size = neuron_config.hf_config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            neuron_config.hf_config.vocab_size,
            neuron_config.hf_config.d_model,
            self.padding_idx,
            dtype=neuron_config.hf_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [NeuronDbrxBlock(neuron_config, block_idx) for block_idx in range(neuron_config.hf_config.n_layers)]
        )
        self.norm = nn.LayerNorm(neuron_config.hf_config.d_model, bias=False)
        self.lm_head = ColumnParallelLinear(neuron_config.hf_config.d_model, neuron_config.hf_config.vocab_size, bias=False)



class NeuronDbrxForCausalLM(NeuronBaseForCausalLM, DbrxPreTrainedModel):
    """
    This class can be used as DbrxForCausalLM
    """
    _STATE_DICT_MODEL_PREFIX = "transformer."

    _model_cls = NeuronDbrxModel

    def __init__(self, model_path: str, neuron_config: NeuronDbrxConfig):
        super().__init__(model_path, neuron_config)
        self.sampler = Sampler(neuron_config)

    @staticmethod
    def load_hf_model(model_path, hf_config):
        return DbrxForCausalLM.from_pretrained(model_path, torch_dtype=hf_config.torch_dtype)

    def enable_token_generation(self):
        # Override to enable weight layout optimization
        new_neuron_config = copy.deepcopy(self.neuron_config)
        new_neuron_config.batch_size = self.neuron_config.tkg_batch_size
        new_neuron_config.n_active_tokens = 1
        new_neuron_config.bucket_n_active_tokens = False

        if not new_neuron_config.enable_bucketing:
            new_neuron_config.buckets = generate_buckets(self.neuron_config.max_length, self.neuron_config.max_length)
        else:
            new_neuron_config.buckets = generate_buckets(128, self.neuron_config.max_length)

        self.token_generation_model = self.model_wrapper(
            neuron_config=new_neuron_config,
            model_cls=self._model_cls,
            tag=TOKEN_GENERATION_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
            # Enable weight layout optimization
            priority_model_idx=0,
        )
        self.models.append(self.token_generation_model)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, neuron_config: NeuronDbrxConfig) -> dict:
        return convert_dbrx_to_neuron_state_dict(state_dict, neuron_config)

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        # Add flags for cc-overlap
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        # Prevent auto-downcasting when running with fp32
        if self.neuron_config.hf_config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        # Enable vector-offset DGE
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        return compiler_args
