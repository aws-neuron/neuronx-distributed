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
import gc
import warnings
from typing import Optional, Tuple, Union

import torch
from modules.custom_calls import CustomRMSNorm
from modules.gqa import (
    GQA,
    BaseGroupQueryAttention,
)
from modules.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from neuronx_distributed.utils.sampling import Sampler

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel
from torch import nn
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import MixtralForCausalLM, MixtralPreTrainedModel
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from modules.attention.attention_base import NeuronAttentionBase
from modules.attention.utils import RotaryEmbedding
from modules.config import NeuronInferenceConfig
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralRMSNorm,
)

from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPs
from neuronx_distributed.modules.moe.model import MoE
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
    assert cfg.glu_mlp is True, "Only GLU MLP is supported for Mixtral Top-K model"

    for l in range(cfg.num_hidden_layers):  # noqa: E741
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
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        down_proj = torch.empty(cfg.num_local_experts, intermediate_size, hidden_size, dtype=dtype, device=device)
        for e in range(cfg.num_local_experts):
            # Copy down_proj
            down_proj_weights = (
                neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w2.weight"].T.detach().clone()
            )
            down_proj_slice = torch.narrow(down_proj, 0, e, 1)
            down_proj_slice.copy_(down_proj_weights)
            del neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w2.weight"]
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    return neuron_state_dict


def preshard_hook_fn(module: torch.nn.Module, model_state_dict: dict, prefix: str) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)
    return False


def get_rmsnorm_cls(config):
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return MixtralRMSNorm if config.on_cpu else CustomRMSNorm


class NeuronMixtralConfig(NeuronInferenceConfig, MixtralConfig):
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

        # capacity_factor = None corresponds to full capacity (no token dropping)
        self.capacity_factor = float(capacity_factor) if capacity_factor is not None else None
        self.glu_mlp = glu_mlp

        self.on_cpu = False

        super().__init__(
            tp_degree=tp_degree,
            batch_size=batch_size,
            padding_side=padding_side,
            seq_len=max_context_length+max_new_tokens,
            max_context_length=max_context_length,
            speculation_length=speculation_length,
            **kwargs,
        )


class NeuronMixtralAttention(NeuronAttentionBase):
    def __init__(self, config: MixtralConfig):
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

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronMixtralAttention has to be initialized in a distributed env. Please use neuronx_distributed"
                " module to initialize a distributed env."
            )
        self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        self.fused_qkv = False
        self.clip_qkv = None

        self.init_gqa_properties()

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )


class NeuronMixtralDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: NeuronMixtralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronMixtralAttention(config=config)

        router = RouterTopK(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
        )
        expert_mlps = ExpertMLPs(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            glu_mlp=config.glu_mlp,
            capacity_factor=config.capacity_factor,
            normalize_top_k_affinities=True,
        )
        self.mlp = MoE(
            router=router,
            expert_mlps=expert_mlps,
        )
        self.mlp.eval()  # Set MoE module in eval mode

        self.input_layernorm = get_rmsnorm_cls(config)(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls(config)(config.hidden_size, eps=config.rms_norm_eps)

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


class NeuronMixtralModel(NeuronBaseModel, MixtralPreTrainedModel):
    """
    NeuronMixtralModel extends the MixtralModel to be traceable.
    The forward function of this class is traced.
    """

    _model_cls = MixtralPreTrainedModel

    def setup_attr_for_model(self, config: NeuronMixtralConfig):
        self.on_device_sampling = config.on_device_sampling
        self.tp_degree = config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.max_batch_size
        self.buckets = config.buckets

    def init_model(self, config: NeuronMixtralConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

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
        self.norm = get_rmsnorm_cls(config)(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(config.hidden_size, self.vocab_size, bias=False)


class NeuronMixtralForCausalLM(NeuronBaseForCausalLM, MixtralPreTrainedModel):
    """
    This class can be used as MixtralForCausalLM
    """

    _model_cls = NeuronMixtralModel

    def __init__(self, model_path: str, config: NeuronMixtralConfig):
        super().__init__(model_path, config)
        self.sampler = Sampler(self.config)

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
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        # Add flags for cc-overlap
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        # Prevent auto-down casting when running with fp32
        if self.config.torch_dtype == torch.float32:
            compiler_args += " --auto-cast=none"
        return compiler_args
