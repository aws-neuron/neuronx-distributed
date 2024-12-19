# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" PyTorch ViT model for NXD inference. """

import logging
import copy
import collections.abc
import random
import math
import os
from typing import Dict, List, Optional, Set, Tuple, Type, Union

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    OutputChannelParallelConv2d,
    ColumnParallelLinear,
    RowParallelLinear,
)
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from transformers import PretrainedConfig, ViTModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
from transformers.modeling_utils import PreTrainedModel

from modules.attention.attention_base import NeuronAttentionBase
from modules.config import NeuronConfig
from modules.checkpoint import load_state_dict
from modules.model_base import NeuronBaseModel, NeuronBaseForCausalLM
from modules.model_wrapper import IMAGE_ENCODING_MODEL_TAG, ModelWrapper


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(0)


class NeuronViTConfig(NeuronConfig):
    def __init__(self, hf_config: PretrainedConfig = None, **kwargs) -> None:
        super().__init__(hf_config, **kwargs)

        if hf_config:
            self.hf_config.attn_cls = "NeuronViTAttention"
        self.trace_tokengen_model = False


class NeuronViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: NeuronViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = NeuronViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class NeuronViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: NeuronViTConfig):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class NeuronViTAttention(NeuronAttentionBase):
    def __init__(self, config: NeuronViTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.tp_degree = config.tp_degree
        self.torch_dtype = config.torch_dtype
        self.fused_qkv = False
        self.clip_qkv = None
        self.bias = True

        self.o_proj_layer_name = "out_proj"

        self.init_gqa_properties()


class NeuronViTIntermediate(nn.Module):
    def __init__(self, config: NeuronViTConfig) -> None:
        super().__init__()
        if parallel_state.model_parallel_is_initialized():
            self.dense = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=config.intermediate_size,
                bias=True,
                gather_output=False,
                dtype=config.torch_dtype
            )
        else:
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class NeuronViTOutput(nn.Module):
    def __init__(self, config: NeuronViTConfig) -> None:
        super().__init__()
        if parallel_state.model_parallel_is_initialized():
            self.dense = RowParallelLinear(
                input_size=config.intermediate_size,
                output_size=config.hidden_size,
                bias=True,
                input_is_parallel=True,
                dtype=config.torch_dtype
            )
        else:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class NeuronViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: NeuronViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = NeuronViTAttention(config)
        self.intermediate = NeuronViTIntermediate(config)
        self.output = NeuronViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs, _ = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
        )
        self_attention_outputs = (self_attention_outputs,)  # change it to a tuple

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class NeuronViTEncoder(nn.Module):
    def __init__(self, config: NeuronViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([NeuronViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class NeuronViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = NeuronViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    _no_split_modules = ["NeuronViTEmbeddings", "NeuronViTLayer"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, NeuronViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)


class NeuronViTModel(NeuronBaseModel, NeuronViTPreTrainedModel):
    def __init__(self, neuron_config: NeuronViTConfig, add_pooling_layer: bool = False, use_mask_token: bool = False):
        super().__init__(neuron_config, optimize_inference=False)
        self.torch_dtype = neuron_config.hf_config.torch_dtype
        self.hf_config = neuron_config.hf_config

        self.embeddings = NeuronViTEmbeddings(self.hf_config, use_mask_token=use_mask_token)
        self.encoder = NeuronViTEncoder(self.hf_config)

        self.layernorm = nn.LayerNorm(self.hf_config.hidden_size, eps=self.hf_config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def setup_attr_for_model(self, neuron_config: NeuronViTConfig):
        self.tp_degree = neuron_config.tp_degree
        self.neuron_config = neuron_config

    def init_model(self, neuron_config: NeuronViTConfig):
        neuron_config.hf_config.tp_degree = neuron_config.tp_degree
        neuron_config.hf_config.torch_dtype = neuron_config.hf_config.torch_dtype
        self.past_key_values = []

    def get_input_embeddings(self) -> NeuronViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.hf_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.hf_config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.hf_config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.hf_config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        outputs = BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        sequence_output = outputs[0]
        return sequence_output[:, 0, :]


class ModelWrapperViT(ModelWrapper):
    """
    A class that wraps the ViT model for vision encoding tasks.
    This class overrides input_generator() to provide additional pixel_values in the sample inputs for tracing
    """
    def __init__(self,
        neuron_config: NeuronViTConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
    ) -> None:
        super().__init__(neuron_config, model_cls, tag, compiler_args, priority_model_idx)
        self.neuron_config = neuron_config

        if not self.neuron_config.hf_config.torch_dtype:
            self.neuron_config.hf_config.torch_dtype = torch.float32

        self.model_cls = model_cls
        self.model = None
        self.is_compiled = False
        self.tag = tag
        if compiler_args is None:
            self.compiler_args = (
                "--enable-saturate-infinity "
                "--auto-cast=none "
                "--model-type=transformer "
                "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
                "-O1"
            )
        else:
            self.compiler_args = compiler_args

    def input_generator(self):
        inputs = []

        # pixel values for images
        pixel_values = torch.zeros(
            (
                self.neuron_config.batch_size,
                3,    # 3 channls
                self.neuron_config.hf_config.image_size,
                self.neuron_config.hf_config.image_size
            ),
            dtype=self.neuron_config.hf_config.torch_dtype
        )

        inputs.append((pixel_values,))

        return inputs

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")

        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() or load_state_dict() making calling forward")

        logging.debug("Processed inputs to the model", self.tag, args)
        outputs = self.model(*args)

        return outputs


class NeuronViTForImageEncoding(NeuronBaseForCausalLM, NeuronViTPreTrainedModel):

    _model_cls = NeuronViTModel

    def __init__(self, model_path: str, neuron_config: NeuronViTConfig):
        super().__init__(model_path, neuron_config)
        self.neuron_config = neuron_config
        self.torch_dtype = neuron_config.hf_config.torch_dtype
        self.model_path = model_path

        self.set_prefixes(model_path)
        self.enable_image_encoding()

    @classmethod
    def set_prefixes(cls, model_path: str):
        """
        Determine prefixes based on the model name extracted from the model path.
        """
        model_name = os.path.basename(os.path.normpath(model_path))
        if model_name in ["vit-base-patch16-224", "vit-large-patch16-224"]:
            cls._STATE_DICT_MODEL_PREFIX = "vit."
            cls._NEW_STATE_DICT_MODEL_PREFIX = ""
        elif model_name in ["vit-huge-patch14-224"]:
            cls._STATE_DICT_MODEL_PREFIX = ""
            cls._NEW_STATE_DICT_MODEL_PREFIX = ""
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
            
    @staticmethod
    def load_hf_model(model_path):
        return ViTModel.from_pretrained(model_path)

    def get_model_wrapper_cls(self):
        return ModelWrapperViT

    def enable_context_encoding(self):
        pass

    def enable_image_encoding(self):
        new_neuron_config = copy.deepcopy(self.neuron_config)
        new_neuron_config.batch_size = self.neuron_config.ctx_batch_size

        self.image_encoding_model = self.model_wrapper(
            neuron_config=new_neuron_config,
            model_cls=self._model_cls,
            tag=IMAGE_ENCODING_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.image_encoding_model)

    @classmethod
    def get_state_dict(cls, model_path: str, neuron_config: NeuronConfig) -> dict:
        # Set the prefixes based on the model path
        cls.set_prefixes(model_path)
        
        model_sd = load_state_dict(model_path)

        # mapping from HF ViT keys to neuron keys
        mapping = {
            "attention.query": "qkv_proj.q_proj",
            "attention.key": "qkv_proj.k_proj",
            "attention.value": "qkv_proj.v_proj",
            "attention.output.dense": "attention.o_proj.o_proj",
        }

        # remove the HF ViT attention layer's dropout
        pass_kes = ["attention.attention.dropout", "attention.output.dropout", "classifier"]

        param_name_list = list(model_sd.keys())
        for param_name in param_name_list:
            if param_name in pass_kes:
                del model_sd[param_name]

            if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                updated_param_name = param_name.replace(cls._STATE_DICT_MODEL_PREFIX, cls._NEW_STATE_DICT_MODEL_PREFIX, 1)

                for key_old, key_new in mapping.items():
                    if key_old in updated_param_name:
                        updated_param_name = updated_param_name.replace(key_old, key_new, 1)
                model_sd[updated_param_name] = model_sd[param_name]
                if param_name != updated_param_name:
                    del model_sd[param_name]

        return model_sd

    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
        ):
        return self.image_encoding_model(pixel_values)
