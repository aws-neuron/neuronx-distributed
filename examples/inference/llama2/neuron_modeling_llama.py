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
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
import math
import logging

from functools import partial

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaMLP,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM
)
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from typing import List, Optional, Tuple, Union, Dict, Any

from neuronx_distributed.parallel_layers.layers import ParallelEmbedding, ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers import parallel_state, utils


class NeuronLlamaMLP(LlamaMLP):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
        )


class NeuronLlamaAttention(LlamaAttention):
    """
    Compared with LlamaAttention, this class just 
    1. replaces the q_proj, k_proj, v_proj with column parallel layer 
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.world_size = parallel_state.get_tensor_model_parallel_size()

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
        )
        self.num_heads = utils.divide(
            config.num_attention_heads, self.world_size)
        self.num_key_value_heads = utils.divide(
            config.num_key_value_heads, self.world_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Just replace the `attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)` with
        `attn_output = attn_output.reshape(bsz, q_len, self.hidden_size // self.world_size)`
        """
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads *
                                 self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i])
                            for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i])
                          for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i])
                            for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(
            bsz, q_len, self.hidden_size // self.world_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i])
                              for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class NeuronLlamaDecoderLayer(LlamaDecoderLayer):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.self_attn = NeuronLlamaAttention(config=config)
        self.mlp = NeuronLlamaMLP(config)


class NeuronLlamaConfig(LlamaConfig):

    def __init__(self,
                 batch_size=1,
                 tp_degree=1,
                 max_context_length=128,
                 max_new_tokens=128,
                 **kwargs):

        self.batch_size = batch_size
        self.tp_degree = tp_degree
        self.max_new_tokens = max_new_tokens
        self.max_context_length = max_context_length
        self.max_length = max_new_tokens + max_context_length
        super().__init__(**kwargs)


class NeuronLlamaModel(LlamaModel):
    """
    NeuronLlamaModel extends the LlamaModel to be traceable.
    The forward function of this class is traced. 
    """

    def __init__(self,
                 config: NeuronLlamaConfig):

        super().__init__(config)

        tp_degree = config.tp_degree
        batch_size = config.batch_size
        max_length = config.max_length

        hidden_dim_per_head = config.hidden_size // config.num_attention_heads
        num_kv_heads_per_partition = config.num_key_value_heads

        if tp_degree > 1:
            # plug in the parallel layers
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size, config.hidden_size, self.padding_idx)
            self.layers = nn.ModuleList([NeuronLlamaDecoderLayer(
                config) for _ in range(config.num_hidden_layers)])

            world_size = parallel_state.get_tensor_model_parallel_size()  # Same as tp_degree
            num_kv_heads_per_partition = utils.divide(
                config.num_key_value_heads, world_size)

        kv_shape = (
            batch_size,
            num_kv_heads_per_partition,
            max_length,
            hidden_dim_per_head
        )
        self.past_key_values = nn.ParameterList(
            [nn.Parameter(torch.zeros(kv_shape, dtype=torch.float32), requires_grad=False)
             for _ in range(config.num_hidden_layers * 2)]
        )

    def forward(self,
                input_ids,
                attention_mask,
                position_ids):

        is_for_context_encoding = input_ids.shape[-1] > 1
        # It is either for context encoding or for token generation
        if is_for_context_encoding:
            past_key_values = None
        else:
            past_key_values = []
            for key_layer_idx in range(0, len(self.past_key_values), 2):
                key_state = self.past_key_values[key_layer_idx]
                value_state = self.past_key_values[key_layer_idx+1]
                past_key_values.append([key_state, value_state])

        outputs = super().forward(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  position_ids=position_ids,
                                  past_key_values=past_key_values,
                                  use_cache=True,
                                  output_attentions=False,
                                  output_hidden_states=False,
                                  return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        past_key_values = outputs.past_key_values

        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(past_key_values):
            if is_for_context_encoding:
                k_cache = kv_per_layer[0] + (self.past_key_values[idx*2] * 0)
                v_cache = kv_per_layer[1] + (self.past_key_values[idx*2+1] * 0)
            else:
                k_cache = kv_per_layer[0][:, :, 1:, :]
                v_cache = kv_per_layer[1][:, :, 1:, :]
            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        return [last_hidden_state] + updated_kv_cache


class NeuronLlamaForCausalLM(LlamaForCausalLM):
    """ 
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron. 

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    def __init__(self,
                 config: NeuronLlamaConfig,
                 context_encoder_model=None,
                 token_generator_model=None):

        super().__init__(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        if context_encoder_model == None or token_generator_model == None:
            assert isinstance(
                config, NeuronLlamaConfig), 'Pass NeuronLlamaConfig as config'
            self.model = NeuronLlamaModel(config)
            self.context_encoding_model = None
            self.token_generation_model = None
            # Initialize weights and apply final processing
            self.post_init()
        else:
            self.context_encoding_model = context_encoder_model
            self.token_generation_model = token_generator_model
            self.post_init()

        self.kv_cache_populated = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
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

        logging.debug("---input---")
        logging.debug("input_ids shape = %s type=%s",
                      input_ids.shape, input_ids.type())
        logging.debug("attention_mask shape = %s type=%s",
                      attention_mask.shape, attention_mask.type())
        logging.debug("position_ids shape = %s type=%s",
                      position_ids.shape, position_ids.type())
        logging.debug("input_ids =%s",  input_ids)
        logging.debug("attention_mask =%s",  attention_mask)
        logging.debug("position_ids =%s",  position_ids)

        # If traced model is present then use that instead
        if self.context_encoding_model and self.token_generation_model:
            if input_ids.shape[-1] > 1:
                outputs = self.context_encoding_model(input_ids,
                                                      attention_mask,
                                                      position_ids)
                # Copy the KV cache from the context_encoding_model to token generation model
                for encoder_model, token_gen_model in zip(self.context_encoding_model.models, self.token_generation_model.models):
                    token_gen_model.load_state_dict(
                        encoder_model.state_dict(), strict=True)
                self.kv_cache_populated = True
            else:
                outputs = self.token_generation_model(input_ids,
                                                      attention_mask,
                                                      position_ids)

            hidden_states = outputs[0]
        else:
            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(input_ids,
                                 attention_mask,
                                 position_ids)
            hidden_states = outputs[0]
            self.kv_cache_populated = True

            # When traced the output kv tensors are aliased to the kv parameter list.
            # The code below mimicks that on CPU.
            new_past_key_values = outputs[1:]
            for i, new_past_key_value in enumerate(new_past_key_values):
                self.model.past_key_values[i].data = new_past_key_value

        logging.debug("---output---")
        logging.debug("last_hidden_state shape = %s type=%s",
                      hidden_states.shape, hidden_states.type())
        logging.debug("last_hidden_state = %s",  hidden_states)

        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i])
                      for i in range(self.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        logging.debug("logits = %s",  logits)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=[],
            hidden_states=outputs[0],
            attentions=None,
        )

    # We override this function because we want to change the way attention_mask
    # is updated each iteration.
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values

        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
            if attention_mask.shape[-1] > self.config.max_length + 1:
                attention_mask = attention_mask[:, 1:]
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
                position_ids = position_ids[:, -1].unsqueeze(-1)

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

    def reset(self):
        # We need to reset the KV cache flag for a new batch of inference.
        # When the flag is reset, the subsequent run will invoke the
        # context encoding model.
        self.kv_cache_populated = False
