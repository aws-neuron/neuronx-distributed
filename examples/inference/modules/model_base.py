import os
import copy
import random
import tempfile
import warnings

from typing import List, Optional, Tuple, Union
import logging

import numpy as np
import torch
from modules.autobucketing import generate_buckets
from modules.checkpoint import load_state_dict
from modules.config import NeuronConfig
from modules.hf_adapter import HuggingFaceGenerationAdapter
from modules.kvcache.kv_cache_manager import KVCacheManager
from safetensors.torch import load_file
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

import neuronx_distributed as nxd
from neuronx_distributed.parallel_layers import utils
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.quantization.quantization_config import QuantizationType
from neuronx_distributed.quantization.quantization_utils import (
    convert_qint8_to_int8_state_dict,
    quantize_pytorch_model_per_channel_symmetric,
    quantize_pytorch_model_per_tensor_symmetric,
)
from neuronx_distributed.trace.model_builder import ModelBuilder
from neuronx_distributed.utils.sampling import Sampler  # noqa: E402

from modules.model_wrapper import (  # noqa: E402
    CONTEXT_ENCODING_MODEL_TAG,  # noqa: E402
    SPECULATION_MODEL_TAG,  # noqa: E402
    MEDUSA_MODEL_TAG,  # noqa: E402
    TOKEN_GENERATION_MODEL_TAG,  # noqa: E402
    ModelWrapper,  # noqa: E402
)
from neuronx_distributed.parallel_layers.parallel_state import get_world_group
from modules.flashdecode.util import mask_util, turn_2d_mask_to_4d


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    nxd.parallel_layers.random.model_parallel_xla_manual_seed(seed)


class NeuronBaseModel(PreTrainedModel):
    """
    Base model that NeuronXXXModel classes inherit from.

    The forward() function will be traced and compiled by NxD.
    """

    def __init__(self, neuron_config: NeuronConfig, optimize_inference=True):
        super().__init__(neuron_config.hf_config)

        self.sampler = None
        self.kv_mgr = None
        self.neuron_config = neuron_config
        self.batch_size = neuron_config.batch_size
        self.n_positions = neuron_config.n_positions
        self.vocab_size = neuron_config.hf_config.vocab_size
        self.speculation_length = neuron_config.speculation_length
        self.padding_side = neuron_config.padding_side
        self.max_length = neuron_config.max_length

        self.rank_util = SPMDRank(world_size=get_world_group().size())
        self.setup_attr_for_model(neuron_config)
        self.init_model(neuron_config)
        if optimize_inference:
            self.init_inference_optimization(neuron_config)
        self.post_init()

    def setup_attr_for_model(self, neuron_config: NeuronConfig):
        """
        Please provide model-specific definition for the following attributes
            self.on_device_sampling
            self.tp_degree
            self.hidden_size
            self.num_attention_heads
            self.num_key_value_heads
            self.max_batch_size
            self.buckets
        """
        raise NotImplementedError("setup_attr_for_model() is not implemented")

    def initialize_process_group(self, world_size, seed:int = 17):
        if not torch.distributed.is_initialized():
            torch.dist.init_process_group(backend="xla")
        else:
            logging.info("torch.distributed was already initialized, skipping...")

        if not nxd.parallel_layers.parallel_state.model_parallel_is_initialized():
            nxd.parallel_layers.initialize_model_parallel(
                tensor_model_parallel_size=world_size,
            )
        else:
            logging.info("nxd was already initialized, skipping...")

        # set seed
        set_random_seed(seed)

    def init_model(self, neuron_config: NeuronConfig):
        """
        Please provide definition for the following components:
            self.embed_tokens
            self.layers
            self.norm
            self.lm_head
        """
        raise NotImplementedError("init_model() is not implemented")

    def init_inference_optimization(self, neuron_config: NeuronConfig):
        if self.on_device_sampling:
            self.sampler = Sampler(neuron_config)
        self.kv_mgr = KVCacheManager(neuron_config, num_kv_head=self.num_key_value_heads)

    def _create_context_attn_mask(self, attention_mask):
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

    def _create_spec_attn_mask(self, attention_mask):
        return (
            attention_mask[:, None, None, :]
            .expand(self.batch_size, 1, self.speculation_length, self.n_positions)
            .to(torch.bool)
        )

    def _create_simple_attn_mask(self, attention_mask):
        return attention_mask[:, None, None, :].expand(self.batch_size, 1, 1, self.n_positions).to(torch.bool)

    def create_attn_mask(self, attention_mask, is_for_context_encoding, is_for_speculation, position_ids):
        if is_for_context_encoding:
            return self._create_context_attn_mask(attention_mask)
        elif is_for_speculation:
            return self._create_spec_attn_mask(attention_mask)
        else:
            return self._create_simple_attn_mask(attention_mask)

    def _medusa_forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            accepted_indices=None,
            current_length=None,
            medusa_mask=None,
            scatter_index=None,
    ):
        is_for_context_encoding = (
                1 < input_ids.shape[-1] != self.medusa_speculation_length
        )
        is_for_medusa_speculation = input_ids.shape[-1] == self.medusa_speculation_length

        # It is either for context encoding or for token generation
        if is_for_context_encoding:
            past_key_values = None
        else:
            medusa_metadata = {
                "current_length": current_length,
                "accepted_indices": accepted_indices,
            }
            past_key_values = self.kv_mgr.get_cache(self.n_positions, medusa_metadata=medusa_metadata)

        # Prepare attention mask(s)
        attention_mask = self.create_attn_mask(
            attention_mask, is_for_context_encoding, False, position_ids
        )
        active_mask = None
        if is_for_medusa_speculation:
            medusa_mask = medusa_mask[0].bool()
            active_mask = medusa_mask[None, None, :, :].expand(
                self.batch_size, 1, self.medusa_speculation_length, self.medusa_speculation_length
            )

        hidden_states, past_key_values = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
        )

        updated_kv_cache = self.kv_mgr.update_cache(is_for_context_encoding, seq_ids, position_ids, past_key_values,
                                              self.n_positions, scatter_index)

        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            if position_ids.shape[-1] == self.medusa_speculation_length:
                index = torch.min(position_ids)
                index = torch.arange(index, index + self.medusa_speculation_length, device=hidden_states.device)
                index = index[None, :, None].expand(
                    self.batch_size, self.medusa_speculation_length, self.hidden_size
                )
                hidden_states = torch.gather(hidden_states, dim=1, index=index)
            else:
                # simple token generation
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        medusa_logits = [logits] + [
            head(hidden_states).float()
            for head in [getattr(self, f"medusa_head_{i}") for i in range(self.num_medusa_heads)]
        ]
        stacked_logits = torch.stack(medusa_logits, dim=0)

        res = logits
        if is_for_context_encoding:
            result = [
                self.sampler.sample(stacked_logits[i: i + 1, -1, :].squeeze(0))
                for i in range(self.neuron_config.num_medusa_heads + 1)
            ]
            res = torch.stack(result, dim=0)  # 5, 1, 10
        else:
            results = []
            for i in range(stacked_logits.shape[1]):
                result = [
                    self.sampler.sample(stacked_logits[j: j + 1, i, :].squeeze(0))
                    for j in range(self.neuron_config.num_medusa_heads + 1)
                ]
                res = torch.stack(result, dim=0)
                results.append(res)

        return [res] + updated_kv_cache

    def forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            accepted_indices=None,
            current_length=None,
            medusa_mask=None,
            scatter_index=None,

            # In llava context encoding model, input_embeds is precomputed
            inputs_embeds: Optional[torch.FloatTensor] = None
    ):
        if self.neuron_config.is_medusa:
            return self._medusa_forward(input_ids, attention_mask, position_ids, seq_ids, accepted_indices,
                                        current_length, medusa_mask, scatter_index)

        is_for_context_encoding = (
                1 < input_ids.shape[-1] != self.speculation_length
        )
        is_for_speculation = input_ids.shape[-1] == self.speculation_length

        cache_size = utils.divide(self.n_positions, self.neuron_config.num_cores_per_group) \
            if self.neuron_config.flash_decoding_enabled else self.n_positions

        # It is either for context encoding or for token generation
        if is_for_context_encoding:
            past_key_values = None
        else:
            past_key_values = self.kv_mgr.get_cache(cache_size)

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

        # FD masks
        active_mask_2d = None
        if self.neuron_config.flash_decoding_enabled and not is_for_context_encoding:
            rank_id = self.rank_util.get_rank()
            active_mask_2d, attention_mask_2d = mask_util(pos_ids=position_ids, rank_id=rank_id,
                                                          num_cores_per_group=self.neuron_config.num_cores_per_group,
                                                          cache_size=cache_size)
            active_mask = turn_2d_mask_to_4d(active_mask_2d, n_positions=1, batch_size=self.batch_size)
            attention_mask = turn_2d_mask_to_4d(attention_mask_2d, n_positions=cache_size, batch_size=self.batch_size)

        hidden_states, past_key_values = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds
        )

        updated_kv_cache = self.kv_mgr.update_cache(is_for_context_encoding, seq_ids, position_ids, past_key_values,
                                                    cache_size, scatter_index, active_mask_2d)
        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            # speculative decoding case; only batch_size=1
            # will need to extend the logic to support multi-batch later
            # maybe just use position_ids for index?
            if position_ids.shape[-1] == self.speculation_length:
                index = torch.min(position_ids)
                index = torch.arange(index, index + self.speculation_length, device=hidden_states.device)
                index = index[None, :, None].expand(self.batch_size, self.speculation_length, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)
            else:
                # simple token generation
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        res = logits
        if self.on_device_sampling:
            # perform sampling on Neuron to get tokens
            res = self.sampler.sample(logits[:, -1, :])

        return [res] + updated_kv_cache

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_model_output(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            active_mask: Optional[List[torch.FloatTensor]] = None,
            # In llava context encoding model, input_embeds is precomputed
            inputs_embeds: Optional[torch.FloatTensor] = None
    ):
        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device  #noqa
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

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


class NeuronBaseForCausalLM(HuggingFaceGenerationAdapter):
    _STATE_DICT_MODEL_PREFIX = "model."
    _NEW_STATE_DICT_MODEL_PREFIX = ""

    _model_cls = None

    def __init__(self, model_path: str, neuron_config: NeuronConfig):
        super().__init__(neuron_config)

        self.neuron_config = neuron_config
        self.vocab_size = neuron_config.hf_config.vocab_size
        self.padding_side = neuron_config.padding_side
        self.kv_cache_populated = False

        self.sampler = None
        self.model_wrapper = self.get_model_wrapper_cls()

        self.models = []
        self.enable_context_encoding()
        if neuron_config.trace_tokengen_model:
            self.enable_token_generation()
        if neuron_config.speculation_length > 0:
            self.enable_speculation()
        if neuron_config.medusa_speculation_length > 0:
            self.enable_medusa_speculation()
        self.model_path = model_path

    @staticmethod
    def load_hf_model(model_path):
        raise NotImplementedError("load_hf_model is not implemented")

    def get_compiler_args(self):
        return None

    def get_model_wrapper_cls(self):
        return ModelWrapper

    def enable_context_encoding(self):
        new_neuron_config = copy.deepcopy(self.neuron_config)
        new_neuron_config.batch_size = self.neuron_config.ctx_batch_size
        new_neuron_config.n_active_tokens = self.neuron_config.max_context_length
        new_neuron_config.bucket_n_active_tokens = True

        if not new_neuron_config.enable_bucketing:
            new_neuron_config.buckets = generate_buckets(new_neuron_config.max_context_length,
                                                         new_neuron_config.max_context_length)
        else:
            new_neuron_config.buckets = generate_buckets(128, new_neuron_config.max_context_length)

        self.context_encoding_model = self.model_wrapper(
            neuron_config=new_neuron_config,
            model_cls=self._model_cls,
            tag=CONTEXT_ENCODING_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.context_encoding_model)

    def enable_token_generation(self):
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
        )
        self.models.append(self.token_generation_model)

    def enable_speculation(self):
        new_neuron_config = copy.deepcopy(self.neuron_config)
        new_neuron_config.batch_size = self.neuron_config.spec_batch_size
        new_neuron_config.n_active_tokens = self.neuron_config.speculation_length
        self.speculation_model = self.model_wrapper(
            neuron_config=new_neuron_config,
            model_cls=self._model_cls,
            tag=SPECULATION_MODEL_TAG,
        )

        self.models.append(self.speculation_model)

    def enable_medusa_speculation(self):
        new_neuron_config = copy.deepcopy(self.neuron_config)
        new_neuron_config.batch_size = self.neuron_config.spec_batch_size
        new_neuron_config.n_active_tokens = self.neuron_config.medusa_speculation_length
        self.medusa_speculation_model = self.model_wrapper(
            neuron_config=new_neuron_config,
            model_cls=self._model_cls,
            tag=MEDUSA_MODEL_TAG
        )

        self.models.append(self.medusa_speculation_model)

    @classmethod
    def get_state_dict(cls, model_path: str, neuron_config: NeuronConfig) -> dict:
        model_sd = load_state_dict(model_path)
        param_name_list = list(model_sd.keys())
        for param_name in param_name_list:
            if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                updated_param_name = param_name.replace(cls._STATE_DICT_MODEL_PREFIX, cls._NEW_STATE_DICT_MODEL_PREFIX,
                                                        1)
                model_sd[updated_param_name] = model_sd[param_name]
                del model_sd[param_name]
        if os.path.exists(model_path + "/medusa_heads.pt"):
            medusa_head = torch.load(model_path + "/medusa_heads.pt", map_location="cpu")
            model_sd.update(medusa_head)
        model_sd = cls.convert_hf_to_neuron_state_dict(model_sd, neuron_config)
        return model_sd

    @classmethod
    def get_quantized_state_dict(cls, neuron_config: NeuronConfig, mmap: bool = False) -> dict:
        """
        This function loads the checkpointed float model state dictionary and weights from the quantized hf model
        This will be removed once we move to safe tensors in NxD
        """
        existing_checkpoint_path = neuron_config.quantized_checkpoints_path
        if not os.path.exists(existing_checkpoint_path):
            raise FileNotFoundError(f"Quantized checkpoint file not found: {existing_checkpoint_path}")

        print(f"Using existing checkpoint: {existing_checkpoint_path}")
        model_quant_sd = torch.load(existing_checkpoint_path)
        model_quant_sd = cls.convert_hf_to_neuron_state_dict(model_quant_sd, neuron_config)

        # Make sure that the non quantized weights are in bfloat16 and not float32
        if neuron_config.hf_config.torch_dtype == torch.bfloat16:
            for name, param in model_quant_sd.items():
                # TODO: Reduce and clean-up these warnings
                if param is not None and param.dtype == torch.float32:
                    if name.endswith(".scale"):
                        warnings.warn(f"Found float32 weights in quantized checkpoint: {name}. Will skip converting to bfloat16 as its scale")
                    else:
                        warnings.warn(f"Found float32 weights in quantized checkpoint: {name}. Will convert to bfloat16")
                        model_quant_sd[name] = param.bfloat16()

        return model_quant_sd

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, neuron_config: NeuronConfig) -> dict:
        """ This function should be over-ridden in child classes as needed """
        return state_dict

    @classmethod
    def generate_quantized_state_dict(cls, model_path: str, neuron_config: NeuronConfig) -> dict:
        hf_model = cls.load_hf_model(model_path)
        quantization_type = QuantizationType(neuron_config.quantization_type)
        if quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            hf_model_quant = quantize_pytorch_model_per_tensor_symmetric(float_model=hf_model, inplace=True)
        elif quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            hf_model_quant = quantize_pytorch_model_per_channel_symmetric(float_model=hf_model, inplace=True)
        else:
            raise RuntimeError(f"{neuron_config.quantization_type} not supported")

        model_quant_sd = hf_model_quant.model.state_dict()
        lm_head_quant_sd = hf_model_quant.lm_head.state_dict()
        convert_qint8_to_int8_state_dict(model_quant_sd)
        convert_qint8_to_int8_state_dict(lm_head_quant_sd)

        model_quant_sd["lm_head.weight"] = lm_head_quant_sd["weight"]
        model_quant_sd["lm_head.scale"] = lm_head_quant_sd["scale"]

        return model_quant_sd

    @classmethod
    def from_pretrained(cls, model_path: str, neuron_config: NeuronConfig):
        return cls(model_path, neuron_config)

    def checkpoint_loader_fn(self, mmap: bool = False):
        """ This function loads the model's state dictionary and weights from the hf model """

        if self.neuron_config.quantized:
            return self.get_quantized_state_dict(self.neuron_config)
        else:
            model_sd = self.get_state_dict(self.model_path, self.neuron_config)
            if self.neuron_config.hf_config.torch_dtype == torch.bfloat16:
                for name, param in model_sd.items():
                    if torch.is_floating_point(param):
                        # only cast floating types
                        model_sd[name] = param.bfloat16()
            return model_sd

    def compile(self, serialize_base_path=None):

        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

        builder = ModelBuilder(
            router=None,
            tp_degree=self.neuron_config.tp_degree,
            pp_degree=self.neuron_config.pp_degree,
            ep_degree=self.neuron_config.ep_degree,
            world_size=self.neuron_config.world_size,
            start_rank_id=self.neuron_config.start_rank_id,
            local_ranks_size=self.neuron_config.local_ranks_size,
            checkpoint_loader=self.checkpoint_loader_fn,
            compiler_workdir=base_compile_work_dir,
        )

        for model in self.models:
            builder.add(
                key=model.tag,
                model_instance=model.get_model_instance(),
                example_inputs=model.input_generator(),
                compiler_args=model.compiler_args,
                bucket_config=model.bucket_config,
                priority_model_idx=model.priority_model_idx,
            )
        if self.neuron_config.flash_decoding_enabled:
            builder.num_cores_per_group = self.neuron_config.num_cores_per_group  # FD needs this to initialize KV groups
        traced_model = builder.trace(initialize_model_weights=False)
        torch.jit.save(traced_model, serialize_base_path + "model.pt")
        del traced_model

        builder.shard_checkpoint(serialize_path=os.path.join(serialize_base_path, "weights/"))
        self.is_loaded_to_neuron = True

    def load(self, serialize_base_path, start_rank_id=None, local_ranks_size=None):
        if start_rank_id is None:
            start_rank_id = 0
        if local_ranks_size is None:
            local_ranks_size = self.neuron_config.local_ranks_size

        traced_model = torch.jit.load(serialize_base_path + "model.pt")
        logging.info(f"loading neuron model for ranks {start_rank_id}...{start_rank_id+local_ranks_size-1}")

        weights = []
        for rank in range(start_rank_id, start_rank_id+local_ranks_size):
            ckpt = load_file(os.path.join(serialize_base_path, f"weights/tp{rank}_sharded_checkpoint.safetensors"))
            weights.append(ckpt)

        start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
        traced_model.nxd_model.initialize(weights, start_rank_tensor)

        for model_wrapper in self.models:
            model_wrapper.model = traced_model

    def to_neuron(self, serialize_base_path=None):
        if serialize_base_path is None:
            with tempfile.TemporaryDirectory(suffix="nxd-temp-serial-path") as tmpdirname:
                self.compile(tmpdirname)
                self.load(tmpdirname)
        else:
            self.compile(serialize_base_path)
            self.load(serialize_base_path)

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # We dont want HF to move parameters to device
        return torch.device("cpu")

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            seq_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            medusa_args=None,
            return_dict: Optional[bool] = None,
            llava_args: Optional[List] = []
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        output_attentions, output_hidden_states, return_dict = self._setup_func_config(output_attentions,
                                                                                       output_hidden_states,
                                                                                       return_dict)

        # infer attention_mask from position_ids if not provided
        if attention_mask is None:
            attention_mask = self._infer_attention_mask(position_ids)

        self._log_input(input_ids, attention_mask, position_ids, seq_ids)

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        outputs, is_run_on_neuron = self._get_model_outputs(input_ids, attention_mask, position_ids, seq_ids,
                                                            medusa_args, llava_args)

        if self.neuron_config.trace_tokengen_model and not self.token_generation_model.is_neuron():
            self._copy_past_key_values(outputs)

        if is_run_on_neuron:
            # When run on neuron, KV cache remains on device
            logits_or_next_tokens = outputs
        else:
            # When run on cpu, KV cache is returned which has to be ignored
            logits_or_next_tokens, *_ = outputs

        logging.debug("---output---")
        logging.debug(f"{'tokens' if self.neuron_config.on_device_sampling else 'logits'} = %s, ",
                      logits_or_next_tokens)

        return self._construct_output(logits_or_next_tokens)

    def _setup_func_config(self, output_attentions, output_hidden_states, return_dict):
        output_attentions = output_attentions if output_attentions is not None else self.neuron_config.hf_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.neuron_config.hf_config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.neuron_config.hf_config.use_return_dict
        return output_attentions, output_hidden_states, return_dict

    def _infer_attention_mask(self, position_ids):
        assert position_ids is not None, "need to call forward with position_ids if attention_mask is not provided"
        batch_size, seq_len = position_ids.shape
        if position_ids.shape[-1] == 1:
            seq_len = self.neuron_config.n_positions
            position_ids_to_compare = position_ids.expand(batch_size, seq_len) - 1
        else:
            seq_len = position_ids.shape[-1]
            position_ids_to_compare = position_ids
        mask = torch.arange(seq_len).view(1, -1).expand(batch_size, seq_len)
        attention_mask = (position_ids_to_compare >= mask).to(dtype=position_ids.dtype)
        return attention_mask

    def _log_input(self, input_ids, attention_mask, position_ids, seq_ids, **kwargs):
        logging.debug("---input---")
        logging.debug("input_ids shape = %s type=%s", input_ids.shape, input_ids.type())
        logging.debug("attention_mask shape = %s type=%s", attention_mask.shape, attention_mask.type())
        logging.debug("position_ids shape = %s type=%s", position_ids.shape, position_ids.type())
        logging.debug("input_ids =%s", input_ids)
        logging.debug("attention_mask =%s", attention_mask)
        logging.debug("position_ids =%s", position_ids)
        logging.debug(f"seq_ids: {seq_ids}")

        if self.neuron_config.trace_tokengen_model and not self.token_generation_model.is_neuron():
            logging.debug(f"first layer kv_cache: {self.token_generation_model.model.kv_mgr.past_key_values[0][:, 0, :, 0]}")

    def _get_model_outputs(self, input_ids, attention_mask, position_ids, seq_ids, medusa_args, llava_args):
        if (
                input_ids.shape[-1] > 1
                and input_ids.shape[-1] != self.neuron_config.speculation_length
                and input_ids.shape[-1] != self.neuron_config.medusa_speculation_length
        ):
            if self.neuron_config.is_medusa:
                medusa_args = self._prepare_inputs()
                outputs = self.context_encoding_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    *medusa_args
                )
            else:
                outputs = self.context_encoding_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    *llava_args
                )
            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        elif input_ids.shape[-1] == self.neuron_config.speculation_length:
            outputs = self.speculation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids
            )
            is_run_on_neuron = self.speculation_model.is_neuron()
        elif input_ids.shape[-1] == self.neuron_config.medusa_speculation_length:
            outputs = self.medusa_speculation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids
            )
            is_run_on_neuron = self.medusa_speculation_model.is_neuron()
        else:
            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                *llava_args
            )
            is_run_on_neuron = self.token_generation_model.is_neuron()

        return outputs, is_run_on_neuron

    def _copy_kv_cache(self, source_model, target_model):
        for source, target in zip(source_model.model.models, target_model.model.models):
            encoder_kv_cache_line = source.states
            token_gen_kv_cache_line = target.states
            for name, _ in token_gen_kv_cache_line._parameters.items():
                token_gen_kv_cache_line._parameters[name] = encoder_kv_cache_line._parameters[name]

    def _copy_past_key_values(self, outputs):
        new_past_key_values = outputs[1:]
        for i, new_past_key_value in enumerate(new_past_key_values):
            self.token_generation_model.model.past_key_values[i].data = new_past_key_value
            self.context_encoding_model.model.past_key_values[i].data = new_past_key_value

    def _construct_output(self, logits_or_next_tokens):
        if self.neuron_config.is_medusa:
            next_tokens = logits_or_next_tokens[:1, :, :]
        else:
            next_tokens = logits_or_next_tokens

        OutputParams = CausalLMOutputWithPast(
            logits=None if self.neuron_config.on_device_sampling else logits_or_next_tokens,
            hidden_states=logits_or_next_tokens,
            attentions=None,
        )

        if self.neuron_config.is_medusa:
            OutputParams.tokens = next_tokens[:1, :, :]
            OutputParams.medusa_tokens = next_tokens[1:, :, :]
        else:
            OutputParams.tokens = next_tokens

        return OutputParams

    def _prepare_inputs(self):
        accepted_indices = torch.zeros((self.neuron_config.batch_size, self.neuron_config.num_medusa_heads + 1),
                                       dtype=torch.int64)
        current_length = torch.zeros((self.neuron_config.batch_size, self.neuron_config.num_medusa_heads + 1),
                                     dtype=torch.int64)
        medusa_mask = torch.zeros(
            (self.neuron_config.batch_size, self.neuron_config.medusa_speculation_length,
             self.neuron_config.medusa_speculation_length),
            dtype=torch.int64,
        )
        scatter_index = torch.zeros((self.neuron_config.batch_size, self.neuron_config.medusa_speculation_length),
                                    dtype=torch.int64)
        return accepted_indices, current_length, medusa_mask, scatter_index

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
