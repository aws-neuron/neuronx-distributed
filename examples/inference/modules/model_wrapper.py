import copy
import logging
import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from modules.autobucketing import get_context_encoder_bk, get_token_generation_bk
from modules.config import NeuronConfig
from torch_neuronx import BucketModelConfig

from neuronx_distributed.quantization.quantization_config import (
    QuantizationType,
    QuantizedDtype,
    get_default_custom_qconfig_dict,
    get_default_per_channel_custom_qconfig_dict,
)
from neuronx_distributed.quantization.quantize import convert
from neuronx_distributed.trace import (
    parallel_model_load,
    parallel_model_trace,
)
from neuronx_distributed.trace.model_builder import BaseModelInstance

IMAGE_ENCODING_MODEL_TAG = "image_encoding_model"
CONTEXT_ENCODING_MODEL_TAG = "context_encoding_model"
TOKEN_GENERATION_MODEL_TAG = "token_generation_model"
SPECULATION_MODEL_TAG = "speculation_model"
MEDUSA_MODEL_TAG = "medusa_speculation_model"


def _reorder_helper(*args: torch.Tensor):
    # sorting is needed due to compiler issues with gather and hence can't support arbitrary order of seq_ids
    seq_ids = args[3]
    indices = torch.argsort(seq_ids)
    reorder_args = []
    for arg in args:
        reorder_args.append(torch.index_select(arg, 0, indices))
    return reorder_args


def get_bucket_model_config_from_tag(tag, neuron_config: NeuronConfig):
    bucket_degree = len(neuron_config.buckets)
    if bucket_degree == 1:
        return None

    pad_token = neuron_config.hf_config.pad_token_id

    # NOTE: KV Cache preprocessing is done within the model and not the
    # shared buffer preprocessor due to lack of support of non-contiguous
    # slicing of nrt tensors via the NRT API.
    if tag == CONTEXT_ENCODING_MODEL_TAG:
        return BucketModelConfig(
            bucket_kernel=get_context_encoder_bk,
            bucket_kernel_constant_args=(torch.tensor(neuron_config.buckets), neuron_config.padding_side, pad_token),
            shared_state_buffer=None,
            func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
        )
    elif tag == TOKEN_GENERATION_MODEL_TAG:
        return BucketModelConfig(
            bucket_kernel=get_token_generation_bk,
            bucket_kernel_constant_args=(torch.tensor(neuron_config.buckets), neuron_config.padding_side),
            shared_state_buffer=None,
            func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
        )
    else:
        raise ValueError(
            f"The supplied tag: {tag} is not supported for Bucketing. Only {CONTEXT_ENCODING_MODEL_TAG} and {TOKEN_GENERATION_MODEL_TAG} are supported"
        )


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        neuron_config: NeuronConfig,
        model_cls,
        tag="",
        compiler_args: str = None,
        priority_model_idx: int = None,
        model_init_kwargs = {},
    ) -> None:
        super().__init__()
        self.neuron_config = neuron_config

        if not self.neuron_config.hf_config.torch_dtype:
            self.neuron_config.hf_config.torch_dtype = torch.float32

        if self.neuron_config.hf_config.pad_token_id is None:
            self.neuron_config.hf_config.pad_token_id = 0

        self.model_cls = model_cls
        self.model = None
        self.is_compiled = False
        self.serialize_base_path = None
        self.tag = tag
        self.is_medusa = neuron_config.is_medusa
        self.model_init_kwargs = model_init_kwargs

        if compiler_args is None:
            self.compiler_args = "--enable-saturate-infinity --auto-cast=none --model-type=transformer --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' -O1 "

        else:
            self.compiler_args = compiler_args

        if self.neuron_config.quantized is True and self.neuron_config.quantization_dtype == 'f8e4m3':
            self.compiler_args += " --internal-hlo2tensorizer-options=--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"

        self.bucket_config = get_bucket_model_config_from_tag(tag, self.neuron_config)
        self.priority_model_idx = priority_model_idx

    def is_neuron(self):
        return self.model is not None and isinstance(self.model, torch.jit.ScriptModule)

    def compile(self, checkpoint_loader, serialize_base_path):
        inputs = self.input_generator()

        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

        # cannot pass partial func with multiprocess using model directly
        parallel_model_trace(
            partial(get_trace_callable, self.model_cls, self.neuron_config),
            inputs,
            tp_degree=self.neuron_config.tp_degree,
            compiler_workdir=os.path.join(base_compile_work_dir, self.tag),
            compiler_args=self.compiler_args,
            inline_weights_to_neff=False,
            spmd_mode=True,
            checkpoint_loader_callable=checkpoint_loader,
            bucket_config=self.bucket_config,
            force_custom_init_on_device=True,
            serialization_path=os.path.join(serialize_base_path, self.tag),
        )
        print(f"Successfully traced the {self.tag}!")

    def load(self, serialize_base_path):
        self.model = parallel_model_load(os.path.join(serialize_base_path, self.tag))

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.model = self.model_cls(self.neuron_config)
        self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def input_generator(
            self,
    ):
        inputs = []
        batch_size = self.neuron_config.batch_size
        for bucket in self.neuron_config.buckets:
            n_active_tokens = bucket if self.neuron_config.bucket_n_active_tokens else self.neuron_config.n_active_tokens

            input_ids = torch.zeros((batch_size, n_active_tokens), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, bucket), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, n_active_tokens), dtype=torch.int64)
            seq_ids = torch.zeros((batch_size), dtype=torch.int64)

            if self.is_medusa:
                accepted_indices = torch.zeros(
                    (batch_size, self.neuron_config.num_medusa_heads + 1), dtype=torch.int64
                )
                current_length = torch.zeros((batch_size, self.neuron_config.num_medusa_heads + 1), dtype=torch.int64)
                medusa_mask = torch.zeros(
                    (batch_size, self.neuron_config.medusa_speculation_length, self.neuron_config.medusa_speculation_length),
                    dtype=torch.int64,
                )
                scatter_index = torch.zeros(
                    (batch_size, self.neuron_config.medusa_speculation_length), dtype=torch.int64
                )
                inputs.append(
                    (
                        input_ids,
                        attention_mask,
                        position_ids,
                        seq_ids,
                        accepted_indices,
                        current_length,
                        medusa_mask,
                        scatter_index,
                    )
                )
            else:
                inputs.append((input_ids, attention_mask, position_ids, seq_ids))

        return inputs

    def get_model_instance(self):
        return DecoderModelInstance(
            model_cls=self.model_cls,
            neuron_config=self.neuron_config,
            **self.model_init_kwargs,
        )

    def _forward_with_pad(self, *args):
        seq_ids = args[3]
        if len(args) > 4:
            medusa_args = args[4:8]
        else:
            medusa_args = None

        # pad the inputs up to the compiled batch size in the end
        def pad_helper(tensor):
            if tensor is None or tensor.shape[0] == self.neuron_config.batch_size:
                return tensor

            padded_shape = list(tensor.shape)
            padded_shape[0] = self.neuron_config.batch_size
            padded_tensor = torch.zeros(padded_shape, dtype=tensor.dtype)
            padded_tensor[: tensor.shape[0]] = tensor
            return padded_tensor

        padded_args = []
        # pad input_ids, attn_mask and position_ids
        for arg in args[0:3]:
            padded_args.append(pad_helper(arg))

        # need to handle seq_ids separately, when compiled batch is 4, if we pad seq_ids from [0,2,1] to [0,2,1,
        # 0]. then the kv cache of padded input could be written into the first cache line, so we need to pad as [0,
        # 2, 1, 3] instead

        seq_ids_list = seq_ids.tolist()
        padded_seq_ids = torch.tensor(
            seq_ids_list + [x for x in range(self.neuron_config.max_batch_size) if x not in seq_ids_list], dtype=seq_ids.dtype
        )
        padded_args.append(padded_seq_ids)

        if medusa_args is not None:
            for arg in medusa_args:
                padded_args.append(pad_helper(arg))

        outputs = self._forward(*padded_args)

        # note that we don't do index select here as it should already be handled, simply sliced out padding here
        if self.is_neuron():
            logits = outputs
            return logits[: seq_ids.shape[0]]
        else:
            logits, *kv_cache = outputs
            return [logits[: seq_ids.shape[0]], *kv_cache]

    def _forward(self, *args):
        if self.neuron_config.is_continuous_batching and self.neuron_config.batch_size == self.neuron_config.max_batch_size:
            logging.debug("running forward and reorder the inputs based on seq_ids")
            preserved_seq_ids = args[3]
            updated_args = _reorder_helper(*args)
            logging.debug(f"Processed inputs to the model. tag={self.tag}, args={args}")
            outputs = self.model(*updated_args)
            if self.is_neuron():
                return torch.index_select(outputs, 0, preserved_seq_ids)
            else:
                return [torch.index_select(outputs[0], 0, preserved_seq_ids), *outputs[1:]]

        logging.debug(f"Processed inputs to the model. tag={self.tag}, args={args}")
        return self.model(*args)

    def pad_to_max_compiled_seq(self, *args):
        if self.tag == CONTEXT_ENCODING_MODEL_TAG:
            to_pad = args[:3]
            pad_lengths = [self.neuron_config.max_context_length - arg.shape[1] for arg in to_pad]
            tensor_pad_vals = [
                self.neuron_config.hf_config.pad_token_id,
                0,
                1
            ]
            padded_args = [F.pad(arg, (0, pad_len), "constant", pad_val) for arg, pad_val, pad_len in zip(to_pad, tensor_pad_vals, pad_lengths)]
            args = (*padded_args,*args[3:])
        else:
            input_ids,attention_mask,*rest_of_args = args
            pad_len = self.neuron_config.max_length - attention_mask.shape[1]
            padded_attention_mask = F.pad(attention_mask, (0, pad_len), "constant", 0)
            args = (input_ids,padded_attention_mask,*rest_of_args)

        return args

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")

        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() or load_state_dict() making calling forward")

        args = self.pad_to_max_compiled_seq(*args)

        seq_ids = args[3]

        input_batch_size = seq_ids.shape[0]

        if input_batch_size == self.neuron_config.batch_size:
            return self._forward(*args)

        cur_batch = 0
        output_logits = []

        logging.debug(f"get input_batch_size as {input_batch_size} but compiled batch_size as {self.neuron_config.batch_size}")
        while cur_batch < input_batch_size:
            if cur_batch + self.neuron_config.batch_size <= input_batch_size:
                # we only process part of the input to run
                logging.debug(f"running foward on batch {cur_batch}:{cur_batch+self.neuron_config.batch_size}")
                outputs = self._forward(
                    *[arg[cur_batch : cur_batch + self.neuron_config.batch_size] for arg in args]
                )
            else:
                # we need to pad the input to run
                logging.debug(
                    f"running forward on batch {cur_batch}:{input_batch_size}, padded up to {self.neuron_config.batch_size}"
                )
                outputs = self._forward_with_pad(*[arg[cur_batch:input_batch_size] for arg in args])

            if self.is_neuron():
                logits = outputs
            else:
                logits, *kv_caches = outputs
                for i, kv_cache in enumerate(kv_caches):
                    self.model.kv_mgr.past_key_values[i].data = kv_cache

            output_logits.append(logits)
            cur_batch += self.neuron_config.batch_size

        if self.is_neuron():
            return torch.cat(output_logits, dim=0)
        else:
            return [torch.cat(output_logits, dim=0), *kv_caches]

class DecoderModelInstance(BaseModelInstance):

    def __init__(self, model_cls, neuron_config: NeuronConfig, **kwargs):
        self.model_cls = model_cls
        self.module = None
        self.input_output_aliases = None
        self.neuron_config = neuron_config
        self.kwargs = kwargs

    def initialize_process_group(self, world_size):
        self.model_cls.initialize_process_group(world_size)

    def load_module(self):
        float_model = self.model_cls(self.neuron_config, **self.kwargs)
        float_model.eval()

        if self.neuron_config.hf_config.torch_dtype == torch.bfloat16:
            float_model.bfloat16()

        if self.neuron_config.quantized is True:
            quantization_type = QuantizationType(self.neuron_config.quantization_type)
            if quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
                q_config = get_default_per_channel_custom_qconfig_dict()
            elif quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
                q_config = get_default_custom_qconfig_dict()
            else:
                raise RuntimeError(f"{self.neuron_config.quantization_type} is not supported")
            if self.neuron_config.quantization_dtype == 'f8e4m3':
                q_config["quantized_dtype"] = QuantizedDtype.F8E4M3

            self.module = convert(float_model, q_config=q_config, inplace=False, mapping=None)
        else:
            self.module = float_model

    def get(self, bucket_rank, **kwargs):
        if bucket_rank is not None:
            self.module.n_positions = self.neuron_config.buckets[bucket_rank]

        # Currently we have to init an input_output_aliases map for
        # each buckets, otherwise it will fail the aliasing setup when
        # generating HLO
        self.input_output_aliases = {}
        num_output_from_trace = 1
        # TODO: This else block is a short-term fix for Llava/ViT models to use DecoderModelInstance.
        #       Long-term, these models should use a different implementation of BaseModelInstance.
        if self.module.kv_mgr is not None:
            past_key_values = self.module.kv_mgr.past_key_values
        else:
            past_key_values = self.module.past_key_values
        for i in range(len(past_key_values)):
            self.input_output_aliases[past_key_values[i]] = (
                num_output_from_trace + i
            )
        return self.module, self.input_output_aliases


def get_trace_callable(model_cls, neuron_config: NeuronConfig, bucket_rank=None):
    if bucket_rank is not None:
        neuron_config.n_positions = neuron_config.buckets[bucket_rank]
    float_model = model_cls(neuron_config)
    float_model.eval()
    if neuron_config.hf_config.torch_dtype == torch.bfloat16:
        float_model.bfloat16()

    if neuron_config.quantized is True:
        quantization_type = QuantizationType(neuron_config.quantization_type)
        if quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            q_config = get_default_per_channel_custom_qconfig_dict()
        elif quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            q_config = get_default_custom_qconfig_dict()
        else:
            raise RuntimeError(f"{neuron_config.quantization_type} is not supported")
        model = convert(float_model, q_config=q_config, inplace=False, mapping=None)
    else:
        model = float_model

    aliases = {}
    num_output_from_trace = 1
    for i in range(len(model.kv_mgr.past_key_values)):
        aliases[model.kv_mgr.past_key_values[i]] = num_output_from_trace + i
    return model, aliases
