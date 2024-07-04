import logging
import os
from functools import partial

import torch
import torch.nn.functional as F
from modules.autobucketing import get_context_encoder_bk, get_token_generation_bk
from torch_neuronx import BucketModelConfig

from neuronx_distributed.quantization.quantize import convert
from neuronx_distributed.trace import (
    parallel_model_load,
    parallel_model_save,
    parallel_model_trace,
)
from neuronx_distributed.trace.trace import ParallelModel

CONTEXT_ENCODING_MODEL_TAG = "context_encoding_model"
TOKEN_GENERATION_MODEL_TAG = "token_generation_model"
SPECULATION_MODEL_TAG = "speculation_model"


def get_bucket_model_config_from_tag(tag, config):
    bucket_degree = len(config.buckets)
    if bucket_degree == 1:
        return None

    # NOTE: KV Cache preprocessing is done within the model and not the
    # shared buffer preprocessor due to lack of support of non-contiguous
    # slicing of nrt tensors via the NRT API.
    if tag == CONTEXT_ENCODING_MODEL_TAG:
        return BucketModelConfig(
            bucket_kernel=get_context_encoder_bk,
            bucket_kernel_constant_args=(torch.tensor(config.buckets), config.padding_side),
            shared_state_buffer=None,
            func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
        )
    elif tag == TOKEN_GENERATION_MODEL_TAG:
        return BucketModelConfig(
            bucket_kernel=get_token_generation_bk,
            bucket_kernel_constant_args=(torch.tensor(config.buckets), config.padding_side),
            shared_state_buffer=None,
            func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
        )
    else:
        raise ValueError(
            f"The supplied tag: {tag} is not supported for Bucketing. Only {CONTEXT_ENCODING_MODEL_TAG} and {TOKEN_GENERATION_MODEL_TAG} are supported"
        )


class ModelWrapper(torch.nn.Module):
    def __init__(self, config, model_cls, tag="", compiler_args: str = None) -> None:
        super().__init__()
        self.config = config

        if not self.config.torch_dtype:
            self.config.torch_dtype = torch.float32

        self.model_cls = model_cls
        self.model = None
        self.is_compiled = False
        self.serialize_base_path = None
        self.tag = tag
        if compiler_args is None:
            self.compiler_args = "--enable-saturate-infinity --auto-cast=none --model-type=transformer -O1"
        else:
            self.compiler_args = compiler_args

        self.bucket_config = get_bucket_model_config_from_tag(tag, self.config)

    def is_neuron(self):
        return self.model is not None and isinstance(self.model, ParallelModel)

    def compile(self, checkpoint_loader, serialize_base_path):
        inputs = self.input_generator()

        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

        # cannot pass partial func with multiprocess using model directly
        parallel_model_trace(
            partial(get_trace_callable, self.model_cls, self.config),
            inputs,
            tp_degree=self.config.tp_degree,
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
        self.model = self.model_cls(self.config)
        self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def input_generator(
        self,
    ):
        inputs = []
        for bucket in self.config.buckets:
            n_active_tokens = bucket if self.config.bucket_n_active_tokens else self.config.n_active_tokens

            input_ids = torch.zeros((self.config.batch_size, n_active_tokens), dtype=torch.int64)
            attention_mask = torch.zeros((self.config.batch_size, bucket), dtype=torch.int64)
            position_ids = torch.zeros((self.config.batch_size, n_active_tokens), dtype=torch.int64)
            seq_ids = torch.zeros((self.config.batch_size), dtype=torch.int64)

            inputs.append((input_ids, attention_mask, position_ids, seq_ids))

        return inputs if len(inputs) > 1 else inputs[0]

    def _forward_with_pad(self, *args):
        tensor, *_, seq_ids = args

        # pad the inputs up to the compiled batch size in the end
        def pad_helper(tensor):
            if tensor is None or tensor.shape[0] == self.config.batch_size:
                return tensor

            padded_shape = list(tensor.shape)
            padded_shape[0] = self.config.batch_size
            padded_tensor = torch.zeros(padded_shape, dtype=tensor.dtype)
            padded_tensor[: tensor.shape[0]] = tensor
            return padded_tensor

        padded_args = []
        for arg in args[:-1]:
            padded_args.append(pad_helper(arg))

        # need to handle seq_ids seperately, when compiled batch is 4, if we pad seq_ids from [0,2,1] to [0,2,1,0].
        # then the kv cache of padded input could be written into the first cache line, so we need to pad as [0, 2, 1, 3] instead

        seq_ids_list = seq_ids.tolist()
        padded_seq_ids = torch.tensor(
            seq_ids_list + [x for x in range(self.config.max_batch_size) if x not in seq_ids_list], dtype=seq_ids.dtype
        )
        padded_args.append(padded_seq_ids)

        logits, *kv_cache = self._forward(*padded_args)

        # note that we don't do index select here as it should already be handled, simply sliced out padding here
        return [logits[: seq_ids.shape[0]], *kv_cache]

    def reorder_helper(self, *args):
        # we then reorder the other inputs based on padded_seq_ids
        # because there are issue with compiler to do gather, we cannot fully support artibrary order of seq_ids for now
        *_, seq_ids = args

        reorder_args = []

        for arg in args:
            reorder_args.append(torch.index_select(arg, 0, seq_ids))

        return [seq_ids] + reorder_args

    def _forward(self, *args):
        if self.config.is_continuous_batching and self.config.batch_size == self.config.max_batch_size:
            logging.debug(f"running forward and reorder the inputs based on seq_ids")
            seq_ids, *args = self.reorder_helper(*args)

        logging.debug("Processed inputs to the model", self.tag, args)

        outputs = self.model(*args)

        if self.config.is_continuous_batching and self.config.batch_size == self.config.max_batch_size:
            return [torch.index_select(outputs[0], 0, seq_ids), *outputs[1:]]

        return outputs

    def pad_on_seq(self, *args):
        """
        Pad on the right, to make the inputs (input_ids, position_ids, attention_mask)
        on sequence dimension to match n_active_tokens, mainly apply to context encoding model
        """
        *to_pad_args, seq_id = args

        padded_args = []

        input_ids, *_ = to_pad_args

        pad_len = self.config.n_active_tokens - input_ids.shape[1]

        if padded_args == 0:
            return args

        logging.debug(f"padding inputs by {pad_len}")
        for arg in to_pad_args:
            arg = F.pad(arg, (0, pad_len), "constant", 0)
            padded_args.append(arg)

        return [*padded_args, seq_id]

    def forward(self, *args):
        logging.debug(f"calling forward on network {self.tag}")

        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() or load_state_dict() making calling forward")

        args = self.pad_on_seq(*args)

        *_, seq_ids = args

        input_batch_size = seq_ids.shape[0]

        if input_batch_size == self.config.batch_size:
            return self._forward(*args)

        cur_batch = 0
        output_logits = []

        logging.debug(f"get input_batch_size as {input_batch_size} but compiled batch_size as {self.config.batch_size}")
        while cur_batch < input_batch_size:
            if cur_batch + self.config.batch_size <= input_batch_size:
                # we only process part of the input to run
                logging.debug(f"running foward on batch {cur_batch}:{cur_batch+self.config.batch_size}")
                logits, *kv_caches = self._forward(
                    *[arg[cur_batch : cur_batch + self.config.batch_size] for arg in args]
                )
            else:
                # we need to pad the input to run
                logging.debug(
                    f"running forward on batch {cur_batch}:{input_batch_size}, padded up to {self.config.batch_size}"
                )
                logits, *kv_caches = self._forward_with_pad(*[arg[cur_batch:input_batch_size] for arg in args])

            if not self.is_neuron():
                for i, kv_cache in enumerate(kv_caches):
                    self.model.past_key_values[i].data = kv_cache

            output_logits.append(logits)
            cur_batch += self.config.batch_size

        return [torch.cat(output_logits, dim=0), *kv_caches]


def get_trace_callable(model_cls, config, bucket_rank=None):
    if bucket_rank is not None:
        config.n_positions = config.buckets[bucket_rank]
    float_model = model_cls(config)
    float_model.eval()
    if config.torch_dtype == torch.bfloat16:
        os.environ["XLA_DOWNCAST_BF16"] = "1"

    if config.quantized is True:
        model = convert(float_model, q_config=None, inplace=False, mapping=None)
    else:
        model = float_model

    aliases = {}
    num_output_from_trace = 1
    for i in range(len(model.past_key_values)):
        aliases[model.past_key_values[i]] = num_output_from_trace + i
    return model, aliases
