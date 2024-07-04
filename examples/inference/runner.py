import json
import logging
import os
from contextlib import contextmanager
from typing import List

import torch
from modules.benchmark import BENCHMARK_REPORT_FILENAME, Benchmark
from torch.profiler import ProfilerActivity, profile
from transformers import AutoTokenizer, GenerationConfig, PreTrainedModel, set_seed

import neuronx_distributed as nxd

END_TO_END_MODEL = "e2e_model"
CONTEXT_ENCODING_MODEL = "context_encoding_model"
TOKEN_GENERATION_MODEL = "token_generation_model"
SPECULATION_MODEL = "speculation_model"
LM_HEAD_NAME = "lm_head.pt"


BASE_COMPILER_WORK_DIR = "/tmp/nxd_model/"
CTX_ENC_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + CONTEXT_ENCODING_MODEL + "/"
TKN_GEN_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + TOKEN_GENERATION_MODEL + "/"
SPEC_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + SPECULATION_MODEL + "/"


class InferenceRunner:
    """
    Use the runner class to trace the model and perform inference.

    Usage:
        trace - Traces the neuron wrapper
        infer - Runs the traced model on Neuron
        infer-on-cpu - Runs the neuron wrapper on CPU
        infer-with-hf - Runs inference with huggingface model on CPU
    """

    def __init__(self, model_path: str = None, tokenizer_path: str = None, generation_config: GenerationConfig = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._is_torch_profile_enabled = False

        if generation_config == None:
            generation_config = GenerationConfig.from_pretrained(model_path)
            generation_config.top_k = 1
            generation_config.do_sample = True

        self.generation_config = generation_config

    def load_hf_model(self):
        # Implement per model
        raise NotImplementedError

    def load_neuron_model_on_cpu(self, max_context_length, max_new_tokens, batch_size, **kwargs):
        # Implement per model
        raise NotImplementedError

    def load_quantized_neuron_model_on_cpu(self, max_context_length, max_new_tokens, batch_size, **kwargs):
        # Implement per model
        raise NotImplementedError

    def load_neuron_model(self, traced_model_path):
        # Implement per model
        raise NotImplementedError

    def load_tokenizer(self, padding_side=None):
        # Implement per model
        raise NotImplementedError

    def get_config_cls(self):
        # Implement per model
        raise NotImplementedError

    def get_model_cls(self):
        # Implement per model
        raise NotImplementedError

    def get_padding_side(self):
        # Implement per model
        raise NotImplementedError

    def enable_torch_profile(self):
        self._is_torch_profile_enabled = True

    def is_torch_profile_enabled(self):
        return self._is_torch_profile_enabled

    @contextmanager
    def torch_profile(self, chrome_trace_path: str = "torch-trace.json", **profile_kwargs):
        if self.is_torch_profile_enabled():
            with profile(activities=[ProfilerActivity.CPU], **profile_kwargs) as prof:
                yield prof
            prof.export_chrome_trace(chrome_trace_path)
        else:
            yield

    def init_ditributed_env(self):
        """
        Initialize a simple neuronx distributed (Tensor Parallelism) environment, where there TP degree is 1.

        This function is just for running NeuronxDistributed models on CPU to validate correctness.
        """
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "2024"

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="xla")

        nxd.parallel_layers.parallel_state.destroy_model_parallel()
        nxd.parallel_layers.parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)

    def get_config_for_nxd(
        self,
        batch_size,
        tp_degree,
        context_lengths,
        new_token_counts,
        **kwargs,
    ):
        if isinstance(context_lengths, int):
            context_lengths = [context_lengths]

        if isinstance(new_token_counts, int):
            new_token_counts = [new_token_counts]

        config_cls = self.get_config_cls()
        config = config_cls.from_pretrained(self.model_path, **kwargs)
        config.tp_degree = tp_degree

        config.max_context_length = context_lengths[-1]
        config.max_new_tokens = new_token_counts[-1]
        max_length = config.max_context_length + config.max_new_tokens
        config.max_length = max_length
        config.n_positions = max_length

        if config.max_position_embeddings <= max_length:
            logging.warning(
                "max_position_embeddings is less than or equal to max_length. Updating max_position_embeddings..."
            )
            config.max_position_embeddings = max_length + 1  # otherwise get error

        config.max_batch_size = batch_size
        config.ctx_batch_size = batch_size
        config.tkg_batch_size = batch_size
        config.batch_size = batch_size

        # bucketing specific
        config.enable_context_encoding_bucketing, config.enable_token_generation_bucketing = [
            len(context_lengths) > 1,
        ] * 2
        config.buckets = [cl + tc for cl, tc in zip(context_lengths, new_token_counts)]
        config.bucket_n_active_tokens = config.enable_context_encoding_bucketing

        config.padding_side = self.get_padding_side()
        config.on_device_sampling = kwargs.get("on_device_sampling", False)

        config.spec_batch_size = batch_size
        config.speculation_length = kwargs.get("speculation_length", 0)
        config.trace_tokengen_model = kwargs.get("trace_tokengen_model", True)

        config.do_sample = self.generation_config.do_sample
        config.top_k = self.generation_config.top_k
        config.quantized = kwargs.get("quantized", False)
        config.quantized_checkpoints_path = kwargs.get("quantized_checkpoints_path", None)
        if config.quantized is True:
            assert config.quantized_checkpoints_path is not None, "quantized_checkpoints_path is required"

        return config

    def generate_with_hf(self, prompt, max_context_length: int, max_new_tokens: int, do_sample=True):
        """
        Use this to generate CPU goldens against which the trace is validated.
        """
        model = self.load_hf_model()
        tokenizer = self.load_tokenizer(padding_side="left")
        return self.generate(model, tokenizer, prompt, max_context_length, max_new_tokens, do_sample=do_sample)

    def generate_on_neuron(self, prompt, model: PreTrainedModel, draft_model: PreTrainedModel = None):
        """
        Runs the trace on Neuron.
        """

        if not isinstance(model, PreTrainedModel):
            raise ValueError(f"Model should be of type PreTrainedModel, got type {type(model)}")

        tokenizer = self.load_tokenizer()
        if len(prompt) != model.config.max_batch_size:
            raise ValueError(f"Number of prompts should match batch size {model.config.max_batch_size}")

        with self.torch_profile(chrome_trace_path="generate-on-neuron.torch-trace.json"):
            generate_ids, outputs = self.generate(
                model, tokenizer, prompt, model.config.max_context_length, model.config.max_new_tokens, draft_model
            )
        model.reset()
        if draft_model is not None:
            draft_model.reset()
        return generate_ids, outputs

    def generate_on_cpu(self, prompt: str, batch_size: int, max_context_length: int, max_new_tokens: int, **kwargs):
        """
        Use generate_on_cpu to confirm the neuron wrapper is correct. If the wrapper works
        on CPU, then the trace should work too. If it does not, it indicates a problem with
        the trace itself.
        """
        if kwargs.get("quantized", False) is False:
            model = self.load_neuron_model_on_cpu(max_context_length, max_new_tokens, batch_size, **kwargs)
        else:
            model = self.load_quantized_neuron_model_on_cpu(max_context_length, max_new_tokens, batch_size, **kwargs)

        tokenizer = self.load_tokenizer()
        generate_ids, outputs = self.generate(model, tokenizer, prompt, max_context_length, max_new_tokens)
        model.reset()
        return generate_ids, outputs

    def generate(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_context_length: int,
        max_new_tokens: int,
        draft_model: PreTrainedModel = None,
        do_sample=True,
    ):
        set_seed(0)  # to avoid randomness in sampling if any
        max_length = max_context_length + max_new_tokens
        inputs = tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        for idx, input in enumerate(inputs["input_ids"]):
            logging.debug("padded tokenized input %s : %s", idx, tokenizer.decode(input))

        if draft_model is not None:
            generate_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                assistant_model=draft_model,
                pad_token_id=tokenizer.eos_token_id,  # Set `pad_token_id` to `eos_token_id` for open-end generation
            )
        else:
            generate_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                top_k=1,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,  # Set `pad_token_id` to `eos_token_id` for open-end generation
            )
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return generate_ids, outputs

    def check_accuracy(
        self,
        traced_model: PreTrainedModel,
        batch_size: int,
        max_context_length: int,
        max_new_tokens: int,
        expected_token_ids: List=None,
        on_cpu: bool=False,
        do_sample: bool=True,
        traced_draft_model: PreTrainedModel=None,
        speculation_length: int=0,
    ):
        """
        Function to compare outputs from huggingface model and neuronx NxD model
        """
        prompt = ["I believe the meaning of life is"] * batch_size
        tokenizer = self.load_tokenizer()

        if expected_token_ids is not None:
            outputs_expected = tokenizer.batch_decode(
                expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        else:
            # Generate goldens with HF on CPU
            expected_token_ids, outputs_expected = self.generate_with_hf(
                prompt, max_context_length, max_new_tokens, do_sample=do_sample
            )
        print(f"Expected output: {outputs_expected}")

        # Generate outputs with NxD
        prompt = ["I believe the meaning of life is"] * batch_size
        if on_cpu:
            output_token_ids, outputs_actual = self.generate_on_cpu(
                prompt, batch_size, max_context_length, max_new_tokens
            )
        else:
            output_token_ids, outputs_actual = self.generate_on_neuron(
                prompt, traced_model, traced_draft_model
            )
        print(f"Actual output  : {outputs_actual}")

        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) if tokenizer else 0
        output_token_ids = output_token_ids[output_token_ids != pad_token_id]
        expected_token_ids = expected_token_ids[expected_token_ids != pad_token_id]
        if traced_draft_model is not None:
            # Handle corner scenario where last few tokens are not generated as part of speculation.
            assert (
                abs(expected_token_ids.shape[-1] - output_token_ids.shape[-1]) <= speculation_length
            ), "Unexpected number of tokens generated by target model"
            tokens_to_compare = min(expected_token_ids.shape[-1], output_token_ids.shape[-1])
            expected_token_ids = expected_token_ids[: tokens_to_compare]
            output_token_ids = output_token_ids[: tokens_to_compare]


        device = "cpu" if on_cpu else "neuron"
        assert torch.equal(
            output_token_ids, expected_token_ids
        ), f"\nActual: ({device}) {output_token_ids} \nExpected (hf-cpu): {expected_token_ids}"
        print(f"The output from Neuronx NxD on {device} is accurate!")

    def trace(
        self,
        traced_model_path,
        tp_degree,
        batch_size,
        context_lengths,
        new_token_counts,
        **kwargs,
    ):
        """
        Function to trace a model with neuronx NxD
        """
        if traced_model_path is not None:
            if not os.path.exists(traced_model_path):
                os.makedirs(traced_model_path)

        # Write the model config into the traced_model_path
        config = self.get_config_for_nxd(
            batch_size,
            tp_degree,
            context_lengths,
            new_token_counts,
            **kwargs,
        )
        if config.torch_dtype != torch.float32 and config.torch_dtype != torch.bfloat16:
            raise ValueError(
                f"Type {config.torch_dtype} is not supported for this model at this time. Please choose float32 or bfloat16."
            )
        # We have the config in the trace_model_path
        config.save_pretrained(traced_model_path)

        # Save config to be used by checkpoint_loader
        self.config = config

        # Copy the tokenizer into the traced_model_path
        tokenizer = self.load_tokenizer()
        if tokenizer:
            tokenizer.save_pretrained(traced_model_path)

        model = self.get_model_cls().from_pretrained(self.model_path, config)
        model.compile(serialize_base_path=traced_model_path)

    def benchmark_sampling(self, model: PreTrainedModel, draft_model: PreTrainedModel=None, target: str=None):
        config = model.config
        tokenizer = self.load_tokenizer()
        target = target if target is not None else "all"

        report = {}

        # Benchmark E2E model
        if target in ["all", "e2e"]:
            batch_encoding = self.get_sample_inputs(END_TO_END_MODEL, config, tokenizer)
            input_param = {
                "input_ids": batch_encoding["input_ids"],
                "attention_mask": batch_encoding["attention_mask"],
                "max_new_tokens": config.max_new_tokens,
                "top_k": 1,
                "do_sample": draft_model is None,
                "assistant_model": draft_model,
            }
            e2e_benchmark = Benchmark(model.generate, input_param, config, preprocess_func=model.reset)
            report[END_TO_END_MODEL] = e2e_benchmark.run()

        # Benchmark context encoding model
        if target in ["all", "context_encode"]:
            input_param = self.get_sample_inputs(CONTEXT_ENCODING_MODEL, config)
            ctx_enc_benchmark = Benchmark(model.context_encoding_model, input_param, config)
            report[CONTEXT_ENCODING_MODEL] = ctx_enc_benchmark.run()

        # Benchmark token generation model
        if hasattr(model, "token_generation_model") and target in ["all", "token_gen"]:
            input_param = self.get_sample_inputs(TOKEN_GENERATION_MODEL, config)
            tkn_gen_benchmark = Benchmark(model.token_generation_model, input_param, config)
            report[TOKEN_GENERATION_MODEL] = tkn_gen_benchmark.run()

        # Benchmark speculation model
        if hasattr(model, "speculation_model") and target in ["all", "speculation"]:
            input_param = self.get_sample_inputs(SPECULATION_MODEL, config)
            spec_benchmark = Benchmark(model.speculation_model, input_param, config)
            report[SPECULATION_MODEL] = spec_benchmark.run()

        print("Benchmark completed and its result is as following")
        print(json.dumps(report, indent=4))
        with open(BENCHMARK_REPORT_FILENAME, "w") as f:
            json.dump(report, f)
        print("Completed saving result to " + BENCHMARK_REPORT_FILENAME)

        return report

    def get_sample_inputs(self, model_type, config, tokenizer=None):
        max_length = config.max_length
        batch_size = config.batch_size

        sample_inputs = None
        if model_type == END_TO_END_MODEL:
            sample_inputs = tokenizer(
                ["I believe the meaning of life is"] * batch_size,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        elif model_type == CONTEXT_ENCODING_MODEL:
            input_ids = torch.zeros((batch_size, max_length), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, max_length), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, max_length), dtype=torch.int64)
            seq_ids = torch.zeros((batch_size), dtype=torch.int64)
            sample_inputs = (input_ids, attention_mask, position_ids, seq_ids)

        elif model_type == TOKEN_GENERATION_MODEL:
            input_ids = torch.zeros((batch_size, 1), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, max_length), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, 1), dtype=torch.int64)
            seq_ids = torch.zeros((batch_size), dtype=torch.int64)
            sample_inputs = (input_ids, attention_mask, position_ids, seq_ids)

        elif model_type == SPECULATION_MODEL:
            spec_len = config.speculation_length
            input_ids = torch.zeros((batch_size, spec_len), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, max_length), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, spec_len), dtype=torch.int64)
            seq_ids = torch.zeros((batch_size), dtype=torch.int64)
            sample_inputs = (input_ids, attention_mask, position_ids, seq_ids)

        return sample_inputs
