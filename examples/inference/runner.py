from enum import Enum
import json
import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import List, Optional, Union, Type

import torch
from modules.benchmark import BENCHMARK_REPORT_FILENAME, Benchmark, LatencyCollector, generate_report
from modules.config import NeuronConfig
from modules.model_base import NeuronBaseModel
from torch.profiler import ProfilerActivity, profile
from transformers import AutoConfig, AutoTokenizer, GenerationConfig, PretrainedConfig, PreTrainedModel, set_seed
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.image_utils import ImageInput

from torch_neuronx.testing.validation import logit_validation
import neuronx_distributed as nxd
from neuronx_distributed.quantization.quantization_config import QuantizationType
from neuronx_distributed.quantization.quantization_utils import (
    quantize_pytorch_model_per_channel_symmetric,
    quantize_pytorch_model_per_tensor_symmetric,
)
from modules.lora_serving import LoraServingConfig


IMAGE_ENCODING_MODEL = "image_encoding_model"
END_TO_END_MODEL = "e2e_model"
CONTEXT_ENCODING_MODEL = "context_encoding_model"
TOKEN_GENERATION_MODEL = "token_generation_model"
SPECULATION_MODEL = "speculation_model"
MEDUSA_MODEL = "medusa_speculation_model"
LM_HEAD_NAME = "lm_head.pt"


BASE_COMPILER_WORK_DIR = "/tmp/nxd_model/"
CTX_ENC_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + CONTEXT_ENCODING_MODEL + "/"
TKN_GEN_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + TOKEN_GENERATION_MODEL + "/"
SPEC_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + SPECULATION_MODEL + "/"


SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

TEST_PROMPT = "I believe the meaning of life is"


class TaskType(Enum):
    GENERATION = "generation_task"
    IMAGE_ENC = "image_encoding_task"


class InferenceRunner:
    """
    Use the runner class to trace the model and perform inference.

    Usage:
        trace - Traces the neuron wrapper
        infer - Runs the traced model on Neuron
        infer-on-cpu - Runs the neuron wrapper on CPU
        infer-with-hf - Runs inference with huggingface model on CPU

    Arguments:
        model_path (str) - The path to the pre-trained model.
        tokenizer_path (str) - The path to the tokenizer associated with the model.
        generation_config (GenerationConfig) - Configuration settings for text generation tasks.
        task_type (TaskType) - The type of task the runner is configured for (default is TaskType.GENERATION).
                              If task type is set as IMAGE_ENC.IMAGE_ENC, the generation_*() related functions
                              will be disabled and the self.generation_config will be initialized as None.
    """

    def __init__(self, model_path: str = None, tokenizer_path: str = None, generation_config: GenerationConfig = None, task_type=TaskType.GENERATION):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._is_torch_profile_enabled = False

        if generation_config is None and task_type == TaskType.GENERATION:
            generation_config = GenerationConfig.from_pretrained(model_path)
            generation_config.top_k = 1
            generation_config.do_sample = True
            generation_config.pad_token_id = 0

        self.generation_config = generation_config

    def load_hf_model(self):
        # Implement per model
        raise NotImplementedError

    def load_neuron_model_on_cpu(self, max_prompt_length, sequence_length, batch_size, **kwargs):
        # Implement per model
        raise NotImplementedError

    def generate_quantized_hf_checkpoints_on_cpu(self, max_prompt_length, sequence_length, batch_size, **kwargs):
        hf_config = self.get_hf_config(sequence_length=sequence_length, **kwargs)
        neuron_config = self.get_config_for_nxd(
            hf_config,
            batch_size,
            1,
            max_prompt_length,
            sequence_length,
            enable_bucketing=False,
            **kwargs)
        neuron_config.hf_config.torch_dtype = torch.float32
        quantized_state_dict = self.get_model_cls().generate_quantized_state_dict(
            model_path=self.model_path, neuron_config=neuron_config
        )
        return quantized_state_dict

    def load_quantized_neuron_model_on_cpu(self, max_prompt_length, sequence_length, batch_size, lora_config: LoraServingConfig=None, **kwargs):
        model = self.load_neuron_model_on_cpu(max_prompt_length, sequence_length, batch_size, lora_config, **kwargs)

        quantization_type = QuantizationType(kwargs.get("quantization_type", "per_tensor_symmetric"))
        if quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            return quantize_pytorch_model_per_tensor_symmetric(model, inplace=True)
        elif quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            return quantize_pytorch_model_per_channel_symmetric(model, inplace=True)
        else:
            raise RuntimeError(f"quantization_type: {quantization_type} not supported")

    def load_neuron_model(self, traced_model_path, start_rank_id=None, local_ranks_size=None):
        neuron_config = self.get_config_cls().load(traced_model_path)
        model = self.get_model_cls().from_pretrained("", neuron_config)
        model.load(traced_model_path, start_rank_id=start_rank_id, local_ranks_size=local_ranks_size)
        if neuron_config.hf_config.torch_dtype == torch.bfloat16:
            model.bfloat16()
        return model

    def load_tokenizer(self, padding_side=None):
        # Implement per model
        raise NotImplementedError

    def load_image_processor(self):
        return None

    def load_processor(self):
        return None

    def get_config_cls(self) -> Type[NeuronConfig]:
        # Implement per model
        raise NotImplementedError

    def get_model_cls(self):
        # Implement per model
        raise NotImplementedError

    def get_padding_side(self):
        # Implement per model
        raise NotImplementedError

    def get_default_hf_generation_config_kwargs(self) -> dict:
        return {
            'do_sample': self.generation_config.do_sample,
            'top_k': self.generation_config.top_k,
            'pad_token_id': self.generation_config.pad_token_id
        }

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

    def get_hf_config(self, sequence_length, **kwargs):
        merged_kwargs = self.get_default_hf_generation_config_kwargs()
        if kwargs is not None:
            merged_kwargs.update(kwargs)

        hf_config: PretrainedConfig = AutoConfig.from_pretrained(self.model_path, **merged_kwargs)
        hf_config.max_length = sequence_length
        if hasattr(hf_config, "max_position_embeddings") and hf_config.max_position_embeddings <= hf_config.max_length:
            logging.warning(
                "max_position_embeddings is less than or equal to max_length. Updating max_position_embeddings..."
            )
            hf_config.max_position_embeddings = hf_config.max_length + 1  # otherwise get error
        hf_config.pad_token_id = kwargs.get("pad_token_id", hf_config.pad_token_id)

        return hf_config

    def get_config_for_nxd(
        self,
        hf_config,
        batch_size,
        tp_degree,
        max_prompt_length,
        sequence_length,
        enable_bucketing,
        **kwargs,
    ) -> NeuronConfig:
        """
        Set up the value for config attributes if needed.

        Please don't add new config attribute here. Instead, please add new
        attributes in NeuronConfig or model-specific config class.
        """
        config_cls = self.get_config_cls()
        try:
            neuron_config = config_cls.load(self.model_path, skip_hf_config=True, **kwargs)
            neuron_config.hf_config = hf_config
            return neuron_config
        except FileNotFoundError:
            return config_cls(hf_config=hf_config,
                              tp_degree=tp_degree,
                              batch_size=batch_size,
                              seq_len=sequence_length,
                              padding_side=self.get_padding_side(),
                              max_context_length=max_prompt_length,
                              enable_bucketing=enable_bucketing,
                              **kwargs)

    def generate_with_hf(self, prompts: List[str], max_length: int, **kwargs):
        """
        Use this to generate CPU goldens against which the trace is validated.
        """
        model = self.load_hf_model()
        if kwargs.get("images") is not None:
            processor = self.load_processor(padding_side="left")
            tokenizer = processor.tokenizer
            kwargs["image_processor"] = processor.image_processor
        else:
            tokenizer = self.load_tokenizer(padding_side="left")
        return self.generate(model, tokenizer, prompts, max_length, **kwargs)

    def generate_on_neuron(self, prompts: List[str], model: PreTrainedModel, draft_model: PreTrainedModel = None, **kwargs):
        """
        Runs the trace on Neuron.
        """

        if not isinstance(model, PreTrainedModel):
            raise ValueError(f"Model should be of type PreTrainedModel, got type {type(model)}")

        if kwargs.get("images") is not None:
            processor = self.load_processor()
            tokenizer = processor.tokenizer
            kwargs["image_processor"] = processor.image_processor
        else:
            tokenizer = self.load_tokenizer()

        if len(prompts) != model.neuron_config.max_batch_size:
            raise ValueError(f"Number of prompts should match batch size {model.neuron_config.max_batch_size}")

        max_length = kwargs.pop("max_length", model.neuron_config.max_length)
        if (max_length > model.neuron_config.max_length):
            ValueError(f"Found user supplied {max_length=} exceeds the compiled model sequence_length={model.neuron_config.max_length}")

        with self.torch_profile(chrome_trace_path="generate-on-neuron.torch-trace.json"):
            outputs, output_tokens = self.generate(
                model, tokenizer, prompts, max_length, draft_model, **kwargs
            )
        model.reset()
        if draft_model is not None:
            draft_model.reset()
        return outputs, output_tokens

    def generate_on_cpu(self, prompts: List[str], batch_size: int, max_prompt_length: int, sequence_length: int, lora_config: LoraServingConfig=None, **kwargs):
        """
        Use generate_on_cpu to confirm the neuron wrapper is correct. If the wrapper works
        on CPU, then the trace should work too. If it does not, it indicates a problem with
        the trace itself.
        """
        if kwargs.get("quantized", False) is False:
            model = self.load_neuron_model_on_cpu(max_prompt_length, sequence_length, batch_size, lora_config, **kwargs)
        else:
            model = self.load_quantized_neuron_model_on_cpu(max_prompt_length, sequence_length, batch_size, lora_config)

        if kwargs.get("images"):
            processor = self.load_processor()
            tokenizer = processor.tokenizer
            kwargs["image_processor"] = processor.image_processor
        else:
            tokenizer = self.load_tokenizer()
        outputs, output_tokens = self.generate(model, tokenizer, prompts, sequence_length, **kwargs)
        model.reset()
        return outputs, output_tokens

    def generate(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        prompts: List[str],
        max_length: int,
        draft_model: PreTrainedModel = None,
        **kwargs
    ):
        set_seed(0)  # to avoid randomness in sampling if any
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")

        # If pixel_values is given, pass to the model
        # Else generate pixel_values from given images
        images = kwargs.pop("images", None)
        pixel_values = kwargs.get("pixel_values")
        if images is not None and pixel_values is None:
            image_processor = kwargs.pop("image_processor", None)
            assert image_processor is not None, "image_processor is required when passing images"
            pixel_values = image_processor(images, return_tensors="pt")["pixel_values"]
            kwargs["pixel_values"] = pixel_values

        for idx, input in enumerate(inputs["input_ids"]):
            logging.debug("tokenized input %s : %s", idx, tokenizer.decode(input))

        if draft_model is not None:
            kwargs.update({
                "assistant_model": draft_model,
                "do_sample": False
            })

        outputs = model.generate(
            inputs.input_ids,
            generation_config=self.generation_config,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            **kwargs,
        )

        if isinstance(outputs, SampleOutput.__args__):
            # Get token ids from output when return_dict_in_generate=True
            output_ids = outputs.sequences
        else:
            output_ids = outputs
        output_tokens = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return outputs, output_tokens

    def check_accuracy(
        self,
        traced_model: PreTrainedModel,
        batch_size: int,
        max_length: int,
        expected_token_ids: Optional[List] = None,
        on_cpu: bool = False,
        do_sample: bool = True,
        traced_draft_model: PreTrainedModel = None,
        speculation_length: int = 0,
        prompt: Optional[str] = None,
        image: Optional[ImageInput] = None,
        **kwargs,
    ):
        """
        Function to compare outputs from huggingface model and neuronx NxD model
        """
        if prompt is None:
            prompts = [TEST_PROMPT] * batch_size
        else:
            prompts = [prompt] * batch_size

        if image is not None:
            kwargs["images"] = [image] * batch_size

        tokenizer = self.load_tokenizer()

        if expected_token_ids is not None:
            outputs_expected = tokenizer.batch_decode(
                expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        else:
            # Generate goldens with HF on CPU
            expected_token_ids, outputs_expected = self.generate_with_hf(
                prompts, max_length, do_sample=do_sample, **kwargs
            )
        print(f"Expected output: {outputs_expected}")

        # Generate outputs with NxD
        print("Generating outputs with NxD")
        if on_cpu:
            max_prompt_length = kwargs.pop("max_prompt_length")
            output_token_ids, outputs_actual = self.generate_on_cpu(
                prompts,
                batch_size,
                max_prompt_length=max_prompt_length,
                sequence_length=max_length,
                **kwargs
            )
        else:
            output_token_ids, outputs_actual = self.generate_on_neuron(
                prompts, traced_model, traced_draft_model, do_sample=do_sample, max_length=max_length,  **kwargs
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

    def check_accuracy_logits(
        self,
        traced_model: PreTrainedModel,
        batch_size: int,
        max_length: int,
        expected_logits: torch.Tensor = None,
        divergence_difference_tol: float = 0.001,
        remove_shift: bool = True,
        tol_map: dict = None,
    ):
        if traced_model.neuron_config.on_device_sampling:
            raise ValueError("Logits validation is not supported with on-device sampling.")

        prompts = [TEST_PROMPT] * batch_size
        tokenizer = self.load_tokenizer()
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")

        if not expected_logits:
            # logit_validation assumes greedy sampling
            expected_outputs, _ = self.generate_with_hf(
                prompts, max_length, do_sample=False, output_logits=True, return_dict_in_generate=True,
            )
            expected_logits = torch.stack(expected_outputs.logits)
        expected_token_ids = expected_logits.argmax(dim=2).T
        expected_tokens = tokenizer.batch_decode(
            expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("Expected Output: ", expected_tokens, expected_token_ids)
        print("Expected Logits Shape: ", expected_logits.shape)

        def generate_logits(model, tokenizer, input_ids):
            prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            actual_outputs, actual_tokens = self.generate_on_neuron(
                prompt, traced_model, do_sample=True, output_logits=True, return_dict_in_generate=True,
                max_length=max_length
            )
            actual_logits = torch.stack(actual_outputs.logits)
            actual_token_ids = actual_logits.argmax(dim=2).T
            print("Actual Output: ", actual_tokens, actual_token_ids)
            print("Actual Logits Shape: ", actual_logits.shape)
            model.reset()
            return actual_logits

        generate_fn = partial(generate_logits, traced_model, tokenizer)
        passed, result, status_msg = logit_validation(inputs.input_ids,
                                                      generate_fn,
                                                      expected_logits,
                                                      divergence_difference_tol=divergence_difference_tol,
                                                      tol_map=tol_map,
                                                      pad_token_id=tokenizer.pad_token_id,
                                                      padding_side=tokenizer.padding_side)
        assert passed, status_msg
        print("Passed logits validation")

    def trace(
        self,
        traced_model_path,
        tp_degree,
        batch_size,
        max_prompt_length=128,
        sequence_length=256,
        enable_bucketing=True,
        **kwargs,
    ):
        """
        Function to trace a model with neuronx NxD
        """
        if traced_model_path is not None:
            if not os.path.exists(traced_model_path):
                os.makedirs(traced_model_path)

        # Write the model config into the traced_model_path
        hf_config = self.get_hf_config(sequence_length=sequence_length, **kwargs)
        if hf_config.torch_dtype != torch.float32 and hf_config.torch_dtype != torch.bfloat16:
            raise ValueError(
                f"Type {hf_config.torch_dtype} is not supported for this model at this time. Please choose float32 or bfloat16."
            )

        self.neuron_config = self.get_config_for_nxd(
            hf_config,
            batch_size,
            tp_degree,
            max_prompt_length,
            sequence_length,
            enable_bucketing,
            **kwargs,
        )

        # Write the model config into the traced_model_path
        self.neuron_config.save(traced_model_path)

        # Copy the tokenizer into the traced_model_path
        tokenizer = self.load_tokenizer()
        if tokenizer:
            tokenizer.save_pretrained(traced_model_path)

        model = self.get_model_cls().from_pretrained(self.model_path, self.neuron_config)

        model.compile(serialize_base_path=traced_model_path)

    def benchmark_sampling(self, model: PreTrainedModel, draft_model: PreTrainedModel = None, target: str = None, **kwargs):
        neuron_config = model.neuron_config
        tokenizer = self.load_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token

        target = target if target is not None else "all"

        report = {}

        # Benchmark E2E model
        if target in ["all", "e2e"]:
            if kwargs.get("image") is not None:
                kwargs["image_processor"] = self.load_image_processor()
            batch_encoding = self.get_sample_inputs(END_TO_END_MODEL, neuron_config, tokenizer, **kwargs)
            input_param = {
                "input_ids": batch_encoding["input_ids"],
                "attention_mask": batch_encoding["attention_mask"],
                "max_new_tokens": neuron_config.max_new_tokens,
                "top_k": 1,
                "do_sample": draft_model is None,
                "assistant_model": draft_model,
            }

            pixel_values = batch_encoding.pop("pixel_values", None)
            if pixel_values is not None:
                input_param["pixel_values"] = pixel_values

            if target == "all":
                latency_collectors = self.create_submodule_latency_collectors(model)

            # Register latency collectors after warm-up to avoid recording warm-up metrics.
            def register_latency_collectors():
                if target == "all":
                    self.register_latency_collectors(latency_collectors, model)

            e2e_benchmark = Benchmark(model.generate, input_param, preprocess_func=model.reset,
                                      post_warmup_func=register_latency_collectors)
            e2e_benchmark.run()
            report[END_TO_END_MODEL] = generate_report(e2e_benchmark.latency_list, neuron_config.max_length, neuron_config.max_batch_size)

            if target == "all":
                report.update(self.generate_submodule_reports(latency_collectors, neuron_config.max_length, neuron_config.max_batch_size))

        # Benchmark context encoding model only
        if target == "context_encode":
            input_param = self.get_sample_inputs(CONTEXT_ENCODING_MODEL, neuron_config, **kwargs)
            ctx_enc_benchmark = Benchmark(model.context_encoding_model, input_param, neuron_config)
            ctx_enc_benchmark.run()
            report[CONTEXT_ENCODING_MODEL] = generate_report(ctx_enc_benchmark.latency_list, neuron_config.max_length, neuron_config.max_batch_size)

        # Benchmark token generation model only
        if hasattr(model, "token_generation_model") and target == "token_gen":
            input_param = self.get_sample_inputs(TOKEN_GENERATION_MODEL, neuron_config)
            tkn_gen_benchmark = Benchmark(model.token_generation_model, input_param)
            tkn_gen_benchmark.run()
            report[TOKEN_GENERATION_MODEL] = generate_report(tkn_gen_benchmark.latency_list, neuron_config.max_length, neuron_config.max_batch_size)

        # Benchmark speculation model only
        if hasattr(model, "speculation_model") and target == "speculation":
            input_param = self.get_sample_inputs(SPECULATION_MODEL, neuron_config)
            spec_benchmark = Benchmark(model.speculation_model, input_param)
            spec_benchmark.run()
            report[SPECULATION_MODEL] = generate_report(spec_benchmark.latency_list, neuron_config.max_length, neuron_config.max_batch_size)

        # Benchmark Medusa speculation model
        if hasattr(model, "medusa_speculation_model") and target == "speculation":
            input_param = self.get_sample_inputs(MEDUSA_MODEL, neuron_config)
            spec_benchmark = Benchmark(model.medusa_speculation_model, input_param)
            spec_benchmark.run()
            report[MEDUSA_MODEL] = generate_report(spec_benchmark.latency_list, neuron_config.max_length, neuron_config.max_batch_size)

        model.reset()
        if draft_model is not None:
            draft_model.reset()

        print("Benchmark completed and its result is as following")
        print(json.dumps(report, indent=4))
        with open(BENCHMARK_REPORT_FILENAME, "w") as f:
            json.dump(report, f)
        print("Completed saving result to " + BENCHMARK_REPORT_FILENAME)

        return report

    def get_sample_inputs(self, model_type, neuron_config: NeuronConfig, tokenizer=None, prompt=None, **kwargs):
        max_length = neuron_config.max_length
        batch_size = neuron_config.batch_size
        num_medusa_heads = neuron_config.num_medusa_heads if neuron_config.num_medusa_heads else 4
        medusa_speculation_length = neuron_config.medusa_speculation_length if neuron_config.medusa_speculation_length else 64

        if prompt is None:
            prompt = TEST_PROMPT

        sample_inputs = None
        if model_type == END_TO_END_MODEL:
            sample_inputs = tokenizer(
                [prompt] * batch_size,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            image = kwargs.get("image")
            if image is not None:
                images = [image] * batch_size
                sample_inputs["pixel_values"] = kwargs.get("image_processor")(images, return_tensors="pt")["pixel_values"]

        elif model_type == CONTEXT_ENCODING_MODEL:
            batch_size = neuron_config.ctx_batch_size
            input_ids = torch.zeros((batch_size, max_length), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, max_length), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, max_length), dtype=torch.int64)
            seq_ids = torch.zeros((batch_size), dtype=torch.int64)

            if neuron_config.is_medusa:
                accepted_indices = torch.zeros((batch_size, num_medusa_heads + 1), dtype=torch.int64)
                current_length = torch.zeros((batch_size, num_medusa_heads + 1), dtype=torch.int64)
                medusa_mask = torch.zeros(
                    (batch_size, medusa_speculation_length, medusa_speculation_length), dtype=torch.int64
                )
                scatter_index = torch.zeros((batch_size, medusa_speculation_length), dtype=torch.int64)
                sample_inputs = (
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    accepted_indices,
                    current_length,
                    medusa_mask,
                    scatter_index,
                )
            elif kwargs.get("image") is not None:
                pixel_values = torch.zeros((batch_size, 3, neuron_config.hf_config.vision_config.image_size, neuron_config.hf_config.vision_config.image_size), dtype=neuron_config.hf_config.torch_dtype)
                text_embedding_indices = torch.zeros((self.config.batch_size, max_length), dtype=torch.int64)
                image_embedding_indices = torch.zeros((self.config.batch_size, max_length), dtype=torch.int64)

                sample_inputs = (
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    pixel_values,
                    text_embedding_indices,
                    image_embedding_indices
                )
            else:
                sample_inputs = (
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                )
        elif model_type == TOKEN_GENERATION_MODEL:
            batch_size = neuron_config.tkg_batch_size
            input_ids = torch.zeros((batch_size, 1), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, max_length), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, 1), dtype=torch.int64)
            seq_ids = torch.zeros((batch_size), dtype=torch.int64)
            sample_inputs = (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
            )
        elif model_type == SPECULATION_MODEL:
            spec_len = neuron_config.speculation_length
            input_ids = torch.zeros((batch_size, spec_len), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, max_length), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, spec_len), dtype=torch.int64)
            seq_ids = torch.zeros((batch_size), dtype=torch.int64)
            sample_inputs = (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
            )

        elif model_type == MEDUSA_MODEL:
            spec_len = neuron_config.medusa_speculation_length
            input_ids = torch.zeros((batch_size, spec_len), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, max_length), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, spec_len), dtype=torch.int64)
            seq_ids = torch.zeros((batch_size), dtype=torch.int64)
            accepted_indices = torch.zeros((batch_size, num_medusa_heads + 1), dtype=torch.int64)
            current_length = torch.zeros((batch_size, num_medusa_heads + 1), dtype=torch.int64)
            medusa_mask = torch.zeros(
                (batch_size, medusa_speculation_length, medusa_speculation_length), dtype=torch.int64
            )
            scatter_index = torch.zeros((batch_size, medusa_speculation_length), dtype=torch.int64)
            sample_inputs = (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                accepted_indices,
                current_length,
                medusa_mask,
                scatter_index,
            )

        elif model_type == IMAGE_ENCODING_MODEL:
            image = kwargs.get("image")
            if image is not None:
                images = [image] * batch_size
                image_processor = self.load_processor()
                pixel_values = image_processor(images, return_tensors="pt")["pixel_values"]
            else:
                pixel_values = torch.zeros(
                    (
                        batch_size,
                        3,   # color images
                        neuron_config.hf_config.image_size,
                        neuron_config.hf_config.image_size
                    ),
                    dtype=neuron_config.hf_config.torch_dtype
                )
            sample_inputs = (
                pixel_values,
            )

        return sample_inputs

    def create_submodule_latency_collectors(self, model):
        collectors = {}
        collectors[CONTEXT_ENCODING_MODEL] = LatencyCollector()
        if hasattr(model, "token_generation_model"):
            collectors[TOKEN_GENERATION_MODEL] = LatencyCollector()
        if hasattr(model, "speculation_model"):
            collectors[SPECULATION_MODEL] = LatencyCollector()
        return collectors

    def register_latency_collectors(self, latency_collectors, model):
        self.register_forward_latency_collector(latency_collectors[CONTEXT_ENCODING_MODEL],
                                                model.context_encoding_model)
        if TOKEN_GENERATION_MODEL in latency_collectors:
            self.register_forward_latency_collector(latency_collectors[TOKEN_GENERATION_MODEL],
                                                    model.token_generation_model)
        if SPECULATION_MODEL in latency_collectors:
            self.register_forward_latency_collector(latency_collectors[SPECULATION_MODEL], model.speculation_model)

    def register_forward_latency_collector(self, latency_collector, model):
        model.register_forward_pre_hook(latency_collector.pre_hook)
        model.register_forward_hook(latency_collector.hook)

    def generate_submodule_reports(self, latency_collectors, max_length, max_batch_size):
        return {key : generate_report(collector.latency_list, max_length, max_batch_size) for key, collector in latency_collectors.items()}
