import argparse
import logging
import os
import pprint
import json
from functools import partial

import torch
from transformers import AutoTokenizer, PreTrainedModel
from neuronx_distributed.trace import parallel_model_save, parallel_model_trace
from benchmark import Benchmark, BENCHMARK_REPORT_FILENAME

END_TO_END_MODEL = "e2e_model"
CONTEXT_ENCODING_MODEL = "context_encoding_model"
TOKEN_GENERATION_MODEL = "token_generation_model"
LM_HEAD_NAME = "lm_head.pt"

BASE_COMPILER_WORK_DIR = "tmp/nxd_model/"
CTX_ENC_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + CONTEXT_ENCODING_MODEL + "/"
TKN_GEN_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + TOKEN_GENERATION_MODEL + "/"

class InferenceRunner:
    """
    Use the runner class to trace the model and perform inference.

    Usage:
        trace - Traces the neuron wrapper
        infer - Runs the traced model on Neuron
        infer-on-cpu - Runs the neuron wrapper on CPU
        infer-with-hf - Runs inference with huggingface model on CPU
    """

    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    def load_hf_model(self):
        # Implement per model
        raise NotImplementedError

    def load_neuron_model_on_cpu(self, max_context_length, max_new_tokens, batch_size):
        # Implement per model
        raise NotImplementedError

    def load_neuron_model(self, traced_model_path):
        # Implement per model
        raise NotImplementedError

    def load_tokenizer(self):
        # Implement per model
        raise NotImplementedError

    def get_config_cls(self):
        # Implement per model
        raise NotImplementedError

    def get_model_cls(self):
        # Implement per model
        raise NotImplementedError

    def get_trace_callable(self):
        raise NotImplementedError

    def generate_with_hf(self, prompt, max_context_length: int, max_new_tokens: int):
        """
        Use this to generate CPU goldens against which the trace is validated.
        """
        model = self.load_hf_model()
        tokenizer = self.load_tokenizer()
        return self.generate(model, tokenizer, prompt, max_context_length, max_new_tokens)

    def generate_on_neuron(self, prompt, traced_model_path: str):
        """
        Runs the trace on Neuron.
        """

        if traced_model_path is None:
            raise ValueError("Set --traced_model_path to save the trace")

        self.tokenizer_path = traced_model_path
        tokenizer = self.load_tokenizer()
        model = self.load_neuron_model(traced_model_path)

        if len(prompt) != model.config.batch_size:
            raise ValueError(f"Number of prompts should match batch size {model.config.batch_size}")

        generate_ids, outputs = self.generate(
            model, tokenizer, prompt, model.config.max_context_length, model.config.max_new_tokens
        )
        model.reset()
        return generate_ids, outputs

    def generate_on_cpu(self, prompt: str, batch_size: int, max_context_length: int, max_new_tokens: int):
        """
        Use generate_on_cpu to confirm the neuron wrapper is correct. If the wrapper works
        on CPU, then the trace should work too. If it does not, it indicates a problem with
        the trace itself.
        """
        model = self.load_neuron_model_on_cpu(max_context_length, max_new_tokens, batch_size)
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
    ):
        max_length = max_context_length + max_new_tokens

        inputs = tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        for idx, input in enumerate(inputs["input_ids"]):
            logging.debug("padded tokenized input %s : %s", idx, tokenizer.decode(input))

        generate_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            top_k=1,
            do_sample=True,
        )
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return generate_ids, outputs

    def check_accuracy(self, traced_model_path, batch_size, max_context_length, max_new_tokens):
        """
        Function to compare outputs from huggingface model and neuronx NxD model
        """
        prompt = ["I believe the meaning of life is"] * batch_size
        generate_ids_actual, outputs_actual = self.generate_on_neuron(prompt, traced_model_path)
        generate_ids_expected, outputs_expected = self.generate_with_hf(prompt, max_context_length, max_new_tokens)

        torch.testing.assert_close(generate_ids_actual, generate_ids_expected)
        print("The output from Neuronx NxD is accurate!")

    def trace(self, traced_model_path, tp_degree, batch_size, max_context_length, max_new_tokens, inline_weights_to_neff):
        """
        Function to trace a model with neuronx NxD
        """
        hf_model = self.load_hf_model()
        config_cls = self.get_config_cls()

        # Save the model state dict. This will be used
        # load the state in each parallel shard when tracing
        saved_dir = self.model_path
        model_save_path = saved_dir + "model.pt"
        if not os.path.exists(model_save_path):
            # Saves the model. This model is then loaded
            # into the parallel NxD layers
            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir)
            model_sd = hf_model.model.state_dict()
            lm_head_sd = hf_model.lm_head.state_dict()
            model_sd["lm_head.weight"] = lm_head_sd["weight"]
            torch.save({"model": model_sd}, model_save_path)

        if traced_model_path is not None:
            if not os.path.exists(traced_model_path):
                os.makedirs(traced_model_path)

        # Write the model config into the traced_model_path
        config = config_cls.from_pretrained(self.model_path)
        config.tp_degree = tp_degree
        max_length = max_context_length + max_new_tokens
        config.max_length = max_length
        config.max_context_length = max_context_length
        config.max_new_tokens = max_new_tokens
        config.batch_size = batch_size

        if (config.torch_dtype != torch.float32 and config.torch_dtype != torch.bfloat16):
            raise ValueError(f"Type {config.torch_dtype} is not supported for this model at this time. Please choose float32 or bfloat16.")


        config.save_pretrained(traced_model_path)
        config_path = traced_model_path  # We have the config in the trace_model_path

        # Copy the tokenizer into the traced_model_path
        self.load_tokenizer().save_pretrained(traced_model_path)

        trace_callable = self.get_trace_callable()
        callable = partial(trace_callable, config_path, model_save_path)

        # Trace the context encoding model
        sample_inputs = self.get_sample_inputs(CONTEXT_ENCODING_MODEL, config)

        traced_model = parallel_model_trace(
            callable,
            sample_inputs,
            tp_degree=tp_degree,
            compiler_workdir=CTX_ENC_MODEL_COMPILER_WORK_DIR,
            compiler_args="--enable-saturate-infinity --auto-cast=none",
            max_parallel_compilations=8,
            inline_weights_to_neff=inline_weights_to_neff
        )

        ctx_encoding_path = traced_model_path + CONTEXT_ENCODING_MODEL
        parallel_model_save(traced_model, ctx_encoding_path)
        print("Successfully traced the context encoding model!")

        # Trace the token generation model
        sample_inputs = self.get_sample_inputs(TOKEN_GENERATION_MODEL, config)

        traced_model = parallel_model_trace(
            callable,
            sample_inputs,
            tp_degree=tp_degree,
            compiler_workdir=TKN_GEN_MODEL_COMPILER_WORK_DIR,
            compiler_args="--enable-saturate-infinity --auto-cast=none",
            max_parallel_compilations=8,
            inline_weights_to_neff=inline_weights_to_neff
        )
        tkn_gen_path = traced_model_path + TOKEN_GENERATION_MODEL
        parallel_model_save(traced_model, tkn_gen_path)

        print("Successfully traced the token generation model!")

    def benchmark_sampling(self, traced_model_path):
        config_cls = self.get_config_cls()
        config = config_cls.from_pretrained(traced_model_path)
        tokenizer = self.load_tokenizer()
        model = self.load_neuron_model(traced_model_path)

        report = {}

        # Benchmark E2E model
        batch_encoding = self.get_sample_inputs(END_TO_END_MODEL, config, tokenizer)
        input_param = {
            "input_ids": batch_encoding["input_ids"], 
            "attention_mask": batch_encoding["attention_mask"], 
            "max_new_tokens": config.max_new_tokens, 
            "top_k": 1,
            "do_sample": True,
        }
        e2e_benchmark = Benchmark(model.generate, input_param, config, 
                                  preprocess_func=model.reset)
        report[END_TO_END_MODEL] = e2e_benchmark.run()

        # Benchmark context encoding model
        input_param = self.get_sample_inputs(CONTEXT_ENCODING_MODEL, config)
        ctx_enc_benchmark = Benchmark(model.context_encoding_model, input_param, config)
        report[CONTEXT_ENCODING_MODEL] = ctx_enc_benchmark.run()

        # Benchmark token generation model
        input_param = self.get_sample_inputs(TOKEN_GENERATION_MODEL, config)
        tkn_gen_benchmark = Benchmark(model.token_generation_model, input_param, config)
        report[TOKEN_GENERATION_MODEL] = tkn_gen_benchmark.run()

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
                return_tensors="pt"
            )
        
        elif model_type == CONTEXT_ENCODING_MODEL:
            input_ids = torch.zeros((batch_size, max_length), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, max_length), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, max_length), dtype=torch.int64)
            sample_inputs = (input_ids, attention_mask, position_ids)

        elif model_type == TOKEN_GENERATION_MODEL:
            input_ids = torch.zeros((batch_size, 1), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, max_length + 1), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, 1), dtype=torch.int64)
            sample_inputs = (input_ids, attention_mask, position_ids)
        
        return sample_inputs


    @classmethod
    def cmd_execute(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "action",
            choices=["trace", "infer", "infer-on-cpu", "infer-with-hf", "benchmark-sampling", "check-accuracy"],
        )
        parser.add_argument(
            "--model",
            required=True,
            type=str,
            help="Model architecture ex. LlamaForCausalLM",
        )
        parser.add_argument(
            "--model_path",
            type=str,
            help="Model path.",
        )
        parser.add_argument(
            "--tokenizer_path",
            type=str,
            help="Tokenizer path.",
        )
        parser.add_argument("-p", "--prompt", action="append", help="prompt for generation", default=[])
        parser.add_argument("--traced_model_path", type=str, help="directory to save the traced model", default=None)
        parser.add_argument(
            "-d",
            "--debug",
            action="store_true",
            help="print debugging statements",
        )
        parser.add_argument(
            "--tp_degree",
            type=int,
            help="tensor parallel degree",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            help="batch size of inference",
        )
        parser.add_argument(
            "--max_context_length",
            type=int,
            help="max context length output",
        )
        parser.add_argument(
            "--max_new_tokens",
            type=int,
            help="max output tokens generated",
        )
        parser.add_argument(
            "--enable_weight_separation",
            action="store_true",
            help="Enables weight separation for tp models"
        )
        parser.set_defaults(enable_weight_separation=False)
        args = parser.parse_args()

        action = args.action
        model_architecture = args.model
        model_path = args.model_path
        tokenizer_path = args.tokenizer_path
        prompt = args.prompt
        traced_model_path = args.traced_model_path
        tp_degree = args.tp_degree
        batch_size = args.batch_size
        max_context_length = args.max_context_length
        max_new_tokens = args.max_new_tokens
        inline_weights_to_neff = not args.enable_weight_separation

        if args.debug:
            from imp import reload

            reload(logging)
            logging.basicConfig(level=logging.DEBUG)

        pp = pprint.PrettyPrinter(indent=2, depth=2)
        print(f"Running {args.action} with arguments: \n {pp.pformat(vars(args))}")

        if model_architecture == "LlamaForCausalLM":
            from llama2.llama2_runner import LlamaRunner

            cls = LlamaRunner

        runner = cls(model_path=model_path, tokenizer_path=tokenizer_path)

        if action == "trace":
            assert model_path != None, "Required parameter --model_path not passed"
            assert tokenizer_path != None, "Required parameter --tokenizer_path not passed"
            assert traced_model_path != None, "Required parameter --traced_model_path not passed"
            assert batch_size != None, "Required parameter --batch_size not passed"
            assert max_context_length != None, "Required parameter --max_context_length not passed"
            assert max_new_tokens != None, "Required parameter --max_new_tokens not passed"
            assert tp_degree != None, "Required parameter --tp_degree not passed"
            runner.trace(traced_model_path, tp_degree, batch_size, max_context_length, max_new_tokens, inline_weights_to_neff)

        elif action == "infer":
            assert prompt != None and len(prompt) > 0, "Required parameter --prompt not passed with valid values"
            assert traced_model_path != None, "Required parameter --traced_model_path not passed"
            _, outputs = runner.generate_on_neuron(prompt, traced_model_path)
            print("Generated outputs:")
            for idx, output in enumerate(outputs):
                print(f"output {idx}: {output}")

        elif action == "infer-on-cpu":
            assert prompt != None and len(prompt) != 0, "Required parameter --prompt not passed"
            assert batch_size != None, "Required parameter --batch_size not passed"
            assert max_context_length != None, "Required parameter --max_context_length not passed"
            assert max_new_tokens != None, "Required parameter --max_new_tokens not passed"
            _, outputs = runner.generate_on_cpu(prompt, batch_size, max_context_length, max_new_tokens)
            print("Generated outputs:")
            for idx, output in enumerate(outputs):
                print(f"output {idx}: {output}")

        elif action == "infer-with-hf":
            assert prompt != None and len(prompt) != 0, "Required parameter --prompt not passed"
            assert max_context_length != None, "Required parameter --max_context_length not passed"
            assert max_new_tokens != None, "Required parameter --max_new_tokens not passed"
            _, outputs = runner.generate_with_hf(prompt, max_context_length, max_new_tokens)
            print("Generated outputs:")
            for idx, output in enumerate(outputs):
                print(f"output {idx}: {output}")

        elif action == "benchmark-sampling":
            assert traced_model_path != None, "Required parameter --traced_model_path not passed"
            runner.benchmark_sampling(traced_model_path)

        elif action == "check-accuracy":
            assert model_path != None, "Required parameter --model_path not passed"
            assert tokenizer_path != None, "Required parameter --tokenizer_path not passed"
            assert traced_model_path != None, "Required parameter --traced_model_path not passed"
            assert batch_size != None, "Required parameter --batch_size not passed"
            assert max_context_length != None, "Required parameter --max_context_length not passed"
            assert max_new_tokens != None, "Required parameter --max_new_tokens not passed"
            runner.check_accuracy(traced_model_path, batch_size, max_context_length, max_new_tokens)


if __name__ == "__main__":
    InferenceRunner.cmd_execute()
