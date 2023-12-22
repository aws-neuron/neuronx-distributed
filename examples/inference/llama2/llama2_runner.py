import os
import torch

import neuronx_distributed
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM

from runner import InferenceRunner, CONTEXT_ENCODING_MODEL_NAME, TOKEN_GENERATION_MODEL_NAME, LM_HEAD_NAME
from llama2.neuron_modeling_llama import NeuronLlamaForCausalLM, NeuronLlamaModel, NeuronLlamaConfig
from neuronx_distributed.trace import parallel_model_load


def get_trace_callable(config_path,
                       model_path):

    config = NeuronLlamaConfig.from_pretrained(config_path)
    model = NeuronLlamaModel(config)
    neuronx_distributed.parallel_layers.load(model_path, 
                                             model_or_optimizer=model, 
                                             sharded=False, 
                                             strict=False)
    aliases = {}
    num_output_from_trace = 1
    for i in range(len(model.past_key_values)):
        aliases[model.past_key_values[i]] = num_output_from_trace + i
    return model, aliases


class LlamaRunner(InferenceRunner):

    def load_hf_model(self):
        return LlamaForCausalLM.from_pretrained(self.model_path)

    def load_neuron_model_on_cpu(self,
                                 max_context_length,
                                 max_new_tokens,
                                 batch_size):

        config = NeuronLlamaConfig.from_pretrained(self.model_path)
        config.tp_degree = 1  # On CPU we do not run tensor parallelism
        config.max_length = max_context_length + max_new_tokens
        config.max_context_length = max_context_length
        config.max_new_tokens = max_new_tokens
        config.batch_size = batch_size

        return NeuronLlamaForCausalLM.from_pretrained(self.model_path, config=config)

    def load_neuron_model(self,
                          traced_model_path):

        if not os.path.exists(traced_model_path + CONTEXT_ENCODING_MODEL_NAME):
            raise ValueError(
                "traced_model_path does not contain " + CONTEXT_ENCODING_MODEL_NAME)
        if not os.path.exists(traced_model_path + TOKEN_GENERATION_MODEL_NAME):
            raise ValueError(
                "traced_model_path does not contain " + TOKEN_GENERATION_MODEL_NAME)
        if not os.path.exists(traced_model_path + LM_HEAD_NAME):
            raise ValueError(
                "traced_model_path does not contain " + LM_HEAD_NAME)

        config = NeuronLlamaConfig.from_pretrained(traced_model_path)
        context_encoding_model = parallel_model_load(
            traced_model_path + CONTEXT_ENCODING_MODEL_NAME)
        token_generation_model = parallel_model_load(
            traced_model_path + TOKEN_GENERATION_MODEL_NAME)
        model = NeuronLlamaForCausalLM(config,
                                       context_encoding_model,
                                       token_generation_model)

        model.lm_head.load_state_dict(
            torch.load(traced_model_path + LM_HEAD_NAME))
        return model

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
        return tokenizer

    def get_config_cls(self):
        return NeuronLlamaConfig

    def get_model_cls(self):
        return NeuronLlamaForCausalLM

    def get_trace_callable(self):
        return get_trace_callable


if __name__ == "__main__":
    LlamaRunner.cmd_execute()
