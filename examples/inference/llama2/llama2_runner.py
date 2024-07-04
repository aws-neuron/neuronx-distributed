import os

import torch
from llama2.neuron_modeling_llama import (
    NeuronLlamaConfig,
    NeuronLlamaForCausalLM,
    NeuronLlamaModel,
    preshard_hook_fn,
)
from runner import InferenceRunner
from transformers import AutoTokenizer

from neuronx_distributed.parallel_layers.checkpointing import _invoke_preshard_hook
from neuronx_distributed.quantization.quantization_utils import (
    convert_float_model_to_pytorch_int8_model,
)


class LlamaRunner(InferenceRunner):
    def load_hf_model(self):
        return NeuronLlamaForCausalLM.load_hf_model(self.model_path)

    def load_neuron_model_on_cpu(self, max_context_length, max_new_tokens, batch_size, **kwargs):
        config = self.get_config_for_nxd(batch_size, 1, max_context_length, max_new_tokens, **kwargs)
        config.torch_dtype = torch.float32

        neuron_model = NeuronLlamaModel(config)

        state_dict = NeuronLlamaForCausalLM.get_state_dict(self.model_path, config=config)
        _invoke_preshard_hook(neuron_model, state_dict)

        neuron_model.load_state_dict(state_dict, strict=False)

        if config.torch_dtype == torch.bfloat16:
            neuron_model.bfloat16()

        model = NeuronLlamaForCausalLM(None, config)
        model.context_encoding_model.model = neuron_model
        model.token_generation_model.model = neuron_model
        return model

    def load_quantized_neuron_model_on_cpu(self, max_context_length, max_new_tokens, batch_size, **kwargs):
        model = self.load_neuron_model_on_cpu(max_context_length, max_new_tokens, batch_size, **kwargs)
        return convert_float_model_to_pytorch_int8_model(model, inplace=True)

    def load_neuron_model(self, traced_model_path):
        config = NeuronLlamaConfig.from_pretrained(traced_model_path)
        model = NeuronLlamaForCausalLM.from_pretrained("", config)

        model.load(traced_model_path)
        if config.torch_dtype == torch.bfloat16:
            os.environ["XLA_DOWNCAST_BF16"] = "1"

        return model

    def load_tokenizer(self, padding_side=None):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = padding_side if padding_side else self.get_padding_side()
        return tokenizer

    def get_config_cls(self):
        return NeuronLlamaConfig

    def get_model_cls(self):
        return NeuronLlamaForCausalLM

    def get_padding_side(self):
        return "right"


if __name__ == "__main__":
    LlamaRunner.cmd_execute()
