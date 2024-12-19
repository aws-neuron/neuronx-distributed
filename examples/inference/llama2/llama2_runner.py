import torch
from modules.config import NeuronConfig
from llama2.neuron_modeling_llama import (
    NeuronLlamaForCausalLM,
    NeuronLlamaModel,
)
from runner import InferenceRunner
from transformers import AutoTokenizer

from neuronx_distributed.parallel_layers.checkpointing import _invoke_preshard_hook
from modules.lora_serving import LoraModel

class LlamaRunner(InferenceRunner):
    def load_hf_model(self):
        return NeuronLlamaForCausalLM.load_hf_model(self.model_path)

    def load_neuron_model_on_cpu(self, max_prompt_length, sequence_length, batch_size, lora_config=None, **kwargs):
        # Create new configs with different properties to run on CPU.
        hf_config = self.get_hf_config(sequence_length=sequence_length, **kwargs)
        neuron_config = self.get_config_for_nxd(
            hf_config,
            batch_size,
            1,
            max_prompt_length=max_prompt_length,
            sequence_length=sequence_length,
            enable_bucketing=False,
            lora_config=lora_config,
            **kwargs)
        hf_config.torch_dtype = torch.float32

        neuron_model = NeuronLlamaModel(neuron_config)

        state_dict = NeuronLlamaForCausalLM.get_state_dict(self.model_path, neuron_config)
        if neuron_config.lora_config is not None:
            # enable LoRA
            neuron_model = LoraModel(neuron_model, neuron_config.lora_config)
            neuron_model = neuron_model.module
            _invoke_preshard_hook(neuron_model, state_dict)
            neuron_model.update_weights_for_lora(state_dict)
        else:
            _invoke_preshard_hook(neuron_model, state_dict)

        if hf_config.torch_dtype == torch.bfloat16:
            neuron_model.bfloat16()

        model = NeuronLlamaForCausalLM(None, neuron_config)
        model.context_encoding_model.model = neuron_model
        model.token_generation_model.model = neuron_model
        return model

    def load_tokenizer(self, padding_side=None):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        # Use eos_token as pad_token which works for both llama2 and llama3
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = padding_side if padding_side else self.get_padding_side()
        return tokenizer

    def get_config_cls(self):
        return NeuronConfig

    def get_model_cls(self):
        return NeuronLlamaForCausalLM

    def get_padding_side(self):
        return "right"

    def get_default_hf_generation_config_kwargs(self):
        config = super().get_default_hf_generation_config_kwargs()
        # set to eos_token_id as that's done in load_tokenizer
        config['pad_token_id'] = self.generation_config.eos_token_id

        return config


if __name__ == "__main__":
    LlamaRunner.cmd_execute()
