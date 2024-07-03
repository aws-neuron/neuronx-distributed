import os
import tempfile
import warnings

import torch
from transformers import PretrainedConfig

from neuronx_distributed.quantization.quantization_utils import (
    convert_float_model_to_pytorch_int8_model,
    convert_qint8_to_int8_state_dict,
)
from neuronx_distributed.utils.speculative_decoding import NeuronSpeculation
from modules.checkpoint import load_state_dict



class NeuronBaseForCausalLM(NeuronSpeculation):

    _STATE_DICT_MODEL_PREFIX = "model."

    @staticmethod
    def load_hf_model(model_path):
        raise NotImplementedError(f"load_hf_model is not implemented")

    @classmethod
    def get_state_dict(cls, model_path: str, config: PretrainedConfig) -> dict:
        model_sd = load_state_dict(model_path)
        param_name_list = list(model_sd.keys())
        for param_name in param_name_list:
            if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                updated_param_name = param_name.replace(cls._STATE_DICT_MODEL_PREFIX, "", 1)
                model_sd[updated_param_name] = model_sd[param_name]
                del model_sd[param_name]
        return model_sd

    @classmethod
    def get_quantized_state_dict(cls, model_path: str, config: PretrainedConfig) -> dict:
        hf_model = cls.load_hf_model(model_path)
        hf_model_quant = convert_float_model_to_pytorch_int8_model(float_model=hf_model)

        model_quant_sd = hf_model_quant.model.state_dict()
        lm_head_quant_sd = hf_model_quant.lm_head.state_dict()
        convert_qint8_to_int8_state_dict(model_quant_sd)
        convert_qint8_to_int8_state_dict(lm_head_quant_sd)

        model_quant_sd["lm_head.weight"] = lm_head_quant_sd["weight"]
        model_quant_sd["lm_head.scale"] = lm_head_quant_sd["scale"]

        return model_quant_sd

    @classmethod
    def from_pretrained(cls, model_path: str, config: PretrainedConfig):
        return cls(model_path, config)

    def checkpoint_loader_fn(self, mmap: bool = False):
        # this function loads the model's state dcitionary and weights from
        # the hf model
        if self.config.quantized is False:
            model_sd = self.get_state_dict(self.model_path, self.config)
            if self.config.torch_dtype == torch.bfloat16:
                for name, param in model_sd.items():
                    model_sd[name] = param.bfloat16()
            return model_sd
        return self.get_quantized_checkpoints()

    def get_quantized_checkpoints(self, mmap: bool = False):
        # this function loads the checkpointed float model state dictionary and weights
        # from the quantized hf model
        # This will be removed once we move to safe tensors in NxD
        existing_checkpoint_path = self.config.quantized_checkpoints_path
        if not os.path.exists(existing_checkpoint_path):
            raise FileNotFoundError(f"Quantized checkpoint file not found: {existing_checkpoint_path}")

        print(f"Using existing checkpoint: {existing_checkpoint_path}")
        model_quant_sd = torch.load(existing_checkpoint_path)

        # Make sure that the non quantized weights are in bfloat16 and not float32
        if self.config.torch_dtype == torch.bfloat16:
            for name, param in model_quant_sd.items():
                if param is not None and param.dtype == torch.float32:
                    warnings.warn(f"Found float32 weights in quantized checkpoint: {name}. Will convert to bfloat16")
                    model_quant_sd[name] = param.bfloat16()

        return model_quant_sd

    def compile(self, serialize_base_path=None):
        if serialize_base_path:
            self.config.save_pretrained(serialize_base_path)
        for model in self.models:
            model.compile(self.checkpoint_loader_fn, serialize_base_path=serialize_base_path)

    def load(self, serialize_base_path):
        for model in self.models:
            model.load(serialize_base_path)

    def to_neuron(self, serialize_base_path=None):
        if serialize_base_path is None:
            with tempfile.TemporaryDirectory(suffix="nxd-temp-serial-path") as tmpdirname:
                self.compile(tmpdirname)
                self.load(tmpdirname)
        else:
            self.compile(serialize_base_path)
            self.load(serialize_base_path)
