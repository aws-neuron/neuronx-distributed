from typing import List, Union
import json
import os
import torch


class LoraServingConfig:
    def __init__(
            self,
            max_loras: int = 1,
            max_lora_rank: int = 16,
            max_loras_on_cpu: int = 2,
            lora_dtype = torch.float32,
            target_modules: List[str] = None,
            lora_bias: str = "none",
            lora_ckpt_paths: List[str] = None):
        # The maximum number of concurrent LoRA adapters in device memory
        self.max_loras = max_loras
        # The highest LoRA rank that needs to be supported
        self.max_lora_rank = max_lora_rank
        # The maximum number of LoRA adapters stored in CPU memory
        self.max_loras_on_cpu = max_loras_on_cpu
        # The data type for LoRA adapters
        self.lora_dtype = lora_dtype
        # List of module names or regex expression of the module names to replace with LoRA.
        self.target_modules = target_modules
        # Bias type for LoRA. Can be 'none', 'all'
        self.lora_bias = lora_bias
        # List of checkpoint paths for LoRA adapters
        self.lora_ckpt_paths = lora_ckpt_paths


    def to_json_file(self, json_file: Union[str, os.PathLike]):
        with open(json_file, "w", encoding="utf-8") as writer:
            config_json = self.to_json_string()
            writer.write(config_json + "\n")


    def to_json_string(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike], **kwargs) -> "LoraServingConfig":
        if not os.path.exists(json_file):
            return None

        with open(json_file, "r", encoding="utf-8") as reader:
            neuron_config = cls.from_json_string(reader.read(), **kwargs)
            return neuron_config

    @classmethod
    def from_json_string(cls, json_string: str, **kwargs) -> "LoraServingConfig":
        merged_kwargs = json.loads(json_string)
        merged_kwargs.update(kwargs)
        return cls(**merged_kwargs)
