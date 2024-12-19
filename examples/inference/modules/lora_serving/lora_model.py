from __future__ import annotations

import re
import torch

from .config import LoraServingConfig
from .lora_module import (
    MultiLoraModuleLinear,
    MultiLoraModuleEmbedding,
    MultiLoraModuleConv2d,
)


class LoraModel(torch.nn.Module):
    prefix: str = "lora_"

    def __init__(self, module, config: LoraServingConfig) -> None:
        assert config is not None
        super().__init__()

        self.module = module
        self.lora_config = config
        self.adapter_ids = None
        self.inject_adapter()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.module.generate(*args, **kwargs)

    def inject_adapter(self):
        r"""
        Creates adapter layers and replaces the target modules with the adapter layers.
        It involves the following four steps:
            Step 1: set the list of target modules rules in wildcard for LoRA injection
            Step 2: For each module in the base model, check if it matches any target module rules. If so
            Step 3: Create a LoraLayer for this module and replace it with the LoraLayer
        """
        lora_config = self.lora_config
        config = self.lora_config
        if config.target_modules is None:
            raise ValueError(
                "Target modules are not set for the base model."
            )

        is_target_modules_in_base_model = False
        key_list = self.get_leave_module_names()

        for key in key_list:
            if not self._check_target_module_exists(key):
                continue
            is_target_modules_in_base_model = True
            parent, target, target_name = self._get_submodules(key)
            self._create_and_replace(target, target_name, parent, current_key=key)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )


    def get_leave_module_names(self):
        key_list = [key for key, _ in self.module.named_modules()]
        key_list = sorted(key_list, key=len, reverse=True)
        result = []
        for s in key_list:
            if not any(other_s.startswith(s) for other_s in result):
                result.append(s)
        return result


    def _get_submodules(self, key):
        module = self.module
        target_name = key.split(".")[-1]
        parent = module.get_submodule(".".join(key.split(".")[:-1]))
        target = module.get_submodule(key)
        return parent, target, target_name


    def _check_target_module_exists(self, key):
        r"""A helper method to check if the passed module's key name matches any of the target modules.

        Args:
            key (`str`): A key to search any matches in config

        Returns:
            `bool` | `re.Match[str]` | `None`: True of match object if key matches any target modules from config, False or
            None if no match found
        """
        config = self.lora_config
        if isinstance(config.target_modules, str):
            target_module_found = re.fullmatch(config.target_modules, key)
        elif key in config.target_modules:
            # this module is specified directly in target_modules
            target_module_found = True
        else:
            target_module_found = any(key.endswith(f".{target_key}") for target_key in config.target_modules)

        return target_module_found


    def _create_and_replace(
        self,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        new_module = self._create_new_module(target)
        self._replace_module(parent, target_name, new_module, target)


    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state


    def _create_new_module(self, target):
        r"""
        Create the corresponding LoraLayer according to its module type, such as torch.nn.Linear and torch.nn.Embedding.
        """
        lora_config = self.lora_config
        # check basic module types
        if isinstance(target, (torch.nn.Embedding)):
            lora_adapters = MultiLoraModuleEmbedding(target, lora_config)
        elif isinstance(target, torch.nn.Linear):
            lora_adapters = MultiLoraModuleLinear(target, lora_config)
        elif isinstance(target, torch.nn.Conv2d):
            lora_adapters = MultiLoraModuleConv2d(target, lora_config)

        if lora_adapters is None:
            # no module could be matched
            raise ValueError(
                f"""Target module {target} is not supported. Currently, only the following modules are supported: "
                    torch.nn.Linear,
                    torch.nn.Embedding,
                    torch.nn.Conv2d,
                """
            )
        return lora_adapters


    def get_base_model(self) -> torch.nn.Module:
        """
        Returns the base model.
        """
        return self.module


    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)
