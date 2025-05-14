from __future__ import annotations

import json
import os
import re
from typing import Optional, Dict, Mapping, Any, Tuple, TYPE_CHECKING
from dataclasses import asdict

import torch
import torch_xla.core.xla_model as xm

from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
from neuronx_distributed.parallel_layers.parallel_state import model_parallel_is_initialized
from neuronx_distributed.parallel_layers.mappings import _gather_along_first_dim, _gather_along_last_dim
from neuronx_distributed.trainer.checkpoint import _get_path
from neuronx_distributed.trainer.checkpoint_storage import create_checkpoint_storage
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.utils.model_utils import is_hf_transformers_available

from .config import LoraConfig
from .layer import LoraConv2d, LoraEmbedding, LoraLayer, LoraLinear
from .tp_layer import LoraParallelLinear, LoraGQAQKVParallelLinear

if TYPE_CHECKING:
    import transformers

# The mapping is based on https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py

#  ------------------------------------------------------------------------------------------
#  Copyright 2023-present the HuggingFace Inc. team.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  ------------------------------------------------------------------------------------------

MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["c_attn"],
    "mpt": ["Wqkv"],
    "RefinedWebModel": ["query_key_value"],
    "RefinedWeb": ["query_key_value"],
    "falcon": ["query_key_value"],
    "btlm": ["c_proj", "c_attn"],
    "codegen": ["qkv_proj"],
    "mistral": ["q_proj", "v_proj"],
    "mixtral": ["q_proj", "v_proj"],
    "stablelm": ["q_proj", "v_proj"],
    "phi": ["q_proj", "v_proj", "fc1", "fc2"],
    "gemma": ["q_proj", "v_proj"],
}

logger = get_logger()
WEIGHTS_NAME = "adapter_model.pt"
CONFIG_NAME = "adapter_config.json"


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (LoRA) model from a pretrained model.

    The method is described in detail in https://arxiv.org/abs/2106.09685.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The model with LoRA adapter injected.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **lora_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    prefix: str = "lora_"
    GQAQKVParallelLinear_lora_type = "LoraGQAQKVParallelLinear"
    ColumnParallelLinear_lora_type = "LoraColumnParallelLinear"
    RowParallelLinear_lora_type = "LoraRowParallelLinear"

    def __init__(self, module: "transformers.PreTrainedModel", config: LoraConfig) -> None:
        assert config is not None
        super().__init__()

        self.module = module
        self.is_lora_merged = False
        self.is_config_saved = False
        self.is_generate_optimum = False
        self.is_verbose_enabled = config.lora_verbose
        self.modules_to_save = config.modules_to_save
        self.is_lora_enabled = False
        self.is_optimum_enabled = False
        self.is_checkpoint_loaded = False
        self.lora_config = config
        self.is_base_model_loaded = False
        self.lora_module_parallel_types: dict = {}
        self.lora_kv_size_multiplier = 1

        if config.load_lora_from_ckpt:
            self.load_checkpoint(config)
        else:
            self.inject_adapter()
            self.print_model_info()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        if self.is_lora_merged:
            self.unmerge_lora()

        return self.module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def _set_optimum_generate(self) -> None:
        if not self.is_optimum_enabled:
            try:
                from optimum.neuron.utils.training_utils import (
                    patch_generation_mixin_to_general_neuron_generation_mixin,
                )

                patch_generation_mixin_to_general_neuron_generation_mixin(self.module)
                self.is_optimum_enabled = True
            except Exception:
                raise ImportError("Failed to import optimum-neuron, generation will not work on Neuron.")

    def generate(self, *args, **kwargs):
        self._set_optimum_generate()
        return self.module.generate(*args, **kwargs)

    def inject_adapter(self) -> None:
        r"""
        Creates adapter layers and replaces the target modules with the adapter layers.
        It involves the following four steps:
            Step 1: set the list of target modules rules in wildcard for LoRA injection
            Step 2: For each module in the base model, check if it matches any target module rules. If so
            Step 3: Create a LoraLayer for this module and replace it with the LoraLayer
            Step 4: freeze the base model and only set the LoRA adapter as trainable
        """
        if self.is_lora_enabled:
            return

        lora_config = self.lora_config
        self._set_target_modules()
        is_target_modules_in_base_model = False
        model = self.module
        key_list = [key for key, _ in model.named_modules()]

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

        self._mark_only_adapters_as_trainable()
        self.is_lora_enabled = True

    def _get_submodules(self, key: str):
        module = self.module
        target_name = key.split(".")[-1]
        parent = module.get_submodule(".".join(key.split(".")[:-1]))
        target = module.get_submodule(key)
        return parent, target, target_name

    def _set_target_modules(self) -> None:
        config = self.lora_config
        if config.target_modules is not None:
            return

        model_type = "unknown"
        model_config = getattr(self.module, "config", None)
        if model_config is not None:
            model_type = getattr(model_config, "model_type", "unknown").lower()

        if model_type not in MODELS_TO_LORA_TARGET_MODULES_MAPPING:
            raise ValueError(
                f"The base model {model_type} is not supported to set target modules automatically. The supported model architectures include {MODELS_TO_LORA_TARGET_MODULES_MAPPING.keys()}"
            )
        else:
            self.lora_config.target_modules = MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_type]

    def _check_target_module_exists(self, key: str) -> bool:
        r"""A helper method to check if the passed module's key name matches any of the target modules.

        Args:
            key (`str`): A key to search any matches in config

        Returns:
            `bool` | `re.Match[str]` | `None`: True of match object if key matches any target modules from config, False or
            None if no match found
        """
        config = self.lora_config
        if not config.target_modules:
            return False
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
        target_name: str,
        parent,
        current_key,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        new_module = self._create_new_module(target, current_key)
        self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name: str, new_module, child) -> None:
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
            new_module.to("xla")

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if self.prefix in name:
                module.to("xla")

    def _mark_only_adapters_as_trainable(self) -> None:
        module = self.module
        for n, p in module.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        # if modules_to_save is specified, we also set these modules as trainable
        if self.modules_to_save is not None:
            for n, p in module.named_parameters():
                if any(target_key in n for target_key in self.modules_to_save):
                    p.requires_grad = True

        bias = self.lora_config.bias
        if bias == "none":
            return

        if bias == "all":
            for n, p in module.named_parameters():
                if "bias" in n:
                    p.requires_grad = True
        elif bias == "lora_only":
            for m in module.modules():
                if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    def _create_new_module(self, target, current_key):
        r"""
        Create the corresponding LoraLayer according to its module type, such as torch.nn.Linear and torch.nn.Embedding.
        """
        lora_config = self.lora_config
        new_module = None

        # check basic module types
        if isinstance(target, torch.nn.Embedding):
            new_module = LoraEmbedding(target, lora_config)
        elif isinstance(target, torch.nn.Conv2d):
            new_module = LoraConv2d(target, lora_config)
        elif isinstance(target, torch.nn.Linear):
            new_module = LoraLinear(target, lora_config)
        elif isinstance(target, ColumnParallelLinear):
            new_module = LoraParallelLinear(base_layer=target, lora_config=lora_config)
            self.lora_module_parallel_types[current_key] = self.ColumnParallelLinear_lora_type
        elif isinstance(target, RowParallelLinear):
            new_module = LoraParallelLinear(base_layer=target, lora_config=lora_config)
            self.lora_module_parallel_types[current_key] = self.RowParallelLinear_lora_type
        elif isinstance(target, GQAQKVColumnParallelLinear):
            new_module = LoraGQAQKVParallelLinear(base_layer=target, lora_config=lora_config)
            self.lora_module_parallel_types[current_key] = self.GQAQKVParallelLinear_lora_type
            self.lora_kv_size_multiplier = target.kv_size_multiplier
        elif is_hf_transformers_available():
            from transformers.pytorch_utils import Conv1D

            if isinstance(target, Conv1D):
                new_module = LoraLinear(target, lora_config, is_conv_1d_layer=True)

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"""Target module {target} is not supported. Currently, only the following modules are supported: "
                    torch.nn.Linear,
                    torch.nn.Embedding,
                    torch.nn.Conv2d,
                    transformers.pytorch_utils.Conv1D,
                    nxd.parallel_layers.ColumnParallelLinear,
                    nxd.parallel_layers.RowParallelLinear,
                    nxd.modules.qkv_linear.GQAQKVColumnParallelLinear,
                """
            )
        return new_module

    def merge_lora(self) -> None:
        if not self.is_lora_merged:
            for module in self.module.modules():
                if isinstance(module, LoraLayer):
                    module.merge()
            self.is_lora_merged = True

    def unmerge_lora(self) -> None:
        if self.is_lora_merged:
            for module in self.module.modules():
                if isinstance(module, LoraLayer):
                    module.unmerge()
            self.is_lora_merged = False

    def get_base_model(self) -> torch.nn.Module:
        """
        Returns the base model.
        """
        return self.module

    def _restore_module_name(self, key: str) -> str:
        key_word = ".base_layer"
        if key_word in key:
            return key.replace(key_word, "")
        else:
            return key

    def _get_lora_adapter_state_dict(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Return the state dict of the LoRA model and the modules specified by modules_to_save.
        There are three cases:
        1) if save_lora_base=True and merge_lora=False, return all modules
        2) if save_lora_base=True and merge_lora=True, we need to
            a) merge the lora modules into the base modules;
            b) selecte the base modules only;
            d) return the selected base modules;
            d) unmerge the lora modules from the base modules (in save_adapter())
        3) if save_lora_base=False, return LoRA adapter only
        """
        config = self.lora_config
        state_dict = self.module_state_dict()
        to_return = {}
        if config.save_lora_base and not config.merge_lora:
            to_return = state_dict
        elif config.save_lora_base and config.merge_lora:
            self.merge_lora()
            for k, value in state_dict.items():
                if "lora_" not in k:
                    k = self._restore_module_name(k)
                    to_return[k] = value
        else:
            bias = config.bias
            if bias == "none":
                to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
            elif bias == "all":
                to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
            elif bias == "lora_only":
                to_return = {}
                for k in state_dict:
                    if "lora_" in k:
                        to_return[k] = state_dict[k]
                        bias_name = k.split("lora_")[0] + "bias"
                        if bias_name in state_dict:
                            to_return[bias_name] = state_dict[bias_name]
            else:
                raise NotImplementedError
            to_return = {k: v for k, v in to_return.items() if (("lora_" in k) or ("bias" in k))}

            # get modules specified by modules_to_save
            if self.modules_to_save is not None:
                for k in state_dict:
                    if any(target_key in k for target_key in self.modules_to_save):
                        to_return[k] = state_dict[k]

        if self.lora_config.save_lora_config_adapter:
            # include the LoRA configuration in the checkpoint
            to_return["lora_config"] = self._get_lora_config_dict()
        else:
            # save the LoRA configuration to a seperate json file
            if not self.is_config_saved and xm.is_master_ordinal(local=True):
                self.save_config(save_dir)

        return to_return

    def _get_lora_config_dict(self) -> dict:
        return self.lora_config.selected_fields_to_save()

    def _save_config_to_json(self, filename: str) -> None:
        r"""save the LoRA configuration to a json file."""
        if not filename.endswith(".json"):
            logger.warn(f"{filename} is not a proper json file name.")
        config = self.lora_config
        output_dict = config.selected_fields_to_save()
        # converting set type to list
        for key, value in output_dict.items():
            if isinstance(value, set):
                output_dict[key] = list(value)
        # save it to a json file
        with open(filename, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    def save_config(self, save_dir: Optional[str] = None) -> None:
        sd = self.lora_config.lora_save_dir if save_dir is None else save_dir
        assert sd
        os.makedirs(sd, exist_ok=True)
        config_filename = os.path.join(sd, CONFIG_NAME)
        self._save_config_to_json(config_filename)
        self.is_config_saved = True

    def save_lora(self, save_dir: Optional[str] = None, adapter_tag: Optional[str] = None) -> None:
        r"""for single-device LoRA saving only."""
        if not model_parallel_is_initialized():
            self._save_single_device_lora(save_dir, adapter_tag)
        else:
            raise RuntimeError("Please use nxd.save_checkpoint() to save LoRA adapter with NxDModel.")

    def _save_single_device_lora(
        self,
        save_dir: Optional[str] = None,
        adapter_tag: Optional[str] = None,
    ) -> None:
        r"""
        Only the master device saves the checkpoint.
        """
        sd = self.lora_config.lora_save_dir if save_dir is None else save_dir
        assert sd

        if xm.is_master_ordinal(local=True):
            state_dict = self._get_lora_adapter_state_dict(sd)
            output_dir = sd if adapter_tag is None else os.path.join(sd, adapter_tag)
            os.makedirs(output_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            if self.is_lora_merged:
                self.unmerge_lora()

    def load_checkpoint(self, lora_config: LoraConfig) -> None:
        r"""
        load the lora configuration and the lora adapter from the checkpoint specified by config.
        Note that this function just parse the checkpoint without loading the state dict into the model.
        We defer the real state dict loading to load_state_dict()
        """
        save_dir = lora_config.lora_save_dir
        assert save_dir
        adapter_tag = lora_config.lora_load_tag

        if not model_parallel_is_initialized():
            # lora adapter checkpoint saved with torch.save()
            output_dir = save_dir if adapter_tag is None else os.path.join(save_dir, adapter_tag)
            ckpt_path = os.path.join(output_dir, WEIGHTS_NAME)
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"{ckpt_path} is not found.")
            ckpt = torch.load(ckpt_path, map_location="cpu")
        else:
            # lora adapter checkpoint saved with nxd.save_checkpoint()
            checkpoint_dir = create_checkpoint_storage(save_dir)
            assert adapter_tag
            ckpt_path = os.path.join(save_dir, _get_path(adapter_tag))
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"{ckpt_path} is not found.")
            ckpt = checkpoint_dir.load_object(ckpt_path, map_location="cpu")

        if "lora_config" in ckpt:
            self.lora_config = self._load_config_from_ckpt(lora_config, ckpt)
            ckpt.pop("lora_config")
            self.lora_ckpt = ckpt
        else:
            config_filename = os.path.join(save_dir, CONFIG_NAME)
            logger.info("LoRA configuration is not save in the checkpoint. Try to load it from %s", config_filename)
            if not os.path.isfile(config_filename):
                raise FileNotFoundError(f"Please name the file for LoRA confiugration as {CONFIG_NAME}.")
            self.lora_config = self._load_config_from_json(lora_config, config_filename)
            self.lora_ckpt = ckpt

        self.is_checkpoint_loaded = True

    def _load_config_from_json(self, lora_config: LoraConfig, filename: str) -> LoraConfig:
        def _from_json_file(path_json_file: str):
            with open(path_json_file, "r") as file:
                json_object = json.load(file)
            return json_object

        if not os.path.isfile(filename):
            raise FileNotFoundError(f"LoRA configuration file {filename} is not found.")

        loaded_attributes = _from_json_file(filename)
        selected_fields = lora_config.get_selected_fields()
        lora_config_dict = asdict(lora_config)
        for key in selected_fields:
            lora_config_dict[key] = loaded_attributes[key]
        return LoraConfig(**lora_config_dict)

    def _load_config_from_ckpt(self, lora_config: LoraConfig, ckpt: Dict[str, Any]) -> LoraConfig:
        config = ckpt.get("lora_config", None)
        if config is None:
            logger.warn("No LoRA configuration is found in checkpoint.")
            return lora_config
        else:
            lora_config_dict = asdict(lora_config)
            for key, value in config.items():
                lora_config_dict[key] = value
            return LoraConfig(**lora_config_dict)

    def load_lora(
        self,
        save_dir: Optional[str] = None,
        adapter_tag: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        adapter_only: bool = True,
    ) -> None:
        r"""
        for single-device LoRA load only.
        save_dir and adapter_tag are used to specify the checkpoint for base model.
        """
        if not model_parallel_is_initialized():
            return self._load_single_device_lora(save_dir, adapter_tag, ckpt_path, adapter_only)
        else:
            raise RuntimeError("Please use nxd.load_checkpoint() to load LoRA adapter when the base model is NxDModel.")

    def _load_single_device_lora(
        self,
        save_dir: Optional[str] = None,
        adapter_tag: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        adapter_only: bool = True,
    ):
        if not adapter_only:
            if ckpt_path is None:
                sd = self.lora_config.lora_save_dir if save_dir is None else save_dir
                assert sd
                output_dir = sd if adapter_tag is None else os.path.join(sd, adapter_tag)
                ckpt_path = os.path.join(output_dir, WEIGHTS_NAME)

            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"The checkpoint file {ckpt_path} is not found.")

            lora_config = self.lora_config
            if not adapter_only and not lora_config.save_lora_base:
                adapters_weights = torch.load(ckpt_path, map_location="cpu")
                self.module.load_state_dict(adapters_weights, strict=False)

        self.is_base_model_loaded = True
        load_result = self.load_lora_adapter()
        return load_result

    def load_lora_adapter(self):
        lora_config = self.lora_config
        if not self.is_base_model_loaded:
            assert lora_config.save_lora_base

        if not (lora_config.save_lora_base and lora_config.merge_lora):
            self.inject_adapter()

        load_result = self.module.load_state_dict(self.lora_ckpt, strict=False)
        self.print_model_info()
        return load_result

    def update_state_dict_keys(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        modules_keys = self.module_state_dict().keys()

        key_word = ".base_layer"
        for mkey in modules_keys:
            if key_word in mkey:
                key = mkey.replace(key_word, "")
                state_dict[mkey] = state_dict[key]
                del state_dict[key]
        return state_dict

    """
    torch.nn.Module APIs
    """

    def named_parameters(self, *args, **kwargs):
        for n, p in self.module.named_parameters(*args, **kwargs):
            yield n, p

    def module_state_dict(self) -> Dict[str, Any]:
        return self.module.state_dict()

    def merge_sharded_lora_weights(self, state_dict):
        def _get_base_module_name(name):
            return name.split(".lora_")[0]

        for name, weights in state_dict.items():
            if name == "lora_config":
                continue
            base_module_name = _get_base_module_name(name)
            if base_module_name in self.lora_module_parallel_types:
                lora_parallel_type = self.lora_module_parallel_types[base_module_name]
                if lora_parallel_type == self.ColumnParallelLinear_lora_type:
                    if ".lora_B" in name:
                        state_dict[name] = _gather_along_first_dim(weights)
                elif lora_parallel_type == self.RowParallelLinear_lora_type:
                    if ".lora_A" in name:
                        state_dict[name] = _gather_along_last_dim(weights)
                elif lora_parallel_type == self.GQAQKVParallelLinear_lora_type:
                    if ".lora_B" in name:
                        merged_lora_weights = _gather_along_first_dim(weights)
                        state_dict[name] = torch.chunk(merged_lora_weights, self.lora_kv_size_multiplier)[0]
                else:
                    raise ValueError("Unknown lora prallel type {lora_parallel_type}")

        xm.mark_step()
        # only the first rank need to save the merged LoRA adapter
        return state_dict

    def state_dict(self, *args, **kwargs):
        config = self.lora_config
        if config.merge_sharded_lora and (config.save_lora_base or config.merge_lora):
            config.save_lora_base = False
            config.merge_lora = False
            logger.info("Since merged_sharded_lora is enabled, we will disable save_lora_base and merge_lora to save merged LoRA adapter only.")
        state_dict = self._get_lora_adapter_state_dict()
        if config.merge_sharded_lora:
            state_dict = self.merge_sharded_lora_weights(state_dict)
        return state_dict

    def load_state_dict(
        self, state_dict: Optional[Mapping[str, Any]] = None, strict: bool = False, assign: bool = False
    ):
        r"""
        There are two steps to load state dict for LoRA model.
        Step 1: load the state dict for the base model
        Step 2: load the state dict for the LoRA adapter
        """
        lora_config = self.lora_config
        load_result = None
        if state_dict is not None and (not lora_config.save_lora_base or not self.is_checkpoint_loaded):
            if self.is_lora_enabled:
                state_dict = self.update_state_dict_keys(dict(state_dict))
            # load the state dict to the base model
            load_result = self.module.load_state_dict(state_dict, strict=strict)
            self.is_base_model_loaded = True

        if lora_config.load_lora_from_ckpt:
            load_result_lora = self.load_lora_adapter()
            if load_result is not None:
                load_result.missing_keys.extend(load_result_lora.missing_keys)
                load_result.unexpected_keys.extend(load_result_lora.unexpected_keys)
            else:
                load_result = load_result_lora

        if load_result is not None and load_result.missing_keys and lora_config.load_lora_from_ckpt:
            raise RuntimeError(f"Missing keys when loading state dictionary: {', '.join(load_result.missing_keys)}")

        return load_result

    """
    common transformers.PreTrainedModel APIs
    """

    @property
    def dtype(self):
        return self.original_module().dtype

    @property
    def config(self):
        return self.original_module().lora_config

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.module, name)

    def get_nb_trainable_parameters(self) -> Tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.module.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        logger.info(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def print_model_info(self) -> None:
        if self.is_verbose_enabled:
            logger.info("LoRA model: %s", self.module)
            logger.info("LoRA configuration: %s", self.lora_config)
            self.print_trainable_parameters()
