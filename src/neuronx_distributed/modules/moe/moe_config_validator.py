"""Validates Mixture of Experts (MoE) configuration settings including dropless mode, activation functions, and capacity factors."""

import json
import logging
import os
from typing import Union
from typing import Dict, Any
from neuronx_distributed.utils.utils import get_dict_from_json

logger = logging.getLogger(__name__)


class MoeConfigValidator:
    def __init__(self, cfg):
        """
        Initialize the MoeConfigValidator class.

        This class is responsible for validating Mixture of Experts (MoE) configuration
        settings, including dropless mode, activation functions, and capacity factors.

        Parameters:
        -----------
        cfg : object
            Configuration object containing model and MoE-specific settings.
            Expected to have attributes like model_source, model.moe, etc.

        Attributes:
        -----------
        cfg : object
            Stores the input configuration object for later use in validation methods.
        hf_model_config : dict
            It's used to store the parsed HuggingFace model config when needed.

        Note:
        -----
        The core parameters for MoE validation (e.g., dropless, capacity_factor, glu_mlp)
        are not stored as attributes but are accessed from cfg.model.moe when needed.
        This ensures that the validator always uses the most up-to-date configuration
        values during validation.
        """
        self.cfg = cfg
        self.hf_model_config = {}

    def _load_hf_config(self) -> Dict[Any, Any]:
        """Load and parse HuggingFace model config."""
        model_config_path = self.cfg.model.model_config
        return get_dict_from_json(model_config_path)

    def _validate_hf_activation(self, dropless: bool) -> None:
        """Validate activation function for HuggingFace models."""
        if not dropless:
            return

        is_dbrx = self.hf_model_config.get("model_type") == "dbrx"
        if is_dbrx:
            ffn_config = self.hf_model_config.get("ffn_config", {})
            ffn_act_fn = ffn_config.get("ffn_act_fn", {})
            if ffn_act_fn.get("name") != "silu":
                raise ValueError(
                    "For DBRX models, dropless mode is only supported with SiLU activation function. "
                    f"Current activation function: {ffn_act_fn.get('name')}. "
                    "Please adjust your configuration."
                )
        else:
            if self.hf_model_config.get("hidden_act") != "silu":
                current_activation = self.hf_model_config.get("hidden_act")
                raise ValueError(
                    "Dropless mode is only supported with SiLU activation function. "
                    f"Current activation function: {current_activation}. "
                    "Please adjust your configuration."
                )

    def _validate_megatron_activation(self, dropless: bool) -> None:
        """Validate activation function for Megatron models."""
        if not dropless:
            return

        activation = getattr(self.cfg.model, "activation", None)
        if not (activation == "silu" or activation == "swiglu"):
            raise ValueError(
                "For Megatron models, dropless mode is only supported with SiLU or SwiGLU activation functions. "
                f"Current activation function: {activation}. "
                "Please adjust your configuration."
            )

    def validate_moe_config(self) -> None:
        """Validate MOE configuration settings."""
        if not hasattr(self.cfg.model, "moe"):
            raise AttributeError(
                "MoE configuration is missing in model config. Please ensure 'moe' attribute is present in the model configuration."
            )

        dropless = getattr(self.cfg.model.moe, "dropless", False)
        capacity_factor = self.cfg.model.moe.capacity_factor
        glu_mlp = getattr(self.cfg.model.moe, "glu_mlp", True)

        if self.cfg.model_source == "hf":
            self.hf_model_config = self._load_hf_config()
            self._validate_hf_activation(dropless)
        elif self.cfg.model_source == "megatron":
            self._validate_megatron_activation(dropless)

        if dropless:
            if not glu_mlp:
                raise ValueError("Dropless mode requires GLU_MLP to be True.")

            if capacity_factor is None or capacity_factor > 0.0:
                logger.warning(
                    "Dropless mode expects a capacity_factor set to 0.0. "
                    f"Current value: {capacity_factor}. Setting capacity_factor to 0.0."
                )
                self.cfg.model.moe.capacity_factor = 0.0

        elif not dropless:
            if capacity_factor is None or capacity_factor > 0.0:
                return  # all_experts
            if capacity_factor <= 0.0:
                raise ValueError(
                    "Dropping requires a capacity factor greater than 0.0 Please adjust your configuration."
                )
