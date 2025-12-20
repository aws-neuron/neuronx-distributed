"""
TensorReplacement: A utility for teacher forcing per step per layer per module tensor outputs for PyTorch models.

This module provides tools to replace outputs of specified modules at given layer and step indices
during execution, which can be useful for debugging, visualization, or analysis.
"""

from .model_modification import (
    modify_model_for_tensor_replacement,
    patch_forward_with_additional_args
)

__all__ = [
    'modify_model_for_tensor_replacement',
    'patch_forward_with_additional_args'
]