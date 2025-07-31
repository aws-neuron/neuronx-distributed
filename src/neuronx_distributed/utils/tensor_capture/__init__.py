"""
TensorCapture: A utility for capturing intermediate tensor outputs from PyTorch models.

This module provides tools to capture intermediate tensor outputs from PyTorch models
during execution, which can be useful for debugging, visualization, or analysis.
"""

from .api import (
    enable_tensor_capture,
    disable_tensor_capture,
    get_available_modules,
    register_tensor,
    get_captured_tensors_dict
)

__all__ = [
    'enable_tensor_capture',
    'disable_tensor_capture',
    'get_available_modules',
    'register_tensor',
    'get_captured_tensors_dict'
]