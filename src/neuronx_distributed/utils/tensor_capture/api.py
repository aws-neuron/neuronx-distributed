from typing import Dict, List, Any, Union, Optional, Callable
import torch
import torch.nn as nn
import copy
from .registry import TensorRegistry
from .model_modification import (
    modify_model_for_tensor_capture, 
    restore_model,
    find_available_modules
)
from neuronx_distributed.utils.logger import get_logger

# Get the logger
logger = get_logger()

def enable_tensor_capture(model: nn.Module, 
                     modules_to_capture: Optional[List[str]] = None, 
                     max_tensors: Optional[int] = None, 
                     capture_inputs: bool = False) -> nn.Module:
    """
    Enable tensor capture for a model
    
    Args:
        model: The model to enable tensor capture for
        modules_to_capture: List of module names whose outputs should be captured
        max_tensors: Maximum number of manually registered tensors to store.
                    If None, no manual tensors will be included in the output.
        capture_inputs: Whether to capture module input tensors
        
    Returns:
        The model with tensor capture enabled
        
    Raises:
        ValueError: If any module in modules_to_capture is not found in the model
    """
    if modules_to_capture is None:
        logger.info("No modules to capture.")
        modules_to_capture = []
    else:
        logger.info(f"Enabling tensor capture for the following modules: {modules_to_capture}")
    return modify_model_for_tensor_capture(model, modules_to_capture, max_tensors, capture_inputs)


def disable_tensor_capture(model: nn.Module) -> nn.Module:
    """
    Disable tensor capture for a model
    
    Args:
        model: The model to disable tensor capture for
        
    Returns:
        The restored model
    """
    # Clear the registry
    registry = TensorRegistry.get_instance()
    registry.clear()
    
    # Restore the model
    return restore_model(model)


def get_available_modules(model: nn.Module) -> List[str]:
    """
    Get a list of all modules in a model that can have their outputs captured
    
    Args:
        model: The model to inspect
        
    Returns:
        List of available module paths
    """
    return find_available_modules(model)


def register_tensor(name: str, tensor: torch.Tensor) -> None:
    """
    Manually register a tensor for capture
    
    Args:
        name: Name/identifier for the tensor
        tensor: Tensor to register
    """
    registry = TensorRegistry.get_instance()
    registry.register_tensor(name, tensor)


def get_captured_tensors_dict() -> Dict[str, torch.Tensor]:
    """
    Get all captured tensors as an ordered dictionary mapping names to tensors.
    
    Returns:
        OrderedDict mapping tensor names to their values
    """
    registry = TensorRegistry.get_instance()
    return registry.get_captured_tensors_dict()
