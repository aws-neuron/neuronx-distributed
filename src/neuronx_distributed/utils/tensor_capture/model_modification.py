import torch
import torch.nn as nn
import types
from typing import List, Dict, Any, Optional, Union, Callable
from .registry import TensorRegistry
from neuronx_distributed.utils.logger import get_logger

# Get the logger
logger = get_logger()

def modify_model_for_tensor_capture(model: nn.Module, 
                            modules_to_capture: Optional[List[str]] = None, 
                            max_tensors: Optional[int] = None, 
                            capture_inputs: bool = False) -> nn.Module:
    """
    Set up tensor capture for a model by registering forward hooks
    
    Args:
        model: The model to set up tensor capture for
        modules_to_capture: List of module names whose outputs should be captured
        max_tensors: Maximum number of tensors to store per model for manually registered tensors
                    This is crucial for tracing as it determines the fixed number of outputs.
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
    
    # Validate that all modules_to_capture exist in the model
    available_modules = set(find_available_modules(model))
    invalid_modules = [module for module in modules_to_capture if module not in available_modules]
    if invalid_modules:
        raise ValueError(f"The following modules were not found in the model: {invalid_modules}")
    
    # Configure the registry
    registry = TensorRegistry.get_instance()
    registry.configure(enabled=True, modules=modules_to_capture, max_tensors=max_tensors, capture_inputs=capture_inputs)
    model_info = registry.model_info

    def make_hook(module_name: str) -> Callable:
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            # Get registry
            registry = TensorRegistry.get_instance()

            def register_tensor_object(prefix: str, obj: Any) -> None:
                """Helper function to register tensors from various object types"""
                if isinstance(obj, torch.Tensor):
                    registry.register_tensor(prefix, obj)
                elif isinstance(obj, tuple):
                    for i, item in enumerate(obj):
                        register_tensor_object(f"{prefix}.{i}", item)
                elif hasattr(obj, '__dataclass_fields__') or (hasattr(obj, '__dict__') and not isinstance(obj, type)):
                    # For dataclass objects or any object with attributes
                    for attr_name in dir(obj):
                        # Skip private attributes, methods, and callables
                        if not attr_name.startswith('_') and not callable(getattr(obj, attr_name)):
                            attr_value = getattr(obj, attr_name)
                            if isinstance(attr_value, torch.Tensor):
                                registry.register_tensor(f"{prefix}.{attr_name}", attr_value)

            # Register input tensors if enabled
            if capture_inputs and input:
                register_tensor_object(f"{module_name}.inputs", input)
            
            # Register output tensors
            register_tensor_object(f"{module_name}.outputs", output)
        return hook
    
    # Register forward hooks for targeted modules
    for name, module in dict(model.named_modules()).items():
        if name in modules_to_capture:
            hook_handle = module.register_forward_hook(make_hook(name))
            model_info.hooks.append(hook_handle)
            logger.info(f"Registered forward hook for module {name} for tensor capture")
    
    return model


def restore_model(model: nn.Module) -> nn.Module:
    """
    Disable tensor capture for a model by removing hooks
    
    Args:
        model: The model to disable tensor capture for
        
    Returns:
        The model with tensor capture disabled
    """
    # Get the registry
    registry = TensorRegistry.get_instance()
    
    # Remove hooks
    registry.remove_hooks()
    
    # Disable tensor capture
    registry.enabled = False
    
    return model


def find_available_modules(model: nn.Module, prefix: str = "") -> List[str]:
    """
    Find all available modules in a model that can be monitored
    
    Args:
        model: The model to inspect
        prefix: Prefix for module names (used in recursion)
        
    Returns:
        List of available module paths
    """
    modules = []
    
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        modules.append(full_name)
        
        # Recursively check child modules
        modules.extend(find_available_modules(module, full_name))
        
        # For ModuleList objects, also include numbered children
        if isinstance(module, torch.nn.ModuleList):
            for i, child in enumerate(module):
                child_name = f"{full_name}.{i}"
                modules.append(child_name)
                modules.extend(find_available_modules(child, child_name))
    
    return modules