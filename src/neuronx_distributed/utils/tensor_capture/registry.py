import torch
from typing import List, Dict, Any, Set, Optional
from collections import OrderedDict, defaultdict

class CapturedModelInfo:
    """
    Class to store information about a model with tensor capture enabled.
    """
    def __init__(self, 
                 modules_to_capture: List[str], 
                 max_tensors: Optional[int] = None,
                 capture_inputs: bool = False):
        self.modules_to_capture = modules_to_capture
        self.max_tensors = max_tensors
        self.capture_inputs = capture_inputs
        self.hooks: List[Any] = []
        self.module_tensors: OrderedDict[str, torch.Tensor] = OrderedDict()  # Store module tensors for this model
        self.manual_tensors: OrderedDict[str, torch.Tensor] = OrderedDict()  # Store manual tensors for this model
        self.manual_tensors_keys: Dict[str, int] = defaultdict(int)

class TensorRegistry:
    """
    TensorRegistry is a singleton class that manages the collection and storage of
    tensors captured during model execution. It handles both automatic capture from
    monitored modules and manual registration of tensors.
    
    The registry maintains two types of tensors:
    1. Module tensors: Outputs from specific modules that are being monitored
    2. Manual tensors: Tensors that are explicitly registered during execution
    
    Usage:
        # Get the singleton instance
        registry = TensorRegistry.get_instance()
        
        # Configure the registry
        registry.configure(enabled=True, modules=["layer1", "layer2"], max_tensors=5)
        
        # Register a tensor manually
        registry.register_tensor("activation", tensor)
        
        # Get all collected tensors
        tensors = registry.get_captured_tensors_dict()
    """
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """
        Get the singleton instance of TensorRegistry.
        
        Returns:
            TensorRegistry: The singleton instance
        """
        if cls._instance is None:
            cls._instance = TensorRegistry()
        return cls._instance
    
    def __init__(self):
        """
        Initialize the TensorRegistry.
        
        Note:
            This should not be called directly. Use get_instance() instead.
        """
        self.enabled = False  # Whether tensor capture is enabled
        self.model_info = CapturedModelInfo([], 10, False)  # Single CapturedModelInfo instance
        
    def clear(self):
        """
        Clear all registered tensors and reset the registry.
        """ 
        self.model_info = CapturedModelInfo([], 10, False)
    
    def configure(self, enabled=False, modules=None, max_tensors=None, capture_inputs=False):
        """
        Configure the tensor registry settings.
        
        Args:
            enabled (bool): Whether to enable tensor collection
            modules (List[str], optional): List of module names whose outputs should be captured
            max_tensors (int, optional): Maximum number of manually registered tensors to store
            capture_inputs (bool): Whether to capture module input tensors
        """
        self.enabled = enabled
        
        # Update the model info with new configuration
        self.model_info = CapturedModelInfo(list(modules or []), max_tensors, capture_inputs)
    
    def register_tensor(self, name, tensor):
        """
        Register a tensor in the registry for capture.
        
        This method handles two types of tensor registration:
        1. Module tensors: When name matches a monitored module or is a module input/output
        2. Manual tensors: When name doesn't match any monitored module
        
        Args:
            name (str or Any): Name/identifier for the tensor
            tensor (torch.Tensor): Tensor to register for capture
        """
        if not self.enabled:
            return
            
        # For module outputs, check if the module is being monitored
        is_monitored = False
        if isinstance(name, str):
            # Direct match with monitored module
            if name in self.model_info.modules_to_capture:
                is_monitored = True
            # Check if name contains any monitored module
            else:
                for module in self.model_info.modules_to_capture:
                    if module in name:
                        is_monitored = True
                        break
        
        if is_monitored:
            self.model_info.module_tensors[name] = tensor.clone().detach()
        # For manual registration
        else:
            # Check if we've reached the limit
            max_tensors = self.model_info.max_tensors
            if max_tensors is not None and len(self.model_info.manual_tensors) >= max_tensors:
                return
                
            # Create a unique key for this tensor
            if isinstance(name, str):
                key = f"manual_{name}"
                if self.model_info.manual_tensors_keys[key] > 0:
                    key = f"{key}_{self.model_info.manual_tensors_keys[key]}"
                self.model_info.manual_tensors_keys[key] += 1
            else:
                key = f"manual_tensor_{len(self.model_info.manual_tensors)}"
                
            self.model_info.manual_tensors[key] = tensor.clone().detach()
    
    def get_captured_tensors_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get all captured tensors as an ordered dictionary mapping names to tensors.
        
        Returns:
            OrderedDict mapping tensor names to their values
        """
        result = OrderedDict()
        
        # Add module tensors
        for module_name in self.model_info.modules_to_capture:
            # First collect input tensors if capture_inputs is enabled
            if self.model_info.capture_inputs:
                input_keys = sorted([k for k in self.model_info.module_tensors.keys() 
                                   if k.startswith(f"{module_name}.inputs")])
                for key in input_keys:
                    result[key] = self.model_info.module_tensors[key]
            
            # Then collect output tensors
            output_keys = sorted([k for k in self.model_info.module_tensors.keys() 
                                if k.startswith(f"{module_name}.outputs") or k == module_name])
            for key in output_keys:
                result[key] = self.model_info.module_tensors[key]
        
        # Add manual tensors
        for key, tensor in self.model_info.manual_tensors.items():
            result[key] = tensor
        
        return result
    
    def get_module_tensors(self):
        """
        Get all tensors from monitored modules.
        
        Returns:
            Dictionary mapping module names to their tensors
        """
        return self.model_info.module_tensors
        
    def get_manual_tensors(self):
        """
        Get all manually registered tensors.
        
        Returns:
            Dictionary mapping tensor names to their values
        """
        return self.model_info.manual_tensors
        
    def get_manual_tensor_count(self):
        """
        Get the count of manually registered tensors.
        
        Returns:
            Number of manually registered tensors
        """
        return len(self.model_info.manual_tensors)
        
    def get_monitored_tensor_count(self):
        """
        Get the count of tensors from monitored modules.
        
        Returns:
            Number of tensors from monitored modules
        """
        return len(self.model_info.module_tensors)
        
    def get_total_tensor_count(self):
        """
        Get the total count of all tensors.
        
        Returns:
            Total number of tensors
        """
        return self.get_monitored_tensor_count() + self.get_manual_tensor_count()
        
    def remove_hooks(self):
        """
        Remove all hooks from the model.
        """
        # Remove hooks
        for hook in self.model_info.hooks:
            hook.remove()
        self.model_info.hooks = []