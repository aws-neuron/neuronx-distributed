"""
Test utilities for tensor capture unit tests.
This module provides common test fixtures, models, and helper functions
to reduce code duplication across test files.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from neuronx_distributed.utils.tensor_capture.registry import TensorRegistry


# Common test models
class SimpleModel(nn.Module):
    """Simple model for basic tensor capture testing"""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 20),
            nn.Linear(20, 30),
            nn.Linear(30, 10)
        ])
        self.activation = nn.ReLU()
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation(x)
        return x


class AttentionLinear(nn.Module):
    """Custom linear layer that accepts attention parameters for testing"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, attention_mask=None, position_ids=None):
        output = self.linear(x)
        
        if attention_mask is not None:
            # Simple mask application - just scale by a scalar factor
            mask_factor = attention_mask.mean().item()
            output = output * mask_factor
        
        if position_ids is not None:
            pos_factor = position_ids.float().mean().item() * 0.1
            output = output + pos_factor
            
        return output


class AttentionModel(nn.Module):
    """Model that passes attention parameters to its modules for testing"""
    def __init__(self):
        super().__init__()
        self.linear = AttentionLinear(10, 20)
        self.norm = nn.LayerNorm(20)
    
    def forward(self, x, attention_mask=None, position_ids=None, use_cache=False):
        # Pass attention parameters to the linear layer
        output = self.linear(x, attention_mask=attention_mask, position_ids=position_ids)
        output = self.norm(output)
        
        if use_cache:
            return output, {"cache": output.mean()}
        
        return output


@dataclass
class ModelOutput:
    """Dataclass for testing tensor capture with dataclasses"""
    logits: torch.Tensor
    hidden_states: torch.Tensor


class DataclassModel(nn.Module):
    """Model that returns a dataclass for testing tensor capture with dataclasses"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
    
    def forward(self, x):
        hidden = self.linear1(x)
        logits = self.linear2(hidden)
        return ModelOutput(logits=logits, hidden_states=hidden)


class TupleOutputModel(nn.Module):
    """Model that returns a tuple for testing tensor capture with tuples"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 10)
    
    def forward(self, x):
        hidden = self.linear1(x)
        logits = self.linear2(hidden)
        return logits, hidden


class TupleOutputLinear(nn.Module):
    """Linear layer that returns a tuple for testing tuple output capture"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
    
    def forward(self, x):
        output = self.linear(x)
        # Return tuple: (output, mean_output)
        return output, output.mean(dim=-1)


class ModelWithTupleOutput(nn.Module):
    """Model with tuple-output module for testing indexed outputs"""
    def __init__(self):
        super().__init__()
        self.tuple_linear = TupleOutputLinear()
    
    def forward(self, x):
        return self.tuple_linear(x)


# Common test fixtures
class TestFixtures:
    """Common test fixtures and data"""
    
    @staticmethod
    def get_input_tensor(batch_size=2, input_size=10):
        """Get a standard input tensor for testing"""
        return torch.randn(batch_size, input_size)
    
    @staticmethod
    def get_attention_mask(batch_size=2, seq_len=10):
        """Get a standard attention mask for testing"""
        return torch.ones(batch_size, seq_len)
    
    @staticmethod
    def get_position_ids(batch_size=2, seq_len=10):
        """Get standard position IDs for testing"""
        return torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)


# Test helper functions
class TestHelpers:
    """Helper functions for common test operations"""
    
    @staticmethod
    def setup_registry():
        """Set up and clear the tensor registry for testing"""
        registry = TensorRegistry.get_instance()
        registry.clear()
        return registry
    
    @staticmethod
    def teardown_registry():
        """Clean up the tensor registry after testing"""
        registry = TensorRegistry.get_instance()
        registry.clear()
    
    @staticmethod
    def assert_tensor_shapes(test_case, tensors_dict, expected_shapes):
        """
        Assert that captured tensors have expected shapes
        
        Args:
            test_case: The unittest.TestCase instance
            tensors_dict: Dictionary of captured tensors
            expected_shapes: Dictionary mapping tensor names to expected shapes
        """
        for name, expected_shape in expected_shapes.items():
            test_case.assertIn(name, tensors_dict, f"Tensor '{name}' not found in captured tensors")
            actual_shape = tensors_dict[name].shape
            test_case.assertEqual(actual_shape, expected_shape, 
                                f"Tensor '{name}' has shape {actual_shape}, expected {expected_shape}")
    
    @staticmethod
    def assert_tensor_keys_present(test_case, tensors_dict, expected_keys):
        """
        Assert that expected tensor keys are present in captured tensors
        
        Args:
            test_case: The unittest.TestCase instance
            tensors_dict: Dictionary of captured tensors
            expected_keys: List of expected tensor keys
        """
        for key in expected_keys:
            test_case.assertIn(key, tensors_dict, f"Expected key '{key}' not found in captured tensors")
    
    @staticmethod
    def assert_tensor_keys_absent(test_case, tensors_dict, absent_keys):
        """
        Assert that certain tensor keys are NOT present in captured tensors
        
        Args:
            test_case: The unittest.TestCase instance
            tensors_dict: Dictionary of captured tensors
            absent_keys: List of keys that should not be present
        """
        for key in absent_keys:
            test_case.assertNotIn(key, tensors_dict, f"Unexpected key '{key}' found in captured tensors")
    
    @staticmethod
    def count_keys_with_pattern(tensors_dict, pattern):
        """
        Count the number of keys that contain a specific pattern
        
        Args:
            tensors_dict: Dictionary of captured tensors
            pattern: String pattern to search for in keys
            
        Returns:
            Number of keys containing the pattern
        """
        return len([k for k in tensors_dict.keys() if pattern in k])
    
    @staticmethod
    def get_keys_with_pattern(tensors_dict, pattern):
        """
        Get all keys that contain a specific pattern
        
        Args:
            tensors_dict: Dictionary of captured tensors
            pattern: String pattern to search for in keys
            
        Returns:
            List of keys containing the pattern
        """
        return [k for k in tensors_dict.keys() if pattern in k]


# Base test class for common setup/teardown
class BaseTensorCaptureTest:
    """Base class for tensor capture tests with common setup/teardown"""
    
    def setUp(self):
        """Common setup for tensor capture tests"""
        self.registry = TestHelpers.setup_registry()
        self.input_tensor = TestFixtures.get_input_tensor()
        self.attention_mask = TestFixtures.get_attention_mask()
        self.position_ids = TestFixtures.get_position_ids()
    
    def tearDown(self):
        """Common teardown for tensor capture tests"""
        TestHelpers.teardown_registry()


# Constants for common test scenarios
class TestConstants:
    """Constants used across multiple test files"""
    
    # Common module names for testing
    SIMPLE_MODULES = ["layers.0", "layers.1", "layers.2"]
    ATTENTION_MODULES = ["linear", "norm"]
    DATACLASS_MODULES = ["linear1", "linear2"]
    
    # Common tensor shapes
    INPUT_SHAPE = (2, 10)
    HIDDEN_SHAPE_20 = (2, 20)
    HIDDEN_SHAPE_30 = (2, 30)
    OUTPUT_SHAPE = (2, 10)
    ATTENTION_MASK_SHAPE = (2, 10)
    POSITION_IDS_SHAPE = (2, 10)
    
    # Common tensor key patterns
    INPUTS_PATTERN = "inputs"
    OUTPUTS_PATTERN = "outputs"
    KWARGS_PATTERN = "kwargs"
    ATTENTION_MASK_KEY = "attention_mask"
    POSITION_IDS_KEY = "position_ids"
