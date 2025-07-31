"""
Unit tests for the tensor_capture API functions.
"""

import unittest
import torch
import torch.nn as nn
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

from neuronx_distributed.utils.tensor_capture import (
    enable_tensor_capture,
    disable_tensor_capture,
    get_available_modules,
    register_tensor,
    get_captured_tensors_dict,
)


class SimpleModel(nn.Module):
    """Simple model for testing tensor capture"""
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


class TestEnableDisableTensorCapture(unittest.TestCase):
    """Test cases for enable_tensor_capture and disable_tensor_capture functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = SimpleModel()
        self.input_tensor = torch.randn(2, 10)
        
    def test_enable_disable_tensor_capture(self):
        """Test enabling and disabling tensor capture on a model"""
        modules_to_capture = ["layers.0", "layers.1"]
        
        # Enable tensor capture
        model_with_capture = enable_tensor_capture(self.model, modules_to_capture)
        
        # Check that the model is the same instance (no modifications)
        self.assertIs(model_with_capture, self.model)
        
        # Run the model to capture tensors
        _ = model_with_capture(self.input_tensor)
        
        # Check that we can get tensors using the API functions
        tensors_dict = get_captured_tensors_dict()
        self.assertEqual(len(tensors_dict), len(modules_to_capture))
        
        # Disable tensor capture
        model_without_capture = disable_tensor_capture(model_with_capture)
        
        # Check that the model is the same instance
        self.assertIs(model_without_capture, self.model)
        
    def test_enable_with_max_tensors(self):
        """Test enabling tensor capture with max_tensors parameter"""
        modules_to_capture = ["layers.0"]
        max_tensors = 5
        
        # Enable tensor capture with max_tensors
        model_with_capture = enable_tensor_capture(self.model, modules_to_capture, max_tensors)
        
        # Check that the model is the same instance (no modifications)
        self.assertIs(model_with_capture, self.model)
        
        # Register some manual tensors
        for i in range(max_tensors + 2):  # Try to register more than max_tensors
            register_tensor(f"manual_tensor_{i}", torch.tensor([float(i)]))
        
        # Run the model
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors
        tensors_dict = get_captured_tensors_dict()
        
        # Count manual tensors (should be limited to max_tensors)
        manual_tensors = [name for name in tensors_dict.keys() if "manual" in name]
        self.assertEqual(len(manual_tensors), max_tensors)
        
    def test_enable_with_capture_inputs(self):
        """Test enabling tensor capture with capture_inputs parameter"""
        modules_to_capture = ["layers.0"]
        
        # Enable tensor capture with capture_inputs=True
        model_with_capture = enable_tensor_capture(self.model, modules_to_capture, capture_inputs=True)
        
        # Run the model to capture tensors
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors as dictionary
        captured_tensors_dict = get_captured_tensors_dict()
        
        # Should have both input and output tensors for layers.0
        self.assertEqual(len(captured_tensors_dict), 2)
        
        # Check that we have both input and output tensors
        input_tensors = [name for name in captured_tensors_dict.keys() if "inputs" in name]
        output_tensors = [name for name in captured_tensors_dict.keys() if "outputs" in name]
        self.assertEqual(len(input_tensors), 1)
        self.assertEqual(len(output_tensors), 1)
        
    def test_enable_with_no_modules(self):
        """Test enabling tensor capture with no modules to capture"""
        # Enable tensor capture with no modules
        model_with_capture = enable_tensor_capture(self.model, None)
        
        # Check that the model is the same instance (no modifications)
        self.assertIs(model_with_capture, self.model)
        
        # Run the model
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors as dictionary - should be empty since no modules were specified
        tensors_dict = get_captured_tensors_dict()
        self.assertEqual(len(tensors_dict), 0)
        
    def test_invalid_module_name(self):
        """Test enabling tensor capture with invalid module names"""
        modules_to_capture = ["invalid_module", "layers.999"]
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            enable_tensor_capture(self.model, modules_to_capture)


class TestGetAvailableModules(unittest.TestCase):
    """Test cases for the get_available_modules function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = SimpleModel()
        
    def test_get_available_modules(self):
        """Test getting available modules from a model"""
        available_modules = get_available_modules(self.model)
        
        # Check that we got the expected modules
        self.assertIn("layers", available_modules)
        self.assertIn("layers.0", available_modules)
        self.assertIn("layers.1", available_modules)
        self.assertIn("layers.2", available_modules)
        self.assertIn("activation", available_modules)
        
    def test_get_available_modules_nested(self):
        """Test getting available modules from a nested model"""
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = SimpleModel()
                self.classifier = nn.Linear(10, 2)
                
            def forward(self, x):
                x = self.feature_extractor(x)
                return self.classifier(x)
        
        nested_model = NestedModel()
        available_modules = get_available_modules(nested_model)
        
        # Check that we got the expected modules
        self.assertIn("feature_extractor", available_modules)
        self.assertIn("feature_extractor.layers", available_modules)
        self.assertIn("feature_extractor.layers.0", available_modules)
        self.assertIn("classifier", available_modules)


class TestRegisterTensor(unittest.TestCase):
    """Test cases for the register_tensor function"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Import the registry directly for testing
        from neuronx_distributed.utils.tensor_capture.registry import TensorRegistry
        self.registry = TensorRegistry.get_instance()
        self.registry.clear()
        self.registry.configure(enabled=True, max_tensors=5)
        
        self.test_tensor = torch.tensor([1.0, 2.0, 3.0])
        
    def tearDown(self):
        """Clean up after tests"""
        self.registry.clear()
        
    def test_register_tensor(self):
        """Test registering a tensor"""
        register_tensor("test_tensor", self.test_tensor)
        
        manual_tensors = self.registry.get_manual_tensors()
        self.assertEqual(len(manual_tensors), 1)
        
        # Check that the tensor was registered correctly
        key = list(manual_tensors.keys())[0]
        self.assertTrue(torch.equal(manual_tensors[key], self.test_tensor))


class TestDataclassCapture(unittest.TestCase):
    """Test cases for tensor capture with dataclasses"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = DataclassModel()
        self.input_tensor = torch.randn(2, 10)
        
    def test_dataclass_capture(self):
        """Test capturing tensors from a model that returns a dataclass"""
        modules_to_capture = ["linear1", "linear2"]
        
        # Enable tensor capture
        model_with_capture = enable_tensor_capture(self.model, modules_to_capture)
        
        # Run the model
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors as dictionary
        captured_tensors_dict = get_captured_tensors_dict()
        
        # Check that we got the expected number of tensors
        self.assertEqual(len(captured_tensors_dict), len(modules_to_capture))
        
        # Get the values as a list to check shapes
        tensors = list(captured_tensors_dict.values())
        
        # Check that the tensors have the expected shapes
        self.assertEqual(tensors[0].shape, (2, 20))  # Output of linear1
        self.assertEqual(tensors[1].shape, (2, 10))  # Output of linear2


class TestTupleOutputCapture(unittest.TestCase):
    """Test cases for tensor capture with tuple outputs"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = TupleOutputModel()
        self.input_tensor = torch.randn(2, 10)
        
    def test_tuple_output_capture(self):
        """Test capturing tensors from a model that returns a tuple"""
        modules_to_capture = ["linear1", "linear2"]
        
        # Enable tensor capture
        model_with_capture = enable_tensor_capture(self.model, modules_to_capture)
        
        # Run the model
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors as dictionary
        captured_tensors_dict = get_captured_tensors_dict()
        
        # Check that we got the expected number of tensors
        self.assertEqual(len(captured_tensors_dict), len(modules_to_capture))
        
        # Get the values as a list to check shapes
        tensors = list(captured_tensors_dict.values())
        
        # Check that the tensors have the expected shapes
        self.assertEqual(tensors[0].shape, (2, 20))  # Output of linear1
        self.assertEqual(tensors[1].shape, (2, 10))  # Output of linear2


class TestCaptureInputs(unittest.TestCase):
    """Test cases for capturing module inputs"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = SimpleModel()
        self.input_tensor = torch.randn(2, 10)
        
    def test_capture_inputs(self):
        """Test capturing module inputs"""
        modules_to_capture = ["layers.1"]  # Capture the middle layer
        
        # Enable tensor capture with capture_inputs=True
        model_with_capture = enable_tensor_capture(self.model, modules_to_capture, capture_inputs=True)
        
        # Run the model
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors as dictionary
        captured_tensors_dict = get_captured_tensors_dict()
        
        # Should have both input and output tensors for layers.1
        self.assertEqual(len(captured_tensors_dict), 2)
        
        # Get the values as a list to check shapes
        tensors = list(captured_tensors_dict.values())
        
        # First tensor should be the input to layers.1, which is the output of layers.0 after activation
        # Second tensor should be the output of layers.1
        self.assertEqual(tensors[0].shape, (2, 20))  # Input to layers.1
        self.assertEqual(tensors[1].shape, (2, 30))  # Output of layers.1


class TestEndToEndAPI(unittest.TestCase):
    """End-to-end tests for the tensor_capture API"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = SimpleModel()
        self.input_tensor = torch.randn(2, 10)
        
    def test_end_to_end_tensor_capture(self):
        """Test end-to-end tensor capture workflow using the API"""
        # Define modules to capture
        modules_to_capture = ["layers.0", "layers.1"]
        
        # Enable tensor capture
        model_with_capture = enable_tensor_capture(self.model, modules_to_capture)
        
        # Run the model
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors as dictionary
        captured_tensors_dict = get_captured_tensors_dict()
        
        # Check that we got the expected number of tensors
        self.assertEqual(len(captured_tensors_dict), len(modules_to_capture))
        
        # Get the values as a list to check shapes
        tensors = list(captured_tensors_dict.values())
        
        # Check that the tensors have the expected shapes
        self.assertEqual(tensors[0].shape, (2, 20))  # Output of layers.0
        self.assertEqual(tensors[1].shape, (2, 30))  # Output of layers.1
        
        # Disable tensor capture
        disable_tensor_capture(model_with_capture)
        
        # After disabling, we should get an empty dictionary
        empty_dict = get_captured_tensors_dict()
        self.assertEqual(len(empty_dict), 0)


if __name__ == '__main__':
    unittest.main()