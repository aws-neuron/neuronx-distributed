"""
Unit tests for the API functions in the tensor_capture module.
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from neuronx_distributed.utils.tensor_capture import (
    enable_tensor_capture,
    disable_tensor_capture,
    get_available_modules,
    register_tensor,
    get_captured_tensors_dict
)
from neuronx_distributed.utils.tensor_capture.registry import TensorRegistry

# Import test utilities for deduplication
from .test_utils import (
    SimpleModel,
    AttentionModel,
    DataclassModel,
    TupleOutputModel,
    TestFixtures,
    TestHelpers,
    TestConstants
)


class TestEnableTensorCapture(unittest.TestCase):
    """Test cases for the enable_tensor_capture function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = TestHelpers.setup_registry()
        self.input_tensor = TestFixtures.get_input_tensor()
        self.attention_mask = TestFixtures.get_attention_mask()
        self.position_ids = TestFixtures.get_position_ids()
        self.model = SimpleModel()
        
    def tearDown(self):
        """Clean up after tests"""
        TestHelpers.teardown_registry()
        
    def test_enable_tensor_capture_basic(self):
        """Test basic tensor capture enabling"""
        modules_to_capture = TestConstants.SIMPLE_MODULES[:2]  # ["layers.0", "layers.1"]
        
        model_with_capture = enable_tensor_capture(self.model, modules_to_capture)
        
        # Should return the same model instance
        self.assertIs(model_with_capture, self.model)
        
        # Registry should be enabled and configured
        self.assertTrue(self.registry.enabled)
        self.assertEqual(self.registry.model_info.modules_to_capture, modules_to_capture)
        
    def test_enable_tensor_capture_with_max_tensors(self):
        """Test enabling tensor capture with max_tensors parameter"""
        modules_to_capture = ["layers.0"]
        max_tensors = 10
        
        model_with_capture = enable_tensor_capture(
            self.model, modules_to_capture, max_tensors=max_tensors
        )
        
        self.assertIs(model_with_capture, self.model)
        self.assertEqual(self.registry.model_info.max_tensors, max_tensors)
        
    def test_enable_tensor_capture_with_capture_inputs(self):
        """Test enabling tensor capture with capture_inputs parameter"""
        modules_to_capture = ["layers.0"]
        
        model_with_capture = enable_tensor_capture(
            self.model, modules_to_capture, capture_inputs=True
        )
        
        self.assertIs(model_with_capture, self.model)
        self.assertTrue(self.registry.model_info.capture_inputs)
        
    def test_enable_tensor_capture_none_modules(self):
        """Test enabling tensor capture with None modules"""
        model_with_capture = enable_tensor_capture(self.model, None)
        
        self.assertIs(model_with_capture, self.model)
        self.assertEqual(self.registry.model_info.modules_to_capture, [])
        
    def test_enable_tensor_capture_invalid_modules(self):
        """Test enabling tensor capture with invalid module names"""
        invalid_modules = ["invalid_module", "layers.999"]
        
        with self.assertRaises(ValueError):
            enable_tensor_capture(self.model, invalid_modules)


class TestDisableTensorCapture(unittest.TestCase):
    """Test cases for the disable_tensor_capture function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = TestHelpers.setup_registry()
        self.input_tensor = TestFixtures.get_input_tensor()
        self.attention_mask = TestFixtures.get_attention_mask()
        self.position_ids = TestFixtures.get_position_ids()
        self.model = SimpleModel()
        
    def tearDown(self):
        """Clean up after tests"""
        TestHelpers.teardown_registry()
        
    def test_disable_tensor_capture(self):
        """Test disabling tensor capture"""
        modules_to_capture = TestConstants.SIMPLE_MODULES[:1]  # ["layers.0"]
        
        # First enable tensor capture
        model_with_capture = enable_tensor_capture(self.model, modules_to_capture)
        self.assertTrue(self.registry.enabled)
        
        # Then disable it
        restored_model = disable_tensor_capture(model_with_capture)
        
        # Should return the same model instance
        self.assertIs(restored_model, self.model)
        
        # Registry should be disabled and cleared
        self.assertFalse(self.registry.enabled)
        self.assertEqual(len(self.registry.model_info.module_tensors), 0)
        self.assertEqual(len(self.registry.model_info.manual_tensors), 0)


class TestGetAvailableModules(unittest.TestCase):
    """Test cases for the get_available_modules function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = SimpleModel()
        
    def test_get_available_modules(self):
        """Test getting available modules from a model"""
        modules = get_available_modules(self.model)
        
        # Should include expected modules
        expected_modules = ["layers", "layers.0", "layers.1", "layers.2", "activation"]
        TestHelpers.assert_tensor_keys_present(self, {m: None for m in modules}, expected_modules)
        
    def test_get_available_modules_attention_model(self):
        """Test getting available modules from attention model"""
        model = AttentionModel()
        modules = get_available_modules(model)
        
        # Should include attention-specific modules
        expected_modules = ["linear", "norm"]
        TestHelpers.assert_tensor_keys_present(self, {m: None for m in modules}, expected_modules)


class TestRegisterTensor(unittest.TestCase):
    """Test cases for the register_tensor function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = TestHelpers.setup_registry()
        self.input_tensor = TestFixtures.get_input_tensor()
        self.attention_mask = TestFixtures.get_attention_mask()
        self.position_ids = TestFixtures.get_position_ids()
        
    def tearDown(self):
        """Clean up after tests"""
        TestHelpers.teardown_registry()
        
    def test_register_tensor_enabled(self):
        """Test registering a tensor when capture is enabled"""
        self.registry.configure(enabled=True, modules=[], max_tensors=5)
        
        tensor = TestFixtures.get_input_tensor()
        register_tensor("test_tensor", tensor)
        
        # Should be registered
        manual_tensors = self.registry.get_manual_tensors()
        self.assertEqual(len(manual_tensors), 1)
        
    def test_register_tensor_disabled(self):
        """Test registering a tensor when capture is disabled"""
        self.registry.configure(enabled=False)
        
        tensor = TestFixtures.get_input_tensor()
        register_tensor("test_tensor", tensor)
        
        # Should not be registered
        manual_tensors = self.registry.get_manual_tensors()
        self.assertEqual(len(manual_tensors), 0)
        
    def test_register_tensor_module_match(self):
        """Test registering a tensor that matches a monitored module"""
        modules = ["linear"]
        self.registry.configure(enabled=True, modules=modules)
        
        tensor = TestFixtures.get_input_tensor()
        register_tensor("linear", tensor)
        
        # Should be registered as module tensor
        module_tensors = self.registry.get_module_tensors()
        manual_tensors = self.registry.get_manual_tensors()
        
        self.assertEqual(len(module_tensors), 1)
        self.assertEqual(len(manual_tensors), 0)


class TestGetCapturedTensorsDict(unittest.TestCase):
    """Test cases for the get_captured_tensors_dict function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = TestHelpers.setup_registry()
        self.input_tensor = TestFixtures.get_input_tensor()
        self.attention_mask = TestFixtures.get_attention_mask()
        self.position_ids = TestFixtures.get_position_ids()
        
    def tearDown(self):
        """Clean up after tests"""
        TestHelpers.teardown_registry()
        
    def test_get_captured_tensors_dict_empty(self):
        """Test getting captured tensors when none are registered"""
        tensors_dict = get_captured_tensors_dict()
        
        self.assertEqual(len(tensors_dict), 0)
        
    def test_get_captured_tensors_dict_with_tensors(self):
        """Test getting captured tensors with registered tensors"""
        modules = ["linear"]
        self.registry.configure(enabled=True, modules=modules, max_tensors=2)
        
        # Register module tensor
        module_tensor = TestFixtures.get_input_tensor()
        register_tensor("linear", module_tensor)
        
        # Register manual tensor
        manual_tensor = TestFixtures.get_input_tensor()
        register_tensor("manual_tensor", manual_tensor)
        
        tensors_dict = get_captured_tensors_dict()
        
        # Should have both tensors
        self.assertEqual(len(tensors_dict), 2)
        
        # Module tensor should come first
        tensor_keys = list(tensors_dict.keys())
        self.assertEqual(tensor_keys[0], "linear")


class TestIntegrationWithAttentionModel(unittest.TestCase):
    """Integration tests using the AttentionModel with professional naming"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = TestHelpers.setup_registry()
        self.input_tensor = TestFixtures.get_input_tensor()
        self.attention_mask = TestFixtures.get_attention_mask()
        self.position_ids = TestFixtures.get_position_ids()
        self.model = AttentionModel()
        
    def tearDown(self):
        """Clean up after tests"""
        TestHelpers.teardown_registry()
        
    def test_attention_model_integration(self):
        """Test complete workflow with attention model"""
        modules_to_capture = TestConstants.ATTENTION_MODULES[:1]  # ["linear"]
        
        # Enable tensor capture with input capture
        model_with_capture = enable_tensor_capture(
            self.model, modules_to_capture, max_tensors=3, capture_inputs=True
        )
        
        # Run model with attention parameters
        _ = model_with_capture(
            self.input_tensor,
            attention_mask=self.attention_mask,
            position_ids=self.position_ids
        )
        
        # Get captured tensors
        tensors_dict = get_captured_tensors_dict()
        
        # Should capture attention parameters as kwargs
        expected_keys = [
            "linear.inputs.0",  # Positional input
            "linear.inputs.kwargs.attention_mask",
            "linear.inputs.kwargs.position_ids",
            "linear.outputs"
        ]
        TestHelpers.assert_tensor_keys_present(self, tensors_dict, expected_keys)
        
        # Verify tensor shapes
        expected_shapes = {
            "linear.inputs.0": TestConstants.INPUT_SHAPE,
            "linear.inputs.kwargs.attention_mask": TestConstants.ATTENTION_MASK_SHAPE,
            "linear.inputs.kwargs.position_ids": TestConstants.POSITION_IDS_SHAPE,
            "linear.outputs": TestConstants.HIDDEN_SHAPE_20  # AttentionLinear outputs (2, 20)
        }
        TestHelpers.assert_tensor_shapes(self, tensors_dict, expected_shapes)
        
        # Disable tensor capture
        restored_model = disable_tensor_capture(model_with_capture)
        self.assertIs(restored_model, self.model)
        
    def test_attention_model_manual_registration(self):
        """Test manual tensor registration with attention model"""
        modules_to_capture = ["linear"]
        
        # Enable tensor capture
        model_with_capture = enable_tensor_capture(
            self.model, modules_to_capture, max_tensors=2
        )
        
        # Register manual tensors
        register_tensor("intermediate_activation", self.input_tensor)
        register_tensor("attention_weights", self.attention_mask)
        
        # Run model
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors
        tensors_dict = get_captured_tensors_dict()
        
        # Should have module output + manual tensors
        self.assertGreaterEqual(len(tensors_dict), 3)
        
        # Check for manual tensors (they get prefixed)
        manual_keys = TestHelpers.get_keys_with_pattern(tensors_dict, "manual_")
        self.assertEqual(len(manual_keys), 2)


if __name__ == '__main__':
    unittest.main()
