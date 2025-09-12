"""
Unit tests for the TensorRegistry class in the tensor_capture module.
"""

import unittest
import unittest.mock
import torch
from collections import OrderedDict

from neuronx_distributed.utils.tensor_capture.registry import TensorRegistry, CapturedModelInfo

# Import test utilities for deduplication
from .test_utils import TestFixtures, TestHelpers, TestConstants


class TestCapturedModelInfo(unittest.TestCase):
    """Test cases for the CapturedModelInfo class"""
    
    def test_init(self):
        """Test CapturedModelInfo initialization"""
        modules = TestConstants.SIMPLE_MODULES[:2]  # ["layers.0", "layers.1"]
        max_tensors = 10
        capture_inputs = True
        
        info = CapturedModelInfo(modules, max_tensors, capture_inputs)
        
        self.assertEqual(info.modules_to_capture, modules)
        self.assertEqual(info.max_tensors, max_tensors)
        self.assertEqual(info.capture_inputs, capture_inputs)
        self.assertEqual(len(info.hooks), 0)
        self.assertIsInstance(info.module_tensors, OrderedDict)
        self.assertIsInstance(info.manual_tensors, OrderedDict)
        self.assertEqual(len(info.module_tensors), 0)
        self.assertEqual(len(info.manual_tensors), 0)


class TestTensorRegistry(unittest.TestCase):
    """Test cases for the TensorRegistry class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = TestHelpers.setup_registry()
        self.input_tensor = TestFixtures.get_input_tensor()
        self.attention_mask = TestFixtures.get_attention_mask()
        self.position_ids = TestFixtures.get_position_ids()
        
    def tearDown(self):
        """Clean up after tests"""
        TestHelpers.teardown_registry()
        
    def test_singleton_pattern(self):
        """Test that TensorRegistry follows singleton pattern"""
        registry1 = TensorRegistry.get_instance()
        registry2 = TensorRegistry.get_instance()
        
        self.assertIs(registry1, registry2)
        
    def test_configure(self):
        """Test configuring the registry"""
        modules = TestConstants.ATTENTION_MODULES  # ["linear", "norm"]
        max_tensors = 15
        capture_inputs = True
        
        self.registry.configure(
            enabled=True, 
            modules=modules, 
            max_tensors=max_tensors, 
            capture_inputs=capture_inputs
        )
        
        self.assertTrue(self.registry.enabled)
        self.assertEqual(self.registry.model_info.modules_to_capture, modules)
        self.assertEqual(self.registry.model_info.max_tensors, max_tensors)
        self.assertEqual(self.registry.model_info.capture_inputs, capture_inputs)
        
    def test_register_tensor_disabled(self):
        """Test that tensor registration is ignored when disabled"""
        self.registry.enabled = False
        tensor = TestFixtures.get_input_tensor()
        
        self.registry.register_tensor("test_tensor", tensor)
        
        # Should not register anything when disabled
        self.assertEqual(len(self.registry.model_info.module_tensors), 0)
        self.assertEqual(len(self.registry.model_info.manual_tensors), 0)
        
    def test_register_module_tensor(self):
        """Test registering a tensor from a monitored module"""
        modules = ["linear"]
        self.registry.configure(enabled=True, modules=modules)
        
        tensor = TestFixtures.get_input_tensor()
        self.registry.register_tensor("linear", tensor)
        
        # Should be registered as module tensor
        self.assertEqual(len(self.registry.model_info.module_tensors), 1)
        self.assertEqual(len(self.registry.model_info.manual_tensors), 0)
        self.assertIn("linear", self.registry.model_info.module_tensors)
        
    def test_register_manual_tensor(self):
        """Test registering a manual tensor"""
        self.registry.configure(enabled=True, modules=[], max_tensors=5)
        
        tensor = TestFixtures.get_input_tensor()
        self.registry.register_tensor("custom_tensor", tensor)
        
        # Should be registered as manual tensor
        self.assertEqual(len(self.registry.model_info.module_tensors), 0)
        self.assertEqual(len(self.registry.model_info.manual_tensors), 1)
        
        # Check that it's stored with the manual prefix
        manual_keys = list(self.registry.model_info.manual_tensors.keys())
        self.assertTrue(any("manual_custom_tensor" in key for key in manual_keys))
        
    def test_register_tensor_max_limit(self):
        """Test that manual tensor registration respects max_tensors limit"""
        max_tensors = 2
        self.registry.configure(enabled=True, modules=[], max_tensors=max_tensors)
        
        # Register more tensors than the limit
        for i in range(max_tensors + 2):
            tensor = torch.tensor([float(i)])
            self.registry.register_tensor(f"tensor_{i}", tensor)
        
        # Should only have max_tensors registered
        self.assertEqual(len(self.registry.model_info.manual_tensors), max_tensors)
        
    def test_get_captured_tensors_dict_empty(self):
        """Test getting captured tensors when none are registered"""
        self.registry.configure(enabled=True, modules=TestConstants.SIMPLE_MODULES)
        
        tensors_dict = self.registry.get_captured_tensors_dict()
        
        self.assertIsInstance(tensors_dict, OrderedDict)
        self.assertEqual(len(tensors_dict), 0)
        
    def test_get_captured_tensors_dict_with_modules(self):
        """Test getting captured tensors with module tensors"""
        modules = ["layers.0", "layers.1"]
        self.registry.configure(enabled=True, modules=modules)
        
        # Register some module tensors
        for module in modules:
            tensor = TestFixtures.get_input_tensor()
            self.registry.register_tensor(module, tensor)
        
        tensors_dict = self.registry.get_captured_tensors_dict()
        
        self.assertEqual(len(tensors_dict), len(modules))
        for module in modules:
            self.assertIn(module, tensors_dict)
            
    def test_get_captured_tensors_dict_with_inputs(self):
        """Test getting captured tensors with input capture enabled"""
        modules = ["linear"]
        self.registry.configure(enabled=True, modules=modules, capture_inputs=True)
        
        # Register input and output tensors
        input_tensor = TestFixtures.get_input_tensor()
        output_tensor = TestFixtures.get_input_tensor(input_size=20)  # Different size for output
        
        self.registry.register_tensor("linear.inputs.0", input_tensor)
        self.registry.register_tensor("linear.outputs", output_tensor)
        
        tensors_dict = self.registry.get_captured_tensors_dict()
        
        # Should have both input and output, with input first
        tensor_keys = list(tensors_dict.keys())
        self.assertEqual(len(tensor_keys), 2)
        self.assertTrue(tensor_keys[0].endswith("inputs.0"))
        self.assertTrue(tensor_keys[1].endswith("outputs"))
        
    def test_get_captured_tensors_dict_with_manual(self):
        """Test getting captured tensors with manual tensors"""
        self.registry.configure(enabled=True, modules=[], max_tensors=3)
        
        # Register some manual tensors
        for i in range(3):
            tensor = torch.tensor([float(i)])
            self.registry.register_tensor(f"manual_{i}", tensor)
        
        tensors_dict = self.registry.get_captured_tensors_dict()
        
        self.assertEqual(len(tensors_dict), 3)
        
    def test_get_captured_tensors_dict_mixed(self):
        """Test getting captured tensors with both module and manual tensors"""
        modules = ["linear"]
        self.registry.configure(enabled=True, modules=modules, max_tensors=2)
        
        # Register module tensor
        module_tensor = TestFixtures.get_input_tensor()
        self.registry.register_tensor("linear", module_tensor)
        
        # Register manual tensors
        for i in range(2):
            manual_tensor = torch.tensor([float(i)])
            self.registry.register_tensor(f"manual_{i}", manual_tensor)
        
        tensors_dict = self.registry.get_captured_tensors_dict()
        
        # Should have module tensors first, then manual tensors
        self.assertEqual(len(tensors_dict), 3)
        tensor_keys = list(tensors_dict.keys())
        self.assertEqual(tensor_keys[0], "linear")  # Module tensor first
        
    def test_clear(self):
        """Test clearing the registry"""
        # Configure and add some tensors
        self.registry.configure(enabled=True, modules=["linear"], max_tensors=5)
        tensor = TestFixtures.get_input_tensor()
        self.registry.register_tensor("linear", tensor)
        self.registry.register_tensor("manual", tensor)
        
        # Verify tensors are registered
        self.assertTrue(len(self.registry.model_info.module_tensors) > 0 or 
                      len(self.registry.model_info.manual_tensors) > 0)
        
        # Clear the registry
        self.registry.clear()
        
        # Should be reset to default state
        self.assertEqual(len(self.registry.model_info.module_tensors), 0)
        self.assertEqual(len(self.registry.model_info.manual_tensors), 0)
        self.assertEqual(self.registry.model_info.modules_to_capture, [])
        self.assertEqual(self.registry.model_info.max_tensors, 10)
        self.assertFalse(self.registry.model_info.capture_inputs)
        
    def test_tensor_cloning(self):
        """Test that registered tensors are properly cloned and detached"""
        self.registry.configure(enabled=True, modules=["linear"])
        
        # Create a tensor that requires grad
        original_tensor = torch.randn(2, 10, requires_grad=True)
        
        self.registry.register_tensor("linear", original_tensor)
        
        # Get the registered tensor
        tensors_dict = self.registry.get_captured_tensors_dict()
        registered_tensor = tensors_dict["linear"]
        
        # Should be cloned and detached
        self.assertFalse(registered_tensor.requires_grad)
        self.assertIsNot(registered_tensor, original_tensor)
        
        # But should have the same data
        self.assertTrue(torch.equal(registered_tensor, original_tensor.detach()))


class TestTensorRegistryHelperMethods(unittest.TestCase):
    """Test cases for TensorRegistry helper methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = TestHelpers.setup_registry()
        self.input_tensor = TestFixtures.get_input_tensor()
        self.attention_mask = TestFixtures.get_attention_mask()
        self.position_ids = TestFixtures.get_position_ids()
        
    def tearDown(self):
        """Clean up after tests"""
        TestHelpers.teardown_registry()
        
    def test_get_module_tensors(self):
        """Test getting module tensors"""
        modules = ["linear", "norm"]
        self.registry.configure(enabled=True, modules=modules)
        
        # Register module tensors
        for module in modules:
            tensor = TestFixtures.get_input_tensor()
            self.registry.register_tensor(module, tensor)
        
        module_tensors = self.registry.get_module_tensors()
        
        self.assertEqual(len(module_tensors), len(modules))
        for module in modules:
            self.assertIn(module, module_tensors)
            
    def test_get_manual_tensors(self):
        """Test getting manual tensors"""
        self.registry.configure(enabled=True, modules=[], max_tensors=3)
        
        # Register manual tensors
        for i in range(3):
            tensor = torch.tensor([float(i)])
            self.registry.register_tensor(f"manual_{i}", tensor)
        
        manual_tensors = self.registry.get_manual_tensors()
        
        self.assertEqual(len(manual_tensors), 3)
        
    def test_get_tensor_counts(self):
        """Test getting tensor counts"""
        modules = ["linear"]
        self.registry.configure(enabled=True, modules=modules, max_tensors=2)
        
        # Register one module tensor and two manual tensors
        module_tensor = TestFixtures.get_input_tensor()
        self.registry.register_tensor("linear", module_tensor)
        
        for i in range(2):
            manual_tensor = torch.tensor([float(i)])
            self.registry.register_tensor(f"manual_{i}", manual_tensor)
        
        # Test counts
        self.assertEqual(self.registry.get_monitored_tensor_count(), 1)
        self.assertEqual(self.registry.get_manual_tensor_count(), 2)
        self.assertEqual(self.registry.get_total_tensor_count(), 3)
        
    def test_remove_hooks(self):
        """Test removing hooks"""
        # Create mock hooks
        mock_hook1 = unittest.mock.MagicMock()
        mock_hook2 = unittest.mock.MagicMock()
        
        self.registry.model_info.hooks = [mock_hook1, mock_hook2]
        
        # Remove hooks
        self.registry.remove_hooks()
        
        # Should have called remove on each hook
        mock_hook1.remove.assert_called_once()
        mock_hook2.remove.assert_called_once()
        
        # Hooks list should be empty
        self.assertEqual(len(self.registry.model_info.hooks), 0)


if __name__ == '__main__':
    unittest.main()
