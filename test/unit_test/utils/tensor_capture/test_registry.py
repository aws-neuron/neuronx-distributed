"""
Unit tests for the TensorRegistry class in the tensor_capture module.
"""

import unittest
import torch
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

from neuronx_distributed.utils.tensor_capture.registry import TensorRegistry


class TestTensorRegistry(unittest.TestCase):
    """Test cases for the TensorRegistry class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Get the singleton instance and reset it
        self.registry = TensorRegistry.get_instance()
        self.registry.clear()
        self.registry.configure(enabled=True, modules=["module1", "module2"], max_tensors=5)
        
        # Create test tensors
        self.test_tensor = torch.tensor([1.0, 2.0, 3.0])
        
    def tearDown(self):
        """Clean up after tests"""
        self.registry.clear()
        
    def test_singleton_pattern(self):
        """Test that the registry follows the singleton pattern"""
        registry1 = TensorRegistry.get_instance()
        registry2 = TensorRegistry.get_instance()
        
        self.assertIs(registry1, registry2)
        
    def test_register_monitored_tensor(self):
        """Test registering a tensor from a monitored module"""
        self.registry.register_tensor("module1", self.test_tensor)
        
        module_tensors = self.registry.get_module_tensors()
        self.assertIn("module1", module_tensors)
        self.assertTrue(torch.equal(module_tensors["module1"], self.test_tensor))
        
    def test_register_manual_tensor(self):
        """Test manually registering a tensor"""
        self.registry.register_tensor("manual_tensor", self.test_tensor)
        
        manual_tensors = self.registry.get_manual_tensors()
        self.assertEqual(len(manual_tensors), 1)
        self.assertEqual(self.registry.get_manual_tensor_count(), 1)
        
    def test_max_tensors_limit(self):
        """Test that the max_tensors limit is enforced"""
        # Register more than max_tensors
        for i in range(10):
            self.registry.register_tensor(f"manual_tensor_{i}", self.test_tensor)
            
        manual_tensors = self.registry.get_manual_tensors()
        self.assertEqual(len(manual_tensors), 5)  # max_tensors
        self.assertEqual(self.registry.get_manual_tensor_count(), 5)
        
    def test_clear_registry(self):
        """Test clearing the registry"""
        self.registry.register_tensor("module1", self.test_tensor)
        self.registry.register_tensor("manual_tensor", self.test_tensor)
        
        self.registry.clear()
        
        module_tensors = self.registry.get_module_tensors()
        manual_tensors = self.registry.get_manual_tensors()
        self.assertEqual(len(module_tensors), 0)
        self.assertEqual(len(manual_tensors), 0)
        self.assertEqual(self.registry.get_manual_tensor_count(), 0)
        
    def test_registry_disabled(self):
        """Test that tensors are not registered when the registry is disabled"""
        self.registry.configure(enabled=False)
        self.registry.register_tensor("module1", self.test_tensor)
        
        module_tensors = self.registry.get_module_tensors()
        self.assertEqual(len(module_tensors), 0)
        
    def test_register_non_string_name(self):
        """Test registering a tensor with a non-string name"""
        # Use an object as a name
        name = object()
        self.registry.register_tensor(name, self.test_tensor)
        
        manual_tensors = self.registry.get_manual_tensors()
        self.assertEqual(len(manual_tensors), 1)
        
        # The key should be "manual_tensor_0"
        self.assertIn("manual_tensor_0", manual_tensors)

    def test_register_tensor_duplicate_name(self):
        """Test registering a tensor with a non-string name"""
        # Use an object as a name
        name = "my_tensor"
        self.registry.register_tensor(name, self.test_tensor)
        self.registry.register_tensor("my_another_tensor", self.test_tensor)
        self.registry.register_tensor(name, self.test_tensor)
 
        manual_tensors = self.registry.get_manual_tensors()

        self.assertEqual(len(manual_tensors), 3)
        self.assertIn("manual_my_tensor", manual_tensors)
        self.assertIn("manual_my_tensor_1", manual_tensors)
        self.assertIn("manual_my_another_tensor", manual_tensors)
        
    def test_register_tensor_with_clone(self):
        """Test that registered tensors are cloned"""
        original = torch.tensor([1.0, 2.0, 3.0])
        self.registry.register_tensor("module1", original)
        
        # Modify the original tensor
        original[0] = 999.0
        
        # Get the registered tensor
        module_tensors = self.registry.get_module_tensors()
        registered = module_tensors["module1"]
        
        # The registered tensor should not be affected by the modification
        self.assertEqual(registered[0].item(), 1.0)
        
    def test_get_monitored_tensor_count(self):
        """Test getting the count of tensors from monitored modules"""
        self.registry.register_tensor("module1", self.test_tensor)
        self.registry.register_tensor("module2", self.test_tensor)
        self.registry.register_tensor("manual_tensor", self.test_tensor)
        
        count = self.registry.get_monitored_tensor_count()
        self.assertEqual(count, 2)
        
    def test_get_total_tensor_count(self):
        """Test getting the total count of all tensors"""
        self.registry.register_tensor("module1", self.test_tensor)
        self.registry.register_tensor("manual_tensor", self.test_tensor)
        
        count = self.registry.get_total_tensor_count()
        self.assertEqual(count, 2)
        
    def test_reconfigure_registry(self):
        """Test reconfiguring the registry"""
        # Register some tensors
        self.registry.register_tensor("module1", self.test_tensor)
        self.registry.register_tensor("manual_tensor", self.test_tensor)
        
        # Reconfigure the registry
        self.registry.configure(enabled=True, modules=["module3"], max_tensors=10)
        
        # Check that the registry was cleared
        module_tensors = self.registry.get_module_tensors()
        manual_tensors = self.registry.get_manual_tensors()
        self.assertEqual(len(module_tensors), 0)
        self.assertEqual(len(manual_tensors), 0)
        
        # Check that the new configuration is in effect
        self.registry.register_tensor("module3", self.test_tensor)
        module_tensors = self.registry.get_module_tensors()
        self.assertIn("module3", module_tensors)
        
    def test_register_tensor_with_detach(self):
        """Test that registered tensors are detached from the computation graph"""
        # Create a tensor that requires grad
        original = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        self.registry.register_tensor("module1", original)
        
        # Get the registered tensor
        module_tensors = self.registry.get_module_tensors()
        registered = module_tensors["module1"]
        
        # The registered tensor should not require grad
        self.assertFalse(registered.requires_grad)
        
    def test_get_captured_tensors_dict(self):
        """Test getting all tensors as an ordered dictionary"""
        # Register some module tensors
        self.registry.register_tensor("module1", torch.tensor([1.0]))
        self.registry.register_tensor("module2", torch.tensor([2.0]))
        
        # Register some manual tensors
        self.registry.register_tensor("manual1", torch.tensor([3.0]))
        self.registry.register_tensor("manual2", torch.tensor([4.0]))
        
        # Get all tensors as dictionary
        tensors_dict = self.registry.get_captured_tensors_dict()
        
        # Check that we got the expected tensors
        self.assertEqual(len(tensors_dict), 4)
        self.assertTrue("module1" in tensors_dict)
        self.assertTrue("module2" in tensors_dict)
        self.assertTrue("manual_manual1" in tensors_dict or "manual1" in tensors_dict)
        self.assertTrue("manual_manual2" in tensors_dict or "manual2" in tensors_dict)
        
        # Check tensor values
        self.assertEqual(tensors_dict["module1"].item(), 1.0)
        self.assertEqual(tensors_dict["module2"].item(), 2.0)
        
    def test_register_tensor_with_module_name_in_string(self):
        """Test registering a tensor with a name that contains a monitored module name"""
        # Register a tensor with a name that contains "module1"
        self.registry.register_tensor("module1.outputs", self.test_tensor)
        
        # Check that it was registered as a module tensor
        module_tensors = self.registry.get_module_tensors()
        self.assertIn("module1.outputs", module_tensors)
        
        # Register a tensor with a name that contains "module2"
        self.registry.register_tensor("module2.inputs", self.test_tensor)
        
        # Check that it was registered as a module tensor
        module_tensors = self.registry.get_module_tensors()
        self.assertIn("module2.inputs", module_tensors)

if __name__ == '__main__':
    unittest.main()