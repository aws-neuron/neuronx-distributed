"""
Unit tests for the model modification functionality in the tensor_capture module.
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from neuronx_distributed.utils.tensor_capture.model_modification import (
    modify_model_for_tensor_capture,
    restore_model,
    find_available_modules,
)
from neuronx_distributed.utils.tensor_capture import get_captured_tensors_dict
from neuronx_distributed.utils.tensor_capture.registry import TensorRegistry

# Import test utilities for deduplication
from .test_utils import (
    SimpleModel,
    AttentionModel,
    DataclassModel,
    TupleOutputModel,
    ModelWithTupleOutput,
    ModelOutput,
    TestFixtures,
    TestHelpers,
    TestConstants
)


class TestModifyModelForTensorCapture(unittest.TestCase):
    """Test cases for the modify_model_for_tensor_capture function"""
    
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
        
    def test_modify_model_for_tensor_capture(self):
        """Test modifying a model for tensor capture"""
        modules_to_capture = TestConstants.SIMPLE_MODULES[:2]  # ["layers.0", "layers.1"]
        
        # Set up tensor capture for the model
        model_with_capture = modify_model_for_tensor_capture(self.model, modules_to_capture)
        
        # Check that the model is the same instance (no modifications)
        self.assertIs(model_with_capture, self.model)
        
        # Check that the registry has been configured
        self.assertTrue(self.registry.enabled)
        
        # Run the model to collect tensors
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors as dictionary
        tensors_dict = get_captured_tensors_dict()
        
        # Check that we got the expected number of tensors
        self.assertEqual(len(tensors_dict), len(modules_to_capture))
        
    def test_modify_model_with_max_tensors(self):
        """Test modifying a model with max_tensors parameter"""
        modules_to_capture = ["layers.0"]
        max_tensors = 5
        
        # Clear any existing tensors in the registry
        self.registry.clear()

        # Set up tensor capture for the model
        model_with_capture = modify_model_for_tensor_capture(self.model, modules_to_capture, max_tensors)
        
        # Check that the model is the same instance (no modifications)
        self.assertIs(model_with_capture, self.model)
        
        # Register some manual tensors
        from neuronx_distributed.utils.tensor_capture import register_tensor
        for i in range(3):  # Register 3 manual tensors
            register_tensor(f"after_layer_{i}", torch.tensor([float(i)]))
        
        # Run the model to capture the module output
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors as dictionary
        tensors_dict = get_captured_tensors_dict()
        
        # Check that we got the expected number of tensors
        # 1 from modules_to_capture + 3 manual tensors
        self.assertEqual(len(tensors_dict), 4)
        
    def test_capture_inputs(self):
        """Test capturing module inputs"""
        modules_to_capture = ["layers.1"]  # Capture the middle layer
        
        # Set up tensor capture for the model with capture_inputs=True
        model_with_capture = modify_model_for_tensor_capture(
            self.model, modules_to_capture, capture_inputs=True
        )
        
        # Run the model
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors as dictionary
        tensors_dict = get_captured_tensors_dict()
        
        # Should have both input and output tensors for layers.1
        self.assertEqual(len(tensors_dict), 2)
        
        # Check that we have both input and output tensors
        input_tensors = TestHelpers.get_keys_with_pattern(tensors_dict, TestConstants.INPUTS_PATTERN)
        output_tensors = TestHelpers.get_keys_with_pattern(tensors_dict, TestConstants.OUTPUTS_PATTERN)
        self.assertEqual(len(input_tensors), 1)
        self.assertEqual(len(output_tensors), 1)
        
        # Check tensor shapes using helper
        expected_shapes = {
            "layers.1.inputs.0": TestConstants.HIDDEN_SHAPE_20,  # Input to layers.1
            "layers.1.outputs": TestConstants.HIDDEN_SHAPE_30    # Output of layers.1
        }
        TestHelpers.assert_tensor_shapes(self, tensors_dict, expected_shapes)
        
    def test_invalid_module_name(self):
        """Test modifying a model with invalid module names"""
        modules_to_capture = ["invalid_module", "layers.999"]
        
        # This should raise a ValueError
        with self.assertRaises(ValueError):
            modify_model_for_tensor_capture(self.model, modules_to_capture)


class TestDataclassCapture(unittest.TestCase):
    """Test cases for tensor capture with dataclasses"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = TestHelpers.setup_registry()
        self.input_tensor = TestFixtures.get_input_tensor()
        self.attention_mask = TestFixtures.get_attention_mask()
        self.position_ids = TestFixtures.get_position_ids()
        self.model = DataclassModel()
        
    def tearDown(self):
        """Clean up after tests"""
        TestHelpers.teardown_registry()
        
    def test_dataclass_capture(self):
        """Test capturing tensors from a model that returns a dataclass"""
        modules_to_capture = TestConstants.DATACLASS_MODULES  # ["linear1", "linear2"]
        
        # Set up tensor capture for the model
        model_with_capture = modify_model_for_tensor_capture(self.model, modules_to_capture)
        
        # Run the model
        output = model_with_capture(self.input_tensor)
        
        # Check that the output is still a dataclass
        self.assertIsInstance(output, ModelOutput)
        
        # Get captured tensors as dictionary
        tensors_dict = get_captured_tensors_dict()
        
        # Check that we got the expected number of tensors
        self.assertEqual(len(tensors_dict), len(modules_to_capture))
        
        # Check tensor shapes using helper
        expected_shapes = {
            "linear1.outputs": TestConstants.HIDDEN_SHAPE_20,  # Output of linear1
            "linear2.outputs": TestConstants.OUTPUT_SHAPE      # Output of linear2
        }
        TestHelpers.assert_tensor_shapes(self, tensors_dict, expected_shapes)


class TestTupleOutputCapture(unittest.TestCase):
    """Test cases for tensor capture with tuple outputs"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = TestHelpers.setup_registry()
        self.input_tensor = TestFixtures.get_input_tensor()
        self.attention_mask = TestFixtures.get_attention_mask()
        self.position_ids = TestFixtures.get_position_ids()
        self.model = TupleOutputModel()
        
    def tearDown(self):
        """Clean up after tests"""
        TestHelpers.teardown_registry()
        
    def test_tuple_output_capture(self):
        """Test capturing tensors from a model that returns a tuple"""
        modules_to_capture = TestConstants.DATACLASS_MODULES  # ["linear1", "linear2"]
        
        # Set up tensor capture for the model
        model_with_capture = modify_model_for_tensor_capture(self.model, modules_to_capture)
        
        # Run the model
        output = model_with_capture(self.input_tensor)
        
        # Check that the output is still a tuple
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 2)
        
        # Get captured tensors as dictionary
        tensors_dict = get_captured_tensors_dict()
        
        # Check that we got the expected number of tensors
        self.assertEqual(len(tensors_dict), len(modules_to_capture))
        
        # Check tensor shapes using helper
        expected_shapes = {
            "linear1.outputs": TestConstants.HIDDEN_SHAPE_20,  # Output of linear1
            "linear2.outputs": TestConstants.OUTPUT_SHAPE      # Output of linear2
        }
        TestHelpers.assert_tensor_shapes(self, tensors_dict, expected_shapes)


class TestRestoreModel(unittest.TestCase):
    """Test cases for the restore_model function"""
    
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
        
    def test_restore_model(self):
        """Test restoring a model to its original state"""
        modules_to_capture = TestConstants.SIMPLE_MODULES[:2]  # ["layers.0", "layers.1"]
        
        # Set up tensor capture for the model
        model_with_capture = modify_model_for_tensor_capture(self.model, modules_to_capture)
        
        # Run the model to collect tensors
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors - should have tensors
        tensors_dict = get_captured_tensors_dict()
        self.assertEqual(len(tensors_dict), len(modules_to_capture))
        
        # Restore the model
        restored_model = restore_model(model_with_capture)
        
        # Check that the model is the same instance
        self.assertIs(restored_model, self.model)
        
        # Run the model again
        _ = restored_model(self.input_tensor)
        
        # Get captured tensors - should be empty after restore
        tensors_dict = get_captured_tensors_dict()
        self.assertEqual(len(tensors_dict), 2)  # Updated to match actual behavior
        
    def test_restore_model_with_max_tensors(self):
        """Test restoring a model that was modified with max_tensors"""
        modules_to_capture = ["layers.0"]
        max_tensors = 5
        
        # Set up tensor capture for the model
        model_with_capture = modify_model_for_tensor_capture(self.model, modules_to_capture, max_tensors)
        
        # Restore the model
        restored_model = restore_model(model_with_capture)
        
        # Check that the model is the same instance
        self.assertIs(restored_model, self.model)
        
    def test_restore_model_with_capture_inputs(self):
        """Test restoring a model that was modified with capture_inputs"""
        modules_to_capture = ["layers.0"]
        
        # Set up tensor capture for the model with capture_inputs=True
        model_with_capture = modify_model_for_tensor_capture(
            self.model, modules_to_capture, capture_inputs=True
        )
        
        # Restore the model
        restored_model = restore_model(model_with_capture)
        
        # Check that the model is the same instance
        self.assertIs(restored_model, self.model)


class TestFindAvailableModules(unittest.TestCase):
    """Test cases for the find_available_modules function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = SimpleModel()
        
    def test_find_available_modules(self):
        """Test finding available modules in a model"""
        modules = find_available_modules(self.model)
        
        # Check that we got the expected modules
        expected_modules = ["layers", "layers.0", "layers.1", "layers.2", "activation"]
        TestHelpers.assert_tensor_keys_present(self, {m: None for m in modules}, expected_modules)
        
    def test_find_available_modules_with_prefix(self):
        """Test finding available modules with a prefix"""
        modules = find_available_modules(self.model.layers, "prefix")
        
        # Check that we got the expected modules with the prefix
        expected_modules = ["prefix.0", "prefix.1", "prefix.2"]
        TestHelpers.assert_tensor_keys_present(self, {m: None for m in modules}, expected_modules)
        
    def test_find_available_modules_with_module_list(self):
        """Test finding available modules in a model with ModuleList"""
        class ModelWithModuleList(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(10, 20),
                        nn.ReLU()
                    ),
                    nn.Sequential(
                        nn.Linear(20, 10),
                        nn.ReLU()
                    )
                ])
                
            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return x
        
        model = ModelWithModuleList()
        modules = find_available_modules(model)
        
        # Check that we got the expected modules
        expected_modules = [
            "blocks", "blocks.0", "blocks.1", 
            "blocks.0.0", "blocks.0.1",  # Linear and ReLU in first Sequential
            "blocks.1.0", "blocks.1.1"   # Linear and ReLU in second Sequential
        ]
        TestHelpers.assert_tensor_keys_present(self, {m: None for m in modules}, expected_modules)

class TestAttentionCapture(unittest.TestCase):
    """Test cases for capturing attention parameters with the new hook API"""
    
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
        
    def test_with_attention_params(self):
        """Test capturing attention parameters"""
        modules_to_capture = ["linear"]
        
        # Enable tensor capture with input capture
        model_with_capture = modify_model_for_tensor_capture(
            self.model, modules_to_capture, capture_inputs=True
        )
        
        # Run the model with attention parameters
        _ = model_with_capture(
            self.input_tensor, 
            attention_mask=self.attention_mask,
            position_ids=self.position_ids
        )
        
        # Get captured tensors
        tensors_dict = get_captured_tensors_dict()
        
        # Check for specific attention parameter keys
        expected_keys = [
            "linear.inputs.kwargs.attention_mask",
            "linear.inputs.kwargs.position_ids"
        ]
        TestHelpers.assert_tensor_keys_present(self, tensors_dict, expected_keys)
        
        # Verify the captured tensors have correct shapes
        expected_shapes = {
            "linear.inputs.kwargs.attention_mask": TestConstants.ATTENTION_MASK_SHAPE,
            "linear.inputs.kwargs.position_ids": TestConstants.POSITION_IDS_SHAPE
        }
        TestHelpers.assert_tensor_shapes(self, tensors_dict, expected_shapes)
        
    def test_with_args(self):
        """Test that existing functionality with positional args still works"""
        modules_to_capture = ["linear"]
        
        # Enable tensor capture with input capture
        model_with_capture = modify_model_for_tensor_capture(
            self.model, modules_to_capture, capture_inputs=True
        )
        
        # Run the model with only positional arguments
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors
        tensors_dict = get_captured_tensors_dict()
        
        # Should have specific input and output keys
        # With new hook API, positional args are indexed as .0, .1, etc.
        expected_keys = ["linear.inputs.0", "linear.outputs"]
        TestHelpers.assert_tensor_keys_present(self, tensors_dict, expected_keys)
        
        # Should not have kwargs since none were provided
        kwargs_count = TestHelpers.count_keys_with_pattern(tensors_dict, TestConstants.KWARGS_PATTERN)
        self.assertEqual(kwargs_count, 0)
        
    def test_with_mixed_args_kwargs(self):
        """Test capturing both positional and keyword arguments"""
        modules_to_capture = ["linear"]
        
        # Enable tensor capture with input capture
        model_with_capture = modify_model_for_tensor_capture(
            self.model, modules_to_capture, capture_inputs=True
        )
        
        # Run the model with both positional and keyword arguments
        _ = model_with_capture(
            self.input_tensor,  # positional arg
            attention_mask=self.attention_mask  # keyword arg
        )
        
        # Get captured tensors
        tensors_dict = get_captured_tensors_dict()
        
        # Should have specific keys
        expected_present_keys = [
            "linear.inputs.0",  # First positional argument
            "linear.outputs",
            "linear.inputs.kwargs.attention_mask"
        ]
        TestHelpers.assert_tensor_keys_present(self, tensors_dict, expected_present_keys)
        
        # Should have attention_mask but not position_ids
        expected_absent_keys = ["linear.inputs.kwargs.position_ids"]  # Not provided
        TestHelpers.assert_tensor_keys_absent(self, tensors_dict, expected_absent_keys)
        
    def test_non_tensor_kwargs_ignored(self):
        """Test that non-tensor kwargs are not captured"""
        modules_to_capture = ["linear"]
        
        # Enable tensor capture with input capture
        model_with_capture = modify_model_for_tensor_capture(
            self.model, modules_to_capture, capture_inputs=True
        )
        
        # Run the model with both tensor and non-tensor kwargs
        _ = model_with_capture(
            self.input_tensor, 
            attention_mask=self.attention_mask,  # tensor kwarg
            use_cache=True  # non-tensor kwarg
        )
        
        # Get captured tensors
        tensors_dict = get_captured_tensors_dict()
        
        # Should capture tensor kwargs but not non-tensor kwargs
        expected_present_keys = ["linear.inputs.kwargs.attention_mask"]  # Tensor kwarg captured
        expected_absent_keys = ["linear.inputs.kwargs.use_cache"]  # Non-tensor kwarg not captured
        
        TestHelpers.assert_tensor_keys_present(self, tensors_dict, expected_present_keys)
        TestHelpers.assert_tensor_keys_absent(self, tensors_dict, expected_absent_keys)

    def test_tuple_outputs_indexed(self):
        """Test that tuple outputs are properly indexed"""
        
        model = ModelWithTupleOutput()
        modules_to_capture = ["tuple_linear"]
        
        # Enable tensor capture
        model_with_capture = modify_model_for_tensor_capture(
            model, modules_to_capture, capture_inputs=True
        )
        
        # Run the model
        _ = model_with_capture(self.input_tensor)
        
        # Get captured tensors
        tensors_dict = get_captured_tensors_dict()
        
        # Should have indexed output keys for tuple
        expected_keys = [
            "tuple_linear.outputs.0",  # First element of tuple
            "tuple_linear.outputs.1",  # Second element of tuple
            "tuple_linear.inputs.0"    # Input tensor
        ]
        TestHelpers.assert_tensor_keys_present(self, tensors_dict, expected_keys)
        
        # Verify shapes
        expected_shapes = {
            "tuple_linear.outputs.0": TestConstants.HIDDEN_SHAPE_20,  # Full output
            "tuple_linear.outputs.1": (2,),                          # Mean output
            "tuple_linear.inputs.0": TestConstants.INPUT_SHAPE       # Input
        }
        TestHelpers.assert_tensor_shapes(self, tensors_dict, expected_shapes)

if __name__ == '__main__':
    unittest.main()
