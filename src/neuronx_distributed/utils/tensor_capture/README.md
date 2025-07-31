# TensorCapture

TensorCapture is a utility for extracting intermediate tensor outputs from PyTorch models during execution. This is useful for debugging, visualization, analysis, or accuracy verification.

## Features

- Capture outputs from specific modules in a model
- Manually register tensors at specific points in your code
- Compatible with NeuronxDistributed's ModelBuilder for traced models

## Basic Usage

```python
from neuronx_distributed.utils.tensor_capture import (
    enable_tensor_capture,
    disable_tensor_capture,
    get_available_modules,
    register_tensor,
    get_captured_tensors_dict
)

# Create a model
model = create_model()

# Find available modules
available_modules = get_available_modules(model)
print(f"Available modules: {available_modules}")

# Define which modules to monitor
modules_to_capture = ["layers.0", "layers.1", "output_layer"]

# Enable tensor capture (outputs only)
model = enable_tensor_capture(model, modules_to_capture, max_tensors=5)

# Or enable tensor capture for both inputs and outputs
# model = enable_tensor_capture(model, modules_to_capture, max_tensors=5, capture_inputs=True)

# Run the model
inputs = create_inputs()
outputs = model(inputs)

# Get captured tensors as an ordered dictionary
captured_tensors_dict = get_captured_tensors_dict()

# Process the captured tensors
for name, tensor in captured_tensors_dict.items():
    print(f"Tensor {name} shape: {tensor.shape}")

# Disable tensor capture when done
model = disable_tensor_capture(model)
```

## API Reference

### `enable_tensor_capture(model, modules_to_capture, max_tensors=None, capture_inputs=False)`

Enable tensor capture for a model.

- **Parameters**:
  - `model`: The model to enable tensor capture for
  - `modules_to_capture`: List of module names to monitor for tensor capture
  - `max_tensors`: Maximum number of manually registered tensors to store (optional)
  - `capture_inputs`: Whether to capture module input tensors in addition to outputs (optional, default=False)
- **Returns**: The model with tensor capture enabled
- **Raises**: ValueError if any module in modules_to_capture is not found in the model

### `disable_tensor_capture(model)`

Disable tensor capture for a model and restore it to its original state.

- **Parameters**:
  - `model`: The model to disable tensor capture for
- **Returns**: The restored model

### `get_available_modules(model)`

Get a list of all modules in a model that can have their outputs captured.

- **Parameters**:
  - `model`: The model to inspect
- **Returns**: List of available module paths

### `register_tensor(name, tensor)`

Manually register a tensor for capture.

- **Parameters**:
  - `name`: Name/identifier for the tensor
  - `tensor`: Tensor to register

### `get_captured_tensors_dict()`

Get all captured tensors as an ordered dictionary.

- **Returns**: OrderedDict mapping tensor names to their values, with the following structure:
  1. Module input tensors (if capture_inputs=True) followed by output tensors, for each module in modules_to_capture
  2. Manually registered tensors with their names preserved

## Advanced Usage

### Manual Tensor Registration

You can manually register tensors at specific points in your code:

```python
def forward(self, x):
    # ... existing code ...
    
    # Register a tensor at a specific point
    register_tensor("after_layer_1", x)
    
    # ... more code ...
    
    return output
```

### Integration with NeuronxDistributed's ModelBuilder

For integration with NeuronxDistributed's ModelBuilder, see the example at `examples/inference/tensor_capture/tensor_capture_example.py`.

This example demonstrates how to:
1. Create a custom BaseModelInstance with tensor capture
2. Update a model's forward method to return captured tensors

### Handling Tensor Capture in Traced Models

When using tensor capture with traced models, you need to ensure that the number of outputs is fixed. The `max_tensors` parameter helps with this by ensuring that the model always returns exactly the specified number of tensors, even if fewer are registered during execution.

```python
# In your model's forward method
def forward(self, x):
    # ... existing code ...
    
    # Get captured tensors if available
    from neuronx_distributed.utils.tensor_capture import get_captured_tensors_dict
    tensor_dict = get_captured_tensors_dict()
    
    # Return captured tensors along with regular outputs
    if tensor_dict:
        return output, tensor_dict
    else:
        return output
```

## Capturing Module Inputs and Outputs

By default, tensor capture only captures the outputs of specified modules. However, you can also capture the inputs to these modules by setting `capture_inputs=True`:

```python
# Enable tensor capture for both inputs and outputs
model = enable_tensor_capture(model, modules_to_capture, capture_inputs=True)
```

When input capture is enabled, the `get_captured_tensors_dict()` method will include input tensors before output tensors for each module. This can be useful for debugging how data flows through your model.

## Working with Complex Data Types

When capturing outputs from modules, the utility can handle various data types:

1. **Single tensors**: Most common case, directly captured
2. **Tuples of tensors**: Each tensor in the tuple is captured separately
3. **Dataclasses with tensor fields**: Each tensor field is extracted and captured

This capability is particularly useful when working with models that return complex outputs.

## Working with Sharded Tensors

When using tensor capture with distributed training or inference, especially with NeuronxDistributed's parallel layers like `ColumnParallelLinear`, you need to handle sharded tensors properly.

### Understanding Sharded Tensors

In tensor-parallel models, certain layers like `ColumnParallelLinear` split their weights and outputs across multiple devices. When capturing tensors from these layers:

1. The captured tensor represents only the rank 0 shard (a portion of the full tensor)
2. The size of the shard depends on the tensor parallelism degree
3. When comparing with non-sharded reference tensors, you need to account for this partitioning

### Example with ColumnParallelLinear

```python
# When comparing outputs from a ColumnParallelLinear layer
if "column_parallel_layer" in module_name:
    # When using TP>1, we only compare the first shard
    if tp_degree > 1:
        shard_size = reference_tensor.size(-1) // tp_degree
        max_diff = torch.max(torch.abs(
            captured_tensor - reference_tensor[..., :shard_size]
        ))
    else:
        max_diff = torch.max(torch.abs(captured_tensor - reference_tensor))
```

### Best Practices for Sharded Tensors

1. **Know your model's parallelism strategy**: Understand which layers are sharded and how
2. **Track tensor shapes**: Log the shapes of captured tensors to identify sharded ones
3. **Proper comparison**: When comparing with reference tensors:
   - For ColumnParallelLinear: Compare only the corresponding shard
   - For RowParallelLinear: The output is typically gathered, so comparison can be direct
4. **Visualization**: When visualizing sharded tensors, note that they represent only a portion of the complete tensor

See the example in the repository at `examples/inference/tensor_capture/tensor_capture_example.py` for a complete demonstration of capturing and comparing sharded tensors from parallel layers.

## Example

The `examples` directory contains a practical demonstration of how to use the tensor_capture utility:

### tensor_capture_example.py

A comprehensive example that demonstrates:

- Creating an MLP model with parallel layers (ColumnParallelLinear and RowParallelLinear)
- Enabling tensor capture on specific modules
- Manually registering tensors during forward pass
- Compiling the model with NeuronxDistributed's ModelBuilder
- Running inference with tensor capture on both CPU and Neuron
- Comparing outputs between CPU and Neuron execution, handling sharded tensors correctly

The example is located in the repository at `examples/inference/tensor_capture/tensor_capture_example.py`.

Run this example with:

```bash
# Demonstrate tensor capture on CPU
python examples/inference/tensor_capture/tensor_capture_example.py demo

# Compile a model with tensor capture
python examples/inference/tensor_capture/tensor_capture_example.py compile --output-path="traced_model_with_capture/" --max-tensors=2

# Run inference with tensor capture
python examples/inference/tensor_capture/tensor_capture_example.py inference --model-path="traced_model_with_capture/"
```

## Use Cases

- **Debugging**: Inspect intermediate tensors to identify issues in model execution
- **Visualization**: Capture activations for visualization tools
- **Analysis**: Collect statistics on intermediate outputs
- **Accuracy Verification**: Compare outputs between different implementations (e.g., CPU vs. Neuron)

## Best Practices

1. **Disable tensor capture when not needed**: Tensor capture adds overhead to model execution, so disable it when not actively debugging.
2. **Be selective about which modules to capture**: Capturing too many modules can slow down execution and consume memory.
3. **Consider tensor size**: Capturing large tensors can consume significant memory.
4. **Clean up**: Always call `disable_tensor_capture` when done to restore the model to its original state.
5. **For traced models**: Ensure that the number of outputs is fixed by using the `max_tensors` parameter.
6. **For accuracy verification**: Capture the same modules in both reference and test models for meaningful comparisons.
7. **For sharded tensors**: Be aware of the tensor parallelism degree and handle comparisons accordingly.