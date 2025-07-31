"""
Example of using tensor_capture utility with NeuronxDistributed parallel layers.

This example demonstrates:
1. Creating an MLP model with parallel layers
2. Enabling tensor capture to monitor intermediate tensors
3. Tracing the model with NeuronxDistributed
4. Running inference with tensor capture on both CPU and Neuron
5. Comparing outputs between CPU and Neuron execution

Usage:
  # Run CPU demonstration
  python tensor_capture_example.py demo

  # Compile model for Neuron with tensor capture
  python tensor_capture_example.py compile

  # Run inference on Neuron with tensor capture
  python tensor_capture_example.py inference
"""

import os
import torch
import argparse
import json
import torch.nn.functional as F
from functools import partial

# Import tensor_capture utilities
from neuronx_distributed.utils.tensor_capture import (
    enable_tensor_capture,
    disable_tensor_capture,
    get_available_modules,
    register_tensor,
    get_captured_tensors_dict
)

# Import NeuronxDistributed components
import torch_neuronx
from neuronx_distributed.trace import ModelBuilder
from neuronx_distributed.trace.model_builder import BaseModelInstance
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.utils.logger import get_logger

# Create a logger for this example
logger = get_logger()


# Define an MLP model with parallel layers
class MLP(torch.nn.Module):
    def __init__(self, hidden_size=2048, intermediate_size=8192):
        super().__init__()
        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False, gather_output=False)
            self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False, gather_output=False)
            self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False, input_is_parallel=True)
        else:
            self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x):
        # Gate and up projections
        gate_output = F.silu(self.gate_proj(x))
        up_output = self.up_proj(x)
        
        # Element-wise multiplication
        intermediate = gate_output * up_output
        
        # Manually register the intermediate tensor
        register_tensor("gate_up_product", intermediate)
        
        # Down projection to final output
        output = self.down_proj(intermediate)
        
        # Get captured tensors
        tensor_dict = get_captured_tensors_dict()
        
        if tensor_dict:
            # Return the output and the tensor dictionary
            return output, tensor_dict
        else:
            return output


class CaptureEnabledInstance(BaseModelInstance):
    """
    Custom BaseModelInstance that enables tensor capture and handles aliasing.
    """
    def __init__(self, modules_to_capture=None, max_tensors=None, hidden_size=2048, intermediate_size=8192):
        self.module = None
        self.modules_to_capture = modules_to_capture or []
        self.max_tensors = max_tensors
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def load_module(self):
        # Create the model instance
        model = MLP(hidden_size=self.hidden_size, intermediate_size=self.intermediate_size)
        
        # Enable tensor capture if modules are specified
        if self.modules_to_capture:
            logger.info(f"Enabling tensor capture for modules: {self.modules_to_capture}")
            self.module = enable_tensor_capture(
                model,
                self.modules_to_capture, 
                self.max_tensors
            )
        else:
            self.module = model

    def get(self, bucket_rank, **kwargs):
        # No aliasing needed for this model
        aliases = {}
        return self.module, aliases


def demonstrate_tensor_capture_cpu():
    """
    Demonstrate tensor capture functionality on CPU.
    """
    logger.info("=== Tensor Capture Demonstration (CPU mode) ===")
    
    # Create a model
    model = MLP()
    
    # Get available modules
    available_modules = get_available_modules(model)
    logger.info(f"Available modules: {available_modules}")
    
    # Enable tensor capture for specific modules
    modules_to_capture = ["gate_proj", "up_proj"]
    model = enable_tensor_capture(model, modules_to_capture, max_tensors=1)
    
    # Create input and run the model
    batch_size = 2
    x = torch.randn(batch_size, 1, 2048)  # [batch_size, seq_len, hidden_size]
    outputs = model(x)
    
    # Display outputs
    logger.info(f"Number of outputs: {len(outputs)}")
    logger.info(f"Main output shape: {outputs[0].shape}")
    
    # Get the tensor dictionary
    tensor_dict = outputs[1]
    
    # Display captured module tensors
    for module_name in modules_to_capture:
        output_key = f"{module_name}.outputs"
        if output_key in tensor_dict:
            logger.info(f"Captured tensor from {module_name} shape: {tensor_dict[output_key].shape}")
    
    # Display manually registered tensor
    if "manual_gate_up_product" in tensor_dict:
        logger.info(f"Manually registered tensor shape: {tensor_dict['manual_gate_up_product'].shape}")
    
    # Disable tensor capture
    model = disable_tensor_capture(model)


def compile_for_neuron(
    modules_to_capture=None,
    max_tensors=2,
    output_path="traced_model_with_capture/",
    batch_sizes=[1, 2],
    tp_degree=2,
    hidden_size=2048,
    intermediate_size=8192
):
    """
    Compile a model with tensor capture for Neuron.
    
    Args:
        modules_to_capture: List of module names to capture outputs from
        max_tensors: Maximum number of manually registered tensors
        output_path: Path to save the compiled model
        batch_sizes: List of batch sizes to compile for
        tp_degree: Tensor parallelism degree
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size
    """
    logger.info("=== Compiling MLP for Neuron with Tensor Capture ===")
    
    # Save model weights for checkpoint loading
    temp_model = MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
    weights_path = os.path.join(output_path, "weights.pt")
    os.makedirs(output_path, exist_ok=True)
    torch.save(temp_model.state_dict(), weights_path)

    # Get available modules to see what can be used for inspection
    available_modules = get_available_modules(temp_model)
    logger.info(f"Available modules: {available_modules}")
        
    # Select some modules to capture if not provided
    if modules_to_capture is None:
        modules_to_capture = ["gate_proj", "up_proj"]

    logger.info(f"Capturing outputs from modules: {modules_to_capture}")
    
    # Create ModelBuilder with the custom CaptureEnabledInstance
    builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=partial(torch.load, weights_path),
        debug=True
    )
    
    # Add phases for each batch size
    for i, batch_size in enumerate(batch_sizes):
        # Create model instance with tensor capture
        model_instance = CaptureEnabledInstance(
            modules_to_capture=modules_to_capture,
            max_tensors=max_tensors,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )
        
        # Add a phase with a unique key
        builder.add(
            key=f"batch{batch_size}",
            model_instance=model_instance,
            example_inputs=[(torch.randn(batch_size, 1, hidden_size),)],  # [batch_size, seq_len, hidden_size]
            compiler_args="--auto-cast=none"
        )
    
    try:
        # Trace the model
        traced_model = builder.trace(initialize_model_weights=True)
        
        # Save the model
        builder.shard_checkpoint(serialize_path=os.path.join(output_path, "weights/"))
        torch.jit.save(traced_model, os.path.join(output_path, "model.pt"))
        
        # Save a config file with model information
        config = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "batch_sizes": batch_sizes,
            "captured_modules": modules_to_capture,
            "max_tensors": max_tensors,
            "tp_degree": tp_degree
        }
        
        with open(os.path.join(output_path, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model compiled and saved to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error during model tracing: {e}")
        import traceback
        traceback.print_exc()
        return None


@torch.inference_mode()
def run_neuron_inference(
    model_path="traced_model_with_capture/",
    compare_with_cpu=True
):
    """
    Run inference on a compiled model with tensor capture.
    
    Args:
        model_path: Path to the compiled model
        compare_with_cpu: Whether to compare outputs with CPU execution
    """
    logger.info("=== Running Inference on Neuron with Tensor Capture ===")
    
    try:
        # Load the model config
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            config = json.load(f)
        
        # Load the traced model
        model = torch.jit.load(os.path.join(model_path, "model.pt"))
        logger.info("Loaded MLP model with tensor capture")
                
        # Load weights for model initialization
        weights = []
        tp_degree = config.get('tp_degree', 2)
        
        # Check if we have sharded weights
        weights_dir = os.path.join(model_path, "weights")
        if os.path.exists(weights_dir):
            # Try to load safetensors first
            try:
                from safetensors.torch import load_file
                for rank in range(tp_degree):
                    safetensor_path = os.path.join(weights_dir, f"tp{rank}_sharded_checkpoint.safetensors")
                    if os.path.exists(safetensor_path):
                        ckpt = load_file(safetensor_path)
                        weights.append(ckpt)
                        logger.info(f"Loaded safetensor weights for rank {rank}")
            except (ImportError, FileNotFoundError):
                # Fall back to PyTorch weights
                for rank in range(tp_degree):
                    pt_path = os.path.join(weights_dir, f"tp{rank}_sharded_checkpoint.pt")
                    if os.path.exists(pt_path):
                        ckpt = torch.load(pt_path)
                        weights.append(ckpt)
                        logger.info(f"Loaded PyTorch weights for rank {rank}")
        
        # Initialize the model with weights
        start_rank_tensor = torch.tensor([0], dtype=torch.int32, device="cpu")
        model.nxd_model.initialize(weights, start_rank_tensor)
        logger.info("Model initialized with weights")
        
        # Create inputs for each batch size
        for batch_size in config['batch_sizes']:
            logger.info(f"Running inference with batch_size={batch_size}")
            
            # Create input tensor
            hidden_size = config.get('hidden_size', 2048)
            input_tensor = torch.randn(batch_size, 1, hidden_size)  # [batch_size, seq_len, hidden_size]
            
            # Run on Neuron
            neuron_outputs = model(input_tensor)
            
            # Print output information
            logger.info(f"Number of outputs: {len(neuron_outputs)}")
            logger.info(f"Main output shape: {neuron_outputs[0].shape}")
            
            # Print captured tensor information
            neuron_tensor_dict = neuron_outputs[1]
            for module_name in config['captured_modules']:
                output_key = f"{module_name}.outputs"
                if output_key in neuron_tensor_dict:
                    logger.info(f"Captured tensor from {module_name} shape: {neuron_tensor_dict[output_key].shape}")
            
            # Display manually registered tensor
            if "manual_gate_up_product" in neuron_tensor_dict:
                logger.info(f"Manually registered tensor shape: {neuron_tensor_dict['manual_gate_up_product'].shape}")

            # Compare with CPU if requested
            if compare_with_cpu:
                # Create CPU model
                cpu_model = MLP(
                    hidden_size=config.get('hidden_size', 2048),
                    intermediate_size=config.get('intermediate_size', 8192)
                )
                
                # Load weights if available
                try:
                    weights_path = os.path.join(model_path, "weights.pt")
                    cpu_model.load_state_dict(torch.load(weights_path))
                except Exception as e:
                    logger.warning(f"Could not load weights for CPU model: {e}")
                
                # Enable tensor capture
                cpu_model = enable_tensor_capture(
                    cpu_model,
                    config['captured_modules'],
                    config['max_tensors']
                )
                
                # Run on CPU
                cpu_outputs = cpu_model(input_tensor)
                
                # Compare main outputs
                try:
                    max_diff = torch.max(torch.abs(neuron_outputs[0] - cpu_outputs[0]))
                    logger.info(f"Max difference between CPU and Neuron main outputs: {max_diff.item()}")
                    
                    # Compare captured tensors
                    cpu_tensor_dict = cpu_outputs[1]
                    neuron_tensor_dict = neuron_outputs[1]
                    
                    for module_name in config['captured_modules']:
                        output_key = f"{module_name}.outputs"
                        
                        # Check that both dictionaries have the key
                        if output_key in cpu_tensor_dict and output_key in neuron_tensor_dict:
                            cpu_tensor = cpu_tensor_dict[output_key]
                            neuron_tensor = neuron_tensor_dict[output_key]
                            
                            # For ColumnParallelLinear outputs, we need to handle sharded tensors
                            if "gate_proj" in module_name or "up_proj" in module_name:
                                # When using TP>1, we only compare the first shard
                                if config.get('tp_degree', 1) > 1:
                                    shard_size = cpu_tensor.size(-1) // config.get('tp_degree', 2)
                                    max_diff = torch.max(torch.abs(
                                        neuron_tensor - cpu_tensor[..., :shard_size]
                                    ))
                                else:
                                    max_diff = torch.max(torch.abs(neuron_tensor - cpu_tensor))
                            else:
                                max_diff = torch.max(torch.abs(neuron_tensor - cpu_tensor))
                            logger.info(f"Max difference for {module_name} outputs: {max_diff.item()}")
                    
                    
                    # Check manually registered tensor
                    if "manual_gate_up_product" in neuron_tensor_dict:
                        neuron_tensor = neuron_tensor_dict["manual_gate_up_product"]
                        cpu_tensor = cpu_tensor_dict["manual_gate_up_product"]
                        if config.get('tp_degree', 1) > 1:
                            shard_size = cpu_tensor.size(-1) // config.get('tp_degree', 2)
                            max_diff = torch.max(torch.abs(
                                neuron_tensor - cpu_tensor[..., :shard_size]
                            ))
                        else:
                            max_diff = torch.max(torch.abs(neuron_tensor - cpu_tensor))

                        logger.info(f"Max difference for manual_gate_up_product: {max_diff.item()}")
                        
                except Exception as e:
                    logger.warning(f"Could not compare all outputs - shapes may differ: {e}")
                    logger.debug(f"CPU outputs: {[o.shape for o in cpu_outputs]}")
                    logger.debug(f"Neuron outputs: {[o.shape for o in neuron_outputs]}")
        
        return True
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to parse arguments and call the appropriate function."""
    parser = argparse.ArgumentParser(description='Tensor capture example with NeuronxDistributed')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Demo command
    subparsers.add_parser('demo', help='Demonstrate tensor capture on CPU')
    
    # Compile command
    compile_parser = subparsers.add_parser('compile', help='Compile model for Neuron with tensor capture')
    compile_parser.add_argument('--output-path', type=str, default="traced_model_with_capture/",
                               help='Path to save the traced model')
    compile_parser.add_argument('--max-tensors', type=int, default=2,
                               help='Maximum number of manually registered tensors to store')
    compile_parser.add_argument('--tp-degree', type=int, default=2,
                               help='Tensor parallelism degree')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference on Neuron with tensor capture')
    inference_parser.add_argument('--model-path', type=str, default="traced_model_with_capture/",
                                help='Path to the traced model')
    inference_parser.add_argument('--no-compare', action='store_true',
                                help='Disable comparison with CPU execution')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        demonstrate_tensor_capture_cpu()
    elif args.command == 'compile':
        compile_for_neuron(
            output_path=args.output_path,
            max_tensors=args.max_tensors,
            tp_degree=args.tp_degree
        )
    elif args.command == 'inference':
        run_neuron_inference(
            model_path=args.model_path,
            compare_with_cpu=not args.no_compare
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()