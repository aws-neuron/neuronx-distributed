"""
Integration tests for tensor_capture with transformer models using NeuronxDistributed parallel layers.
"""

import os
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import json
import math

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding
)
from neuronx_distributed.trace import ModelBuilder
from neuronx_distributed.trace.model_builder import BaseModelInstance

from neuronx_distributed.utils.tensor_capture import (
    enable_tensor_capture,
    disable_tensor_capture,
    get_available_modules,
    register_tensor,
    get_captured_tensors_dict
)


class Config:
    """Configuration class for transformer model"""
    def __init__(self):
        self.vocab_size = 1000
        self.hidden_size = 64
        self.intermediate_size = 256
        self.n_layers = 2
        self.n_heads = 4
        self.n_kv_heads = 4
        self.head_dim = 16
        self.dtype = torch.float32
        self.rms_norm_eps = 1e-5
        self.pad_token = 0
        self.capture_tensors = True  # Flag to control tensor capture


class RMSNorm(nn.Module):
    """RMSNorm implementation"""
    def __init__(self, cfg):
        super().__init__()
        self.eps = cfg.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(cfg.hidden_size, dtype=cfg.dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MLP(nn.Module):
    """MLP module using parallel layers"""
    def __init__(self, cfg):
        super().__init__()
        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(
                cfg.hidden_size, 
                cfg.intermediate_size, 
                bias=False, 
                gather_output=False, 
                dtype=cfg.dtype
            )
            self.up_proj = ColumnParallelLinear(
                cfg.hidden_size, 
                cfg.intermediate_size, 
                bias=False, 
                gather_output=False, 
                dtype=cfg.dtype
            )
            self.down_proj = RowParallelLinear(
                cfg.intermediate_size, 
                cfg.hidden_size, 
                bias=False, 
                input_is_parallel=True, 
                dtype=cfg.dtype
            )
        else:
            self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, dtype=cfg.dtype)
            self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, dtype=cfg.dtype)
            self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False, dtype=cfg.dtype)

    def forward(self, x):
        # Gate and up projections
        gate_output = F.silu(self.gate_proj(x))
        up_output = self.up_proj(x)
        
        # Register gateproj
        register_tensor("gate_proj", gate_output)

        # Element-wise multiplication
        intermediate = gate_output * up_output
        
        # Down projection to final output
        output = self.down_proj(intermediate)
        
        return output


class Attention(nn.Module):
    """Attention module using parallel layers"""
    def __init__(self, cfg):
        super().__init__()
        
        if parallel_state.model_parallel_is_initialized():
            tp_degree = parallel_state.get_tensor_model_parallel_group().size()
            
            if cfg.n_heads % tp_degree != 0:
                raise ValueError("n_heads not evenly divisible by tp degree")
                
            self.n_heads = cfg.n_heads // tp_degree
            self.n_kv_heads = max(cfg.n_kv_heads // tp_degree, 1)
            
            self.query = ColumnParallelLinear(
                cfg.hidden_size, 
                self.n_heads * tp_degree * cfg.head_dim, 
                bias=False, 
                gather_output=False, 
                dtype=cfg.dtype
            )
            self.key = ColumnParallelLinear(
                cfg.hidden_size, 
                self.n_kv_heads * tp_degree * cfg.head_dim, 
                bias=False, 
                gather_output=False, 
                dtype=cfg.dtype
            )
            self.value = ColumnParallelLinear(
                cfg.hidden_size, 
                self.n_kv_heads * tp_degree * cfg.head_dim, 
                bias=False, 
                gather_output=False, 
                dtype=cfg.dtype
            )
            self.output = RowParallelLinear(
                self.n_heads * tp_degree * cfg.head_dim, 
                cfg.hidden_size, 
                bias=False, 
                input_is_parallel=True, 
                dtype=cfg.dtype
            )
        else:
            self.n_heads = cfg.n_heads
            self.n_kv_heads = cfg.n_kv_heads
            
            self.query = nn.Linear(cfg.hidden_size, self.n_heads * cfg.head_dim, bias=False, dtype=cfg.dtype)
            self.key = nn.Linear(cfg.hidden_size, self.n_kv_heads * cfg.head_dim, bias=False, dtype=cfg.dtype)
            self.value = nn.Linear(cfg.hidden_size, self.n_kv_heads * cfg.head_dim, bias=False, dtype=cfg.dtype)
            self.output = nn.Linear(self.n_heads * cfg.head_dim, cfg.hidden_size, bias=False, dtype=cfg.dtype)
        
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads if self.n_kv_heads > 0 else 1

    def forward(self, x, attention_mask=None, position_ids=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Register only one intermediate tensor to avoid too many registrations
        register_tensor("qkv_tensors", q)  # Just register q as representative
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply position encoding if provided
        if position_ids is not None:
            # Simple position encoding simulation
            pos_factor = position_ids.float().unsqueeze(-1).unsqueeze(-1) * 0.01
            q = q + pos_factor
            k = k + pos_factor
        
        # With GQA, k/v heads are shared among different q heads
        # repeat k/v heads to match q heads
        if self.n_rep > 1:
            k = torch.repeat_interleave(k, dim=2, repeats=self.n_rep)
            v = torch.repeat_interleave(v, dim=2, repeats=self.n_rep)
        
        # Transpose for attention calculation
        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        
        # Attention calculation
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Simple mask application - expand mask to match scores shape
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores * mask_expanded + (1 - mask_expanded) * (-1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.output(context)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block using parallel layers"""
    def __init__(self, cfg):
        super().__init__()
        self.attention = Attention(cfg)
        self.mlp = MLP(cfg)
        self.attention_norm = RMSNorm(cfg)
        self.mlp_norm = RMSNorm(cfg)

    def forward(self, x, attention_mask=None, position_ids=None):
        # Attention with residual connection - pass kwargs to attention
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, attention_mask=attention_mask, position_ids=position_ids)
        x = residual + x
        
        # MLP with residual connection
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class ParallelTransformer(nn.Module):
    """Transformer model using parallel layers"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # Store the config
        
        if parallel_state.model_parallel_is_initialized():
            self.embedding = ParallelEmbedding(
                cfg.vocab_size,
                cfg.hidden_size,
                shard_across_embedding=True,
                dtype=cfg.dtype
            )
            self.output = ColumnParallelLinear(
                cfg.hidden_size,
                cfg.vocab_size,
                bias=False,
                gather_output=True,
                dtype=cfg.dtype
            )
        else:
            self.embedding = nn.Embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                padding_idx=cfg.pad_token,
                dtype=cfg.dtype
            )
            self.output = nn.Linear(
                cfg.hidden_size,
                cfg.vocab_size,
                bias=False,
                dtype=cfg.dtype
            )
        
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg)
        
    def forward(self, input_ids, attention_mask=None, position_ids=None):
        # Embedding
        x = self.embedding(input_ids)
        register_tensor("embedding_output", x)
        
        # Create position_ids if not provided
        if position_ids is None:
            seq_len = input_ids.size(1)
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        
        # Process through transformer layers with kwargs
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask=attention_mask, position_ids=position_ids)
            if self.cfg.capture_tensors and i == 0:  # Only register first layer to limit registrations
                register_tensor(f"layer_{i}_output", x)
        
        # Final norm and output
        x = self.norm(x)
        register_tensor("norm_output", x)
        
        logits = self.output(x)
        register_tensor("logits_output", logits)
        
        # Get captured tensors as a dictionary
        tensor_dict = get_captured_tensors_dict()
        if tensor_dict:
            return logits, tensor_dict
        else:
            return logits



class TransformerCaptureInstance(BaseModelInstance):
    """Custom BaseModelInstance for transformer with tensor capture"""
    def __init__(self, cfg, modules_to_capture=None, max_tensors=None, capture_inputs=False):
        super().__init__(ParallelTransformer, (cfg,))
        self.cfg = cfg
        self.modules_to_capture = modules_to_capture or []
        self.max_tensors = max_tensors
        self.capture_inputs = capture_inputs

    def load_module(self):
        # Create the transformer model
        self.module = ParallelTransformer(self.cfg)
        
        # Enable tensor capture
        self.module = enable_tensor_capture(
            self.module, 
            self.modules_to_capture, 
            self.max_tensors,
            capture_inputs=self.capture_inputs
        )

    def get(self, bucket_rank, **kwargs):
        # No aliasing needed for this model
        aliases = {}
        return self.module, aliases


class TestTensorCaptureTransformerIntegration(unittest.TestCase):
    """Integration tests for tensor capture with transformer models using parallel layers"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cfg = Config()
        self.input_tensor = torch.randint(0, self.cfg.vocab_size, (2, 10))  # Batch size 2, sequence length 10
        
    def tearDown(self):
        """Clean up after tests"""
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
    
    def test_tensor_capture_cpu(self):
        """Test tensor capture on CPU"""
        # Create model
        model = ParallelTransformer(self.cfg)
        
        # Get available modules
        available_modules = get_available_modules(model)
        
        # Define modules to capture
        modules_to_capture = [
            "embedding",
            "layers.0.attention",
            "layers.1.mlp",
            "norm",
            "output"
        ]
        
        # Verify that the modules exist
        for module in modules_to_capture:
            self.assertIn(module, available_modules)
        
        # Enable tensor capture
        max_tensors = 5
        modified_model = enable_tensor_capture(model, modules_to_capture, max_tensors=max_tensors)
        
        # Run the model
        outputs = modified_model(self.input_tensor)
        
        # Check that outputs is a tuple with the main output and captured tensors
        self.assertIsInstance(outputs, tuple)
        
        # First element should be the main output
        main_output = outputs[0]
        self.assertEqual(main_output.shape, (2, 10, self.cfg.vocab_size))
        
        # Second element should be the tensor dictionary
        tensor_dict = outputs[1]
        self.assertIsInstance(tensor_dict, dict)
        
        # Check that we have the expected module outputs in the dictionary
        for module_name in modules_to_capture:
            output_key = f"{module_name}.outputs"
            self.assertIn(output_key, tensor_dict, f"Missing output for {module_name}")
        
        # Check specific tensor shapes
        self.assertEqual(tensor_dict["embedding.outputs"].shape, (2, 10, self.cfg.hidden_size))
        self.assertEqual(tensor_dict["layers.0.attention.outputs"].shape, (2, 10, self.cfg.hidden_size))
        self.assertEqual(tensor_dict["layers.1.mlp.outputs"].shape, (2, 10, self.cfg.hidden_size))
        self.assertEqual(tensor_dict["norm.outputs"].shape, (2, 10, self.cfg.hidden_size))
        self.assertEqual(tensor_dict["output.outputs"].shape, (2, 10, self.cfg.vocab_size))
        
        # Check that we have the manually registered tensors
        self.assertIn("manual_gate_proj", tensor_dict)
        self.assertIn("manual_gate_proj_1", tensor_dict)
        self.assertIn("manual_qkv_tensors", tensor_dict)
        self.assertIn("manual_qkv_tensors_1", tensor_dict)
  
        # Disable tensor capture
        model = disable_tensor_capture(modified_model)

    def test_tensor_capture_attention_params_cpu(self):
        """Test capturing attention parameters along with inputs and outputs on CPU"""
        # Create model
        cfg = Config()
        model = ParallelTransformer(cfg)
        
        # Create input data
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # Enable tensor capture for attention layer with input capture
        modules_to_capture = ["layers.0.attention"]
        model = enable_tensor_capture(
            model, 
            modules_to_capture, 
            capture_inputs=True  # Enable input capture to test attention parameters
        )
        
        # Run model with attention parameters
        with torch.no_grad():
            _ = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        
        # Get captured tensors
        captured_tensors = get_captured_tensors_dict()
        
        # Verify we captured both inputs and outputs
        self.assertGreater(len(captured_tensors), 0)
        
        # Check for input tensors (positional args)
        input_keys = [k for k in captured_tensors.keys() if "inputs" in k and "kwargs" not in k]
        self.assertGreater(len(input_keys), 0)
        
        # Check for kwargs tensors
        kwargs_keys = [k for k in captured_tensors.keys() if "kwargs" in k]
        self.assertGreater(len(kwargs_keys), 0)
        
        # Verify specific kwargs were captured
        attention_mask_keys = [k for k in kwargs_keys if "attention_mask" in k]
        position_ids_keys = [k for k in kwargs_keys if "position_ids" in k]
        
        self.assertGreater(len(attention_mask_keys), 0, "attention_mask kwargs should be captured")
        self.assertGreater(len(position_ids_keys), 0, "position_ids kwargs should be captured")
        
        # Check for output tensors
        output_keys = [k for k in captured_tensors.keys() if "outputs" in k]
        self.assertGreater(len(output_keys), 0)
        
        # Verify tensor shapes are reasonable
        for key, tensor in captured_tensors.items():
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertGreater(tensor.numel(), 0)
        
        # Clean up
        model = disable_tensor_capture(model)

    def test_tensor_capture_inputs_cpu(self):
        """Test capturing inputs along with outputs on CPU"""
        # Create model
        model = ParallelTransformer(self.cfg)
        
        # Define modules to capture
        modules_to_capture = [
            "layers.0.attention",
            "layers.0.mlp"
        ]
        
        # Enable tensor capture with input capture enabled
        max_tensors = 5
        modified_model = enable_tensor_capture(
            model, 
            modules_to_capture, 
            max_tensors=max_tensors,
            capture_inputs=True  # Enable input capture
        )
        
        # Run the model
        outputs = modified_model(self.input_tensor)
        
        # Check that outputs is a tuple with the main output and captured tensors
        self.assertIsInstance(outputs, tuple)
        
        # Second element should be the tensor dictionary
        tensor_dict = outputs[1]
        self.assertIsInstance(tensor_dict, dict)
        
        # Check that we have both inputs and outputs in the dictionary
        for module_name in modules_to_capture:
            # Check for inputs - inputs might have a suffix like .0
            input_key_prefix = f"{module_name}.inputs"
            input_key_found = False
            for key in tensor_dict.keys():
                if key.startswith(input_key_prefix):
                    input_key_found = True
                    break
            self.assertTrue(input_key_found, f"Missing input for {module_name}. Expected key starting with '{input_key_prefix}' in {list(tensor_dict.keys())}")
            
            # Check for outputs
            output_key = f"{module_name}.outputs"
            self.assertIn(output_key, tensor_dict, f"Missing output for {module_name}")
            
            # Verify shapes - inputs and outputs should have the same batch and sequence dimensions
            if input_key_found:
                # Find the actual input tensor using the prefix
                for key in tensor_dict.keys():
                    if key.startswith(input_key_prefix):
                        input_tensor = tensor_dict[key]
                        output_tensor = tensor_dict[output_key]
                        self.assertEqual(input_tensor.shape[0], output_tensor.shape[0], "Batch size mismatch")
                        self.assertEqual(input_tensor.shape[1], output_tensor.shape[1], "Sequence length mismatch")
                        break
        
        # Disable tensor capture
        model = disable_tensor_capture(modified_model)

    def compile_transformer_model(self, tp_degree, output_path, capture_inputs=False):
        """Helper function to compile transformer model with tensor capture"""
        # Create model weights for checkpoint loading
        model = ParallelTransformer(self.cfg)
        weights_path = os.path.join(output_path, "weights.pt")
        torch.save(model.state_dict(), weights_path)
        
        # Define modules to capture
        modules_to_capture = [
            "embedding",
            "layers.0.attention",
            "output"
        ]
        
        # Create ModelBuilder
        builder = ModelBuilder(
            router=None,
            tp_degree=tp_degree,
            checkpoint_loader=partial(torch.load, weights_path),
            debug=True
        )
        
        # Add model instance with tensor capture
        builder.add(
            key="transformer",
            model_instance=TransformerCaptureInstance(
                self.cfg,
                modules_to_capture=modules_to_capture,
                max_tensors=5,
                capture_inputs=capture_inputs  # Pass the capture_inputs parameter
            ),
            example_inputs=[(torch.randint(0, self.cfg.vocab_size, (1, 10)),)],
            compiler_args="--auto-cast=none"
        )
        
        # Trace the model
        traced_model = builder.trace(initialize_model_weights=True)
        
        # Save the model
        model_path = os.path.join(output_path, "model.pt")
        torch.jit.save(traced_model, model_path)
        
        # Save config
        config = {
            "vocab_size": self.cfg.vocab_size,
            "hidden_size": self.cfg.hidden_size,
            "intermediate_size": self.cfg.intermediate_size,
            "n_layers": self.cfg.n_layers,
            "captured_modules": modules_to_capture,
            "max_tensors": 5,
            "tp_degree": tp_degree,
            "capture_inputs": capture_inputs
        }
        
        with open(os.path.join(output_path, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
            
        # Save weights directory
        builder.shard_checkpoint(serialize_path=os.path.join(output_path, "weights/"))
        
        return model_path, modules_to_capture
    
    def run_neuron_inference(self, model_path, input_tensor):
        """Helper function to run inference on Neuron"""
        # Load the model config
        with open(os.path.join(model_path, "config.json"), 'r') as f:
            config = json.load(f)
        
        # Load the traced model
        model = torch.jit.load(os.path.join(model_path, "model.pt"))
        
        # Load weights for model initialization
        weights = []
        tp_degree = config.get('tp_degree', 1)
        
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
            except (ImportError, FileNotFoundError):
                # Fall back to PyTorch weights
                for rank in range(tp_degree):
                    pt_path = os.path.join(weights_dir, f"tp{rank}_sharded_checkpoint.pt")
                    if os.path.exists(pt_path):
                        ckpt = torch.load(pt_path)
                        weights.append(ckpt)
        
        # Initialize the model with weights
        start_rank_tensor = torch.tensor([0], dtype=torch.int32, device="cpu")
        model.nxd_model.initialize(weights, start_rank_tensor)
        
        # Run inference
        with torch.no_grad():
            neuron_outputs = model(input_tensor)
        
        return neuron_outputs, config

    def test_tensor_capture_neuron_tp1_with_attention_params(self):
        """Test tensor capture on Neuron with TP=1 including attention parameter capture"""
        # Create a temporary directory for weights and model
        output_path = "test_transformer_output_tp1_attention"
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Compile the model with input capture enabled to test attention parameters
            model_path, modules_to_capture = self.compile_transformer_model(
                tp_degree=1, 
                output_path=output_path,
                capture_inputs=True  # Enable input capture for attention parameters
            )
            
            # Create input data with attention parameters
            input_data = torch.randint(0, self.cfg.vocab_size, (1, 10))
            
            # Run inference on Neuron
            neuron_outputs, config = self.run_neuron_inference(output_path, input_data)
            
            # Check that we got the expected number of outputs
            # 1 main output + 1 tensor dictionary
            self.assertIsInstance(neuron_outputs, tuple)
            self.assertEqual(len(neuron_outputs), 2, "Expected 2 outputs: logits and tensor dictionary")
            
            # Check that the second output is a dictionary
            self.assertIsInstance(neuron_outputs[1], dict, "Second output should be a dictionary of tensors")
            
            # Get the tensor dictionary
            tensor_dict = neuron_outputs[1]
            
            # Check the shapes of captured tensors in the dictionary
            self.assertEqual(tensor_dict["embedding.outputs"].shape, (1, 10, self.cfg.hidden_size))  # embedding output
            self.assertEqual(tensor_dict["layers.0.attention.outputs"].shape, (1, 10, self.cfg.hidden_size))  # attention output
            self.assertEqual(tensor_dict["output.outputs"].shape, (1, 10, self.cfg.vocab_size))   # output layer output
            
            # Check that we have both inputs and outputs for attention layer
            attention_input_keys = [k for k in tensor_dict.keys() if k.startswith("layers.0.attention.inputs")]
            attention_output_keys = [k for k in tensor_dict.keys() if k.startswith("layers.0.attention.outputs")]
            
            self.assertGreater(len(attention_input_keys), 0, "Should have attention input tensors")
            self.assertGreater(len(attention_output_keys), 0, "Should have attention output tensors")
            
            # Check that we have the manually registered tensors
            self.assertIn("manual_gate_proj", tensor_dict)
            self.assertIn("manual_qkv_tensors", tensor_dict)
            
            # Check shapes of manually registered tensors
            self.assertEqual(tensor_dict["manual_gate_proj"].shape[0], 1)  # batch size
            self.assertEqual(tensor_dict["manual_qkv_tensors"].shape[0], 1)  # batch size

            # Create CPU model for comparison
            cpu_model = ParallelTransformer(self.cfg)
            
            # Load weights if available
            try:
                weights_path = os.path.join(output_path, "weights.pt")
                cpu_model.load_state_dict(torch.load(weights_path))
            except Exception as e:
                print(f"Warning: Could not load weights for CPU model: {e}")
            
            # Enable tensor capture with same configuration
            cpu_model = enable_tensor_capture(
                cpu_model,
                modules_to_capture,
                config['max_tensors'],
                capture_inputs=config.get('capture_inputs', False)
            )
            
            # Run on CPU
            with torch.no_grad():
                cpu_outputs = cpu_model(input_data)
            
            # First element is the main output
            cpu_main_output = cpu_outputs[0]
            
            # Compare main outputs
            max_diff = torch.max(torch.abs(neuron_outputs[0] - cpu_main_output))
            self.assertLess(max_diff.item(), 1e-5, "Main output differs too much between CPU and Neuron")
            
            # Get the tensor dictionaries
            cpu_tensor_dict = cpu_outputs[1]
            neuron_tensor_dict = neuron_outputs[1]
            
            # Compare captured tensors
            for output_key in neuron_tensor_dict.keys():
                # Check that both dictionaries have the key
                self.assertIn(output_key, cpu_tensor_dict, f"Missing {output_key} in CPU outputs")
                self.assertIn(output_key, neuron_tensor_dict, f"Missing {output_key} in Neuron outputs")
                
                cpu_tensor = cpu_tensor_dict[output_key]
                neuron_tensor = neuron_tensor_dict[output_key]
                
                # Check that the shapes match
                self.assertEqual(cpu_tensor.shape, neuron_tensor.shape, 
                                f"Shape mismatch for {output_key}: CPU {cpu_tensor.shape}, Neuron {neuron_tensor.shape}")
                
                # Check that the values are close
                max_diff = torch.max(torch.abs(cpu_tensor - neuron_tensor))
                self.assertLess(max_diff.item(), 1e-5, f"Tensor {output_key} differs too much between CPU and Neuron")
                
        finally:
            # Clean up
            import shutil
            if os.path.exists(output_path):
                shutil.rmtree(output_path)

    def test_tensor_capture_inputs_neuron(self):
        """Test capturing inputs along with outputs on Neuron"""
        # Create a temporary directory for weights and model
        output_path = "test_transformer_output_inputs"
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Compile the model with input capture enabled
            model_path, modules_to_capture = self.compile_transformer_model(
                tp_degree=1, 
                output_path=output_path,
                capture_inputs=True  # Enable input capture
            )
            
            # Create input data
            input_data = torch.randint(0, self.cfg.vocab_size, (1, 10))
            
            # Run inference on Neuron
            neuron_outputs, config = self.run_neuron_inference(output_path, input_data)
            
            # Check that we got the expected number of outputs
            self.assertIsInstance(neuron_outputs, tuple)
            self.assertEqual(len(neuron_outputs), 2, "Expected 2 outputs: logits and tensor dictionary")
            
            # Check that the second output is a dictionary
            self.assertIsInstance(neuron_outputs[1], dict, "Second output should be a dictionary of tensors")
            
            # Get the tensor dictionary
            tensor_dict = neuron_outputs[1]
            
            # Check that we have both inputs and outputs in the dictionary
            for module_name in modules_to_capture:
                # Check for inputs - inputs might have a suffix like .0
                input_key_prefix = f"{module_name}.inputs"
                input_key_found = False
                for key in tensor_dict.keys():
                    if key.startswith(input_key_prefix):
                        input_key_found = True
                        input_key = key  # Save the actual key for later use
                        break
                self.assertTrue(input_key_found, f"Missing input for {module_name}. Expected key starting with '{input_key_prefix}' in {list(tensor_dict.keys())}")
                
                # Check for outputs
                output_key = f"{module_name}.outputs"
                self.assertIn(output_key, tensor_dict, f"Missing output for {module_name}")
                
                # Verify shapes - inputs and outputs should have the same batch and sequence dimensions
                # Use the actual input key we found
                if input_key_found:
                    input_tensor = tensor_dict[input_key]
                    output_tensor = tensor_dict[output_key]
                    self.assertEqual(input_tensor.shape[0], output_tensor.shape[0], "Batch size mismatch")
                    self.assertEqual(input_tensor.shape[1], output_tensor.shape[1], "Sequence length mismatch")
        finally:
            # Clean up
            import shutil
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
        
    def test_tensor_capture_neuron_tp2_with_sharded_tensors(self):
        """Test tensor capture on Neuron with TP=2 including sharded tensor handling"""
        # Create a temporary directory for weights and model
        output_path = "test_transformer_output_tp2_sharded"
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Compile the model
            model_path, modules_to_capture = self.compile_transformer_model(tp_degree=2, output_path=output_path)
            
            # Create input data
            input_data = torch.randint(0, self.cfg.vocab_size, (1, 10))
            
            # Run inference on Neuron
            neuron_outputs, config = self.run_neuron_inference(output_path, input_data)
            
            # Check that we got the expected number of outputs
            # 1 main output + 1 tensor dictionary
            self.assertIsInstance(neuron_outputs, tuple)
            self.assertEqual(len(neuron_outputs), 2, "Expected 2 outputs: logits and tensor dictionary")
            
            # Check that the second output is a dictionary
            self.assertIsInstance(neuron_outputs[1], dict, "Second output should be a dictionary of tensors")
            
            # Create CPU model for comparison
            cpu_model = ParallelTransformer(self.cfg)
            
            # Load weights if available
            try:
                weights_path = os.path.join(output_path, "weights.pt")
                cpu_model.load_state_dict(torch.load(weights_path))
            except Exception as e:
                print(f"Warning: Could not load weights for CPU model: {e}")
            
            # Enable tensor capture
            cpu_model = enable_tensor_capture(
                cpu_model,
                modules_to_capture,
                config['max_tensors'],
                capture_inputs=config.get('capture_inputs', False)
            )
            # Run on CPU
            with torch.no_grad():
                cpu_outputs = cpu_model(input_data)
            
            # First element is the main output
            cpu_main_output = cpu_outputs[0]
            
            # Compare main outputs
            max_diff = torch.max(torch.abs(neuron_outputs[0] - cpu_main_output))
            self.assertLess(max_diff.item(), 1e-5, "Main output differs too much between CPU and Neuron")
            
            # Get the tensor dictionaries
            cpu_tensor_dict = cpu_outputs[1]
            neuron_tensor_dict = neuron_outputs[1]

            # Check that we have the manually registered tensors
            self.assertIn("manual_gate_proj", neuron_tensor_dict)
            self.assertIn("manual_qkv_tensors", neuron_tensor_dict)
            
            # Compare captured tensors with sharding awareness
            tp_degree = config.get('tp_degree', 2)
            
            for module_name in modules_to_capture:
                output_key = f"{module_name}.outputs"
                
                # Check that both dictionaries have the key
                self.assertIn(output_key, cpu_tensor_dict, f"Missing {output_key} in CPU outputs")
                self.assertIn(output_key, neuron_tensor_dict, f"Missing {output_key} in Neuron outputs")
                
                cpu_tensor = cpu_tensor_dict[output_key]
                neuron_tensor = neuron_tensor_dict[output_key]
                
                # Handle sharded tensors for parallel layers
                if "attention" in module_name and tp_degree > 1:
                    # For attention layers with TP>1, we only compare the first shard
                    if cpu_tensor.size(-1) != neuron_tensor.size(-1):
                        shard_size = cpu_tensor.size(-1) // tp_degree
                        max_diff = torch.max(torch.abs(cpu_tensor[..., :shard_size] - neuron_tensor))
                        self.assertLess(max_diff.item(), 1e-5, 
                                      f"Sharded tensor {output_key} differs too much between CPU and Neuron")
                    else:
                        # Non-sharded comparison
                        max_diff = torch.max(torch.abs(cpu_tensor - neuron_tensor))
                        self.assertLess(max_diff.item(), 1e-5, 
                                      f"Tensor {output_key} differs too much between CPU and Neuron")
                else:
                    # Regular comparison for non-sharded tensors
                    self.assertEqual(cpu_tensor.shape, neuron_tensor.shape, 
                                    f"Shape mismatch for {output_key}: CPU {cpu_tensor.shape}, Neuron {neuron_tensor.shape}")
                    max_diff = torch.max(torch.abs(cpu_tensor - neuron_tensor))
                    self.assertLess(max_diff.item(), 1e-5, 
                                  f"Tensor {output_key} differs too much between CPU and Neuron")

            # Compare manually registered tensors with sharding awareness
            for key in cpu_tensor_dict:
                # Skip module outputs as we've already compared them
                if any(key.startswith(f"{module}.outputs") for module in modules_to_capture):
                    continue
                
                # Check if the key exists in both dictionaries
                if key not in neuron_tensor_dict:
                    print(f"Warning: Key {key} exists in CPU outputs but not in Neuron outputs")
                    continue
                
                cpu_tensor = cpu_tensor_dict[key]
                neuron_tensor = neuron_tensor_dict[key]
                
                # Handle sharded tensors for parallel layers
                if "gate_proj" in key or "qkv" in key:
                    # These might be sharded with TP>1
                    if cpu_tensor.shape == neuron_tensor.shape:
                        max_diff = torch.max(torch.abs(cpu_tensor - neuron_tensor))
                        self.assertLess(max_diff.item(), 1e-5, 
                                      f"Tensor {key} differs too much between CPU and Neuron")
                    elif len(cpu_tensor.shape) > 0 and len(neuron_tensor.shape) > 0:
                        # Compare sharded portion
                        shard_size = cpu_tensor.size(-1) // tp_degree
                        max_diff = torch.max(torch.abs(cpu_tensor[..., :shard_size] - neuron_tensor))
                        self.assertLess(max_diff.item(), 1e-5, 
                                      f"Sharded tensor {key} differs too much between CPU and Neuron")
                    else:
                        print(f"Warning: Shape mismatch for tensor {key}: CPU {cpu_tensor.shape}, Neuron {neuron_tensor.shape}")
                else:
                    # Regular comparison for non-sharded tensors
                    if cpu_tensor.shape == neuron_tensor.shape:
                        max_diff = torch.max(torch.abs(cpu_tensor - neuron_tensor))
                        self.assertLess(max_diff.item(), 1e-5, 
                                      f"Tensor {key} differs too much between CPU and Neuron")
                    else:
                        print(f"Warning: Shape mismatch for tensor {key}: CPU {cpu_tensor.shape}, Neuron {neuron_tensor.shape}")
            
        finally:
            # Clean up
            import shutil
            if os.path.exists(output_path):
                shutil.rmtree(output_path)


def run_test(test_name=None):
    """
    Run a specific test function or all tests if none specified.
    
    Args:
        test_name: Name of the test function to run (e.g., 'test_tensor_capture_cpu')
    """
    # Create test instance
    test_instance = TestTensorCaptureTransformerIntegration()
    
    # Set up the test instance
    test_instance.setUp()
    
    try:
        if test_name is None:
            # Run all tests
            print("Running all tests...")
            test_functions = [name for name in dir(test_instance) if name.startswith('test_')]
            for func_name in test_functions:
                print(f"\n=== Running {func_name} ===")
                try:
                    getattr(test_instance, func_name)()
                    print(f"✅ {func_name} PASSED")
                except Exception as e:
                    print(f"❌ {func_name} FAILED: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Run specific test
            if not hasattr(test_instance, test_name):
                print(f"❌ Test '{test_name}' not found!")
                print(f"Available tests: {[name for name in dir(test_instance) if name.startswith('test_')]}")
                return
            
            print(f"\n=== Running {test_name} ===")
            try:
                getattr(test_instance, test_name)()
                print(f"✅ {test_name} PASSED")
            except Exception as e:
                print(f"❌ {test_name} FAILED: {e}")
                import traceback
                traceback.print_exc()
    finally:
        # Clean up
        test_instance.tearDown()

if __name__ == '__main__':
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        run_test(test_name)
    else:
        # List available tests
        test_instance = TestTensorCaptureTransformerIntegration()
        test_functions = [name for name in dir(test_instance) if name.startswith('test_')]
        print("Available tests:")
        for func in test_functions:
            print(f"  - {func}")
        print("\nRun a specific test with: python test_transformer_tensor_capture.py <test_name>")
        print("Example: python test_transformer_tensor_capture.py test_tensor_capture_cpu")
        