import unittest
import torch
import math
import torch.nn as nn
from torch_neuronx.proto import metaneff_pb2

import neuronx_distributed.trace.nxd_model
from neuronx_distributed.trace.model_builder import ModelBuilderV2
from neuronx_distributed.trace.model_builder_utils import TraceArtifacts
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.mock_torchdist import mock_distributed

class TestModelBuilderV2(unittest.TestCase):

    def test_single_trace(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            def forward(self, x):
                return self.linear(x) + 1
        
        model = SimpleModel()
        example_inputs = torch.rand(3, 10)
        
        traced_model = ModelBuilderV2(model) \
                        .trace(args=example_inputs, tag="key1")

        self.assertIsInstance(traced_model.trace_artifacts_collection["key1"], TraceArtifacts)
        
        torch.classes.neuron.Runtime().unsafe_close()
        
    def test_multi_hlo_trace(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            def forward(self, x):
                return self.linear(x) + 1
        
        model = SimpleModel()
        example_inputs1 = torch.rand(3, 10)
        example_inputs2 = torch.rand(2, 10)
        example_inputs3 = torch.rand(4, 10)
        
        traced_model = ModelBuilderV2(model) \
                        .trace(args=example_inputs1, tag="key1") \
                        .trace(args=example_inputs2, tag="key2") \
                        .trace(args=example_inputs3, tag="key3")

        for key in ["key1", "key2", "key3"]:
            self.assertIsInstance(traced_model.trace_artifacts_collection[key], TraceArtifacts)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_module_with_multiple_inputs_trace(self):
        class MultiInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(5, 3)
                self.linear2 = nn.Linear(11, 3)

            def forward(self, x, y):
                return self.linear1(x) + self.linear2(y)
        
        model = MultiInputModel()
        example_inputs1 = (torch.rand(3, 5), torch.rand(3, 11))
        example_inputs2 = (torch.rand(12, 5), torch.rand(12, 11))

        traced_model = ModelBuilderV2(model) \
                        .trace(args=example_inputs1, tag="bkt1") \
                        .trace(args=example_inputs2, tag="bkt2")

        for key in ["bkt1", "bkt2"]:
            self.assertIsInstance(traced_model.trace_artifacts_collection[key], TraceArtifacts)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_kwargs_only_trace(self):
        class KwargsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 5)
                self.linear2 = nn.Linear(8, 5)

            def forward(self, input1=None, input2=None):
                out1 = self.linear1(input1) if input1 is not None else 0
                out2 = self.linear2(input2) if input2 is not None else 0
                return out1 + out2

        model = KwargsModel()
        traced_model = ModelBuilderV2(model) \
            .trace(kwargs={"input1": torch.rand(3, 10), "input2": torch.rand(3, 8)}, tag="both_inputs") \
            .trace(kwargs={"input1": torch.rand(2, 10)}, tag="input1_only")

        # Verify trace artifacts
        self.assertIsInstance(traced_model.trace_artifacts_collection["both_inputs"], TraceArtifacts)
        self.assertIsInstance(traced_model.trace_artifacts_collection["input1_only"], TraceArtifacts)

        # Verify user_input_keys in metaneff
        both_inputs_keys = set()
        for tensor in traced_model.trace_artifacts_collection["both_inputs"].metaneff.input_tensors:
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT:
                both_inputs_keys.add(tensor.user_input_key.decode())
        self.assertEqual(both_inputs_keys, {"input1", "input2"})
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_mixed_args_kwargs_trace(self):
        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x, scale=None, bias=None):
                out = self.linear(x)
                if scale is not None:
                    out = out * scale
                if bias is not None:
                    out = out + bias
                return out

        model = MixedModel()
        traced_model = ModelBuilderV2(model) \
            .trace(
                args=(torch.rand(3, 10),),
                kwargs={"scale": torch.rand(3, 5), "bias": torch.rand(3, 5)},
                tag="full_inputs"
            ) \
            .trace(
                args=(torch.rand(2, 10),),
                kwargs={"scale": torch.rand(2, 5)},
                tag="partial_inputs"
            )

        # Verify trace artifacts
        self.assertIsInstance(traced_model.trace_artifacts_collection["full_inputs"], TraceArtifacts)
        self.assertIsInstance(traced_model.trace_artifacts_collection["partial_inputs"], TraceArtifacts)

        # Verify user_input_keys in metaneff
        full_input_params = set()
        for tensor in traced_model.trace_artifacts_collection["full_inputs"].metaneff.input_tensors:
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT and tensor.user_input_key:
                full_input_params.add(tensor.user_input_key.decode())
        self.assertEqual(full_input_params, {"x", "scale", "bias"})
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_compile_simple_model(self):
        """Test compilation of a simple model without priority model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        example_inputs = torch.rand(3, 10)
        
        nxd_model = ModelBuilderV2(model) \
            .trace(args=example_inputs, tag="key1") \
            .compile()
        
        self.assertIsInstance(nxd_model, neuronx_distributed.trace.nxd_model.NxDModel)

        torch.classes.neuron.Runtime().unsafe_close()

    def test_compile_with_priority_model(self):
        """Test compilation with a priority model specified."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        example_inputs1 = torch.rand(3, 10)
        example_inputs2 = torch.rand(2, 10)
        
        nxd_model = ModelBuilderV2(model) \
            .trace(args=example_inputs1, tag="priority") \
            .trace(args=example_inputs2, tag="secondary") \
            .compile(priority_model_key="priority")
        
        self.assertIsInstance(nxd_model, neuronx_distributed.trace.nxd_model.NxDModel)

        torch.classes.neuron.Runtime().unsafe_close()


class TestModelBuilderV2Distributed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_dist = mock_distributed(world_size=32)
        cls.mock_dist.__enter__()
        torch.distributed.init_process_group(backend="xla", rank=0, world_size=32)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=32, skip_collective_init=True)
        parallel_state.set_aot_mode(True)

    @classmethod
    def tearDownClass(cls):
        parallel_state.set_aot_mode(False)
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
        cls.mock_dist.__exit__(None, None, None)

    def test_single_trace_CPL(self):
        class ColumnParallelModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = ColumnParallelLinear(1024, 1024, gather_output=False)

            def forward(self, x):
                return self.layer(x)
        
        model = ColumnParallelModel()
        example_inputs = torch.rand(32, 1024)
        
        traced_model = ModelBuilderV2(model) \
                        .trace(args=example_inputs, tag="key1")

        self.assertIsInstance(traced_model.trace_artifacts_collection["key1"], TraceArtifacts)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_single_trace_RPL(self):
        class RowParallelModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = RowParallelLinear(1024, 1024, input_is_parallel=True)

            def forward(self, x):
                return self.layer(x)
        
        model = RowParallelModel()
        example_inputs = torch.rand(32, 1024)
        
        traced_model = ModelBuilderV2(model) \
                        .trace(args=example_inputs, tag="key1")

        self.assertIsInstance(traced_model.trace_artifacts_collection["key1"], TraceArtifacts)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_single_trace_CPLRPL(self):
        class CPLRPLModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)

            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)

        model = CPLRPLModel()
        example_inputs = torch.rand(32, 1024)
        
        traced_model = ModelBuilderV2(model) \
                        .trace(args=example_inputs, tag="key1")

        self.assertIsInstance(traced_model.trace_artifacts_collection["key1"], TraceArtifacts)
        
        torch.classes.neuron.Runtime().unsafe_close()
        
    def test_multi_hlo_trace_CPL(self):
        class ColumnParallelModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = ColumnParallelLinear(1024, 1024, gather_output=False)

            def forward(self, x):
                return self.layer(x)
        
        model = ColumnParallelModel()
        example_inputs1 = torch.rand(3, 1024)
        example_inputs2 = torch.rand(2, 1024)
        example_inputs3 = torch.rand(4, 1024)
        
        traced_model = ModelBuilderV2(model) \
                        .trace(args=example_inputs1, tag="key1") \
                        .trace(args=example_inputs2, tag="key2") \
                        .trace(args=example_inputs3, tag="key3")

        for key in ["key1", "key2", "key3"]:
            self.assertIsInstance(traced_model.trace_artifacts_collection[key], TraceArtifacts)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_multi_hlo_trace_RPL(self):
        class RowParallelModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = RowParallelLinear(1024, 1024, input_is_parallel=True)

            def forward(self, x):
                return self.layer(x)
        
        model = RowParallelModel()
        example_inputs1 = torch.rand(3, 1024)
        example_inputs2 = torch.rand(2, 1024)
        example_inputs3 = torch.rand(4, 1024)
        
        traced_model = ModelBuilderV2(model) \
                        .trace(args=example_inputs1, tag="key1") \
                        .trace(args=example_inputs2, tag="key2") \
                        .trace(args=example_inputs3, tag="key3")

        for key in ["key1", "key2", "key3"]:
            self.assertIsInstance(traced_model.trace_artifacts_collection[key], TraceArtifacts)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_multi_hlo_trace_CPLRPL(self):
        class CPLRPLModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)

            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)

        model = CPLRPLModel()
        example_inputs1 = torch.rand(3, 1024)
        example_inputs2 = torch.rand(2, 1024)
        example_inputs3 = torch.rand(4, 1024)
        
        traced_model = ModelBuilderV2(model) \
                        .trace(args=example_inputs1, tag="key1") \
                        .trace(args=example_inputs2, tag="key2") \
                        .trace(args=example_inputs3, tag="key3")

        for key in ["key1", "key2", "key3"]:
            self.assertIsInstance(traced_model.trace_artifacts_collection[key], TraceArtifacts)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_kwargs_only_trace_CPL(self):
        class ColumnParallelKwargsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = ColumnParallelLinear(1024, 1024, gather_output=False)

            def forward(self, input_ids=None, attention_mask=None):
                out = self.layer(input_ids)
                if attention_mask is not None:
                    out = out * attention_mask
                return out

        model = ColumnParallelKwargsModel()
        traced_model = ModelBuilderV2(model) \
            .trace(
                kwargs={
                    "input_ids": torch.rand(32, 1024),
                    "attention_mask": torch.rand(32, 1)
                },
                tag="with_mask"
            ) \
            .trace(
                kwargs={"input_ids": torch.rand(16, 1024)},
                tag="no_mask"
            )

        # Verify trace artifacts
        self.assertIsInstance(traced_model.trace_artifacts_collection["with_mask"], TraceArtifacts)
        self.assertIsInstance(traced_model.trace_artifacts_collection["no_mask"], TraceArtifacts)

        # Verify user_input_keys in metaneff
        with_mask_keys = set()
        for tensor in traced_model.trace_artifacts_collection["with_mask"].metaneff.input_tensors:
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT:
                with_mask_keys.add(tensor.user_input_key.decode())
        self.assertEqual(with_mask_keys, {"input_ids", "attention_mask"})
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_mixed_args_kwargs_trace_CPLRPL(self):
        class MixedCPLRPLModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)

            def forward(self, x, scale_factor=None):
                x = self.layer1(x)
                if scale_factor is not None:
                    x = x * scale_factor
                return self.layer2(x)

        model = MixedCPLRPLModel()
        traced_model = ModelBuilderV2(model) \
            .trace(
                args=(torch.rand(32, 1024),),
                kwargs={"scale_factor": torch.rand(32, 1)},
                tag="with_scale"
            ) \
            .trace(
                args=(torch.rand(16, 1024),),
                tag="no_scale"
            )

        # Verify trace artifacts
        self.assertIsInstance(traced_model.trace_artifacts_collection["with_scale"], TraceArtifacts)
        self.assertIsInstance(traced_model.trace_artifacts_collection["no_scale"], TraceArtifacts)

        # Verify user_input_key in metaneff
        with_scale_keys = set()
        for tensor in traced_model.trace_artifacts_collection["with_scale"].metaneff.input_tensors:
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT and tensor.user_input_key:
                with_scale_keys.add(tensor.user_input_key.decode())
        self.assertEqual(with_scale_keys, {"x", "scale_factor"})
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_compile_distributed_model(self):
        """Test compilation of a distributed model."""
        class DistributedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        
        model = DistributedModel()
        example_inputs = torch.rand(32, 1024)
        
        nxd_model = ModelBuilderV2(model) \
            .trace(args=example_inputs, tag="key1") \
            .compile()
        
        self.assertIsInstance(nxd_model, neuronx_distributed.trace.nxd_model.NxDModel)

        torch.classes.neuron.Runtime().unsafe_close()

    def test_compile_distributed_with_priority(self):
        """Test compilation of a distributed model with priority model."""
        class DistributedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        
        model = DistributedModel()
        example_inputs1 = torch.rand(32, 1024)
        example_inputs2 = torch.rand(16, 1024)
        
        nxd_model = ModelBuilderV2(model) \
            .trace(args=example_inputs1, tag="priority") \
            .trace(args=example_inputs2, tag="secondary") \
            .compile(priority_model_key="priority")
        
        self.assertIsInstance(nxd_model, neuronx_distributed.trace.nxd_model.NxDModel)

        torch.classes.neuron.Runtime().unsafe_close()

    def test_compile_distributed_multiple_buckets(self):
        """Test compilation of a complex distributed model with multiple buckets using mixed args and kwargs."""
        class ComplexDistributedBlock(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.attention = nn.ModuleDict({
                    'q': ColumnParallelLinear(hidden_size, hidden_size, gather_output=False),
                    'k': ColumnParallelLinear(hidden_size, hidden_size, gather_output=False),
                    'v': ColumnParallelLinear(hidden_size, hidden_size, gather_output=False),
                    'o': RowParallelLinear(hidden_size, hidden_size, input_is_parallel=True)
                })
                self.mlp = nn.ModuleDict({
                    'fc1': ColumnParallelLinear(hidden_size, 4 * hidden_size, gather_output=False),
                    'fc2': RowParallelLinear(4 * hidden_size, hidden_size, input_is_parallel=True)
                })
                self.norm1 = nn.LayerNorm(hidden_size)
                self.norm2 = nn.LayerNorm(hidden_size)
                self.dropout1 = nn.Dropout(0.1)
                self.dropout2 = nn.Dropout(0.1)
                self.hidden_size = hidden_size

            def forward(self, hidden_states, attention_mask=None):
                # Self-attention
                norm_states = self.norm1(hidden_states)
                q = self.attention['q'](norm_states)
                k = self.attention['k'](norm_states)
                v = self.attention['v'](norm_states)

                # Scaled dot-product attention
                attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hidden_size)
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask
                attention_probs = torch.softmax(attention_scores, dim=-1)
                attention_probs = self.dropout1(attention_probs)
                
                context = torch.matmul(attention_probs, v)
                attention_output = self.attention['o'](context)

                # Add & Norm
                hidden_states = hidden_states + attention_output
                
                # MLP
                norm_states = self.norm2(hidden_states)
                mlp_hidden = self.mlp['fc1'](norm_states)
                mlp_hidden = torch.nn.functional.gelu(mlp_hidden)
                mlp_hidden = self.dropout2(mlp_hidden)
                mlp_output = self.mlp['fc2'](mlp_hidden)

                # Add & Norm
                output = hidden_states + mlp_output
                return output

        class ComplexDistributedModel(nn.Module):
            def __init__(self, hidden_size=1024, num_layers=2):
                super().__init__()
                self.layers = nn.ModuleList([
                    ComplexDistributedBlock(hidden_size) for _ in range(num_layers)
                ])
                self.final_norm = nn.LayerNorm(hidden_size)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, 
                    hidden_states, 
                    attention_mask=None, 
                    position_embeddings=None, 
                    token_type_ids=None):
                
                # Add position embeddings if provided
                if position_embeddings is not None:
                    hidden_states = hidden_states + position_embeddings
                    
                # Add token type embeddings if provided
                if token_type_ids is not None:
                    hidden_states = hidden_states + token_type_ids
                    
                hidden_states = self.dropout(hidden_states)
                
                for layer in self.layers:
                    hidden_states = layer(hidden_states, attention_mask)
                        
                return self.final_norm(hidden_states)

        # Initialize model
        model = ComplexDistributedModel(hidden_size=1024, num_layers=2)
        
        # Define various input configurations
        input_configs = [
            # (batch_size, seq_len, has_attention_mask, has_position_emb, has_token_type)
            (32, 128, True, True, True),     # All inputs
            (16, 256, True, True, False),    # No token_type_ids
            (8, 512, True, False, True),     # No position_embeddings
            (64, 64, False, True, True),     # No attention_mask
            (24, 384, True, True, True),     # All inputs, different size
            (48, 128, True, False, False),   # Only attention_mask
            (40, 256, False, True, False),   # Only position_embeddings
            (56, 128, False, False, True),   # Only token_type_ids
            (12, 512, True, False, False),   # Only attention_mask, different size
            (4, 1024, False, False, False),  # No optional inputs
        ]
        
        builder = ModelBuilderV2(model)
        
        # Trace with different input configurations
        for idx, (batch_size, seq_len, has_attn, has_pos, has_token) in enumerate(input_configs, 1):
            # Base input
            hidden_states = torch.rand(batch_size, seq_len, 1024)
            kwargs = {}
            
            # Optional inputs
            if has_attn:
                attention_mask = torch.zeros(batch_size, 1, 1, seq_len)
                attention_mask[:, :, :, :seq_len//2] = -10000.0  # Simulate padding
                kwargs['attention_mask'] = attention_mask
                
            if has_pos:
                kwargs['position_embeddings'] = torch.rand(batch_size, seq_len, 1024)
                
            if has_token:
                kwargs['token_type_ids'] = torch.rand(batch_size, seq_len, 1024)
                
            builder.trace(
                args=(hidden_states,),
                kwargs=kwargs,
                tag=f"bucket{idx}"
            )
        
        # Compile with first bucket as priority
        nxd_model = builder.compile(priority_model_key="bucket1")

        self.assertIsInstance(nxd_model, neuronx_distributed.trace.nxd_model.NxDModel)

        torch.classes.neuron.Runtime().unsafe_close()

    def test_compile_with_weight_to_skip_layout_optimization(self):
        """Test compilation with specific weights skipped during layout optimization."""
        class SimpleParallelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.key = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.value = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.output = RowParallelLinear(1024, 1024, input_is_parallel=True)
                self.fc = ColumnParallelLinear(1024, 1024, gather_output=True)
                
            def forward(self, hidden_states, attention_mask=None):
                q = self.query(hidden_states)
                k = self.key(hidden_states)
                v = self.value(hidden_states)
                
                # Simple attention
                scores = torch.matmul(q, k.transpose(-1, -2))
                if attention_mask is not None:
                    scores = scores + attention_mask
                attn = torch.softmax(scores, dim=-1)
                context = torch.matmul(attn, v)
                
                output = self.output(context)
                return self.fc(output)

        model = SimpleParallelModel()
        
        # Specify weights to skip layout optimization
        weights_to_skip = {
            'query->weight',
            'key->weight',
            'value->weight'
        }
        
        # Create different input configurations
        input_configs = [
            # (batch_size, has_attention_mask)
            (32, True),   # With attention mask
            (16, False),  # Without attention mask
            (8, True),    # Different batch size with mask
        ]
        
        builder = ModelBuilderV2(
            model, 
            weights_to_skip_layout_optimization=weights_to_skip
        )
        
        # Trace with different configurations
        for idx, (batch_size, has_mask) in enumerate(input_configs, 1):
            hidden_states = torch.rand(batch_size, 128, 1024)
            kwargs = {}
            
            if has_mask:
                attention_mask = torch.zeros(batch_size, 1, 1, 128)
                attention_mask[:, :, :, 64:] = -10000.0
                kwargs['attention_mask'] = attention_mask
                
            builder.trace(
                args=(hidden_states,),
                kwargs=kwargs,
                tag=f"bucket{idx}"
            )
        
        # Compile with first bucket as priority
        nxd_model = builder.compile(priority_model_key="bucket1")
        
        self.assertIsInstance(nxd_model, neuronx_distributed.trace.nxd_model.NxDModel)
        
        # Verify weights were properly marked for skipping
        priority_trace = builder.trace_artifacts_collection["bucket1"]
        
        # Check that weight names were properly mapped to indices
        weight_name_to_idx = priority_trace.weight_name_to_idx
        for weight_name in weights_to_skip:
            self.assertIn(weight_name, weight_name_to_idx, 
                        f"Weight {weight_name} not found in weight_name_to_idx")
        
        # Verify these weights are in the skip set
        self.assertEqual(
            weights_to_skip,
            priority_trace.weight_names_to_skip,
            "Mismatch in weights marked for skipping layout optimization"
        )
        
        torch.classes.neuron.Runtime().unsafe_close()

if __name__ == '__main__':
    unittest.main()
