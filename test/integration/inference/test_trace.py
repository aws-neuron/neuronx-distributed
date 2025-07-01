import unittest
import torch
import torch.nn as nn
from torch_neuronx.proto import metaneff_pb2

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.model_builder import trace
from neuronx_distributed.trace.model_builder_utils import TraceArtifacts
from neuronx_distributed.trace.mock_torchdist import mock_distributed

torch.manual_seed(0)

class TestTrace(unittest.TestCase):
    def test_function_model(self):
        def func(x, y):
            return 2 * x + y
        
        input_shapes = [
            (3,),
            (4, 5),
            (2, 3, 4)
        ]
        
        for shape in input_shapes:
            with self.subTest(shape=shape):
                example_inputs = (torch.rand(shape), torch.rand(shape))
                trace_artifacts = trace(func, args=example_inputs, preserve_parameters=False)
                
                self.assertIsInstance(trace_artifacts, TraceArtifacts)
                self.assertIsNotNone(trace_artifacts.hlo)
                self.assertIsNotNone(trace_artifacts.metaneff)
                self.assertIsNotNone(trace_artifacts.flattener)
                self.assertIsNotNone(trace_artifacts.packer)
                self.assertIsNotNone(trace_artifacts.weight_name_to_idx)
                self.assertIsNotNone(trace_artifacts.provided_args)
                self.assertIsNotNone(trace_artifacts.model_params)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_module_model(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            def forward(self, x):
                return self.linear(x) + 1
        
        model = Model()
        model.eval()
        example_inputs = torch.rand(3, 10)
        
        trace_artifacts = trace(model, args=example_inputs, preserve_parameters=False)
        
        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        self.assertIsNotNone(trace_artifacts.hlo)
        self.assertIsNotNone(trace_artifacts.metaneff)
        self.assertIsNotNone(trace_artifacts.flattener)
        self.assertIsNotNone(trace_artifacts.packer)
        self.assertIsNotNone(trace_artifacts.weight_name_to_idx)
        self.assertIsNotNone(trace_artifacts.provided_args)
        self.assertIsNotNone(trace_artifacts.model_params)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_function_with_multiple_inputs(self):
        def multi_input_func(x, y, z):
            return x * y + z
        
        example_inputs = (torch.rand(3), torch.rand(3), torch.rand(3))
        trace_artifacts = trace(multi_input_func, args=example_inputs, preserve_parameters=False)
        
        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_module_with_multiple_inputs(self):
        class MultiInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(5, 3)
                self.linear2 = nn.Linear(11, 3)

            def forward(self, x, y):
                return self.linear1(x) + self.linear2(y)
        
        model = MultiInputModel()
        example_inputs = (torch.rand(3, 5), torch.rand(3, 11))
        
        trace_artifacts = trace(model, args=example_inputs, preserve_parameters=False)
        
        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_invalid_input_type(self):
        def func(x):
            return x * 2
        
        invalid_input = "not a tensor"
        with self.assertRaises(ValueError):
            trace(func, args=invalid_input, preserve_parameters=False)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_tuple_with_non_tensor(self):
        def func(x, y):
            return x + y
        
        invalid_input = (torch.randn(3), "not a tensor")
        with self.assertRaises(ValueError):
            trace(func, args=invalid_input, preserve_parameters=False)
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_non_spmd_tracing(self):
        def func(x):
            return x * 2
        
        example_inputs = torch.rand(3)
        with self.assertRaises(NotImplementedError):
            trace(func, args=example_inputs, spmd=False, preserve_parameters=False)
       
        torch.classes.neuron.Runtime().unsafe_close()

    def test_kwargs_only(self):
        class KwargsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x=None, y=None):
                if y is None:
                    return self.linear(x)
                return self.linear(x) + y
        
        model = KwargsModel()
        
        # Test with single kwarg
        trace_artifacts = trace(
            model,
            kwargs={"x": torch.rand(3, 10)}
        )
        
        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        # Verify ``user_input_key`` is set for x
        for tensor in trace_artifacts.metaneff.input_tensors:
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT:
                self.assertEqual(tensor.user_input_key.decode(), "x")
        
        # Test with multiple kwargs
        trace_artifacts = trace(
            model,
            kwargs={
                "x": torch.rand(3, 10),
                "y": torch.rand(3, 5)
            }
        )
        
        # Verify user_input_keys are set correctly
        input_keys = []
        for tensor in trace_artifacts.metaneff.input_tensors:
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT:
                input_keys.append(tensor.user_input_key.decode())
        self.assertEqual(set(input_keys), {"x", "y"})
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_args_and_kwargs_mixed(self):
        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x, y=None, z=None):
                out = self.linear(x)
                if y is not None:
                    out = out + y
                if z is not None:
                    out = out * z
                return out
        
        model = MixedModel()
        
        # Test with args and kwargs
        trace_artifacts = trace(
            model,
            args=(torch.rand(3, 10),),
            kwargs={
                "z": torch.rand(3, 5),
                "y": torch.rand(3, 5),
            }
        )
        
        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        
        # Verify correct number of USER_INPUT tensors
        user_inputs = [
            tensor for tensor in trace_artifacts.metaneff.input_tensors
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT
        ]
        self.assertEqual(len(user_inputs), 3)

        # Verify kwargs have ``user_input_key`` set
        params = []
        for tensor in user_inputs:
            self.assertTrue(tensor.user_input_key)
            params.append(tensor.user_input_key.decode())
        self.assertEqual(set(params), {"x", "y", "z"})
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_args_and_kwargs_mixed_some_not_used_in_computation(self):
        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x, y=None, z=None):
                out = self.linear(x)
                if z is not None:
                    out = out * z
                return out
        
        model = MixedModel()
        
        # Test with args and kwargs
        trace_artifacts = trace(
            model,
            args=(torch.rand(3, 10),),
            kwargs={
                "z": torch.rand(3, 5),
                "y": torch.rand(3, 5),
            }
        )
        
        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        
        # Verify correct number of USER_INPUT tensors
        user_inputs = [
            tensor for tensor in trace_artifacts.metaneff.input_tensors
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT
        ]
        self.assertEqual(len(user_inputs), 2)

        # Verify user_input_key
        params = []
        for tensor in user_inputs:
            self.assertTrue(tensor.user_input_key)
            params.append(tensor.user_input_key.decode())
        self.assertEqual(set(params), {"x", "z"})
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_invalid_kwargs(self):
        def func(x, y=None):
            return x + (y if y is not None else 0)
        
        # Test with invalid kwarg name
        with self.assertRaises(ValueError):
            trace(
                func,
                kwargs={"invalid_arg": torch.rand(3)}
            )
        
        # Test with non-tensor kwarg value
        with self.assertRaises(ValueError):
            trace(
                func,
                kwargs={"y": "not a tensor"}
            )
        
        torch.classes.neuron.Runtime().unsafe_close()

    def test_required_args_as_kwargs(self):
        def func(a, b=None, c=None):
            if b is None and c is None:
                return a
            if c is None:
                return a + b
            return a + b + c
        
        # Test providing required argument 'a' as kwarg
        trace_artifacts = trace(
            func,
            kwargs={
                "a": torch.rand(3, 4),
                "b": torch.rand(3, 4),
                "c": torch.rand(3, 4)
            }
        )
        
        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        
        # Verify user_input_keys are set correctly
        input_keys = []
        for tensor in trace_artifacts.metaneff.input_tensors:
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT:
                input_keys.append(tensor.user_input_key.decode())
        self.assertEqual(set(input_keys), {"a", "b", "c"})
        
        # Verify is_positional flag is set correctly in provided_args
        param_info = {arg.param_name: arg.is_positional for arg in trace_artifacts.provided_args}
        self.assertTrue(param_info["a"])  # 'a' should be marked as positional
        self.assertFalse(param_info["b"])  # 'b' should be marked as non-positional
        self.assertFalse(param_info["c"])  # 'c' should be marked as non-positional

        torch.classes.neuron.Runtime().unsafe_close()

    def test_mix_required_and_optional_args(self):
        def func(a, b=None, c=None):
            if b is None and c is None:
                return a
            if c is None:
                return a + b
            return a + b + c
        
        # Test providing mix of positional and keyword arguments
        trace_artifacts = trace(
            func,
            args=(torch.rand(3, 4), torch.rand(3, 4)),
            kwargs={"c": torch.rand(3, 4)}
        )
        
        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        
        # Verify user_input_keys
        input_keys = []
        for tensor in trace_artifacts.metaneff.input_tensors:
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT:
                input_keys.append(tensor.user_input_key.decode())
        self.assertEqual(set(input_keys), {"a", "b", "c"})
        
        # Verify is_positional flags
        param_info = {arg.param_name: arg.is_positional for arg in trace_artifacts.provided_args}
        self.assertTrue(param_info["a"])
        self.assertFalse(param_info["b"])
        self.assertFalse(param_info["c"])

        torch.classes.neuron.Runtime().unsafe_close()


class TestTraceDistributed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize process group and model parallel
        cls.mock_dist = mock_distributed(world_size=32)
        cls.mock_dist.__enter__()
        torch.distributed.init_process_group(backend="xla", rank=0, world_size=32)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=32, skip_collective_init=True)

    @classmethod
    def tearDownClass(cls):
        # Clean up the distributed environment
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
        cls.mock_dist.__exit__(None, None, None)

    def test_trace_column_parallel_linear(self):
        class ColumnParallelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = ColumnParallelLinear(1024, 1024, gather_output=False)

            def forward(self, x):
                return self.layer(x)

        model = ColumnParallelModel()
        example_inputs = torch.randn(32, 1024)
        trace_artifacts = trace(model, args=example_inputs, preserve_parameters=False)

        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        self.assertIsNotNone(trace_artifacts.hlo)
        self.assertIsNotNone(trace_artifacts.metaneff)

        torch.classes.neuron.Runtime().unsafe_close()

    def test_trace_row_parallel_linear(self):
        class RowParallelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = RowParallelLinear(1024, 1024, input_is_parallel=True)

            def forward(self, x):
                return self.layer(x)

        model = RowParallelModel()
        example_inputs = torch.randn(32, 1024)
        trace_artifacts = trace(model, args=example_inputs, preserve_parameters=False)

        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        self.assertIsNotNone(trace_artifacts.hlo)

        torch.classes.neuron.Runtime().unsafe_close()

    def test_trace_combined_parallel_layers(self):
        class CombinedParallelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)

            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)

        model = CombinedParallelModel()
        example_inputs = torch.randn(32, 1024)
        trace_artifacts = trace(model, args=example_inputs, preserve_parameters=False)

        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        self.assertIsNotNone(trace_artifacts.hlo)

        torch.classes.neuron.Runtime().unsafe_close()

    def test_trace_column_parallel_linear_with_kwargs(self):
        class ColumnParallelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = ColumnParallelLinear(1024, 1024, gather_output=False)

            def forward(self, input_ids=None, attention_mask=None):
                out = self.layer(input_ids)
                if attention_mask is not None:
                    out = out * attention_mask
                return out

        model = ColumnParallelModel()
        trace_artifacts = trace(
            model,
            kwargs={
                "input_ids": torch.randn(32, 1024),
                "attention_mask": torch.ones(32, 1)
            }
        )

        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        
        # Verify user_input_keys are set correctly
        input_keys = []
        for tensor in trace_artifacts.metaneff.input_tensors:
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT:
                self.assertTrue(tensor.user_input_key)
                input_keys.append(tensor.user_input_key.decode())
        
        self.assertEqual(set(input_keys), {"input_ids", "attention_mask"})

        torch.classes.neuron.Runtime().unsafe_close()

    def test_trace_combined_parallel_layers_with_mixed_inputs(self):
        class CombinedParallelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)

            def forward(self, x, scale_factor=None):
                x = self.layer1(x)
                if scale_factor is not None:
                    x = x * scale_factor
                return self.layer2(x)

        model = CombinedParallelModel()
        trace_artifacts = trace(
            model,
            args=(torch.randn(32, 1024),),
            kwargs={"scale_factor": torch.randn(32, 1)}
        )

        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        
        # Verify user_input_key
        input_keys = []
        for tensor in trace_artifacts.metaneff.input_tensors:
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT:
                input_keys.append(tensor.user_input_key.decode())
        self.assertEqual(set(input_keys), {"x", "scale_factor"})

        torch.classes.neuron.Runtime().unsafe_close()

    def test_distributed_required_args_as_kwargs(self):
        class DistributedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.col_layer = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.row_layer = RowParallelLinear(1024, 1024, input_is_parallel=True)
            
            def forward(self, input_ids, attention_mask=None, position_ids=None):
                x = self.col_layer(input_ids)
                if attention_mask is not None:
                    x = x * attention_mask
                x = self.row_layer(x)
                if position_ids is not None:
                    x = x + position_ids
                return x

        model = DistributedModel()
        
        # Test providing required argument as kwarg
        trace_artifacts = trace(
            model,
            kwargs={
                "input_ids": torch.randn(32, 1024),
                "attention_mask": torch.ones(32, 1),
                "position_ids": torch.randn(32, 1024)
            }
        )

        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        
        # Verify user_input_keys
        input_keys = []
        for tensor in trace_artifacts.metaneff.input_tensors:
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT:
                input_keys.append(tensor.user_input_key.decode())
        self.assertEqual(set(input_keys), {"input_ids", "attention_mask", "position_ids"})
        
        # Verify is_positional flags
        param_info = {arg.param_name: arg.is_positional for arg in trace_artifacts.provided_args}
        self.assertTrue(param_info["input_ids"])
        self.assertFalse(param_info["attention_mask"])
        self.assertFalse(param_info["position_ids"])

        torch.classes.neuron.Runtime().unsafe_close()

    def test_distributed_mix_required_and_optional_args(self):
        class DistributedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.col_layer = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.row_layer = RowParallelLinear(1024, 1024, input_is_parallel=True)
            
            def forward(self, input_ids, attention_mask=None, position_ids=None):
                x = self.col_layer(input_ids)
                if attention_mask is not None:
                    x = x * attention_mask
                x = self.row_layer(x)
                if position_ids is not None:
                    x = x + position_ids
                return x

        model = DistributedModel()
        
        # Test mix of positional and keyword arguments
        trace_artifacts = trace(
            model,
            args=(torch.randn(32, 1024),torch.ones(32, 1),),
            kwargs={
                "position_ids": torch.randn(32, 1024)
            }
        )

        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        
        # Verify user_input_keys
        input_keys = []
        for tensor in trace_artifacts.metaneff.input_tensors:
            if tensor.type == metaneff_pb2.MetaTensor.Type.USER_INPUT:
                input_keys.append(tensor.user_input_key.decode())
        self.assertEqual(set(input_keys), {"input_ids", "attention_mask", "position_ids"})
        
        # Verify is_positional flags
        param_info = {arg.param_name: arg.is_positional for arg in trace_artifacts.provided_args}
        self.assertTrue(param_info["input_ids"])
        self.assertFalse(param_info["attention_mask"])
        self.assertFalse(param_info["position_ids"])

        torch.classes.neuron.Runtime().unsafe_close()


if __name__ == '__main__':
    unittest.main()