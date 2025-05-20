import unittest
import torch
import torch.nn as nn

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
                trace_artifacts = trace(func, example_inputs, preserve_parameters=False)
                
                self.assertIsInstance(trace_artifacts, TraceArtifacts)
                self.assertIsNotNone(trace_artifacts.hlo)
                self.assertIsNotNone(trace_artifacts.metaneff)
                self.assertIsNotNone(trace_artifacts.flattener)
                self.assertIsNotNone(trace_artifacts.packer)
                self.assertIsNotNone(trace_artifacts.weight_name_to_idx)

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
        
        trace_artifacts = trace(model, example_inputs, preserve_parameters=False)
        
        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        self.assertIsNotNone(trace_artifacts.hlo)
        self.assertIsNotNone(trace_artifacts.metaneff)
        self.assertIsNotNone(trace_artifacts.flattener)
        self.assertIsNotNone(trace_artifacts.packer)
        self.assertIsNotNone(trace_artifacts.weight_name_to_idx)

    def test_function_with_multiple_inputs(self):
        def multi_input_func(x, y, z):
            return x * y + z
        
        example_inputs = (torch.rand(3), torch.rand(3), torch.rand(3))
        trace_artifacts = trace(multi_input_func, example_inputs, preserve_parameters=False)
        
        self.assertIsInstance(trace_artifacts, TraceArtifacts)

    def test_module_with_multiple_inputs(self):
        class MultiInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(5, 3)
                self.linear2 = nn.Linear(11, 3)

            def forward(self, x, y):
                return self.linear1(x) + self.linear2(y)
        
        model = MultiInputModel()
        model.eval()
        example_inputs = (torch.rand(3, 5), torch.rand(3, 11))
        
        trace_artifacts = trace(model, example_inputs, preserve_parameters=False)
        
        self.assertIsInstance(trace_artifacts, TraceArtifacts)

    def test_invalid_input_type(self):
        def func(x):
            return x * 2
        
        invalid_input = "not a tensor"
        with self.assertRaises(ValueError):
            trace(func, invalid_input, preserve_parameters=False)

    def test_tuple_with_non_tensor(self):
        def func(x, y):
            return x + y
        
        invalid_input = (torch.randn(3), "not a tensor")
        with self.assertRaises(ValueError):
            trace(func, invalid_input, preserve_parameters=False)

    def test_non_spmd_tracing(self):
        def func(x):
            return x * 2
        
        example_inputs = torch.rand(3)
        with self.assertRaises(NotImplementedError):
            trace(func, example_inputs, spmd=False, preserve_parameters=False)


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
        trace_artifacts = trace(model, example_inputs, preserve_parameters=False)

        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        self.assertIsNotNone(trace_artifacts.hlo)
        self.assertIsNotNone(trace_artifacts.metaneff)

    def test_trace_row_parallel_linear(self):
        class RowParallelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = RowParallelLinear(1024, 1024, input_is_parallel=True)

            def forward(self, x):
                return self.layer(x)

        model = RowParallelModel()
        example_inputs = torch.randn(32, 1024)
        trace_artifacts = trace(model, example_inputs, preserve_parameters=False)

        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        self.assertIsNotNone(trace_artifacts.hlo)

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
        trace_artifacts = trace(model, example_inputs, preserve_parameters=False)

        self.assertIsInstance(trace_artifacts, TraceArtifacts)
        self.assertIsNotNone(trace_artifacts.hlo)


if __name__ == '__main__':
    unittest.main()