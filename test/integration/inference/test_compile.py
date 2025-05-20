import unittest
import torch
import os
import shutil
import tempfile

from neuronx_distributed.trace.model_builder import trace, compile
from neuronx_distributed.trace.model_builder_utils import (
    ModelBuilderConstants,
    CompilationArtifacts,
    generate_key,
)
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.mock_torchdist import mock_distributed

class TestCompile(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(10, 5)

    def tearDown(self):
        pass

    def test_compile_basic(self):
        input_tensor = torch.randn(3, 10)
        trace_artifacts = trace(self.model, input_tensor, preserve_parameters=False)
        compilation_artifacts = compile(
            hlo_module=trace_artifacts.hlo,
            flattener=trace_artifacts.flattener,
            packer=trace_artifacts.packer,
            metaneff=trace_artifacts.metaneff,
            weight_name_to_idx=trace_artifacts.weight_name_to_idx
        )

        self.assertIsInstance(compilation_artifacts, CompilationArtifacts)
        self.assertTrue(os.path.exists(compilation_artifacts.neff_filename))
        generated_key = generate_key(trace_artifacts.hlo)
        expected_path = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, generated_key)
        self.assertTrue(os.path.isdir(expected_path), f"Directory {expected_path} does not exist")

        shutil.rmtree(expected_path)

    def test_compile_with_custom_compiler_workdir(self):
        temp_dir = tempfile.mkdtemp()
        input_tensor = torch.randn(3, 10)
        trace_artifacts = trace(self.model, input_tensor, preserve_parameters=False)
        compilation_artifacts = compile(
            hlo_module=trace_artifacts.hlo,
            flattener=trace_artifacts.flattener,
            packer=trace_artifacts.packer,
            metaneff=trace_artifacts.metaneff,
            weight_name_to_idx=trace_artifacts.weight_name_to_idx,
            compiler_workdir=temp_dir
        )        

        self.assertIsInstance(compilation_artifacts, CompilationArtifacts)
        self.assertTrue(os.path.exists(compilation_artifacts.neff_filename))
        generated_key = generate_key(trace_artifacts.hlo)
        expected_path = os.path.join(temp_dir, generated_key)
        self.assertTrue(os.path.isdir(expected_path), f"Directory {expected_path} does not exist")

        shutil.rmtree(temp_dir)

    def test_compile_with_compiler_args(self):
        input_tensor = torch.randn(3, 10)
        trace_artifacts = trace(self.model, input_tensor, preserve_parameters=False)
        compiler_args = "--enable-mixed-precision-accumulation"
        compilation_artifacts = compile(
            hlo_module=trace_artifacts.hlo,
            flattener=trace_artifacts.flattener,
            packer=trace_artifacts.packer,
            metaneff=trace_artifacts.metaneff,
            weight_name_to_idx=trace_artifacts.weight_name_to_idx,
            compiler_args=compiler_args
        )        

        self.assertIsInstance(compilation_artifacts, CompilationArtifacts)
        self.assertTrue(os.path.exists(compilation_artifacts.neff_filename))
        generated_key = generate_key(trace_artifacts.hlo)
        expected_path = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, generated_key)
        self.assertTrue(os.path.isdir(expected_path), f"Directory {expected_path} does not exist")

        shutil.rmtree(expected_path)

    def test_compile_complex_model(self):
        class ComplexModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 20)
                self.linear2 = torch.nn.Linear(20, 5)
            
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                return self.linear2(x)

        model = ComplexModel()
        input_tensor = torch.randn(3, 10)
        trace_artifacts = trace(model, input_tensor, preserve_parameters=False)
        compilation_artifacts = compile(
            hlo_module=trace_artifacts.hlo,
            flattener=trace_artifacts.flattener,
            packer=trace_artifacts.packer,
            metaneff=trace_artifacts.metaneff,
            weight_name_to_idx=trace_artifacts.weight_name_to_idx
        )
        
        self.assertIsInstance(compilation_artifacts, CompilationArtifacts)
        self.assertTrue(os.path.exists(compilation_artifacts.neff_filename))
        generated_key = generate_key(trace_artifacts.hlo)
        expected_path = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, generated_key)
        self.assertTrue(os.path.isdir(expected_path), f"Directory {expected_path} does not exist")

        shutil.rmtree(expected_path)

    def test_compile_with_multiple_inputs(self):
        class MultiInputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(15, 5)
            
            def forward(self, x, y):
                return self.linear(torch.cat([x, y], dim=1))

        model = MultiInputModel()
        input_tensors = (torch.randn(3, 10), torch.randn(3, 5))
        trace_artifacts = trace(model, input_tensors, preserve_parameters=False)
        compilation_artifacts = compile(
            hlo_module=trace_artifacts.hlo,
            flattener=trace_artifacts.flattener,
            packer=trace_artifacts.packer,
            metaneff=trace_artifacts.metaneff,
            weight_name_to_idx=trace_artifacts.weight_name_to_idx
        )
        
        self.assertIsInstance(compilation_artifacts, CompilationArtifacts)
        self.assertTrue(os.path.exists(compilation_artifacts.neff_filename))
        generated_key = generate_key(trace_artifacts.hlo)
        expected_path = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, generated_key)
        self.assertTrue(os.path.isdir(expected_path), f"Directory {expected_path} does not exist")

        shutil.rmtree(expected_path)

    def test_compile_multi_hlos(self):
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        input1 = torch.randn(3, 10)
        input2 = torch.randn(4, 10)
        input3 = torch.randn(5, 10)

        trace_artifacts = {
            'key1': trace(model, input1),
            'key2': trace(model, input2),
            'key3': trace(model, input3)
        }

        for key in ['key1', 'key2', 'key3']:
            compilation_artifacts = compile(
                hlo_module=trace_artifacts[key].hlo,
                flattener=trace_artifacts[key].flattener,
                packer=trace_artifacts[key].packer,
                metaneff=trace_artifacts[key].metaneff,
                weight_name_to_idx=trace_artifacts[key].weight_name_to_idx,
                key=key
            )

            self.assertIsInstance(compilation_artifacts, CompilationArtifacts)
            self.assertTrue(os.path.exists(compilation_artifacts.neff_filename))
            expected_path = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, key)
            self.assertTrue(os.path.isdir(expected_path), f"Directory {expected_path} does not exist")

            shutil.rmtree(expected_path)


class TestCompileDistributed(unittest.TestCase):
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
    
    def test_compile_column_parallel_linear(self):
        class ColumnParallelModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = ColumnParallelLinear(1024, 1024, gather_output=False)

            def forward(self, x):
                return self.layer(x)

        model = ColumnParallelModel()
        input_tensor = torch.randn(32, 1024)
        trace_artifacts = trace(model, input_tensor, preserve_parameters=False)
        compilation_artifacts = compile(
            hlo_module=trace_artifacts.hlo,
            flattener=trace_artifacts.flattener,
            packer=trace_artifacts.packer,
            metaneff=trace_artifacts.metaneff,
            weight_name_to_idx=trace_artifacts.weight_name_to_idx
        )

        self.assertIsInstance(compilation_artifacts, CompilationArtifacts)
        self.assertTrue(os.path.exists(compilation_artifacts.neff_filename))
        generated_key = generate_key(trace_artifacts.hlo)
        expected_path = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, generated_key)
        self.assertTrue(os.path.isdir(expected_path), f"Directory {expected_path} does not exist")

        shutil.rmtree(expected_path)

    def test_compile_row_parallel_linear(self):
        class RowParallelModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = RowParallelLinear(1024, 1024, input_is_parallel=True)

            def forward(self, x):
                return self.layer(x)

        model = RowParallelModel()
        input_tensor = torch.randn(32, 1024)
        trace_artifacts = trace(model, input_tensor, preserve_parameters=False)
        compilation_artifacts = compile(
            hlo_module=trace_artifacts.hlo,
            flattener=trace_artifacts.flattener,
            packer=trace_artifacts.packer,
            metaneff=trace_artifacts.metaneff,
            weight_name_to_idx=trace_artifacts.weight_name_to_idx
        )

        self.assertIsInstance(compilation_artifacts, CompilationArtifacts)
        self.assertTrue(os.path.exists(compilation_artifacts.neff_filename))
        generated_key = generate_key(trace_artifacts.hlo)
        expected_path = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, generated_key)
        self.assertTrue(os.path.isdir(expected_path), f"Directory {expected_path} does not exist")

        shutil.rmtree(expected_path)

    def test_compile_combined_parallel_layers(self):
        class CombinedParallelModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)

            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)

        model = CombinedParallelModel()
        input_tensor = torch.randn(32, 1024)
        trace_artifacts = trace(model, input_tensor, preserve_parameters=False)
        compilation_artifacts = compile(
            hlo_module=trace_artifacts.hlo,
            flattener=trace_artifacts.flattener,
            packer=trace_artifacts.packer,
            metaneff=trace_artifacts.metaneff,
            weight_name_to_idx=trace_artifacts.weight_name_to_idx
        )

        self.assertIsInstance(compilation_artifacts, CompilationArtifacts)
        self.assertTrue(os.path.exists(compilation_artifacts.neff_filename))
        generated_key = generate_key(trace_artifacts.hlo)
        expected_path = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, generated_key)
        self.assertTrue(os.path.isdir(expected_path), f"Directory {expected_path} does not exist")

        shutil.rmtree(expected_path)

    def test_compile_multi_hlos(self):
        class CombinedParallelModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = ColumnParallelLinear(1024, 1024, gather_output=False)
                self.layer2 = RowParallelLinear(1024, 1024, input_is_parallel=True)

            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)

        model = CombinedParallelModel()
        input1 = torch.randn(32, 1024)
        input2 = torch.randn(64, 1024)
        input3 = torch.randn(128, 1024)

        trace_artifacts = {
            'key1': trace(model, input1),
            'key2': trace(model, input2),
            'key3': trace(model, input3)
        }

        for key in ['key1', 'key2', 'key3']:
            compilation_artifacts = compile(
                hlo_module=trace_artifacts[key].hlo,
                flattener=trace_artifacts[key].flattener,
                packer=trace_artifacts[key].packer,
                metaneff=trace_artifacts[key].metaneff,
                weight_name_to_idx=trace_artifacts[key].weight_name_to_idx,
                key=key
            )

            self.assertIsInstance(compilation_artifacts, CompilationArtifacts)
            self.assertTrue(os.path.exists(compilation_artifacts.neff_filename))
            expected_path = os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, key)
            self.assertTrue(os.path.isdir(expected_path), f"Directory {expected_path} does not exist")

            shutil.rmtree(expected_path)

if __name__ == '__main__':
    unittest.main()