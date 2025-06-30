import unittest
import torch
import os
import shutil
import tempfile

from neuronx_distributed.trace.model_builder import (
    trace,
    compile,
    compile_wlo,
    compile_layout_transformer,
) 
from neuronx_distributed.trace.model_builder_utils import (
    ModelBuilderConstants,
    CompilationArtifacts,
    WLOArtifacts,
    LayoutTransformerArtifacts,
)
from neuronx_distributed.trace.hlo_utils import mark_weights_for_wlo, apply_layout_transformation
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.mock_torchdist import mock_distributed

class TestCompileWLO(unittest.TestCase):
    def test_compile_multi_hlos(self):
        class DenseMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5120)
                self.layer2 = torch.nn.Linear(5120, 12000)
                self.layer3 = torch.nn.Linear(12000, 2048)
                self.layer4 = torch.nn.Linear(2048, 4096)
                self.layer5 = torch.nn.Linear(4096, 2048)
                self.layer6 = torch.nn.Linear(2048, 1024)
                self.layer7 = torch.nn.Linear(1024, 512)
                self.layer8 = torch.nn.Linear(512, 256)
                self.layer9 = torch.nn.Linear(256, 128)
                self.layer10 = torch.nn.Linear(128, 5)
                
                self.relu = torch.nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.layer1(x))
                x = self.relu(self.layer2(x))
                x = self.relu(self.layer3(x))
                x = self.relu(self.layer4(x))
                x = self.relu(self.layer5(x))
                x = self.relu(self.layer6(x))
                x = self.relu(self.layer7(x))
                x = self.relu(self.layer8(x))
                x = self.relu(self.layer9(x))
                x = self.layer10(x)
                return x

        model = DenseMLP()
        input1 = torch.randn(3, 10)
        input2 = torch.randn(4, 10)
        input3 = torch.randn(5, 10)

        trace_artifacts = {
            'key1': trace(model, args=input1),
            'key2': trace(model, args=input2),
            'key3': trace(model, args=input3)
        }

        # Let 'key1' be the priority model
        mark_weights_for_wlo(
            priority_model_trace_hlo=trace_artifacts['key1'].hlo,
            priority_model_weight_name_to_idx=trace_artifacts['key1'].weight_name_to_idx,
        )

        wlo_artifacts = compile_wlo(
            hlo_module=trace_artifacts['key1'].hlo,
            metaneff=trace_artifacts['key1'].metaneff,
            key='key1'
        )

        self.assertIsInstance(wlo_artifacts, WLOArtifacts)
        self.assertTrue(os.path.exists(wlo_artifacts.neff_filepath))
        self.assertTrue(os.path.exists(wlo_artifacts.wrapped_neff_hlo_filepath))
        
        # Subsequent model compilation
        for key in ['key2', 'key3']:
            apply_layout_transformation(
                hlo_module=trace_artifacts[key].hlo,
                flattener=trace_artifacts[key].flattener,
                packer=trace_artifacts[key].packer,
                metaneff=trace_artifacts[key].metaneff,
                weight_name_to_idx=trace_artifacts[key].weight_name_to_idx,
                wlo_artifacts=wlo_artifacts,
                key=key
            )

            compilation_artifacts = compile(
                hlo_module=trace_artifacts[key].hlo,
                metaneff=trace_artifacts[key].metaneff,
                key=key
            )

            self.assertIsInstance(compilation_artifacts, CompilationArtifacts)
            self.assertTrue(os.path.exists(compilation_artifacts.neff_filepath))

        layout_transformer = compile_layout_transformer(
            wlo_artifacts=wlo_artifacts,
            priority_model_weight_name_to_idx=trace_artifacts['key1'].weight_name_to_idx,
        )

        self.assertIsInstance(layout_transformer, LayoutTransformerArtifacts)
        self.assertTrue(os.path.exists(layout_transformer.hlo_filepath))
        self.assertTrue(os.path.exists(layout_transformer.neff_filepath))
        self.assertTrue(os.path.exists(layout_transformer.metaneff_filepath))

        for key in ['key1', 'key2', 'key3', 'layout_opt']:
            shutil.rmtree(os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, key))
        
        torch.classes.neuron.Runtime().unsafe_close()


class TestCompileWLODistributed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_dist = mock_distributed(world_size=2)
        cls.mock_dist.__enter__()
        torch.distributed.init_process_group(backend="xla", rank=0, world_size=2)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=2, skip_collective_init=True)
        parallel_state.set_aot_mode(True)

    @classmethod
    def tearDownClass(cls):
        parallel_state.set_aot_mode(False)
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()
        cls.mock_dist.__exit__(None, None, None)
    
    def test_compile_multi_hlos(self):
        class CPLRPLModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lay1 = ColumnParallelLinear(input_size=4, output_size=4, bias=False, gather_output=False, dtype=torch.float32)
                self.lay2 = RowParallelLinear(input_size=4, output_size=4, bias=False, input_is_parallel=True, dtype=torch.float32)

            def forward(self, x):
                x = self.lay1(x)
                return self.lay2(x)


        model = CPLRPLModel()
        input1 = torch.randn(3, 4)
        input2 = torch.randn(2, 4)
        input3 = torch.randn(4, 4)

        trace_artifacts = {
            'key1': trace(model, args=input1),
            'key2': trace(model, args=input2),
            'key3': trace(model, args=input3),
        }

        # Let 'key1' be the priority model
        mark_weights_for_wlo(
            priority_model_trace_hlo=trace_artifacts['key1'].hlo,
            priority_model_weight_name_to_idx=trace_artifacts['key1'].weight_name_to_idx,
        )

        wlo_artifacts = compile_wlo(
            hlo_module=trace_artifacts['key1'].hlo,
            metaneff=trace_artifacts['key1'].metaneff,
            key='key1'
        )

        self.assertIsInstance(wlo_artifacts, WLOArtifacts)
        self.assertTrue(os.path.exists(wlo_artifacts.neff_filepath))
        self.assertTrue(os.path.exists(wlo_artifacts.wrapped_neff_hlo_filepath))
        
        # Subsequent model compilation
        for key in ['key2', 'key3']:
            apply_layout_transformation(
                hlo_module=trace_artifacts[key].hlo,
                flattener=trace_artifacts[key].flattener,
                packer=trace_artifacts[key].packer,
                metaneff=trace_artifacts[key].metaneff,
                weight_name_to_idx=trace_artifacts[key].weight_name_to_idx,
                wlo_artifacts=wlo_artifacts,
                key=key
            )

            compilation_artifacts = compile(
                hlo_module=trace_artifacts[key].hlo,
                metaneff=trace_artifacts[key].metaneff,
                key=key
            )

            self.assertIsInstance(compilation_artifacts, CompilationArtifacts)
            self.assertTrue(os.path.exists(compilation_artifacts.neff_filepath))

        layout_transformer = compile_layout_transformer(
            wlo_artifacts=wlo_artifacts,
            priority_model_weight_name_to_idx=trace_artifacts['key1'].weight_name_to_idx,
        )

        self.assertIsInstance(layout_transformer, LayoutTransformerArtifacts)
        self.assertTrue(os.path.exists(layout_transformer.hlo_filepath))
        self.assertTrue(os.path.exists(layout_transformer.neff_filepath))
        self.assertTrue(os.path.exists(layout_transformer.metaneff_filepath))

        for key in ['key1', 'key2', 'key3', 'layout_opt']:
            shutil.rmtree(os.path.join(ModelBuilderConstants.DEFAULT_COMPILER_WORKDIR, key))
        
        torch.classes.neuron.Runtime().unsafe_close()


if __name__ == '__main__':
    unittest.main()