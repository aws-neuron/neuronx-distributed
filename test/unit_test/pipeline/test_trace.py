# Standard Library
import unittest
from unittest.mock import patch, MagicMock

# Third Party
import torch

import neuronx_distributed.pipeline.trace as tracer
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 2)

    def _transform(self, x):
        return x.transpose(1, 0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x) + self.linear3(x)
        trans_x = self._transform(x)
        return trans_x
    
class NestedModule(torch.nn.Module):
    
        def __init__(self):
            super().__init__()
            self.my_mod = MyModule()
            self.linear4 = torch.nn.Linear(2, 2)
    
        def forward(self, x):
            x = self.my_mod(x)
            return self.linear4(x)
    
class NestedNxDModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.my_mod = MyModule()
        self.cpl = ColumnParallelLinear(10,10)
        self.linear4 = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = self.my_mod(x)
        x = self.cpl(x)
        x = parallel_cross_entropy(x)
        return self.linear4(x)

class TestTrace(unittest.TestCase):
    def test_get_concrete_args(self):
        mod = MyModule()
        args = tracer.get_concrete_args(mod, [])
        assert 'x' in args
        assert len(args) == 1

    def test_get_tracer_class_torch_model(self):
        mod = MyModule()
        tracer_cls = tracer.get_tracer_class(mod)
        assert tracer_cls == tracer.TorchTracerWrapper

    def test_get_tracer_class_hf_cls_input(self):
        mod = MyModule()
        tracer_cls = tracer.get_tracer_class(mod, "hf")
        assert tracer_cls == tracer.HFTracerWrapper

    def test_get_tracer_class_torch_cls_input(self):
        mod = MyModule()
        tracer_cls = tracer.get_tracer_class(mod, "torch")
        assert tracer_cls == tracer.TorchTracerWrapper

    def test_get_tracer_class_invalid_input(self):
        mod = MyModule()
        with self.assertRaises(ValueError):
            tracer.get_tracer_class(mod, "test")

    @patch('torch.distributed.get_rank')   
    def test_trace_model(self, rank_mock):
        mod = MyModule()
        traced_model = tracer.trace_model(model=mod, input_names=['x'])
        expected_nodes = [{'op': 'placeholder', 'name': 'x'}, {'op': 'call_module', 'name': 'linear1'}, 
                          {'op': 'call_module', 'name': 'linear2'}, {'op': 'call_module', 'name': 'linear3'}, 
                          {'op': 'call_function', 'name': 'add'}, {'op': 'call_method', 'name': 'transpose'}, 
                          {'op': 'output', 'name': 'output'}]
        ops = [{"op": node.op, "name": node.name} for node in traced_model.graph.nodes]
        assert ops == expected_nodes

    @patch('torch.distributed.get_rank')   
    def test_trace_model_will_leaf_module(self, rank_mock):
        mod = NestedModule()
        traced_model = tracer.trace_model(model=mod, input_names=['x'], leaf_modules=['MyModule'])
        expected_nodes = [{'op': 'placeholder', 'name': 'x'}, {'op': 'call_module', 'name': 'my_mod'}, 
                        {'op': 'call_module', 'name': 'linear4'}, {'op': 'output', 'name': 'output'}]
        ops = [{"op": node.op, "name": node.name} for node in traced_model.graph.nodes]
        assert ops == expected_nodes

    @patch('neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size', MagicMock(return_value=1))
    @patch('neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank', MagicMock(return_value=1))
    @patch('neuronx_distributed.parallel_layers.layers._initialize_affine_weight_cpu', MagicMock(return_value=None))
    @patch('neuronx_distributed.parallel_layers.layers._initialize_affine_weight_neuron', MagicMock(return_value=None))
    @patch('torch.distributed.get_rank')   
    def test_nxd_trace_model_will_leaf_module(self, rank_mock):
        mod = NestedNxDModule()
        traced_model = tracer.trace_model(model=mod, input_names=['x'], leaf_modules=['MyModule', 'ColumnParallelLinear'], 
                                          autowrap_functions=[parallel_cross_entropy])
        expected_nodes = [{'op': 'placeholder', 'name': 'x'}, {'op': 'call_module', 'name': 'my_mod'}, 
                          {'op': 'call_module', 'name': 'cpl'}, {'op': 'call_function', 'name': 'parallel_cross_entropy'}, 
                          {'op': 'call_module', 'name': 'linear4'}, {'op': 'output', 'name': 'output'}]
        ops = [{"op": node.op, "name": node.name} for node in traced_model.graph.nodes]
        assert ops == expected_nodes


if __name__ == "__main__":
    unittest.main()