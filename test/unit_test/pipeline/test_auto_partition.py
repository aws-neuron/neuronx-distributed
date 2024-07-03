# Standard Library
import unittest
from unittest.mock import MagicMock, patch

# Third Party
import torch

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy
from neuronx_distributed.pipeline.model import NxDPPModel
from neuronx_distributed.pipeline.partition import create_partitions

from .. import update_result


class NxDModule(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.rpl = RowParallelLinear(10, 10)
        self.cpl = ColumnParallelLinear(10, 10)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(num_layers)])

    def forward(self, x):
        x = self.rpl(x)
        x = self.cpl(x)
        x = parallel_cross_entropy(x)
        return self.linear4(x)


def get_model_nxd(num_layers):
    model = NxDPPModel(module=NxDModule(num_layers), transformer_layer_cls=torch.nn.Linear, tracer_cls="torch")
    return model


class TestAutoPartition(unittest.TestCase):
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_parameter_cpu", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_affine_weight_neuron", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=2)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("neuronx_distributed.pipeline.model.parallel_state")
    @patch("torch.distributed.get_rank")
    def test_model_autopartition(self, rank_mock, state_mock):
        try:
            num_layers = 40
            model = get_model_nxd(num_layers)
            transformer_layer_cls = torch.nn.Linear

            pipeline_parallel_size = 4
            model_layers = model.get_model_layers(model.original_torch_module, transformer_layer_cls)
            partitions = create_partitions(pipeline_parallel_size, model_layers)

            expected_model_layers = [f"layers.{x}" for x in range(num_layers)]
            expected_partitions = [f"layers.{x}" for x in range(9, num_layers, 10)]
            expected_partitions = expected_partitions[:-1]
            assert model_layers == expected_model_layers
            assert partitions == expected_partitions
        except:
            update_result({"inference_success": 0})
            raise

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_parameter_cpu", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_affine_weight_neuron", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=2)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("neuronx_distributed.pipeline.model.parallel_state")
    @patch("torch.distributed.get_rank")
    def test_model_autopartition_unevenly_divisible_layers(self, rank_mock, state_mock):
        try:
            num_layers = 19
            model = get_model_nxd(num_layers)
            transformer_layer_cls = torch.nn.Linear

            pipeline_parallel_size = 4
            model_layers = model.get_model_layers(model.original_torch_module, transformer_layer_cls)
            partitions = create_partitions(pipeline_parallel_size, model_layers)
            expected_model_layers = [f"layers.{x}" for x in range(num_layers)]
            expected_partitions = ["layers.3", "layers.8", "layers.13"]
            assert model_layers == expected_model_layers
            assert partitions == expected_partitions
        except:
            update_result({"inference_success": 0})
            raise


if __name__ == "__main__":
    unittest.main()
