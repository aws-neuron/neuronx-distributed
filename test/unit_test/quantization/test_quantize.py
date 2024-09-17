import unittest

import torch

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.quantization.quantization_layers import (
    QuantizedColumnParallel,
    QuantizedRowParallel,
)
from neuronx_distributed.quantization.quantize import convert


class TestConvert(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_world_size = parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
        self.initial_rank = parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK

        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0

    def tearDown(self) -> None:
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = self.initial_world_size
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = self.initial_rank

    def test_convert(self):
        class ParallelLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lay1 = ColumnParallelLinear(
                    input_size=4, output_size=6, device=torch.device("cpu"), bias=True, dtype=torch.float32
                )
                self.lay2 = RowParallelLinear(
                    input_size=6, output_size=4, device=torch.device("cpu"), bias=True, dtype=torch.float32
                )
                self.lay3 = torch.nn.Linear(
                    in_features=6, out_features=4, bias=False, device=torch.device("cpu"), dtype=torch.float32
                )

            def forward(self, x):
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = ParallelLinear()
                self.linear2 = torch.nn.Linear(
                    in_features=6, out_features=4, bias=False, device=torch.device("cpu"), dtype=torch.float32
                )

        model1 = Model()
        model2 = convert(module=model1, q_config=None, inplace=False, mapping=None)

        assert isinstance(model1.linear1.lay1, ColumnParallelLinear)
        assert isinstance(model1.linear1.lay2, RowParallelLinear)
        assert isinstance(model2.linear1.lay1, QuantizedColumnParallel)
        assert isinstance(model2.linear1.lay2, QuantizedRowParallel)

        del model1, model2

        # With inplace
        model1 = Model()
        convert(module=model1, q_config=None, inplace=True, mapping=None)

        assert isinstance(model1.linear1.lay1, QuantizedColumnParallel)
        assert isinstance(model1.linear1.lay2, QuantizedRowParallel)
