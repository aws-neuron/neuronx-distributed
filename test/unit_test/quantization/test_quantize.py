import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

import torch

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.utils import is_torch_version_greater_than_2
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
        self.initial_tensor_model_parallel_group = parallel_state._TENSOR_MODEL_PARALLEL_GROUP
        self.initial_world_size = parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
        self.initial_rank = parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK
        self.initial_data_parallel_group = parallel_state._DATA_PARALLEL_GROUP

        parallel_state._TENSOR_MODEL_PARALLEL_GROUP = MagicMock(spec=torch.distributed.ProcessGroup)
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP.size.return_value = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0
        parallel_state._DATA_PARALLEL_GROUP = MagicMock(spec=torch.distributed.ProcessGroup)
        parallel_state._DATA_PARALLEL_GROUP.size.return_value = 1

    def tearDown(self) -> None:
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP = self.initial_tensor_model_parallel_group
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = self.initial_world_size
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = self.initial_rank
        parallel_state._DATA_PARALLEL_GROUP = self.initial_data_parallel_group
    
    @unittest.skipIf(not is_torch_version_greater_than_2(),
                    "There is no torch.device context manager in torch<=1.*")

    def test_convert(self):
        #### Define a dummy model
        class ParallelLinearBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lay1 = ColumnParallelLinear(
                    input_size=4, output_size=6, device=torch.device("meta"), bias=True, dtype=torch.float32
                )
                self.lay2 = RowParallelLinear(
                    input_size=6, output_size=4, device=torch.device("meta"), bias=True, dtype=torch.float32
                )
                self.lay3 = torch.nn.Linear(
                    in_features=6, out_features=4, bias=False, device=torch.device("meta"), dtype=torch.float32
                )

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_block = ParallelLinearBlock()
                self.linear = torch.nn.Linear(
                    in_features=6, out_features=4, bias=False, device=torch.device("meta"), dtype=torch.float32
                )

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_block = ParallelLinearBlock()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()
                self.attention = Attention()

        with torch.device('meta'):
            model1 = torch.nn.ModuleList([Model() for i in range(2)])

        include = ["*mlp.linear_block"]
        model2 = convert(module=model1, q_config=None, inplace=False, mapping=None, include=include)

        for i in range(2):
            assert isinstance(model2[i].mlp.linear_block.lay1, QuantizedColumnParallel)
            assert isinstance(model2[i].mlp.linear_block.lay2, QuantizedRowParallel)

        include = ["*mlp.linear_block.lay2"]
        model3 = convert(module=model1, q_config=None, inplace=False, mapping=None, include=include)
        for i in range(2):
            assert isinstance(model3[i].mlp.linear_block.lay1, ColumnParallelLinear)
            assert isinstance(model3[i].mlp.linear_block.lay2, QuantizedRowParallel)


        del model1, model2, model3

        # With inplace
        with torch.device("meta"):
            model1 = torch.nn.ModuleList([Model() for i in range(2)])
            _  = convert(module=model1, q_config=None, inplace=True, mapping=None)
            for i in range(2):
                assert isinstance(model1[i].mlp.linear_block.lay1, QuantizedColumnParallel)
                assert isinstance(model1[i].mlp.linear_block.lay2, QuantizedRowParallel)
                assert isinstance(model1[i].attention.linear_block.lay1, QuantizedColumnParallel)
                assert isinstance(model1[i].attention.linear_block.lay2, QuantizedRowParallel)
