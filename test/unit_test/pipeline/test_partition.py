# Standard Library
import unittest
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np

# Third Party
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

import neuronx_distributed.pipeline.partition as partition
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy
from neuronx_distributed.pipeline.model import NxDPPModel

from .. import update_result
from .test_base import get_traced_model_gpt


class NestedNxDModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rpl = RowParallelLinear(10, 10)
        self.cpl = ColumnParallelLinear(10, 10)
        self.linear4 = torch.nn.Linear(2, 2)

    def forward(self, x):
        x = self.rpl(x)
        x = self.cpl(x)
        x = parallel_cross_entropy(x)
        return self.linear4(x)


def get_traced_model_nxd():
    model = NxDPPModel(module=NestedNxDModule(), transformer_layer_cls=torch.nn.Linear, tracer_cls="torch")
    model.trace(
        input_names=["x"],
        leaf_modules=["RowParallelLinear", "ColumnParallelLinear"],
        autowrap_functions=[parallel_cross_entropy],
    )
    model.cut_pipeline_stage("cpl")
    return model.traced_model


class TestPartition(unittest.TestCase):
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
    def test_partition_traced_model(self, rank_mock, state_mock):
        try:
            traced_model = get_traced_model_nxd()
            split_mod = partition.partition_traced_model(traced_model)
            for name, module in split_mod.named_children():
                if name == "submod_0":
                    for n, child_module in module.named_children():
                        if n == "rpl":
                            assert isinstance(child_module, RowParallelLinear)
                        elif n == "cpl":
                            assert isinstance(child_module, ColumnParallelLinear)
                        else:
                            assert False, "Unexpected node in submoule 0"
                elif name == "submod_1":
                    for n, child_module in module.named_children():
                        if n == "linear4":
                            assert isinstance(child_module, torch.nn.Linear)
                        else:
                            assert False, "Unexpected node in submoule 1"
                else:
                    assert False, "Unexpected number of submodule"
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
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=8)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("neuronx_distributed.pipeline.model.parallel_state")
    @patch("torch.distributed.get_rank")
    def test_partition_traced_model_gpt2(self, rank_mock, state_mock):
        try:
            traced_model = get_traced_model_gpt()
            split_mod = partition.partition_traced_model(traced_model)
            partition_count = 0
            for name, module in split_mod.named_children():
                partition_count += 1
                if name != "submod_7":
                    for n, child_module in module.named_children():
                        if "transformer_h_" in n:
                            assert isinstance(child_module, GPT2Block)
                num_params = sum([np.prod(p.size()) for p in module.parameters()])
                if partition_count == 1:
                    assert num_params == 53166336
                elif partition_count == 8:
                    assert num_params == 38598912
                else:
                    assert num_params == 7087872
            assert partition_count == 8
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
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=8)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("neuronx_distributed.pipeline.model.parallel_state")
    @patch("torch.distributed.get_rank")
    def test_analyze_pipeline_module(self, rank_mock, state_mock):
        try:
            traced_model = get_traced_model_gpt()
            split_mod = partition.partition_traced_model(traced_model)
            (
                stage_id_to_IO_input_names,
                stage_id_to_model_input_names,
                stage_id_to_input_count,
                stage_id_to_output_count,
            ) = partition.analyze_pipeline_module(split_mod)
            expected_io_names = {0: OrderedDict()}
            # Note: Renamed "add_2" to "add_3" when upgrading transformers from 4.28.1 to 4.36.2
            for i in range(7):
                if i == 0:
                    io_dict = OrderedDict(
                        [
                            (
                                "transformer_h_" + str(i + 1),
                                partition.PipelineIO("transformer_h_" + str(i + 1), input_idx=0, output_idx=0),
                            ),
                            ("mul", partition.PipelineIO("mul", input_idx=1, output_idx=1)),
                            ("add_3", partition.PipelineIO("add_3", output_idx=2)),
                        ]
                    )
                elif i < 6:
                    io_dict = OrderedDict(
                        [
                            (
                                "transformer_h_" + str(i + 1),
                                partition.PipelineIO("transformer_h_" + str(i + 1), input_idx=0, output_idx=0),
                            ),
                            ("mul", partition.PipelineIO("mul", input_idx=1)),
                            ("add_3", partition.PipelineIO("add_3")),
                        ]
                    )
                else:
                    io_dict = OrderedDict(
                        [
                            (
                                "transformer_h_" + str(i + 1),
                                partition.PipelineIO("transformer_h_" + str(i + 1), input_idx=0, output_idx=0),
                            ),
                            ("add_3", partition.PipelineIO("add_3", input_idx=1)),
                        ]
                    )
                expected_io_names.update({i + 1: io_dict})
            expected_stage_id_to_model_input_names = {
                0: {"input_ids": 0, "attention_mask": 1},
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
                6: {},
                7: {"labels": 2},
            }
            expected_stage_id_to_input_count = {0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3}
            expected_stage_id_to_output_count = {0: 3, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2}
            assert str(stage_id_to_IO_input_names) == str(expected_io_names)
            assert stage_id_to_model_input_names == expected_stage_id_to_model_input_names
            assert stage_id_to_input_count == expected_stage_id_to_input_count
            assert stage_id_to_output_count == expected_stage_id_to_output_count
        except:
            update_result({"inference_success": 0})
            raise


if __name__ == "__main__":
    unittest.main()
