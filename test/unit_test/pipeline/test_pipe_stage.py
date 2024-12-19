"""
This file tests basic creation of using manual pipeline stage module
"""

from neuronx_distributed.pipeline.manual_pipe_stage import PipelineStageModule
import torch

import unittest


class ToyModel(torch.nn.Module):
    def __init__(self, nlayers, hidden_size=8):
        super(ToyModel, self).__init__()
        self.layers = [
            torch.nn.Linear(hidden_size, hidden_size) for _ in range(nlayers)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_layers(
        self,
    ):
        return self.layers


class TestPipelineStageCreation(unittest.TestCase):

    def test_stage_creation(self):
        """"""
        model = ToyModel(nlayers=4).get_layers()
        n_stage = 4

        for stage_idx in range(n_stage):
            stage = PipelineStageModule(model, n_stage, stage_idx)
            assert (
                len(stage.stage_modules) == 1
            ), "Expected to get 1 module in each stage"

    def test_mark_shared_weights(
        self,
    ):
        layers = ToyModel(nlayers=4).get_layers()
        PipelineStageModule.mark_weight_sharing(
            [(layers[0], "weight"), (layers[-1], "weight")],
            "shared_first_last_layer_weight",
        )

        assert hasattr(
            layers[0], "weight_sharing"
        ), "Expected to have weight_sharing attribute in the first layer"
        assert hasattr(
            layers[-1], "weight_sharing"
        ), "Expected to have weight_sharing attribute in the last layer"


class TestPipelinePartition(unittest.TestCase):

    def test_partition_evenly(self):
        layers = ToyModel(nlayers=4).get_layers()
        n_stage = 4
        expected_stage_idx_mapping = {
            layers[0]: 0,
            layers[1]: 1,
            layers[2]: 2,
            layers[3]: 3,
        }

        stage_idx_mapping = PipelineStageModule._partition_evenly(layers, n_stage)
        for idx, layer in enumerate(layers):
            expected_idx = expected_stage_idx_mapping[layer]
            assigned_idx = stage_idx_mapping[layer]
            assert (
                expected_idx == assigned_idx
            ), f"Expected to have layer {idx} mapping to stage {expected_idx}, but got {assigned_idx}"

    def test_partition_evenly_uneven(self):
        layers = ToyModel(nlayers=5).get_layers()
        n_stage = 4
        expected_stage_idx_mapping = {
            layers[0]: 0,
            layers[1]: 1,
            layers[2]: 2,
            layers[3]: 3,
            layers[4]: 3,
        }

        stage_idx_mapping = PipelineStageModule._partition_evenly(layers, n_stage)
        for idx, layer in enumerate(layers):
            expected_idx = expected_stage_idx_mapping[layer]
            assigned_idx = stage_idx_mapping[layer]
            assert expected_idx == assigned_idx, (
                f"Expected to have layer {idx} mapping "
                f"to stage {expected_idx}, but got {assigned_idx}"
            )

    def test_partition_evenly_fail(self):
        layers = ToyModel(nlayers=4).get_layers()
        n_stage = 5
        with self.assertRaises(AssertionError):
            PipelineStageModule._partition_evenly(layers, n_stage)

    def test_use_custom_partition(self):
        layers = ToyModel(nlayers=4).get_layers()
        n_stage = 4

        def custom_partition_fn(layers, _):
            rst = {}
            for i, layer in enumerate(layers):
                rst[layer] = 0 if i == 0 else 1
            return rst

        first_stage_layers = PipelineStageModule._partition(
            layers, n_stage, 0, custom_partition_fn
        )
        second_stage_layers = PipelineStageModule._partition(
            layers, n_stage, 1, custom_partition_fn
        )
        assert (
            len(first_stage_layers) == 1
        ), "Expected to have only 1 layer in the first stage"
        assert (
            len(second_stage_layers) == 3
        ), "Expected to have 3 layers in the second stage"


if __name__ == "__main__":
    unittest.main()
