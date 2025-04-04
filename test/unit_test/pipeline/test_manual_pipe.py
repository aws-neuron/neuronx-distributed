"""
This file tests basic creation of using manual pipeline stage module
"""

from neuronx_distributed.pipeline.manual_pipe_stage import PipelineStageModule
from neuronx_distributed.pipeline.model import NxDPPModel
import torch

import unittest
from unittest.mock import MagicMock, patch


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
            ), f"Expected to get 1 module in each stage, got {len(stage.stage_modules)}"

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

        pipe_module = PipelineStageModule(layers, 4, 0)
        weight_sharing_info = pipe_module.shared_weights_on_pp_stages["shared_first_last_layer_weight"]
        layer, weight_path = weight_sharing_info[1][0], weight_sharing_info[1][1]
        shared_on_pp_ranks = weight_sharing_info[0]

        assert torch.allclose(getattr(layer, weight_path), layers[0].weight), (
            "Expected to have shared weight in the first layer, "
        )

        assert set(shared_on_pp_ranks) == set([0, 3]), (
            f"Expected to have shared weight on pipeline stages 0 and 1"
            f"but got {shared_on_pp_ranks}"
        )

        # test construction on middle stage
        pipe_module = PipelineStageModule(layers, 4, 2)
        assert (
            len(pipe_module.shared_weights_on_pp_stages) == 0
        ), "Expected to have no shared weights on stage 2"
    
    def test_gather_pp_stage_idxs(self,):
        # mock the case for a 4-layer model with 4 PP stages
        layers = ToyModel(nlayers=4).get_layers()
        group_name = "shared_first_last_layer_weight"
        PipelineStageModule.mark_weight_sharing(
            [(layers[0], "weight"), (layers[-1], "weight")],
            group_name,
        )
        layers_to_stage_idx = {
            layers[0]: 0,
            layers[1]: 1,
            layers[2]: 2,
            layers[3]: 3,
        }

        weight_sharing_info = PipelineStageModule._gather_pp_stage_idxs_for_weight_sharing(
            stage_idx_mapping=layers_to_stage_idx,
            current_stage_idx=0,
        )

        assert len(weight_sharing_info) == 1, "Expected to have 1 shared weight"
        assert len(weight_sharing_info[group_name]) == 2, "Expected to have [shared_pp_stage_idxs, [layer, weight_path]]"
        assert weight_sharing_info[group_name][0] == [0, 3], "Expected to have shared weight on stage 0 and 3"
        assert weight_sharing_info[group_name][1][0] == layers[0], "Expected to have shared weight in the first layer"
        assert weight_sharing_info[group_name][1][1] == "weight", "Expected to have shared weight in the first layer"

        weight_sharing_info = PipelineStageModule._gather_pp_stage_idxs_for_weight_sharing(
            stage_idx_mapping=layers_to_stage_idx,
            current_stage_idx=3,
        )

        assert len(weight_sharing_info) == 1, "Expected to have 1 shared weight"
        assert len(weight_sharing_info[group_name]) == 2, "Expected to have [shared_pp_stage_idxs, [layer, weight_path]]"
        assert weight_sharing_info[group_name][0] == [0, 3], "Expected to have shared weight on stage 0 and 3"
        assert weight_sharing_info[group_name][1][0] == layers[3], "Expected to have shared weight in the 4th layer"
        assert weight_sharing_info[group_name][1][1] == "weight", "Expected to have shared weight in the 4th layer"

        weight_sharing_info = PipelineStageModule._gather_pp_stage_idxs_for_weight_sharing(
            stage_idx_mapping=layers_to_stage_idx,
            current_stage_idx=2,
        )
        assert len(weight_sharing_info) == 0, "Expected to have no shared weight on stage 2"

        weight_sharing_info = PipelineStageModule._gather_pp_stage_idxs_for_weight_sharing(
            stage_idx_mapping=layers_to_stage_idx,
            current_stage_idx=1,
        )
        assert len(weight_sharing_info) == 0, "Expected to have no shared weight on stage 1"

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
        n_stage = 2

        def custom_partition_fn(layers, _):
            rst = {}
            for i, layer in enumerate(layers):
                rst[layer] = 0 if i == 0 else 1
            return rst

        first_stage_layers, _  = PipelineStageModule._partition(
            layers, n_stage, 0, custom_partition_fn
        )
        second_stage_layers, _ = PipelineStageModule._partition(
            layers, n_stage, 1, custom_partition_fn
        )
        assert (
            len(first_stage_layers) == 1
        ), "Expected to have only 1 layer in the first stage"
        assert (
            len(second_stage_layers) == 3
        ), "Expected to have 3 layers in the second stage"


class TestStageDataPreprocess(unittest.TestCase):

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_parameter_cpu", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_affine_weight_neuron", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=4)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=0)
    )
    @patch("neuronx_distributed.pipeline.model.NxDPPModel.get_current_stage", MagicMock(return_value=0))
    @patch("neuronx_distributed.pipeline.model.NxDPPModel.move_model_to_device", MagicMock(return_value=lambda x: x))
    @patch("torch.distributed.get_rank")
    @patch("neuronx_distributed.pipeline.model.parallel_state")
    def test_preprocess_first_stage_io(self, rank_mock, state_mock):
        """Testing the preprocess for first stage, 
        should read data from data iterator
        """
        toy_model = ToyModel(nlayers=4)
        nxdpp_model = NxDPPModel(
            toy_model.get_layers(), 
            num_microbatches=1, 
            manual_pp_partition=True
        )
        # put some mock data to the data iterator
        mock_data_point = "mock_data"
        nxdpp_model.model_inputs_iter = [iter([mock_data_point])]
        nxdpp_model.current_model_chunk = 0

        # expected to iterate to the next data loader
        nxdpp_model._manual_pp_preprocess_stage_io()

        assert nxdpp_model.current_mb_stage_input[0] == (mock_data_point, ), (
            f"Expected to have the first data point, but got {nxdpp_model.current_mb_stage_input[0]}"
        )
        assert nxdpp_model.current_mb_stage_label[0] == (None, ), (
            f"Expected to not have label data, but got {nxdpp_model.current_mb_stage_label[0]}"
        )

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_parameter_cpu", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_affine_weight_neuron", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=4)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=3)
    )
    @patch("neuronx_distributed.pipeline.model.NxDPPModel.get_current_stage", MagicMock(return_value=3))
    @patch("neuronx_distributed.pipeline.model.NxDPPModel.move_model_to_device", MagicMock(return_value=lambda x: x))
    @patch("neuronx_distributed.utils.serialization.SerializationManager.deserialize", MagicMock(return_value="deserialized_dummy_tensor"))
    @patch("torch.distributed.get_rank")
    @patch("neuronx_distributed.pipeline.model.parallel_state")
    def test_preprocess_last_stage_io(self, rank_mock, state_mock):
        """Testing the preprocess for last stage,
        should receive data from previous stage, 
        and read data from data iterator for labels 
        """
        toy_model = ToyModel(nlayers=4)
        nxdpp_model = NxDPPModel(
            toy_model.get_layers(), 
            num_microbatches=1, 
            manual_pp_partition=True,
            manual_pp_loss_fn=torch.nn.CrossEntropyLoss()
        )
        nxdpp_model.current_model_chunk = 0
        # put some mock data to the data iterator
        mock_label = "mock_label"
        nxdpp_model.model_labels_iter = [iter([mock_label])]

        # bypass shape tracing step
        nxdpp_model.shape_traced = True
        # inject the meta data for testing
        pipe_io_obj = nxdpp_model.stage_id_to_IO_input_names[3]["stage_3_input"]
        pipe_io_obj.metadata = ["dummy_meta"]
        def __dummy_recv_op(*arg, **kwargs):
            return "dummy_tensor"
        nxdpp_model.recv_op = __dummy_recv_op

        nxdpp_model._manual_pp_preprocess_stage_io()

        assert nxdpp_model.current_mb_stage_label[0] == (mock_label, ), (
            f"Expected to have label data, but got {nxdpp_model.current_mb_stage_label[0]}"
        )

        assert nxdpp_model.current_mb_stage_input[0] == ("deserialized_dummy_tensor", ), (
            f"Expected to have received dummy data, but got {nxdpp_model.current_mb_stage_input[0]}"
        )


    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_parameter_cpu", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_affine_weight_neuron", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=4)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=2)
    )
    @patch("neuronx_distributed.pipeline.model.NxDPPModel.get_current_stage", MagicMock(return_value=2))
    @patch("neuronx_distributed.pipeline.model.NxDPPModel.move_model_to_device", MagicMock(return_value=lambda x: x))
    @patch("neuronx_distributed.utils.serialization.SerializationManager.deserialize", MagicMock(return_value="deserialized_dummy_tensor"))
    @patch("torch.distributed.get_rank")
    @patch("neuronx_distributed.pipeline.model.parallel_state")
    def test_preprocess_middle_stage_io(self, rank_mock, state_mock):
        """Testing the preprocess for middle stage,
        should receive data from previous stage 
        """
        toy_model = ToyModel(nlayers=4)
        nxdpp_model = NxDPPModel(
            toy_model.get_layers(), 
            num_microbatches=1, 
            manual_pp_partition=True,
            manual_pp_loss_fn=torch.nn.CrossEntropyLoss()
        )
        nxdpp_model.current_model_chunk = 0

        # bypass shape tracing step
        nxdpp_model.shape_traced = True
        current_stage = nxdpp_model.get_current_stage()
        # inject the meta data for testing
        pipe_io_obj = nxdpp_model.stage_id_to_IO_input_names[current_stage][f"stage_{current_stage}_input"]
        pipe_io_obj.metadata = ["dummy_meta"]
        def __dummy_recv_op(*arg, **kwargs):
            return "dummy_tensor"
        nxdpp_model.recv_op = __dummy_recv_op

        nxdpp_model._manual_pp_preprocess_stage_io()

        assert nxdpp_model.current_mb_stage_label[0] == (None, ), (
            f"Expected to not have label data, but got {nxdpp_model.current_mb_stage_label[0]}"
        )

        assert nxdpp_model.current_mb_stage_input[0] == ("deserialized_dummy_tensor", ), (
            f"Expected to have received dummy data, but got {nxdpp_model.current_mb_stage_input[0]}"
        )

if __name__ == "__main__":
    unittest.main()
