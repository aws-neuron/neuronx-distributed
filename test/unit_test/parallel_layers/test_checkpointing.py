import unittest
from unittest.mock import patch

from neuronx_distributed.parallel_layers.checkpointing import load

MODULE = "neuronx_distributed.parallel_layers.checkpointing"


class CheckpointingTest(unittest.TestCase):
    @patch(f"{MODULE}.get_tensor_model_parallel_size", return_value=1)
    @patch(f"{MODULE}.get_pipeline_model_parallel_rank", return_value=0)
    @patch(f"{MODULE}.get_tensor_model_parallel_rank", return_value=0)
    @patch(f"{MODULE}.torch")
    def test_load(self, mock_torch, mock_tp_rk, mock_pp_rk, mock_tp_sz):
        # Act
        res = load(path := "some_path")
        # Assert
        mock_torch.load.assert_called_once_with(
            path + "/tp_rank_00_pp_rank_00/checkpoint.pt", map_location="cpu", weights_only=False
        )
        assert res == mock_torch.load.return_value

    @patch(f"{MODULE}.get_tensor_model_parallel_size", return_value=1)
    @patch(f"{MODULE}.get_pipeline_model_parallel_rank", return_value=0)
    @patch(f"{MODULE}.get_tensor_model_parallel_rank", return_value=0)
    @patch(f"{MODULE}.torch")
    def test_load_xser(self, mock_torch, mock_tp_rk, mock_pp_rk, mock_tp_sz):
        # Act
        res = load(path := "some_path", load_xser=True)
        # Assert
        mock_torch.load.assert_called_once_with(path + "/tp_rank_00_pp_rank_00", weights_only=False)
        assert res == mock_torch.load.return_value

    @patch(f"{MODULE}.get_tensor_model_parallel_size", return_value=1)
    @patch(f"{MODULE}.get_pipeline_model_parallel_rank", return_value=0)
    @patch(f"{MODULE}.get_tensor_model_parallel_rank", return_value=0)
    @patch(f"{MODULE}.torch")
    def test_load_weights_only(self, mock_torch, mock_tp_rk, mock_pp_rk, mock_tp_sz):
        # Act
        res = load(path := "some_path", weights_only=True)
        # Assert
        mock_torch.load.assert_called_once_with(
            path + "/tp_rank_00_pp_rank_00/checkpoint.pt", map_location="cpu", weights_only=True
        )
        assert res == mock_torch.load.return_value

    @patch(f"{MODULE}.get_tensor_model_parallel_size", return_value=1)
    @patch(f"{MODULE}.get_pipeline_model_parallel_rank", return_value=0)
    @patch(f"{MODULE}.get_tensor_model_parallel_rank", return_value=0)
    @patch(f"{MODULE}.torch")
    def test_load_xser_weights_only(self, mock_torch, mock_tp_rk, mock_pp_rk, mock_tp_sz):
        # Act
        res = load(path := "some_path", load_xser=True, weights_only=True)
        # Assert
        mock_torch.load.assert_called_once_with(path + "/tp_rank_00_pp_rank_00", weights_only=True)
        assert res == mock_torch.load.return_value


if __name__ == "__main__":
    unittest.main()
