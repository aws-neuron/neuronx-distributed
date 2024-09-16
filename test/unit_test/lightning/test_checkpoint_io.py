import unittest
from unittest.mock import patch

from neuronx_distributed.lightning.checkpoint_io import NeuronCheckpointIO

MODULE = "neuronx_distributed.lightning.checkpoint_io"


class NeuronCheckpointIOTest(unittest.TestCase):
    @patch(f"{MODULE}.load")
    def test_load(self, mock_load):
        # Arrange
        io = NeuronCheckpointIO()
        # Act
        io.load_checkpoint(path := "some_path", master_dp_only := True)
        # Assert
        mock_load.assert_called_once_with(
            chkpt_path=path, load_xser=True, master_dp_only=master_dp_only, weights_only=False
        )

    @patch(f"{MODULE}.load")
    def test_load_weights_only(self, mock_load):
        # Arrange
        io = NeuronCheckpointIO(weights_only=True)
        # Act
        io.load_checkpoint(path := "some_path", master_dp_only := True)
        # Assert
        mock_load.assert_called_once_with(
            chkpt_path=path, load_xser=True, master_dp_only=master_dp_only, weights_only=True
        )


if __name__ == "__main__":
    unittest.main()
