import unittest
from unittest.mock import patch

import neuronx_distributed as nxd

MODULE = "neuronx_distributed.trainer.checkpoint_storage"


class S3CheckpointStorageTest(unittest.TestCase):
    @patch(f"{MODULE}.boto3")
    def test_uses_default_session(self, mock_boto3):
        # Arrange
        storage = nxd.trainer.checkpoint_storage.S3CheckpointStorage("s3://some_bucket/some_dir")
        # Act
        resource = storage.get_resource()
        # Assert
        self.assertEqual(resource, mock_boto3._get_default_session.return_value.resource.return_value)


if __name__ == "__main__":
    unittest.main()
