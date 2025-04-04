import unittest
from unittest.mock import patch, MagicMock

import s3transfer
import boto3
import torch
from io import BytesIO
from boto3.resources.base import ServiceResource
from botocore.client import BaseClient
import traceback


import neuronx_distributed as nxd

try:
    import awscrt
    use_crt = True
except ImportError:
    use_crt = False

MODULE = "neuronx_distributed.trainer.checkpoint_storage"


class S3CheckpointStorageTest(unittest.TestCase):

    def setUp(self):
        # Reset persistent variables
        nxd.trainer.checkpoint_storage._s3_resource = None
        nxd.trainer.checkpoint_storage._s3_client = None
        nxd.trainer.checkpoint_storage._s3_transfer_manager = None

        # For CRT
        self.bucket_name = 'some_bucket'
        self.key_name = 'some_dir'
        self.filename = 'test_tensor.pt'
        self.dirname = f's3://{self.bucket_name}/{self.key_name}'

        crt_config = {
            'num_threads': 16,
            'part_size_mb': 1,
        }
        self.storage = nxd.trainer.checkpoint_storage.S3CheckpointStorage("s3://some_bucket/some_dir", crt_config)

    
    @patch(f'{MODULE}.S3CheckpointStorage._create_transfer_manager')
    @patch(f'{MODULE}.S3CheckpointStorage.get_transfer_manager')
    @patch(f'{MODULE}.use_crt', False)
    def test_save_object(self, mock_get_transfer_manager, mock_create_transfer_manager):
        # Setup
        mock_manager = MagicMock()
        mock_get_transfer_manager.return_value = mock_manager
        mock_future = MagicMock()
        mock_manager.upload.return_value = mock_future

        # Test object
        test_obj = torch.tensor([1, 2, 3])

        # Call save_object
        self.storage.save_object(test_obj, self.filename)

        # Assertions
        mock_manager.upload.assert_called_once()
        args, kwargs = mock_manager.upload.call_args
        self.assertEqual(kwargs['bucket'], self.bucket_name)
        self.assertEqual(kwargs['key'], f'{self.key_name}/{self.filename}')


    @patch(f'{MODULE}.S3CheckpointStorage.get_client')
    def test_load_object(self, mock_get_client):
        # Setup
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Test object
        test_obj = torch.tensor([1, 2, 3])

        # Mock S3 response
        mock_body = BytesIO()
        torch.save(test_obj, mock_body)
        mock_body.seek(0)
        mock_response = {'Body': mock_body}
        mock_client.get_object.return_value = mock_response

        # Call load_object
        loaded_obj = self.storage.load_object(self.filename)

        # Assertions
        mock_client.get_object.assert_called_once_with(
            Bucket="some_bucket",
            Key=f'some_dir/{self.filename}'
        )
        self.assertTrue(torch.equal(loaded_obj, test_obj))

    @unittest.skipIf(not use_crt, "Skipping CRT test as awscrt could not be imported")
    @patch(f'{MODULE}.S3CheckpointStorage._create_transfer_manager')
    @patch(f'{MODULE}.S3CheckpointStorage.get_transfer_manager')
    @patch(f'{MODULE}.use_crt', True)
    @patch(f'{MODULE}.awscrt.__version__', "0.19.19")
    def test_save_object_crt(self, mock_get_transfer_manager, mock_create_transfer_manager):
        # Define a custom side effect to print the call stack
        def custom_side_effect(*args, **kwargs):
            print("Call stack at method_name call:")
            traceback.print_stack()
            return None  # or any other return value expected from method_name
        
        # Setup
        mock_crt_manager = MagicMock()
        mock_get_transfer_manager.return_value = mock_crt_manager
        # mock_get_transfer_manager.side_effect = custom_side_effect
        # mock_create_transfer_manager.side_effect = custom_side_effect
        mock_future = MagicMock()
        mock_crt_manager.upload.return_value.result.return_value = mock_future

        # Test object
        test_obj = torch.tensor([1, 2, 3])

        # Call save_object
        self.storage.save_object(test_obj, self.filename)

        # Assertions
        mock_crt_manager.upload.assert_called_once()
        _, kwargs = mock_crt_manager.upload.call_args
        self.assertEqual(kwargs['bucket'], self.bucket_name)
        self.assertEqual(kwargs['key'], f'{self.key_name}/{self.filename}')

    
    def test_get_resource(self):
        resource = self.storage.get_resource()
        assert isinstance(resource, ServiceResource)

    def test_get_client(self):
        client = self.storage.get_client()
        assert isinstance(client, BaseClient)

    @unittest.skipIf(use_crt, "Skipping because TransferConfig.preferred_transfer_client is unsupported type if CRT installed")
    def test_classic_transfer_manager(self):
        transfer_config = boto3.s3.transfer.TransferConfig(preferred_transfer_client="classic")
        tm = self.storage.get_transfer_manager(transfer_config)
        assert isinstance(tm, s3transfer.manager.TransferManager)

    @unittest.skipIf(not use_crt, "Skipping CRT test as awscrt could not be imported")
    def test_crt_transfer_manager(self):

        transfer_config = boto3.s3.transfer.TransferConfig()
        tm = self.storage.get_transfer_manager(transfer_config)
        assert isinstance(tm, s3transfer.crt.CRTTransferManager)


if __name__ == "__main__":
    unittest.main()
