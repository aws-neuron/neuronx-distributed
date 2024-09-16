import unittest
from unittest.mock import MagicMock

from neuronx_distributed.utils.model_utils import (
    get_delay_tracing,
    check_delay_tracing
)
from neuronx_distributed.pipeline.model import NxDPPModel

class TestGetDelayTracing(unittest.TestCase):
    @unittest.skip("disabled delayed tracing")
    def test_nxdppmodel_with_delay_tracing(self):

        mock_model = MagicMock(spec=NxDPPModel)
        mock_model._delay_tracing = True

        result = get_delay_tracing(mock_model)
        self.assertTrue(result)

    @unittest.skip("disabled delayed tracing")
    def test_nxdppmodel_without_delay_tracing(self):

        mock_model = MagicMock(spec=NxDPPModel)
        mock_model._delay_tracing = False

        result = get_delay_tracing(mock_model)
        self.assertEqual(result, False)

    @unittest.skip("disabled delayed tracing")
    def test_dict_with_pipeline_config(self):
        arg = {
            "pipeline_config": {
                "_delay_tracing": True
            }
        }

        result = get_delay_tracing(arg)
        self.assertTrue(result)

    @unittest.skip("disabled delayed tracing")
    def test_dict_without_pipeline_config(self):
        arg = {
            "pipeline_config": {
                "other_config": True
            }
        }

        result = get_delay_tracing(arg)
        self.assertEqual(result, None)

    @unittest.skip("disabled delayed tracing")
    def test_dict_without_delay_tracing(self):
        arg = {
            "pipeline_config": {}
        }

        result = get_delay_tracing(arg)
        self.assertEqual(result, None)

    @unittest.skip("disabled delayed tracing")
    def test_non_nxdppmodel_and_non_dict(self):
        result = get_delay_tracing("some string")
        self.assertEqual(result, None)


class TestCheckDelayTracing(unittest.TestCase):
    @unittest.skip("disabled delayed tracing")
    def test_pipeline_config_with_use_model_wrapper_and_no_input_names(self):
        nxd_config = {
            "pipeline_config": {
                "use_model_wrapper": True
            }
        }

        result = check_delay_tracing(nxd_config)
        self.assertTrue(result)

    @unittest.skip("disabled delayed tracing")
    def test_pipeline_config_with_use_model_wrapper_and_input_names(self):
        nxd_config = {
            "pipeline_config": {
                "use_model_wrapper": True,
                "input_names": ["x"]
            }
        }

        result = check_delay_tracing(nxd_config)
        self.assertFalse(result)

    @unittest.skip("disabled delayed tracing")
    def test_pipeline_config_without_use_model_wrapper(self):
        nxd_config = {
            "pipeline_config": {
                "use_model_wrapper": False
            }
        }

        result = check_delay_tracing(nxd_config)
        self.assertFalse(result)

    @unittest.skip("disabled delayed tracing")
    def test_pipeline_config_missing_use_model_wrapper(self):
        nxd_config = {
            "pipeline_config": {}
        }

        result = check_delay_tracing(nxd_config)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
