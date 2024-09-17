# Standard Library
import unittest
from unittest.mock import MagicMock, patch

import neuronx_distributed as nxd

# Third Party


class TestObject:
    pass


class TestNxDConfig(unittest.TestCase):
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.model.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=4)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("torch.distributed.get_rank")
    def test_neuronx_distributed_config0(self, rank_mock):
        nxd_config = nxd.neuronx_distributed_config(
            tensor_parallel_size=8,
            pipeline_parallel_size=4,
            pipeline_config=None,
            optimizer_config=None,
            activation_checkpoint_config=TestObject,
            pad_model=False,
            sequence_parallel=False,
            model_init_config=None,
        )

        assert nxd_config["optimizer_config"] == {
            "zero_one_enabled": False,
            "grad_clipping": True,
            "max_grad_norm": 1.0,
        }
        assert nxd_config["model_init_config"] == {
            "sequential_move_factor": 11,
            "meta_device_init": False,
            "param_init_fn": None,
        }
        assert nxd_config["pipeline_config"] is None
        assert nxd_config["activation_checkpoint_config"] == TestObject
        assert nxd_config["pad_model"] is False
        assert nxd_config["sequence_parallel"] is False

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.model.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=4)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("torch.distributed.get_rank")
    def test_neuronx_distributed_config1(self, rank_mock):
        nxd_config = nxd.neuronx_distributed_config(
            tensor_parallel_size=8,
            pipeline_parallel_size=4,
            pipeline_config=None,
            optimizer_config={"zero_one_enabled": True},
            activation_checkpoint_config=None,
            pad_model=False,
            sequence_parallel=False,
            model_init_config=None,
        )

        assert nxd_config["optimizer_config"] == {
            "zero_one_enabled": True,
            "grad_clipping": True,
            "max_grad_norm": 1.0,
        }

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.model.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=4)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("torch.distributed.get_rank")
    def test_neuronx_distributed_config2(self, rank_mock):
        model_init_config = {"meta_device_init": True, "param_init_fn": lambda x: None}
        nxd_config = nxd.neuronx_distributed_config(
            tensor_parallel_size=8,
            pipeline_parallel_size=4,
            pipeline_config=None,
            optimizer_config=None,
            activation_checkpoint_config=None,
            pad_model=False,
            sequence_parallel=False,
            model_init_config=model_init_config,
        )

        assert nxd_config["model_init_config"] == model_init_config

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.model.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=4)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("torch.distributed.get_rank")
    def test_neuronx_distributed_config_check_mixed_precision_setting(self, rank_mock):
        mixed_precision_config = {
            "use_master_weights": True,
            "use_fp32_grad_acc": True,
            "use_master_weights_in_ckpt": False,
        }
        nxd_config = nxd.neuronx_distributed_config(
            tensor_parallel_size=8,
            pipeline_parallel_size=4,
            pipeline_config=None,
            optimizer_config=None,
            activation_checkpoint_config=None,
            pad_model=False,
            sequence_parallel=False,
            mixed_precision_config=mixed_precision_config
        )

        assert nxd_config["mixed_precision_config"] == mixed_precision_config

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.model.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=4)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("torch.distributed.get_rank")
    def test_neuronx_distributed_config_check_default_mixed_precision_setting_with_optimizer_config_none(self, rank_mock):
        mixed_precision_config = {
            "use_master_weights": False,
            "use_fp32_grad_acc": False,
            "use_master_weights_in_ckpt": False,
        }
        nxd_config = nxd.neuronx_distributed_config(
            tensor_parallel_size=8,
            pipeline_parallel_size=4,
            pipeline_config=None,
            optimizer_config=None,
            activation_checkpoint_config=None,
            pad_model=False,
            sequence_parallel=False,
        )

        assert nxd_config["mixed_precision_config"] == mixed_precision_config

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.model.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=4)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("torch.distributed.get_rank")
    def test_neuronx_distributed_config_check_default_mixed_precision_setting_with_optimizer_config_has_zero(self, rank_mock):
        mixed_precision_config = {
            "use_master_weights": True,
            "use_fp32_grad_acc": True,
            "use_master_weights_in_ckpt": False,
        }
        nxd_config = nxd.neuronx_distributed_config(
            tensor_parallel_size=8,
            pipeline_parallel_size=4,
            pipeline_config=None,
            optimizer_config={"zero_one_enabled": True},
            activation_checkpoint_config=None,
            pad_model=False,
            sequence_parallel=False,
        )

        assert nxd_config["mixed_precision_config"] == mixed_precision_config

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.model.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=4)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("torch.distributed.get_rank")
    def test_neuronx_distributed_config_check_default_mixed_precision_setting_with_optimizer_config_has_no_zero(self, rank_mock):
        mixed_precision_config = {
            "use_master_weights": False,
            "use_fp32_grad_acc": False,
            "use_master_weights_in_ckpt": False,
        }
        nxd_config = nxd.neuronx_distributed_config(
            tensor_parallel_size=8,
            pipeline_parallel_size=4,
            pipeline_config=None,
            optimizer_config={"zero_one_enabled": False},
            activation_checkpoint_config=None,
            pad_model=False,
            sequence_parallel=False,
        )

        assert nxd_config["mixed_precision_config"] == mixed_precision_config


if __name__ == "__main__":
    unittest.main()
