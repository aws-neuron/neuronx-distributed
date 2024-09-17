# Standard Library
import unittest
from unittest.mock import MagicMock, patch

# Third Party
import torch
from neuronx_distributed.parallel_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
import neuronx_distributed as nxd
from neuronx_distributed.modules.lora import LoraConfig, LoraModel, get_lora_model
from neuronx_distributed.pipeline.model import NxDPPModel


class NxDModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rpl = ColumnParallelLinear(32, 32)
        self.cpl = RowParallelLinear(32, 32)
        self.linear = torch.nn.Linear(32, 32)


def get_nxd_model():
    return NxDModule()


class NxDPPModule(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.rpl = RowParallelLinear(10, 10)
        self.cpl = ColumnParallelLinear(10, 10)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(num_layers)])


def get_pp_model(num_layers=4):
    return NxDPPModule(num_layers)


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
        self.conv2d = torch.nn.Conv2d(32, 32, 4)
        self.embedding = torch.nn.Embedding(32, 32)


def get_nxd_lora_config():
    return LoraConfig(
        enable_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["rpl", "cpl"],
    )

def get_lora_config():
    return LoraConfig(
        enable_lora=True,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["linear", "conv2d", "embedding"],
    )

class TestModelWrapper(unittest.TestCase):
    def test_model_wrapper_single_device(self):
        model = Module()
        lora_config = get_lora_config()
        lora_model = LoraModel(model, lora_config)
        assert isinstance(lora_model, LoraModel)
        assert lora_model.lora_config == lora_config
        assert type(lora_model.get_base_model()) is type(model)

    def test_unified_model_wrapper_single_device(self):
        model = Module()
        lora_config = get_lora_config()
        lora_model = get_lora_model(model, lora_config)
        assert isinstance(lora_model, LoraModel)

        assert lora_model.lora_config == lora_config
        assert type(lora_model.get_base_model()) is type(model)

    @patch("neuronx_distributed.parallel_layers.layers._initialize_parameter_cpu", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_affine_weight_neuron", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("neuronx_distributed.utils.model_utils.move_model_to_device", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.move_model_to_device", MagicMock(return_value=None))
    def test_model_wrapper(self):
        nxd_config = nxd.neuronx_distributed_config(
            tensor_parallel_size=8,
            lora_config=get_lora_config(),
        )
        model = nxd.initialize_parallel_model(nxd_config, get_nxd_model)

        assert isinstance(model, nxd.trainer.model.NxDModel)
        assert model.nxd_config == nxd_config
        assert isinstance(model.module, LoraModel)
        model_str = str(model)
        assert "LoraModel" in model_str

    @patch("neuronx_distributed.parallel_layers.layers._initialize_parameter_cpu", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_affine_weight_neuron", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=8))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.parallel_layers.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("neuronx_distributed.utils.model_utils.move_model_to_device", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.move_model_to_device", MagicMock(return_value=None))
    def test_unified_model_wrapper(self):
        lora_config = get_nxd_lora_config()
        model = NxDModule()
        model = get_lora_model(model, lora_config)

        assert isinstance(model, LoraModel)
        model_str = str(model)
        assert "LoraModel" in model_str

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_parameter_cpu", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_affine_weight_neuron", MagicMock(return_value=None))
    @patch("neuronx_distributed.pipeline.model.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=2)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_tensor_model_parallel_size", MagicMock(return_value=2)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_tensor_model_parallel_rank", MagicMock(return_value=0)
    )
    @patch("neuronx_distributed.pipeline.partition.get_pipeline_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.partition.get_pipeline_model_parallel_size", MagicMock(return_value=2))
    @patch("neuronx_distributed.pipeline.model.NxDPPModel._create_pg_with_ranks", MagicMock(return_value=None))
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("neuronx_distributed.utils.model_utils.move_model_to_device", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.move_model_to_device", MagicMock(return_value=None))
    def test_pp_model_wrapper(self):
        pipeline_cuts = [
            "layers.1",
        ]
        nxd_config = nxd.neuronx_distributed_config(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            pipeline_config={
                "transformer_layer_cls": torch.nn.Linear,
                "tracer_cls": "torch",
                "pipeline_cuts": pipeline_cuts,
                "param_init_fn": None,
                "use_zero1_optimizer": True,
                "use_optimizer_wrapper": True,
                "input_names": ["input_ids", "attention_mask", "labels"],
            },
            optimizer_config={
                "zero_one_enabled": True,
                "grad_clipping": True,
                "max_grad_norm": 1.0,
            },
            sequence_parallel=True,
            activation_checkpoint_config="full",
            lora_config=get_nxd_lora_config(),
        )
        model = nxd.initialize_parallel_model(nxd_config, get_pp_model)

        assert isinstance(model, nxd.trainer.model.NxDModel)
        assert model.nxd_config == nxd_config
        assert model.pp_enabled
        assert isinstance(model.module, LoraModel)
        assert isinstance(model.module.get_base_model(), nxd.pipeline.NxDPPModel)
        model_str = str(model)
        assert "NxDModel" in model_str
        assert "NxDPPModel" in model_str
        assert "LoraModel" in model_str

    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_size", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers.get_tensor_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_parameter_cpu", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.layers._initialize_affine_weight_neuron", MagicMock(return_value=None))
    @patch("neuronx_distributed.pipeline.model.parallel_state.initialize_model_parallel", MagicMock(return_value=None))
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.model_parallel_is_initialized", MagicMock(return_value=True)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_size", MagicMock(return_value=2)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_pipeline_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_tensor_model_parallel_size", MagicMock(return_value=2)
    )
    @patch(
        "neuronx_distributed.pipeline.model.parallel_state.get_tensor_model_parallel_rank", MagicMock(return_value=1)
    )
    @patch("neuronx_distributed.pipeline.partition.get_pipeline_model_parallel_rank", MagicMock(return_value=1))
    @patch("neuronx_distributed.pipeline.partition.get_pipeline_model_parallel_size", MagicMock(return_value=2))
    @patch("neuronx_distributed.pipeline.model.NxDPPModel._create_pg_with_ranks", MagicMock(return_value=None))
    @patch("neuronx_distributed.utils.model_utils.get_local_world_size", MagicMock(return_value=32))
    @patch("neuronx_distributed.utils.model_utils.move_model_to_device", MagicMock(return_value=None))
    @patch("neuronx_distributed.parallel_layers.move_model_to_device", MagicMock(return_value=None))
    def test_unified_pp_model_wrapper(self):
        model = NxDPPModel(module=NxDPPModule(4), transformer_layer_cls=torch.nn.Linear, tracer_cls="torch")
        model = get_lora_model(model, get_nxd_lora_config())
        assert isinstance(model, LoraModel)
        assert isinstance(model.module, NxDPPModel)
        model_str = str(model)
        assert "NxDPPModel" in model_str
        assert "LoraModel" in model_str


if __name__ == "__main__":
    unittest.main()
