import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_xla.core.xla_model as xm

from neuronx_distributed.modules.moe.moe_parallel_layers import (
    ExpertFusedColumnParallelLinear,
    ExpertFusedRowParallelLinear,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.quantization.quantization_config import (
    get_default_custom_qconfig_dict,
    get_default_per_channel_custom_qconfig_dict,
)
from neuronx_distributed.quantization.quantization_layers import (
    BaseQuantizeParallelLinear,
    QuantizedColumnParallel,
    QuantizedExpertFusedColumnParallel,
    QuantizedExpertFusedRowParallel,
    QuantizedParallelLinearLayerStateDictAdaptor,
    QuantizedRowParallel,
)


class TestQuantizedParallelLinearLayerStateDictAdaptor(unittest.TestCase):
    def test_get_weight_from_state_dict(self):
        # Weight field present
        weight = MagicMock()
        state_dict = {"lay1.weight": weight}
        assert weight == QuantizedParallelLinearLayerStateDictAdaptor.get_weight_from_state_dict("lay1.", state_dict)

        # Torch quint present
        weight = torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0, 2.0]), 1.0, 0, torch.qint8)
        state_dict = {
            "lay1._packed_params.dtype": torch.qint8,
            "lay1._packed_params._packed_params": (weight, MagicMock()),
        }
        torch.testing.assert_close(
            torch.tensor([-1, 0, 1, 2], dtype=torch.int8),
            QuantizedParallelLinearLayerStateDictAdaptor.get_weight_from_state_dict("lay1.", state_dict),
        )

        # Runtime error check
        state_dict = {"lay1.something": MagicMock()}
        with self.assertRaisesRegex(RuntimeError, "Cannot find lay1.weight in the state_dict"):
            QuantizedParallelLinearLayerStateDictAdaptor.get_weight_from_state_dict("lay1.", state_dict)

    @pytest.mark.skip("Currently not testing Bias")
    def test_get_bias_from_state_dict(self):
        pass

    def test_get_scale_from_state_dict(self):
        # Test when qint present
        weight = torch.quantize_per_tensor(
            torch.tensor([-1.0, 0.0, 1.0, 2.0]), scale=0.0033, zero_point=0, dtype=torch.qint8
        )
        state_dict = {
            "lay1._packed_params.dtype": torch.qint8,
            "lay1._packed_params._packed_params": (weight, MagicMock()),
        }
        torch.testing.assert_close(
            torch.tensor([0.0033]),
            QuantizedParallelLinearLayerStateDictAdaptor.get_scale_from_state_dict("lay1.", state_dict),
        )

        weight = torch.quantize_per_channel(
            torch.randn(4, 4), torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.tensor([0, 1, 2, 3]), 0, torch.qint8
        )
        state_dict = {
            "lay1._packed_params.dtype": torch.qint8,
            "lay1._packed_params._packed_params": (weight, MagicMock()),
        }

        assert torch.allclose(
            torch.Tensor([[0.1], [0.2], [0.3], [0.4]]),
            QuantizedParallelLinearLayerStateDictAdaptor.get_scale_from_state_dict("lay1.", state_dict),
        )


class TestBaseQuantizeParallelLinear(unittest.TestCase):
    @patch.multiple(BaseQuantizeParallelLinear, __abstractmethods__=set())
    def test_init(self):
        with self.assertRaises(AssertionError) as context:
            BaseQuantizeParallelLinear(quantization_type="something")

        self.assertTrue(
            "something quantization is not supported currently. Specify from [['per_tensor_symmetric', 'per_channel_symmetric']]"
            in str(context.exception)
        )

        with self.assertRaises(AssertionError) as context:
            BaseQuantizeParallelLinear(quantization_type="per_tensor_symmetric", quantized_dtype=torch.float16)

        self.assertTrue("torch.float16 quantization is not supported currently. Specify from [['torch.int8']]")

        test_class = BaseQuantizeParallelLinear(quantization_type="per_tensor_symmetric", quantized_dtype=torch.int8)
        assert test_class.scale is None

    @patch.multiple(BaseQuantizeParallelLinear, __abstractmethods__=set())
    def test_init_weight(self):
        test_class = BaseQuantizeParallelLinear(quantization_type="per_tensor_symmetric", quantized_dtype=torch.int8)
        weight = torch.empty((5, 5), dtype=torch.int8)
        test_class._init_weight(weight=weight)
        torch.testing.assert_close(weight, torch.zeros(5, 5, dtype=torch.int8))

    @patch.multiple(BaseQuantizeParallelLinear, __abstractmethods__=set())
    def test_init_bias(self):
        test_class = BaseQuantizeParallelLinear(quantization_type="per_tensor_symmetric", quantized_dtype=torch.int8)
        bias = torch.empty((5,), dtype=torch.bfloat16)
        test_class._init_bias(bias=bias)
        torch.testing.assert_close(bias, torch.zeros(5, dtype=torch.bfloat16))

    @patch.multiple(BaseQuantizeParallelLinear, __abstractmethods__=set())
    def test_setup_for_scale(self):
        # for per_tensor_symmetric
        test_class = BaseQuantizeParallelLinear(quantization_type="per_tensor_symmetric", quantized_dtype=torch.int8)
        test_class._setup_for_scale(
            weight_shape=MagicMock(), quantization_type=test_class.quantization_type, weight_partition_dim=MagicMock()
        )
        assert hasattr(test_class.scale, "get_tensor_from_state_dict")
        assert test_class.scale.tensor_model_parallel is False
        assert torch.allclose(test_class.scale, torch.Tensor([1.0]))

        del test_class

        # for per_channel_symmetric, per channel same as partition dim
        test_class = BaseQuantizeParallelLinear(quantization_type="per_channel_symmetric", quantized_dtype=torch.int8)
        test_class.weight = MagicMock(device=torch.device("cpu"))
        test_class._setup_for_scale(
            weight_shape=(8, 10),
            quantization_type=test_class.quantization_type,
            weight_partition_dim=0,
            per_channel_axis=0,
        )
        assert hasattr(test_class.scale, "get_tensor_from_state_dict")
        assert torch.allclose(test_class.scale, torch.ones((8, 1)))
        assert test_class.scale.tensor_model_parallel is True
        assert test_class.scale.partition_dim == 0

        # for per_channel_symmetric, per channel not same as partition dim
        test_class = BaseQuantizeParallelLinear(quantization_type="per_channel_symmetric", quantized_dtype=torch.int8)
        test_class.weight = MagicMock(device=torch.device("cpu"))
        test_class._setup_for_scale(
            weight_shape=(8, 10),
            quantization_type=test_class.quantization_type,
            weight_partition_dim=0,
            per_channel_axis=1,
        )
        assert hasattr(test_class.scale, "get_tensor_from_state_dict")
        assert torch.allclose(test_class.scale, torch.ones((1, 10)))
        assert test_class.scale.tensor_model_parallel is False


class TestQuantizedColumnParallel(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_world_size = parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
        self.initial_rank = parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK

        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0

    def tearDown(self) -> None:
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = self.initial_world_size
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = self.initial_rank
        return

    @patch("neuronx_distributed.quantization.quantization_layers.get_tensor_model_parallel_size", return_value=1)
    @patch("neuronx_distributed.quantization.quantization_layers.BaseQuantizeParallelLinear._setup_for_weight")
    @patch("neuronx_distributed.quantization.quantization_layers.QuantizedColumnParallel._setup_for_bias")
    @patch("neuronx_distributed.quantization.quantization_layers.BaseQuantizeParallelLinear._setup_for_scale")
    @patch("neuronx_distributed.quantization.quantization_layers.QuantizedColumnParallel._setup_for_parallelism")
    def test_init(
        self,
        mock_setup_for_parallelism,
        mock_setup_for_scale,
        mock_setup_for_bias,
        mock_setup_for_weight,
        mock_get_tensor_model_parallel_size,
    ):
        _ = QuantizedColumnParallel(
            input_size=5,
            output_size=5,
            quantization_type="per_tensor_symmetric",
            quantized_dtype=torch.int8,
            dtype=torch.bfloat16,
        )
        mock_get_tensor_model_parallel_size.assert_called_once()
        mock_setup_for_weight.assert_called_once()
        mock_setup_for_bias.assert_called_once()
        mock_setup_for_scale.assert_called_once()
        mock_setup_for_parallelism.assert_called_once()

    @patch("neuronx_distributed.quantization.quantization_layers._initialize_affine_weight_neuron", return_value=1)
    @patch("neuronx_distributed.quantization.quantization_layers._initialize_parameter_cpu")
    @patch("neuronx_distributed.quantization.quantization_layers.get_tensor_model_parallel_size", return_value=2)
    def test_setup_for_weight(
        self,
        mock_get_tensor_model_parallel_size,
        mock_initialize_parameter_cpu,
        mock_initialize_affine_weight_neuron,
    ):
        input_size = 4
        output_size = 6

        layer = QuantizedColumnParallel(input_size, output_size, device=torch.device("cpu"), bias=False)

        # Assert _initialize_parameter_cpu called with right inputs
        mock_initialize_parameter_cpu.assert_called_once_with(
            param=layer.weight,
            partition_dim=0,
            init_method=layer._init_weight,
            param_dtype=torch.int8,
            stride=layer.stride,
            return_master_param=layer.keep_master_weight,
        )

        # Assert weight properties
        self.assertEqual(layer.weight.shape, (3, 4))  # Adjusted for partition_dim=0
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertFalse(layer.weight.requires_grad)

        del layer

        layer = QuantizedColumnParallel(input_size, output_size, device=xm.xla_device(), bias=False)

        # Assert _initialize_affine_weight_neuron called with right inputs
        mock_initialize_affine_weight_neuron.assert_called_once_with(
            weight=layer.weight, init_method=layer._init_weight, partition_dim=0, stride=layer.stride
        )

        # Assert weight properties
        self.assertEqual(layer.weight.shape, (3, 4))  # Adjusted for partition_dim=0
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertFalse(layer.weight.requires_grad)

    @pytest.mark.skip(reason="Not testing bias currently as its not being used")
    def test_setup_for_bias(self):
        pass

    def _setup_for_parallelism(self, device):
        pass

    def test_from_float(self):
        cpl = ColumnParallelLinear(
            input_size=4, output_size=6, device=torch.device("cpu"), bias=True, dtype=torch.float32
        )
        q_config = get_default_custom_qconfig_dict()
        qcpl = QuantizedColumnParallel.from_float(cpl, q_config=q_config)
        assert qcpl.weight.dtype == torch.int8
        assert qcpl.scale.shape == (1,)
        assert qcpl.bias is not None

        # Channel axis = 0
        q_config = get_default_per_channel_custom_qconfig_dict()
        qcpl = QuantizedColumnParallel.from_float(cpl, q_config=q_config)
        assert qcpl.weight.dtype == torch.int8
        assert qcpl.scale.shape == (6, 1)
        assert qcpl.bias is not None

        # Channel axis = 1
        q_config["quantization_per_channel_axis"] = 1
        qcpl = QuantizedColumnParallel.from_float(cpl, q_config=q_config)
        assert qcpl.weight.dtype == torch.int8
        assert qcpl.scale.shape == (1, 4)
        assert qcpl.bias is not None


class TestQuantizedRowParallel(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_world_size = parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
        self.initial_rank = parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK

        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0

    def tearDown(self) -> None:
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = self.initial_world_size
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = self.initial_rank
        return

    @patch("neuronx_distributed.quantization.quantization_layers.BaseQuantizeParallelLinear._setup_for_scale")
    @patch("neuronx_distributed.quantization.quantization_layers.QuantizedRowParallel._setup_for_bias")
    @patch("neuronx_distributed.quantization.quantization_layers.QuantizedRowParallel._setup_for_weight")
    def test_init(
        self,
        mock_setup_for_weight,
        mock_setup_for_bias,
        mock_setup_for_scale,
    ):
        _ = QuantizedRowParallel(input_size=6, output_size=4, dtype=torch.bfloat16)
        mock_setup_for_weight.assert_called_once()
        mock_setup_for_bias.assert_called_once()
        mock_setup_for_scale.assert_called_once()

    @patch("neuronx_distributed.quantization.quantization_layers._initialize_affine_weight_neuron", return_value=1)
    @patch("neuronx_distributed.quantization.quantization_layers._initialize_parameter_cpu")
    def test_setup_for_weight(self, mock_initialize_parameter_cpu, mock_initialize_affine_weight_neuron):
        input_size = 6
        output_size = 4
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 2
        layer = QuantizedRowParallel(input_size, output_size, device=torch.device("cpu"), bias=False)

        # Assert _initialize_parameter_cpu called with right inputs
        mock_initialize_parameter_cpu.assert_called_once_with(
            param=layer.weight,
            partition_dim=1,
            init_method=layer._init_weight,
            param_dtype=torch.int8,
            stride=layer.stride,
            return_master_param=layer.keep_master_weight,
        )

        # Assert weight properties
        self.assertEqual(layer.weight.shape, (4, 3))
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertFalse(layer.weight.requires_grad)
        assert hasattr(layer.weight, "get_tensor_from_state_dict")

        del layer

        layer = QuantizedRowParallel(input_size, output_size, device=xm.xla_device(), bias=False)

        # Assert _initialize_affine_weight_neuron called with right inputs
        mock_initialize_affine_weight_neuron.assert_called_once_with(
            weight=layer.weight, init_method=layer._init_weight, partition_dim=1, stride=layer.stride
        )

        # Assert weight properties
        self.assertEqual(layer.weight.shape, (4, 3))  # Adjusted for partition_dim=1
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertFalse(layer.weight.requires_grad)

    @pytest.mark.skip(reason="Not testing bias currently as its not being used")
    def test_setup_for_bias(self):
        pass

    def test_from_float(self):
        rpl = RowParallelLinear(input_size=6, output_size=4, device=torch.device("cpu"), bias=True, dtype=torch.float32)
        q_config = get_default_custom_qconfig_dict()
        qrpl = QuantizedRowParallel.from_float(rpl, q_config=q_config)
        assert qrpl.weight.dtype == torch.int8
        assert qrpl.scale.shape == (1,)
        assert qrpl.bias is not None

        # Channel axis = 0
        q_config = get_default_per_channel_custom_qconfig_dict()
        qrpl = QuantizedRowParallel.from_float(rpl, q_config=q_config)
        assert qrpl.weight.dtype == torch.int8
        assert qrpl.bias is not None
        assert qrpl.scale.shape == (4, 1)

        # Channel axis = 1
        q_config["quantization_per_channel_axis"] = 1
        qrpl = QuantizedRowParallel.from_float(rpl, q_config=q_config)
        assert qrpl.weight.dtype == torch.int8
        assert qrpl.bias is not None
        assert qrpl.scale.shape == (1, 6)


class TestQuantizedExpertFusedColumnParallel(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_world_size = parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
        self.initial_rank = parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK
        self.initial_expert_world_size = parallel_state._MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE


        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0
        parallel_state._MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = 1

    def tearDown(self) -> None:
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = self.initial_world_size
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = self.initial_rank
        parallel_state._MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = self.initial_expert_world_size
        return

    @patch("neuronx_distributed.quantization.quantization_layers.get_tensor_model_parallel_size", return_value=1)
    @patch("neuronx_distributed.quantization.quantization_layers.QuantizedExpertFusedColumnParallel._setup_for_weight")
    @patch("neuronx_distributed.quantization.quantization_layers.QuantizedExpertFusedColumnParallel._setup_for_bias")
    @patch("neuronx_distributed.quantization.quantization_layers.BaseQuantizeParallelLinear._setup_for_scale")
    @patch(
        "neuronx_distributed.quantization.quantization_layers.QuantizedExpertFusedColumnParallel._setup_for_parallelism"
    )
    def test_init(
        self,
        mock_setup_for_parallelism,
        mock_setup_for_bias,
        mock_setup_for_weight,
        mock_setup_for_scale,
        mock_get_tensor_model_parallel_size,
    ):
        _ = QuantizedExpertFusedColumnParallel(
            num_experts=2,
            input_size=3,
            output_size=4,
            quantization_type="per_tensor_symmetric",
            quantized_dtype=torch.int8,
            dtype=torch.bfloat16,
        )
        mock_get_tensor_model_parallel_size.assert_called_once()
        mock_setup_for_weight.assert_called_once()
        mock_setup_for_bias.assert_called_once()
        mock_setup_for_scale.assert_called_once()
        mock_setup_for_parallelism.assert_called_once()

    @patch("neuronx_distributed.quantization.quantization_layers._initialize_affine_weight_neuron", return_value=1)
    @patch("neuronx_distributed.quantization.quantization_layers._initialize_parameter_cpu")
    @patch("neuronx_distributed.quantization.quantization_layers.get_tensor_model_parallel_size", return_value=2)
    def test_setup_for_weight(
        self,
        mock_get_tensor_model_parallel_size,
        mock_initialize_parameter_cpu,
        mock_initialize_affine_weight_neuron,
    ):
        num_experts = 2
        input_size = 4
        output_size = 6

        layer = QuantizedExpertFusedColumnParallel(num_experts, input_size, output_size, device=torch.device("cpu"))

        # Assert _initialize_parameter_cpu called with right inputs
        mock_initialize_parameter_cpu.assert_called_once_with(
            param=layer.weight,
            partition_dim=2,
            init_method=layer._init_weight,
            param_dtype=torch.int8,
            stride=layer.stride,
            return_master_param=layer.keep_master_weight,
        )

        # Assert weight properties
        self.assertEqual(layer.weight.shape, (2, 4, 3))  # Adjusted for partition_dim=2
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertFalse(layer.weight.requires_grad)

        del layer

        layer = QuantizedExpertFusedColumnParallel(num_experts, input_size, output_size, device=xm.xla_device())

        # Assert _initialize_affine_weight_neuron called with right inputs
        mock_initialize_affine_weight_neuron.assert_called_once_with(
            weight=layer.weight, init_method=layer._init_weight, partition_dim=2, stride=layer.stride
        )

        # Assert weight properties
        self.assertEqual(layer.weight.shape, (2, 4, 3))  # Adjusted for partition_dim=2
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertFalse(layer.weight.requires_grad)

    def test_from_float(self):
        cpl = ExpertFusedColumnParallelLinear(
            num_experts=2, input_size=4, output_size=6, device=torch.device("cpu"), dtype=torch.float32
        )
        q_config = get_default_custom_qconfig_dict()
        qcpl = QuantizedExpertFusedColumnParallel.from_float(cpl, q_config=q_config)
        assert qcpl.weight.dtype == torch.int8
        assert qcpl.bias is None
        qcpl.scale.shape == (1,)

        q_config = get_default_per_channel_custom_qconfig_dict()
        q_config["quantization_per_channel_axis"] = 1  # First dimension is reserved for experts
        qcpl = QuantizedExpertFusedColumnParallel.from_float(cpl, q_config=q_config)
        assert qcpl.scale.shape == (1, 4, 1)
        assert qcpl.scale.tensor_model_parallel is False

        q_config = get_default_per_channel_custom_qconfig_dict()
        q_config["quantization_per_channel_axis"] = 2  # First dimension is reserved for experts
        qcpl = QuantizedExpertFusedColumnParallel.from_float(cpl, q_config=q_config)
        qcpl.scale.shape == (1, 1, 6)
        assert qcpl.scale.tensor_model_parallel is True and qcpl.scale.partition_dim == 2


class TestQuantizedExpertFusedRowParallel(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_world_size = parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
        self.initial_rank = parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK
        self.initial_expert_world_size = parallel_state._MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE

        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 1
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = 0
        parallel_state._MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = 1

    def tearDown(self) -> None:
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = self.initial_world_size
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_RANK = self.initial_rank
        parallel_state._MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = self.initial_expert_world_size
        return

    @patch("neuronx_distributed.quantization.quantization_layers.BaseQuantizeParallelLinear._setup_for_scale")
    @patch("neuronx_distributed.quantization.quantization_layers.QuantizedExpertFusedRowParallel._setup_for_bias")
    @patch("neuronx_distributed.quantization.quantization_layers.QuantizedExpertFusedRowParallel._setup_for_weight")
    def test_init(
        self,
        mock_setup_for_weight,
        mock_setup_for_bias,
        mock_setup_for_scale,
    ):
        _ = QuantizedExpertFusedRowParallel(num_experts=2, input_size=6, output_size=4, dtype=torch.bfloat16)
        mock_setup_for_weight.assert_called_once()
        mock_setup_for_bias.assert_called_once()
        mock_setup_for_scale.assert_called_once()

    @patch("neuronx_distributed.quantization.quantization_layers._initialize_affine_weight_neuron", return_value=1)
    @patch("neuronx_distributed.quantization.quantization_layers._initialize_parameter_cpu")
    def test_setup_for_weight(self, mock_initialize_parameter_cpu, mock_initialize_affine_weight_neuron):
        num_experts = 2
        input_size = 6
        output_size = 4
        parallel_state._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = 2
        layer = QuantizedExpertFusedRowParallel(num_experts, input_size, output_size, device=torch.device("cpu"))

        # Assert _initialize_parameter_cpu called with right inputs
        mock_initialize_parameter_cpu.assert_called_once_with(
            param=layer.weight,
            partition_dim=1,
            init_method=layer._init_weight,
            param_dtype=torch.int8,
            stride=layer.stride,
            return_master_param=layer.keep_master_weight,
        )

        # Assert weight properties
        self.assertEqual(layer.weight.shape, (2, 3, 4))
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertFalse(layer.weight.requires_grad)
        assert hasattr(layer.weight, "get_tensor_from_state_dict")

        del layer

        layer = QuantizedExpertFusedRowParallel(num_experts, input_size, output_size, device=xm.xla_device())

        # Assert _initialize_affine_weight_neuron called with right inputs
        mock_initialize_affine_weight_neuron.assert_called_once_with(
            weight=layer.weight, init_method=layer._init_weight, partition_dim=1, stride=layer.stride
        )

        # Assert weight properties
        self.assertEqual(layer.weight.shape, (2, 3, 4))  # Adjusted for partition_dim=1
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertFalse(layer.weight.requires_grad)

    def test_from_float(self):
        rpl = ExpertFusedRowParallelLinear(
            num_experts=2,
            input_size=6,
            output_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        q_config = get_default_custom_qconfig_dict()
        qrpl = QuantizedExpertFusedRowParallel.from_float(rpl, q_config=q_config)
        assert qrpl.weight.dtype == torch.int8
        assert qrpl.bias is None
        assert qrpl.scale.shape == (1,)

        q_config = get_default_per_channel_custom_qconfig_dict()
        q_config["quantization_per_channel_axis"] = 1  # First dimension is reserved for experts
        qrpl = QuantizedExpertFusedRowParallel.from_float(rpl, q_config=q_config)
        qrpl.scale.shape == (1, 6, 1)
        assert qrpl.scale.tensor_model_parallel is True and qrpl.scale.partition_dim == 1

        q_config = get_default_per_channel_custom_qconfig_dict()
        q_config["quantization_per_channel_axis"] = 2  # First dimension is reserved for experts
        qrpl = QuantizedExpertFusedRowParallel.from_float(rpl, q_config=q_config)
        qrpl.scale.shape == (1, 1, 4)
        assert qrpl.scale.tensor_model_parallel is False


if __name__ == "__main__":
    unittest.main(verbosity=3, failfast=False)
