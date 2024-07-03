"""Module for Quantization Linear Layers

This Module contains layers for Quantization.
Currently supported Layer are

ColumnParallelLinear => QuantizedColumnParallel
RowParallelLinear => QuantizedRowParallel

NOTE: With the growing code base, we need to figure out a way so that CPL and QCPL layers do not diverge in the forward pass.
This should be done in the way pytorch is doing: torch/nn/modules/linear.py where in the forward pass it calls the function
'torch.nn.functional.linear. This functional API is more scalable and can adapt to different inputs.
We would be creating neuronx_distributed.functional API for this purpose. NAPP-2202
"""

import enum
import warnings
from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Union

import torch
from torch.nn.parameter import Parameter

from neuronx_distributed.parallel_layers.layers import (
    _initialize_affine_weight_neuron,
    _initialize_parameter_cpu,
    linear_with_async_allreduce,
)
from neuronx_distributed.parallel_layers.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_size,
)
from neuronx_distributed.parallel_layers.utils import (
    divide,
    set_tensor_model_parallel_attributes,
)
from neuronx_distributed.quantization.dequantize import dequantize
from neuronx_distributed.utils.logger import get_logger

logger = get_logger()


### Create Enum to define the type of quantization possible
class MyEnumMeta(enum.EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        else:
            return True


class QuantizationType(Enum, metaclass=MyEnumMeta):
    SCALAR = "scalar"


class QuantizedDtype(Enum, metaclass=MyEnumMeta):
    INT8 = torch.int8


class BaseQuantizeParallelLinear(torch.nn.Module, metaclass=ABCMeta):
    def __init__(
        self,
        quantization_type: Union[QuantizationType, str] = "scalar",
        dequantized_dtype: torch.dtype = torch.bfloat16,
        quantized_dtype: torch.dtype = torch.int8,
        device: torch.device = None,
    ) -> None:
        """_summary_

        Args:
            quantization_type (Union[QuantizationType, str], optional): Quantization type. Defaults to "scalar".
            dequantized_dtype (torch.dtype, optional): Detype to dequantize the weight to. Defaults to torch.bfloat16.
            quantized_dtype (torch.dtype, optional): Dtype to qunatize the weight to. Defaults to torch.int8.
            device (torch.device, optional): Device to which initialize the Parameters. Defaults to None.
        """
        super().__init__()
        assert (
            quantization_type in QuantizationType
        ), f"{quantization_type} quantization is not supported currently. Specify from {[[e.value for e in QuantizationType]]}"
        assert (
            quantized_dtype in QuantizedDtype
        ), f"{quantized_dtype} quantization is not supported currently. Specify from {[e.value for e in QuantizedDtype]}"
        self.quantization_type = QuantizationType(quantization_type)
        self.dequantized_dtype = dequantized_dtype
        self.quantized_dtype = QuantizedDtype(quantized_dtype).value
        self.device = device
        self.register_parameter("scale", None)

    def _init_weight(self, weight: torch.Tensor):
        """Init the weight in Quantized Parallel layers with zeroes.

        We fill with zeroes just to put up some value in it. The actualy value would come from loading the checkpoints.

        Args:
            weight (torch.Tensor): Weight to initialize
        """
        torch.nn.init._no_grad_fill_(weight, 0.0)

    def _init_bias(self, bias: torch.Tensor):
        """Init the bias in Quantized Parallel layers with zeroes.

        Args:
            bias (torch.Tensor): bias to initialize
        """
        torch.nn.init._no_grad_fill_(bias, 0.0)

    @classmethod
    @abstractmethod
    def from_float(
        cls,
        mod,
        quantization_type: Union[QuantizationType, str] = QuantizationType.SCALAR,
        quantized_dtype: Union[QuantizedDtype, torch.device] = QuantizedDtype.INT8,
    ):
        """Create Quantized class from non quantized version

        Args:
            mod (BaseParallelLinear): non quantized linear layer
        """


class TensorParallelDim(enum.Enum):
    QUANTIZED_COLUMN_PARALLEL = 0
    QUANTIZED_ROW_PARALLEL = 1


class QuantizedParallelLinearLayerStateDictAdaptor(object):
    """
    A utility class that modifies the state dict to the form required by the Quantized Linear Layers
    """

    @staticmethod
    def get_weight_from_state_dict(prefix: str, state_dict: dict) -> torch.Tensor:
        """Get weight tesor from the state dict

        Args:
            prefix (str): layer prefix
            state_dict (dict): model state dict from the checkpoint

        Raises:
            RuntimeError: If 'weight' field is not found

        Returns:
            torch.Tensor: weight tensor
        """
        if (prefix + "weight") in state_dict:
            # No mofification required
            return state_dict[prefix + "weight"]
        elif (prefix + "_packed_params.dtype") in state_dict:
            return torch.int_repr(state_dict[prefix + "_packed_params._packed_params"][0])
        else:
            raise RuntimeError(f"Cannot find {(prefix + 'weight')} in the state_dict")

    @staticmethod
    def set_weight_to_state_dict(prefix: str, tensor: torch.Tensor, state_dict: dict) -> None:
        if (prefix + "weight") in state_dict:
            state_dict[prefix + "weight"] = tensor
        elif (prefix + "_packed_params.dtype") in state_dict:
            state_dict[prefix + "_packed_params._packed_params"][0] = tensor
        else:
            raise RuntimeError(f"Cannot find {(prefix + 'weight')} in the state_dict")

    @staticmethod
    def get_bias_from_state_dict(prefix: str, state_dict: dict) -> torch.Tensor:
        """Get bias from state dict

        Args:
            prefix (str): layer prefix
            state_dict (dict): model state dict from the checkpoint

        Raises:
            RuntimeError: if 'bias' field is not found

        Returns:
            torch.Tensor: bias tensor
        """
        if (prefix + "bias") in state_dict:
            return state_dict[prefix + "weight"]
        elif (prefix + "_packed_params.dtype") in state_dict:
            bias = state_dict[prefix + "_packed_params._packed_params"][1]
            if isinstance(bias, torch.nn.Parameter):
                bias = bias.data
            return bias
        else:
            warnings.warn(f"Cannot find {(prefix + 'bias')} in the state_dict")
            return None

    @staticmethod
    def set_bias_to_state_dict(prefix: str, tensor: torch.Tensor, state_dict: dict) -> None:
        raise NotImplementedError()

    @staticmethod
    def get_scale_from_state_dict(prefix: str, state_dict) -> torch.Tensor:
        """Get scale value from state dict

        Args:
            prefix (str): layer prefix
            state_dict (dict): model state dict from the checkpoint

        Raises:
            RuntimeError: if scale is not found

        Returns:
            torch.Tensor: scale tensor
        """
        if (prefix + "_packed_params.dtype") in state_dict and state_dict[prefix + "_packed_params._packed_params"][
            0
        ].dtype == torch.qint8:
            return torch.tensor([state_dict[prefix + "_packed_params._packed_params"][0].q_scale()])
        elif (prefix + "scale") in state_dict:
            scale: torch.Tensor = state_dict[prefix + "scale"]
            # If dict already contains the scale in the form of torch tensor of dimension 1
            if scale.shape == (1,):
                return scale
            elif scale.shape == ():
                return scale.unsqueeze(0)
            else:
                raise RuntimeError(f"Scale shape is not valid {(prefix + 'scale')}: {scale}")
        else:
            raise RuntimeError(f"Cannot find {(prefix + 'scale')} in state_dict")


class QuantizedColumnParallel(BaseQuantizeParallelLinear):
    """Quantized Linear layer with column parallelism.

    Notes: See documentation for ColumnParallel for the implementation details

    Args:
        input_size (int): first dimension of matrix A.
        output_size (int): second dimension of matrix A.
        bias (bool, optional): If true, add bias. Defaults to True.
        quantization_type (Union[QuantizationType, str], optional): Type of quantization to use. Defaults to "scalar".
        gather_output (bool, optional): If true, call all-gether on output and make Y avaiable
                    to all Neuron devices, otherwise, every Neuron device will have its output
                    which is Y_i = XA_i. Defaults to True.
        dtype (torch.dtype, optional):dtype of the weights, not in quantized format. Defaults to torch.float32.
        quantized_dtype (Union[QuantizedDtype, torch.dtype], optional): dtype of the weights, in quantized format. Defaults to QuantizedDtype.INT8.
        device (torch.device, optional): Device for parameter initialization. Defaults to None.
        stride (int, optional): stride. Defaults to 1.
        sequence_parallel_enabled (bool, optional): Defaults to False.
        keep_master_weight (bool, optional): Defaults to False.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        quantization_type: Union[QuantizationType, str] = "scalar",
        gather_output: bool = True,
        dtype: torch.dtype = torch.float32,
        quantized_dtype: Union[QuantizedDtype, torch.dtype] = QuantizedDtype.INT8,
        device: torch.device = None,
        stride: int = 1,
        sequence_parallel_enabled: bool = False,
        keep_master_weight: bool = False,
    ):
        super().__init__(
            quantization_type=quantization_type, dequantized_dtype=dtype, quantized_dtype=quantized_dtype, device=device
        )

        if self.quantization_type == QuantizationType.SCALAR:
            self.scale = Parameter(torch.tensor([1.0]), requires_grad=False)

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        world_size = get_tensor_model_parallel_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.stride = stride
        self.keep_master_weight = keep_master_weight
        self.sequence_parallel_enabled = sequence_parallel_enabled

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.

        ###### Weight setup #####
        self._setup_for_weight()
        ###### Bias setup #####
        self._setup_for_bias(bias=bias)
        ##### Parallelism setup #####
        self._setup_for_parallelism(world_size=world_size)

        ###### Quantization Scale setup #####
        setattr(self.scale, "get_tensor_from_state_dict", QuantizedColumnParallel.get_scale_from_state_dict)
        set_tensor_model_parallel_attributes(tensor=self.scale, is_parallel=False, dim=None, stride=None)

        self._forward_impl = linear_with_async_allreduce

    def _setup_for_weight(self):
        init_device = self.device

        weight = torch.empty(
            self.output_size_per_partition, self.input_size, dtype=self.quantized_dtype, device=init_device
        )
        self.weight = Parameter(weight, requires_grad=False)

        self.device = self.weight.device

        if self.device.type == "cpu":
            self.master_weight = _initialize_parameter_cpu(
                param=self.weight,
                partition_dim=TensorParallelDim.QUANTIZED_COLUMN_PARALLEL.value,
                init_method=self._init_weight,
                param_dtype=self.quantized_dtype,
                stride=self.stride,
                return_master_param=self.keep_master_weight,
            )
        elif self.device.type == "meta":
            set_tensor_model_parallel_attributes(
                tensor=self.weight,
                is_parallel=True,
                dim=TensorParallelDim.QUANTIZED_COLUMN_PARALLEL.value,
                stride=self.stride,
            )
        else:
            _initialize_affine_weight_neuron(
                weight=self.weight,
                init_method=self._init_weight,
                partition_dim=TensorParallelDim.QUANTIZED_COLUMN_PARALLEL.value,
                stride=self.stride,
            )

        setattr(self.weight, "get_tensor_from_state_dict", QuantizedColumnParallel.get_weight_from_state_dict)
        setattr(self.weight, "set_tensor_to_state_dict", QuantizedColumnParallel.set_weight_to_state_dict)

    def _setup_for_bias(self, bias: bool):
        if bias:
            self.bias_size = self.output_size if self.gather_output else self.output_size_per_partition
            if self.device is None or self.device.type == "cpu":
                self.bias = Parameter(torch.empty(self.bias_size, dtype=self.dequantized_dtype), requires_grad=False)
            else:
                self.bias = Parameter(
                    torch.empty(self.bias_size, device=self.device, dtype=self.dequantized_dtype), requires_grad=False
                )
            if self.bias.device != torch.device("meta"):
                self._init_bias(bias=self.bias)

            if not self.gather_output:
                set_tensor_model_parallel_attributes(self.bias, True, 0, stride=self.stride)

            setattr(self.bias, "get_tensor_from_state_dict", QuantizedColumnParallel.get_scale_from_state_dict)
            setattr(self.bias, "set_tensor_to_state_dict", QuantizedColumnParallel.set_bias_to_state_dict)
        else:
            self.register_parameter("bias", None)

    def _setup_for_parallelism(self, world_size: int):
        self.async_tensor_model_parallel_allreduce = not self.sequence_parallel_enabled and world_size > 1
        if self.sequence_parallel_enabled:
            if world_size <= 1:
                warnings.warn(f"`sequence_parallel_enabled` is set to `True`, but got world_size of {world_size}")

        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel_enabled:
            raise RuntimeError(
                "`async_tensor_model_parallel_allreduce` and `sequence_parallel_enabled` cannot be enabled at the same time."
            )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input)

        # Matrix multiply.
        weight_for_matmul = dequantize(self.weight, self.scale, input_parallel.dtype)
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=weight_for_matmul,
            bias=None,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            save_for_backward=False,
        )
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel_enabled
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output = (output + self.bias) if self.bias is not None else output
        return output

    @staticmethod
    def get_weight_from_state_dict(prefix: str, state_dict: dict) -> torch.Tensor:
        return QuantizedParallelLinearLayerStateDictAdaptor.get_weight_from_state_dict(
            prefix=prefix, state_dict=state_dict
        )

    @staticmethod
    def set_weight_to_state_dict(prefix: str, tensor: torch.Tensor, state_dict: dict) -> None:
        return QuantizedParallelLinearLayerStateDictAdaptor.set_weight_to_state_dict(
            prefix=prefix, tensor=tensor, state_dict=state_dict
        )

    @staticmethod
    def get_bias_from_state_dict(prefix: str, state_dict: dict) -> torch.Tensor:
        return QuantizedParallelLinearLayerStateDictAdaptor.get_bias_from_state_dict(
            prefix=prefix, state_dict=state_dict
        )

    @staticmethod
    def set_bias_to_state_dict(prefix: str, tensor: torch.Tensor, state_dict: dict) -> torch.Tensor:
        return QuantizedParallelLinearLayerStateDictAdaptor.set_bias_to_state_dict(
            prefix=prefix, tensor=tensor, state_dict=state_dict
        )

    @staticmethod
    def get_scale_from_state_dict(prefix: str, state_dict):
        return QuantizedParallelLinearLayerStateDictAdaptor.get_scale_from_state_dict(
            prefix=prefix, state_dict=state_dict
        )

    @classmethod
    def from_float(
        cls,
        mod,
        quantization_type: Union[QuantizationType, str] = QuantizationType.SCALAR,
        quantized_dtype: Union[QuantizedDtype, torch.device] = QuantizedDtype.INT8,
    ):
        """Create a QuantizedColumnParallel from a float module."""
        assert mod.__class__.__name__ == "ColumnParallelLinear", "ColumnParallelLinear expected"
        new_mod = QuantizedColumnParallel(
            input_size=mod.input_size,
            output_size=mod.output_size,
            quantization_type=quantization_type,
            bias=mod.bias is not None,
            quantized_dtype=quantized_dtype,
            gather_output=mod.gather_output,
            dtype=mod.dtype,
            device=mod.weight.device,
            stride=mod.stride,
            sequence_parallel_enabled=mod.sequence_parallel_enabled,
            keep_master_weight=mod.keep_master_weight,
        )
        return new_mod


class QuantizedRowParallel(BaseQuantizeParallelLinear):
    """Quantized Linear layer with row parallelism.

    Notes: See documentation for RowParallelLinear for the implementation details

    Args:
        input_size (int): input size
        output_size (int): output size
        bias (bool, optional): whether to use bias. Defaults to True.
        quantization_type (Union[QuantizationType, str], optional): type of quantization to use. Defaults to "scalar".
        input_is_parallel (bool, optional): Defaults to False.
        dtype (torch.dtype, optional):dtype of the weights, not in quantized format. Defaults to torch.float32.
        quantized_dtype (Union[QuantizedDtype, torch.dtype], optional): dtype of the weights, in quantized format. Defaults to QuantizedDtype.INT8.
        device (torch.device, optional): Device for parameter initialization. Defaults to None.
        stride (int, optional): stride. Defaults to 1.
        sequence_parallel_enabled (bool, optional): Defaults to False.
        keep_master_weight (bool, optional):Defaults to False.

    Raises:
        RuntimeError: When sequence parallel is enabled without input is parallel
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        quantization_type: Union[QuantizationType, str] = "scalar",
        input_is_parallel: bool = False,
        dtype: torch.dtype = torch.float32,
        quantized_dtype: Union[QuantizedDtype, torch.dtype] = QuantizedDtype.INT8,
        device: torch.device = None,
        stride: int = 1,
        sequence_parallel_enabled: bool = False,
        keep_master_weight: bool = False,
    ):
        super().__init__(
            quantization_type=quantization_type, dequantized_dtype=dtype, quantized_dtype=quantized_dtype, device=device
        )
        if self.quantization_type == QuantizationType.SCALAR:
            self.scale = Parameter(torch.tensor([1.0]), requires_grad=False)

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        world_size = get_tensor_model_parallel_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.sequence_parallel_enabled and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel_enabled`, `input_is_parallel` must be `True`")
        self.stride = stride
        self.keep_master_weight = keep_master_weight

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.

        ###### Weight setup #####
        self._setup_for_weight()
        ###### Bias setup #####
        self._setup_for_bias(bias=bias)

        ###### Quantization Scale setup #####
        setattr(self.scale, "get_tensor_from_state_dict", QuantizedRowParallel.get_scale_from_state_dict)
        set_tensor_model_parallel_attributes(tensor=self.scale, is_parallel=False, dim=None, stride=None)

        self._forward_impl = linear_with_async_allreduce

    def _setup_for_weight(self):
        init_device = self.device
        weight = torch.empty(
            self.output_size, self.input_size_per_partition, device=init_device, dtype=self.quantized_dtype
        )
        self.weight = Parameter(weight, requires_grad=False)
        self.device = self.weight.device

        if self.device.type == "cpu":
            self.master_weight = _initialize_parameter_cpu(
                param=self.weight,
                partition_dim=TensorParallelDim.QUANTIZED_ROW_PARALLEL.value,
                init_method=self._init_weight,
                param_dtype=self.quantized_dtype,
                stride=self.stride,
                return_master_param=self.keep_master_weight,
            )
        elif self.device.type == "meta":
            set_tensor_model_parallel_attributes(
                tensor=self.weight,
                is_parallel=True,
                dim=TensorParallelDim.QUANTIZED_ROW_PARALLEL.value,
                stride=self.stride,
            )
        else:
            _initialize_affine_weight_neuron(
                weight=self.weight,
                init_method=self._init_weight,
                partition_dim=TensorParallelDim.QUANTIZED_ROW_PARALLEL.value,
                stride=self.stride,
            )

        setattr(self.weight, "get_tensor_from_state_dict", QuantizedRowParallel.get_weight_from_state_dict)
        setattr(self.weight, "set_tensor_to_state_dict", QuantizedRowParallel.set_weight_to_state_dict)

    def _setup_for_bias(self, bias: bool):
        if bias:
            if self.device is None or self.device.type == "cpu":
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        dtype=self.dequantized_dtype,
                    ),
                    requires_grad=False,
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=self.device,
                        dtype=self.dequantized_dtype,
                    ),
                    requires_grad=False,
                )
            if self.bias.device != torch.device("meta"):
                self._init_bias(self.bias)
            setattr(self.bias, "sequence_parallel_enabled", self.sequence_parallel_enabled)
            setattr(self.bias, "get_tensor_from_state_dict", QuantizedRowParallel.get_scale_from_state_dict)
            setattr(self.bias, "set_tensor_to_state_dict", QuantizedRowParallel.set_bias_to_state_dict)
        else:
            self.register_parameter("bias", None)

    def forward(self, input_: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward of QuantizedRowParallel

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel_enabled
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        weight_for_matmul = dequantize(self.weight, self.scale, input_parallel.dtype)
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=weight_for_matmul,
            bias=None,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
            save_for_backward=False,
        )
        # All-reduce across all the partitions.
        if self.sequence_parallel_enabled:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        output = (output_ + self.bias) if self.bias is not None else output_
        return output

    @staticmethod
    def get_weight_from_state_dict(prefix: str, state_dict: dict) -> torch.Tensor:
        return QuantizedParallelLinearLayerStateDictAdaptor.get_weight_from_state_dict(
            prefix=prefix, state_dict=state_dict
        )

    @staticmethod
    def set_weight_to_state_dict(prefix: str, tensor: torch.Tensor, state_dict: dict) -> None:
        return QuantizedParallelLinearLayerStateDictAdaptor.set_weight_into_state_dict(
            prefix=prefix, tensor=tensor, state_dict=state_dict
        )

    @staticmethod
    def get_bias_from_state_dict(prefix: str, state_dict: dict) -> torch.Tensor:
        return QuantizedParallelLinearLayerStateDictAdaptor.get_bias_from_state_dict(
            prefix=prefix, state_dict=state_dict
        )

    @staticmethod
    def set_bias_to_state_dict(prefix: str, tensor: torch.Tensor, state_dict: dict) -> torch.Tensor:
        return QuantizedParallelLinearLayerStateDictAdaptor.set_bias_to_state_dict(
            prefix=prefix, tensor=tensor, state_dict=state_dict
        )

    @staticmethod
    def get_scale_from_state_dict(prefix: str, state_dict):
        return QuantizedParallelLinearLayerStateDictAdaptor.get_scale_from_state_dict(
            prefix=prefix, state_dict=state_dict
        )

    @classmethod
    def from_float(
        cls,
        mod,
        quantization_type: Union[QuantizationType, str] = QuantizationType.SCALAR,
        quantized_dtype: Union[QuantizedDtype, torch.device] = QuantizedDtype.INT8,
    ):
        """Create a QuantizedRowParallel from a float module

        Args:
            mod: float module
        """
        assert mod.__class__.__name__ == "RowParallelLinear", "RowParallelLinear expected"
        return QuantizedRowParallel(
            input_size=mod.input_size,
            output_size=mod.output_size,
            bias=mod.bias is not None,
            quantization_type=quantization_type,
            input_is_parallel=mod.input_is_parallel,
            dtype=mod.dtype,
            quantized_dtype=quantized_dtype,
            device=mod.weight.device,
            stride=mod.stride,
            sequence_parallel_enabled=mod.sequence_parallel_enabled,
            keep_master_weight=mod.keep_master_weight,
        )
