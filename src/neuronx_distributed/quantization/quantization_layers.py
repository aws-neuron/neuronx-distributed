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

import warnings
from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Any, Sequence, Dict, cast, Type

import torch
from torch.nn.parameter import Parameter
from torch.distributed import ProcessGroup

from neuronx_distributed.modules.moe.moe_parallel_layers import (
    ExpertFusedLinearWithAsyncCommunication,
    ExpertFusedLinear,
)
from neuronx_distributed.parallel_layers.layers import (
    LinearWithAsyncCommunication,
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
    get_expert_model_parallel_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_size,
)
from neuronx_distributed.parallel_layers.utils import (
    divide,
    set_tensor_model_parallel_attributes,
    get_padding_length,
    is_torch_version_greater_than_2
)
from neuronx_distributed.quantization.dequantize import direct_cast_dequantize, scale_dequantize
from neuronx_distributed.quantization.quantization_config import (
    _DEFAULT_CUSTOM_QCONFIG_DICT,
    BASE_QCONFIG_DICT_TYPE,
    PER_CHANNEL_QCONFIG_DICT_TYPE,
    QuantizationType,
    QuantizedDtype,
    ActivationQuantizationType
)
from neuronx_distributed.quantization.quantization_utils import extract_q_scale, quantize_fp8_per_channel
from neuronx_distributed.utils.logger import get_logger

logger = get_logger()


class BaseQuantizeParallelLinear(torch.nn.Module, metaclass=ABCMeta):
    autograd_func_class: Type[torch.autograd.Function] = LinearWithAsyncCommunication

    def __init__(
        self,
        quantization_type: Union[QuantizationType, str] = "per_tensor_symmetric",
        dequantized_dtype: torch.dtype = torch.bfloat16,
        quantized_dtype: Union[QuantizedDtype, torch.dtype] = torch.int8,
        device: Optional[torch.device] = None,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
    ) -> None:
        """_summary_

        Args:
            quantization_type (Union[QuantizationType, torch.qscheme], optional): Quantization type. Defaults to per_tensor_symmetric.
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
        self.tensor_parallel_group = tensor_model_parallel_group if \
            tensor_model_parallel_group is not None else cast(ProcessGroup, get_tensor_model_parallel_group())

        self.register_parameter("scale", None)

        self.keep_master_weight: Optional[bool] = None
        self.weight_shape: Optional[Sequence[int]] = None
        self.weight_partition_dim: Optional[int] = None
        self.stride: Optional[int] = None
        self.bias_shape: Optional[Sequence[int]] = None

    def _init_weight(self, weight: torch.Tensor) -> None:
        """Init the weight in Quantized Parallel layers with zeroes.

        We fill with zeroes just to put up some value in it. The actualy value would come from loading the checkpoints.

        Args:
            weight (torch.Tensor): Weight to initialize
        """
        if not is_torch_version_greater_than_2() or weight.dtype not in [QuantizedDtype.F8E4M3, QuantizedDtype.F8E5M2]:
            # For FP8 dtypes , RuntimeError: "fill_cpu" not implemented for 'Float8_e4m3fn'
            torch.nn.init._no_grad_fill_(weight, 0.0)

    def _init_bias(self, bias: torch.Tensor) -> None:
        """Init the bias in Quantized Parallel layers with zeroes.

        Args:
            bias (torch.Tensor): bias to initialize
        """
        torch.nn.init._no_grad_fill_(bias, 0.0)

    def _setup_for_weight(self) -> None:
        init_device = self.device
        assert self.weight_shape is not None, "self.weight_shape is None and it should be a proper value."
        weight = torch.empty(*self.weight_shape, dtype=self.quantized_dtype, device=init_device)
        self.weight = Parameter(weight, requires_grad=False)
        self.device = self.weight.device

        assert self.stride is not None
        assert self.weight_partition_dim is not None
        if self.device.type == "cpu":
            assert self.keep_master_weight is not None
            self.master_weight = _initialize_parameter_cpu(
                param=self.weight,
                partition_dim=self.weight_partition_dim,
                num_partitions=self.tensor_parallel_group.size(),
                init_method=self._init_weight,
                param_dtype=self.quantized_dtype,
                stride=self.stride,
                return_master_param=self.keep_master_weight,
            )
        elif self.device.type == "meta":
            set_tensor_model_parallel_attributes(
                tensor=self.weight,
                is_parallel=True,
                dim=self.weight_partition_dim,
                stride=self.stride,
                num_partitions=self.tensor_parallel_group.size()
            )
        else:
            _initialize_affine_weight_neuron(
                weight=self.weight,
                init_method=self._init_weight,
                partition_dim=self.weight_partition_dim,
                num_partitions=self.tensor_parallel_group.size(),
                stride=self.stride,
            )

        setattr(self.weight, "get_tensor_from_state_dict", self.get_weight_from_state_dict)
        setattr(self.weight, "set_tensor_to_state_dict", self.set_weight_to_state_dict)

    def _base_setup_for_bias(self, bias: bool) -> None:
        if bias:
            assert self.bias_shape is not None, "self.bias_shape is None and it should be a proper value."
            if self.device is None or self.device.type == "cpu":
                self.bias = Parameter(torch.empty(*self.bias_shape, dtype=self.dequantized_dtype), requires_grad=False)
            else:
                self.bias = Parameter(
                    torch.empty(*self.bias_shape, device=self.device, dtype=self.dequantized_dtype), requires_grad=False
                )
            if self.bias.device != torch.device("meta"):
                self._init_bias(self.bias)

            setattr(self.bias, "get_tensor_from_state_dict", self.get_bias_from_state_dict)
            setattr(self.bias, "set_tensor_to_state_dict", self.set_bias_to_state_dict)
        else:
            self.register_parameter("bias", None)

    def _setup_for_scale(
        self,
        weight_shape: Sequence[int],
        quantization_type: QuantizationType,
        weight_partition_dim: Optional[int] = None,
        per_channel_axis: Optional[int] = None,
    ):
        """Setup required for scale

        Args:
            weight_shape (tuple): Weight shape
            quantization_type (QuantizationType): Quantization Type
            weight_partition_dim (Optional[int], optional): Weight partition dimension. Defaults to None.
                This is required if per channel quantization is used.
            per_channel_axis (Optional[int], optional): Scale dimension. Defaults to None.
                This is required if per channel quantization is used.

        Raises:
            ValueError: If quantization_type is not within QuantizationType.PER_TENSOR_SYMMETRIC and QuantizationType.PER_CHANNEL_SYMMETRIC

        NOTE: Currently we are setting the attribute for tensor model parallel even for per tensor symmetric case.
        This is to make it uniform. After KVCache quantization is implemented(as K and V have different quantization schemes) and if the uniformity is not required, remove it.
        """
        if quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            self.scale = Parameter(torch.tensor([1.0], device=self.weight.device), requires_grad=False)
            set_tensor_model_parallel_attributes(
                tensor=self.scale, is_parallel=False, dim=0, stride=1, num_partitions=1,
            )
        elif quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            assert (
                per_channel_axis is not None
            ), "per_channel_axis cannot be None for per_channel_symmetric quantization"
            scale_shape = [1] * len(weight_shape)
            scale_shape[per_channel_axis] = weight_shape[per_channel_axis]

            self.scale = Parameter(torch.ones(scale_shape, device=self.weight.device), requires_grad=False)

            # Only when the weight partition dim is the per_channel_axis, we need to partition the scale
            if weight_partition_dim == per_channel_axis:
                # we need to partition scale as well
                set_tensor_model_parallel_attributes(
                    tensor=self.scale,
                    is_parallel=True,
                    dim=weight_partition_dim, # type: ignore
                    stride=self.stride, # type: ignore
                    num_partitions=self.tensor_parallel_group.size(),
                )
            else:
                set_tensor_model_parallel_attributes(
                    tensor=self.scale, is_parallel=False, dim=0, stride=1, num_partitions=1,
                )
        else:
            raise ValueError(f"scale for quantization_type: {quantization_type} not supported")

        setattr(self.scale, "get_tensor_from_state_dict", BaseQuantizeParallelLinear.get_scale_from_state_dict)

    @staticmethod
    def get_weight_from_state_dict(prefix: str, state_dict: Dict[str, Any]) -> torch.Tensor:
        return QuantizedParallelLinearLayerStateDictAdaptor.get_weight_from_state_dict(
            prefix=prefix, state_dict=state_dict
        )

    @staticmethod
    def set_weight_to_state_dict(prefix: str, tensor: torch.Tensor, state_dict: Dict[str, Any]) -> None:
        QuantizedParallelLinearLayerStateDictAdaptor.set_weight_to_state_dict(
            prefix=prefix, tensor=tensor, state_dict=state_dict
        )

    @staticmethod
    def get_bias_from_state_dict(prefix: str, state_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
        return QuantizedParallelLinearLayerStateDictAdaptor.get_bias_from_state_dict(
            prefix=prefix, state_dict=state_dict
        )

    @staticmethod
    def set_bias_to_state_dict(prefix: str, tensor: torch.Tensor, state_dict: Dict[str, Any]) -> None:
        QuantizedParallelLinearLayerStateDictAdaptor.set_bias_to_state_dict(
            prefix=prefix, tensor=tensor, state_dict=state_dict
        )

    @staticmethod
    def get_scale_from_state_dict(prefix: str, state_dict: Dict[str, Any]) -> torch.Tensor:
        return QuantizedParallelLinearLayerStateDictAdaptor.get_scale_from_state_dict(
            prefix=prefix, state_dict=state_dict
        )

    def _apply_post_quantization_hook(mod, new_mod):
        if hasattr(mod, "post_create_quantized_module_hook"):
            post_process_quantized_layer = mod.post_create_quantized_module_hook
            post_process_quantized_layer(new_mod)
        return new_mod

    @classmethod
    @abstractmethod
    def from_float(cls, mod, q_config: BASE_QCONFIG_DICT_TYPE = _DEFAULT_CUSTOM_QCONFIG_DICT):
        """Create Quantized class from non quantized version

        Args:
            mod (BaseParallelLinear): non quantized linear layer
        """


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
    def set_weight_to_state_dict(prefix: str, tensor: torch.Tensor, state_dict: Dict[str, Any]) -> None:
        if (prefix + "weight") in state_dict:
            state_dict[prefix + "weight"] = tensor
        elif (prefix + "_packed_params.dtype") in state_dict:
            state_dict[prefix + "_packed_params._packed_params"][0] = tensor
        else:
            raise RuntimeError(f"Cannot find {(prefix + 'weight')} in the state_dict")

    @staticmethod
    def get_bias_from_state_dict(prefix: str, state_dict: Dict[str, Any]) -> Optional[torch.Tensor]:
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
            return state_dict[prefix + "bias"]
        elif (prefix + "_packed_params.dtype") in state_dict:
            bias = state_dict[prefix + "_packed_params._packed_params"][1]
            if isinstance(bias, torch.nn.Parameter):
                bias = bias.data
            return bias
        else:
            warnings.warn(f"Cannot find {(prefix + 'bias')} in the state_dict")
            return None

    @staticmethod
    def set_bias_to_state_dict(prefix: str, tensor: torch.Tensor, state_dict: Dict[str, Any]) -> None:
        raise NotImplementedError()

    @staticmethod
    def get_scale_from_state_dict(prefix: str, state_dict: Dict[str, Any]) -> torch.Tensor:
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
            return extract_q_scale(state_dict[prefix + "_packed_params._packed_params"][0])
        elif (prefix + "scale") in state_dict:
            scale: torch.Tensor = state_dict[prefix + "scale"]
            return scale
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
        quantization_type: Union[QuantizationType, str] = "per_tensor_symmetric",
        gather_output: bool = True,
        dtype: torch.dtype = torch.float32,
        quantized_dtype: Union[QuantizedDtype, torch.dtype] = QuantizedDtype.INT8,
        device: Optional[torch.device] = None,
        stride: int = 1,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        keep_master_weight: bool = False,
        quantization_per_channel_axis: Optional[int] = None,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        pad: bool = False,
        activation_quantization_type: Optional[Union[ActivationQuantizationType, str]] = None,
        clamp_bound: float = float('inf')
    ):
        super().__init__(
            quantization_type=quantization_type,
            dequantized_dtype=dtype,
            quantized_dtype=quantized_dtype,
            device=device,
            tensor_model_parallel_group=tensor_model_parallel_group,
        )

        # Keep input parameters
        self.activation_quantization_type = ActivationQuantizationType(activation_quantization_type)
        self.clamp_bound = clamp_bound
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        world_size = self.tensor_parallel_group.size()
        self.pad = pad
        if self.pad:
            self.pad_size = get_padding_length(self.output_size, world_size)
            self.output_size = self.output_size + self.pad_size

        self.output_size_per_partition = divide(self.output_size, world_size)
        self.stride: int = stride
        self.keep_master_weight = keep_master_weight

        if sequence_parallel_enabled:
            if world_size <= 1:
                warnings.warn(f"`sequence_parallel_enabled` is set to `True`, but got world_size of {world_size}")

            if sequence_dimension is None:
                warnings.warn(
                    "`sequence_parallel_enabled` is set to `True`, but got `sequence_dimension` as `None`. Defaulting `sequence_dimension` to 0."
                )
                sequence_dimension = 0

        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.sequence_dimension = sequence_dimension

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.

        ###### Weight config and bias config setup #####
        self._setup_for_weight_and_bias_config(bias=bias)
        ###### Weight setup #####
        self._setup_for_weight()
        ###### Bias setup #####
        self._setup_for_bias(bias=bias)
        ###### Scale setup #####
        if quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            if quantization_per_channel_axis is None:
                quantization_per_channel_axis = 0  # Default to 0
            if self.__class__.__name__ == "QuantizedColumnParallel":
                # For post-matmul output dequantization, we need the per_channel_axis = output_dim
                assert (
                    quantization_per_channel_axis == 0
                ), "Only per_channel_axis=0 supported for post-matmul output dequantization"

        assert self.weight_shape is not None, "self.weight_shape is None and it should be a proper value."
        self._setup_for_scale(
            weight_shape=self.weight_shape,
            quantization_type=self.quantization_type,
            weight_partition_dim=self.weight_partition_dim,
            per_channel_axis=quantization_per_channel_axis,
        )
        ##### Parallelism setup #####
        self._setup_for_parallelism(world_size=world_size)

        self._forward_impl = linear_with_async_allreduce

    def _setup_for_weight_and_bias_config(self, bias: bool):
        self.weight_shape = (
            self.output_size_per_partition,
            self.input_size,
        )
        self.weight_partition_dim = 0

        if bias:
            self.bias_size = self.output_size if self.gather_output else self.output_size_per_partition
            self.bias_shape = (self.bias_size,)
        else:
            self.bias_shape = None

    def _setup_for_bias(self, bias: bool):
        self._base_setup_for_bias(bias=bias)
        if bias:
            if not self.gather_output:
                set_tensor_model_parallel_attributes(
                    self.bias, True, 0, stride=self.stride, num_partitions=self.tensor_parallel_group.size(),
                )

    def _setup_for_parallelism(self, world_size: int):
        self.async_tensor_model_parallel_allreduce = not self.sequence_parallel_enabled and world_size > 1
        if self.sequence_parallel_enabled:
            if world_size <= 1:
                warnings.warn(f"`sequence_parallel_enabled` is set to `True`, "
                              f"but got world_size of {world_size}")

        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel_enabled:
            raise RuntimeError(
                "`async_tensor_model_parallel_allreduce` and `sequence_parallel_enabled` cannot "
                "be enabled at the same time."
            )

    def forward(self, input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [batch, sequence, hidden]

        Returns:
            - output
        """

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input
        else:
            input_parallel = copy_to_tensor_model_parallel_region(
                input, process_group=self.tensor_parallel_group,
            )
        
        ## TODO: add a flow for static quantization once zhankuil@ is done
        if self.activation_quantization_type == ActivationQuantizationType.DYNAMIC:
            # Matrix multiply.
            original_dtype = input_parallel.dtype
            quantized_input, input_scale = quantize_fp8_per_channel(input_parallel, dtype=torch.float8_e4m3fn, channel_axis=1, clamp_bound=self.clamp_bound)
            output_parallel = self._forward_impl(
                input=quantized_input,
                weight=self.weight,
                bias=None,
                async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                autograd_func_class=self.autograd_func_class,
                save_for_backward=False,
                process_group=self.tensor_parallel_group,
            )
            ## dequantize twice, because we quantize twice, once for input, once for weights
            output_parallel = scale_dequantize(tensor=scale_dequantize(tensor=output_parallel, scale=input_scale, upcast_dtype=original_dtype), scale=self.scale.t(), upcast_dtype=original_dtype)
        else:
            # Matrix multiply.
            weight_for_matmul = direct_cast_dequantize(tensor=self.weight, upcast_dtype=input_parallel.dtype)
            output_parallel = self._forward_impl(
                input=input_parallel,
                weight=weight_for_matmul,
                bias=None,
                async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
                sequence_parallel_enabled=self.sequence_parallel_enabled,
                sequence_dimension=self.sequence_dimension,
                autograd_func_class=self.autograd_func_class,
                save_for_backward=False,
                process_group=self.tensor_parallel_group,
            )
            output_parallel = scale_dequantize(
                tensor=output_parallel, scale=self.scale.T, upcast_dtype=output_parallel.dtype
            )

        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel_enabled
            output = gather_from_tensor_model_parallel_region(
                output_parallel, process_group=self.tensor_parallel_group,
            )
        else:
            output = output_parallel
        output = (output + self.bias) if self.bias is not None else output
        return output

    @classmethod
    def from_float(
        cls, mod, q_config: Union[BASE_QCONFIG_DICT_TYPE, PER_CHANNEL_QCONFIG_DICT_TYPE] = _DEFAULT_CUSTOM_QCONFIG_DICT):
        """Create a QuantizedColumnParallel from a float module."""
        assert mod.__class__.__name__ == "ColumnParallelLinear", "ColumnParallelLinear expected"
        new_mod = QuantizedColumnParallel(
            input_size=mod.input_size,
            output_size=mod.output_size if not mod.pad else mod.output_size - mod.pad_size,
            quantization_type=q_config["quantization_type"],
            bias=mod.bias is not None,
            quantized_dtype=q_config["quantized_dtype"],
            gather_output=mod.gather_output,
            dtype=mod.dtype,
            device=mod.weight.device,
            stride=mod.stride,
            sequence_parallel_enabled=mod.sequence_parallel_enabled,
            sequence_dimension=mod.sequence_dimension,
            keep_master_weight=mod.keep_master_weight,
            quantization_per_channel_axis=cast(int, q_config.get("quantization_per_channel_axis")),
            tensor_model_parallel_group=mod.tensor_parallel_group,
            pad=mod.pad,
            activation_quantization_type=q_config["activation_quantization_type"],
            clamp_bound=q_config["clamp_bound"],
        )
        
        return cls._apply_post_quantization_hook(mod, new_mod)

    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        if not self.pad or self.pad_size == 0:
            return
        if self.output_size != model_state_dict[prefix].shape[0] + self.pad_size:
            size = model_state_dict[prefix].shape[0]
            raise RuntimeError(f"State dict {prefix} is of an unexpected size {size} expected {size - self.pad_size}")

        model_state_dict[prefix] = torch.nn.functional.pad(model_state_dict[prefix], (0, 0, 0, self.pad_size))
        assert prefix.endswith(".weight")
        scale_prefix = prefix.replace(".weight", ".scale")
        model_state_dict[scale_prefix] = torch.nn.functional.pad(model_state_dict[scale_prefix], (0, 0, 0, self.pad_size))


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
        quantization_type: Union[QuantizationType, str] = "per_tensor_symmetric",
        input_is_parallel: bool = False,
        dtype: torch.dtype = torch.float32,
        quantized_dtype: Union[QuantizedDtype, torch.dtype] = QuantizedDtype.INT8,
        device: Optional[torch.device] = None,
        stride: int = 1,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        keep_master_weight: bool = False,
        quantization_per_channel_axis: Optional[int] = None,
        reduce_output: bool = True,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        pad: bool = False,
        activation_quantization_type: Optional[Union[ActivationQuantizationType, str]] = None,
        clamp_bound = float('inf')
    ):
        super().__init__(
            quantization_type=quantization_type,
            dequantized_dtype=dtype,
            quantized_dtype=quantized_dtype,
            device=device,
            tensor_model_parallel_group=tensor_model_parallel_group,
        )

        # Keep input parameters
        self.activation_quantization_type = ActivationQuantizationType(activation_quantization_type)
        self.clamp_bound = clamp_bound
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        world_size = self.tensor_parallel_group.size()
        self.pad = pad
        if self.pad:
            self.pad_size = get_padding_length(self.input_size, world_size)
            self.input_size = self.input_size + self.pad_size

        self.input_size_per_partition = divide(self.input_size, world_size)

        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.sequence_parallel_enabled and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel_enabled`, `input_is_parallel` must be `True`")

        if self.sequence_parallel_enabled and sequence_dimension is None:
            warnings.warn(
                "`sequence_parallel_enabled` is set to `True`, but got `sequence_dimension` as `None`. Defaulting `sequence_dimension` to 0."
            )
            sequence_dimension = 0

        self.sequence_dimension: int = sequence_dimension # type: ignore
        
        self.stride = stride
        self.keep_master_weight = keep_master_weight
        self.reduce_output = reduce_output

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.

        ###### Weight config and bias config setup #####
        self._setup_for_weight_and_bias_config(bias=bias)
        ###### Weight setup #####
        self._setup_for_weight()
        ###### Bias setup #####
        self._setup_for_bias(bias=bias)

        ###### Quantization Scale setup #####
        if quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            if quantization_per_channel_axis is None:
                quantization_per_channel_axis = 0  # Default to 0
            if self.__class__.__name__ == "QuantizedRowParallel":
                # For post-matmul output dequantization, we need the per_channel_axis = output_dim
                assert (
                    quantization_per_channel_axis == 0
                ), "Only per_channel_axis=0 supported for post-matmul output dequantization"

        assert self.weight_shape, "self.weight_shape is None and it should be a proper value."
        self._setup_for_scale(
            weight_shape=self.weight_shape,
            quantization_type=self.quantization_type,
            weight_partition_dim=self.weight_partition_dim,
            per_channel_axis=quantization_per_channel_axis,
        )

        self._forward_impl = linear_with_async_allreduce

    def _setup_for_weight_and_bias_config(self, bias: bool):
        self.weight_shape = (
            self.output_size,
            self.input_size_per_partition,
        )
        self.weight_partition_dim = 1

        if bias:
            self.bias_size = self.output_size
            self.bias_shape = (self.bias_size,)
        else:
            self.bias_shape = None

    def _setup_for_bias(self, bias: bool):
        self._base_setup_for_bias(bias=bias)
        if bias:
            setattr(self.bias, "sequence_parallel_enabled", self.sequence_parallel_enabled)

    def forward(self, input_: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
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
            input_parallel = scatter_to_tensor_model_parallel_region(
                input_, process_group=self.tensor_parallel_group,
            )
        # Matrix multiply.
        if self.activation_quantization_type == ActivationQuantizationType.DYNAMIC:
            original_dtype = input_parallel.dtype
            #TODO: combine with zhankuil's implementation of static quantization for input
            quantized_input, input_scale = quantize_fp8_per_channel(input_parallel, dtype=torch.float8_e4m3fn, channel_axis=1)
            
            
            output_ = self._forward_impl(
                input=quantized_input,
                weight=self.weight,
                bias=None,
                async_grad_allreduce=False,
                sequence_parallel_enabled=False,
                sequence_dimension=self.sequence_dimension,
                autograd_func_class=self.autograd_func_class,
                save_for_backward=False,
                process_group=self.tensor_parallel_group,
            )
            ## dequantize twice, because we quantize twice, once for input, once for weights
            output_ = scale_dequantize(tensor=scale_dequantize(tensor=output_, scale=input_scale, upcast_dtype=original_dtype), scale=self.scale.t(), upcast_dtype=original_dtype)
        else:
            # Matrix multiply.
            weight_for_matmul = direct_cast_dequantize(tensor=self.weight, upcast_dtype=input_parallel.dtype)
            output_ = self._forward_impl(
                input=input_parallel,
                weight=weight_for_matmul,
                bias=None,
                async_grad_allreduce=False,
                sequence_parallel_enabled=False,
                sequence_dimension=self.sequence_dimension,
                autograd_func_class=self.autograd_func_class,
                save_for_backward=False,
                process_group=self.tensor_parallel_group,
            )
            output_ = scale_dequantize(
                tensor=output_, scale=self.scale.T, upcast_dtype=output_.dtype,
            )

        

        if self.reduce_output:
            # All-reduce across all the partitions.
            if self.sequence_parallel_enabled:
                output_ = reduce_scatter_to_sequence_parallel_region(
                    output_, self.sequence_dimension, process_group=self.tensor_parallel_group,
                )
            else:
                output_ = reduce_from_tensor_model_parallel_region(
                    output_, process_group=self.tensor_parallel_group,
                )
        output = (output_ + self.bias) if self.bias is not None else output_
        return output

    @classmethod
    def from_float(
        cls,
        mod,
        q_config: Union[BASE_QCONFIG_DICT_TYPE, PER_CHANNEL_QCONFIG_DICT_TYPE] = _DEFAULT_CUSTOM_QCONFIG_DICT,
    ):
        """Create a QuantizedRowParallel from a float module

        Args:
            mod: float module
        """
        assert mod.__class__.__name__ == "RowParallelLinear", "RowParallelLinear expected"
        new_mod = QuantizedRowParallel(
            input_size=mod.input_size if not mod.pad else mod.input_size - mod.pad_size,
            output_size=mod.output_size,
            bias=mod.bias is not None,
            quantization_type=q_config["quantization_type"],
            input_is_parallel=mod.input_is_parallel,
            dtype=mod.dtype,
            quantized_dtype=q_config["quantized_dtype"],
            device=mod.weight.device,
            stride=mod.stride,
            sequence_parallel_enabled=mod.sequence_parallel_enabled,
            sequence_dimension=mod.sequence_dimension,
            keep_master_weight=mod.keep_master_weight,
            quantization_per_channel_axis=cast(int, q_config.get("quantization_per_channel_axis")),
            reduce_output=mod.reduce_output,
            tensor_model_parallel_group=mod.tensor_parallel_group,
            pad=mod.pad,
            activation_quantization_type=q_config["activation_quantization_type"],
            clamp_bound=q_config["clamp_bound"],
        )
        
        return cls._apply_post_quantization_hook(mod, new_mod)

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> None:
        if not self.pad or self.pad_size == 0:
            return
        if self.input_size != model_state_dict[prefix].shape[1] + self.pad_size:
            size = model_state_dict[prefix].shape[1]
            raise RuntimeError(f"State dict {prefix} is of an unexpected size {size} expected {size - self.pad_size}")
        model_state_dict[prefix] = torch.nn.functional.pad(model_state_dict[prefix], (0, self.pad_size))

class QuantizedExpertFusedColumnParallel(QuantizedColumnParallel, ExpertFusedLinear):
    """
    Quantized version of the ExpertFusedColumnParallelLinear class
    """

    autograd_func_class = ExpertFusedLinearWithAsyncCommunication

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        quantization_type: Union[QuantizationType, str] = "per_tensor_symmetric",
        dtype: torch.dtype = torch.float32,
        quantized_dtype: Union[QuantizedDtype, torch.dtype] = QuantizedDtype.INT8,
        device: Optional[torch.device] = None,
        stride: int = 1,
        keep_master_weight: bool = False,
        quantization_per_channel_axis: Optional[int] = None,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
    ):
        self.num_experts = num_experts
        self._n_local_experts = divide(num_experts, get_expert_model_parallel_size())

        if quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            if quantization_per_channel_axis is None:
                quantization_per_channel_axis = 2  # Default to 2
            # For post-matmul output dequantization, we need the per_channel_axis = output_dim
            assert (
                quantization_per_channel_axis == 2
            ), "Only per_channel_axis=2 supported for post-matmul output dequantization"

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            quantization_type=quantization_type,
            gather_output=False,
            dtype=dtype,
            quantized_dtype=quantized_dtype,
            device=device,
            stride=stride,
            sequence_parallel_enabled=False,
            keep_master_weight=keep_master_weight,
            quantization_per_channel_axis=quantization_per_channel_axis,
            tensor_model_parallel_group=tensor_model_parallel_group,
        )

    def _setup_for_weight_and_bias_config(self, bias: bool):
        """
        Same as the ExpertFusedColumnParallelLinear.set_weight_and_bias_config()

        TODO: modularize both quantization layers and moe layers
        """
        # Define 3D weight tensor, one linear layer per expert
        self.weight_shape = (self._n_local_experts, self.input_size, self.output_size_per_partition)
        # Column parallel partitioning for each expert
        self.weight_partition_dim = 2
        self.bias_shape = None

    def forward(self, input: torch.Tensor, expert_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Same as the forward of ExpertFusedColumnParallelLinear, except with weight dequantization,
        and save_for_backward=False."""

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input
        else:
            input_parallel = copy_to_tensor_model_parallel_region(
                input, process_group=self.tensor_parallel_group,
            )

        # Matrix multiply.
        weight = self.weight[expert_indices, :, :] if expert_indices is not None else self.weight
        weight_for_matmul = direct_cast_dequantize(tensor=weight, upcast_dtype=input_parallel.dtype)
        output = self._forward_impl(
            input=input_parallel,
            weight=weight_for_matmul,
            bias=None,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            autograd_func_class=self.autograd_func_class,
            save_for_backward=False,
            process_group=self.tensor_parallel_group,
        )
        output = scale_dequantize(tensor=output, scale=self.scale, upcast_dtype=output.dtype)
        return output

    @classmethod
    def from_float(
        cls,
        mod,
        q_config: Union[BASE_QCONFIG_DICT_TYPE, PER_CHANNEL_QCONFIG_DICT_TYPE] = _DEFAULT_CUSTOM_QCONFIG_DICT,
    ):
        """Create a QuantizedExpertFusedColumnParallel from a float module."""
        assert mod.__class__.__name__ == "ExpertFusedColumnParallelLinear", "ExpertFusedColumnParallelLinear expected"

        new_mod = QuantizedExpertFusedColumnParallel(
            num_experts=mod.num_experts,
            input_size=mod.input_size,
            output_size=mod.output_size,
            quantization_type=q_config["quantization_type"],
            quantized_dtype=q_config["quantized_dtype"],
            dtype=mod.dtype,
            device=mod.weight.device,
            stride=mod.stride,
            keep_master_weight=mod.keep_master_weight,
            quantization_per_channel_axis=cast(int, q_config.get("quantization_per_channel_axis")),
            tensor_model_parallel_group=mod.tensor_parallel_group,
        )
        return new_mod


class QuantizedExpertFusedRowParallel(QuantizedRowParallel, ExpertFusedLinear):
    """
    Quantized version of the ExpertFusedRowParallelLinear class
    """

    autograd_func_class = ExpertFusedLinearWithAsyncCommunication

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        reduce_output: bool = False,
        quantization_type: Union[QuantizationType, str] = "per_tensor_symmetric",
        quantized_dtype: Union[QuantizedDtype, torch.dtype] = QuantizedDtype.INT8,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        stride: int = 1,
        keep_master_weight: bool = False,
        quantization_per_channel_axis: Optional[int] = None,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
    ):
        self.num_experts = num_experts
        self._n_local_experts = divide(num_experts, get_expert_model_parallel_size())

        if quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            if quantization_per_channel_axis is None:
                quantization_per_channel_axis = 2  # Default to 2
            # For post-matmul output dequantization, we need the per_channel_axis = output_dim
            assert (
                quantization_per_channel_axis == 2
            ), "Only per_channel_axis=2 supported for post-matmul output dequantization"

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            quantization_type=quantization_type,
            input_is_parallel=True,
            dtype=dtype,
            quantized_dtype=quantized_dtype,
            device=device,
            stride=stride,
            sequence_parallel_enabled=False,
            keep_master_weight=keep_master_weight,
            quantization_per_channel_axis=quantization_per_channel_axis,
            reduce_output=reduce_output,
            tensor_model_parallel_group=tensor_model_parallel_group,
        )

    def _setup_for_weight_and_bias_config(self, bias: bool):
        """
        Same as the ExpertFusedRowParallelLinear.set_weight_and_bias_config()

        TODO: modularize both quantization layers and moe layers
        """
        # Define 3D weight tensor, one linear layer per expert
        self.weight_shape = (self._n_local_experts, self.input_size_per_partition, self.output_size)
        # Row parallel partitioning for each expert
        self.weight_partition_dim = 1
        self.bias_shape = None

    def forward(self, input_: torch.Tensor, expert_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Same as the forward of ExpertFusedRowParallelLinear, except with weight dequantization,
        and save_for_backward=False."""

        # Matrix multiply.
        weight = self.weight[expert_indices, :, :] if expert_indices is not None else self.weight
        weight_for_matmul = direct_cast_dequantize(tensor=weight, upcast_dtype=input_.dtype)
        output_parallel = self._forward_impl(
            input=input_,
            weight=weight_for_matmul,
            bias=None,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
            autograd_func_class=self.autograd_func_class,
            save_for_backward=False,
            process_group=self.tensor_parallel_group,
        )
        output_parallel = scale_dequantize(tensor=output_parallel, scale=self.scale, upcast_dtype=output_parallel.dtype)

        if self.reduce_output:
            output = reduce_from_tensor_model_parallel_region(
                output_parallel, process_group=self.tensor_parallel_group,
            )
            return output
        else:
            # Return without output all-reduce, in favor of an all-reduce or reduce-scatter after the MoE output combine.
            return output_parallel

    @classmethod
    def from_float(
        cls,
        mod,
        q_config: Union[BASE_QCONFIG_DICT_TYPE, PER_CHANNEL_QCONFIG_DICT_TYPE] = _DEFAULT_CUSTOM_QCONFIG_DICT,
    ):
        """
        Create a QuantizedExpertFusedRowParallel from a float module
        """
        assert mod.__class__.__name__ == "ExpertFusedRowParallelLinear", "ExpertFusedRowParallelLinear expected"

        return QuantizedExpertFusedRowParallel(
            num_experts=mod.num_experts,
            input_size=mod.input_size,
            output_size=mod.output_size,
            reduce_output=mod.reduce_output,
            quantization_type=q_config["quantization_type"],
            dtype=mod.dtype,
            quantized_dtype=q_config["quantized_dtype"],
            device=mod.weight.device,
            stride=mod.stride,
            keep_master_weight=mod.keep_master_weight,
            quantization_per_channel_axis=cast(int, q_config.get("quantization_per_channel_axis")),
            tensor_model_parallel_group=mod.tensor_parallel_group,
        )
