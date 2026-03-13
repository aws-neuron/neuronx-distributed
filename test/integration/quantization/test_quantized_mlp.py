import os
import shutil
import traceback
from functools import partial

import torch
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.quantization.dequantize import scale_dequantize, direct_cast_dequantize
from neuronx_distributed.quantization.quantization_config import (
    BASE_QCONFIG_DICT_TYPE,
    QuantizationType,
    ActivationQuantizationType,
    get_default_custom_qconfig_dict,
    get_default_per_channel_custom_qconfig_dict,
)
from neuronx_distributed.quantization.quantization_utils import (
    convert_qint8_to_int8_state_dict,
    quantize_pytorch_model_per_channel_symmetric,
    quantize_pytorch_model_per_tensor_symmetric,
    quantize_static_quant_activations
)
from neuronx_distributed.quantization.quantize import convert
from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder

dim = 6
DTYPE = torch.float32
Q_DTYPE = torch.int8
PT_SAVE_PATH = "quantized_model.pt"


class QuantizedCpuLinear(torch.nn.Module):
    """
    CPU version of our Dequant logic . Used for testing.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        quantization_type=QuantizationType.PER_TENSOR_SYMMETRIC,
        per_channel_axis=None,
        activation_quantization_type=ActivationQuantizationType.NONE,
        
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8, device=device), requires_grad=False
        )
        self.dtype = dtype
        if quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=dtype))
            if activation_quantization_type == ActivationQuantizationType.STATIC:
                self.input_scale = torch.nn.Parameter(torch.tensor([1.0], dtype=dtype))
        else:
            per_channel_axis = per_channel_axis if per_channel_axis is not None else 0
            weight_shape = self.weight.shape
            scale_shape = [1] * len(weight_shape)
            scale_shape[per_channel_axis] = weight_shape[per_channel_axis]
            self.scale = torch.nn.Parameter(torch.ones(scale_shape, device=self.weight.device), requires_grad=False)
        if bias:
            raise NotImplementedError()
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor):
        if hasattr(self, "input_scale"):
            input = quantize_static_quant_activations(input, self.input_scale, Q_DTYPE)
            input = direct_cast_dequantize(input, upcast_dtype=DTYPE)
            weight = direct_cast_dequantize(self.weight, upcast_dtype=DTYPE)
            scale = self.input_scale * self.scale
        else:
            weight = direct_cast_dequantize(self.weight, upcast_dtype=input.dtype)
            scale = self.scale

        output = F.linear(input, weight, self.bias)
        return scale_dequantize(tensor=output, scale=scale.T, upcast_dtype=self.dtype)

    @classmethod
    def from_float(
        cls,
        mod,
        q_config,
    ):
        assert mod.__class__.__name__ == "Linear", "Linear expected"
        return QuantizedCpuLinear(
            in_features=mod.in_features,
            out_features=mod.out_features,
            bias=mod.bias,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
            quantization_type=q_config["quantization_type"],
            per_channel_axis=q_config.get("quantization_per_channel_axis"),
            activation_quantization_type=q_config["activation_quantization_type"],
        )


class Model(torch.nn.Module):
    def __init__(self, is_parallel):
        super().__init__()
        self.is_parallel = is_parallel
        self.layers = torch.nn.ModuleList()
        for i in range(4):
            if is_parallel:
                if i % 2 == 0:
                    self.layers.append(ColumnParallelLinear(
                        input_size=dim, output_size=dim, bias=False, gather_output=False, dtype=DTYPE
                    ))
                else:
                    self.layers.append(RowParallelLinear(
                        input_size=dim, output_size=dim, bias=False, input_is_parallel=True, dtype=DTYPE
                    ))
            else:
                self.layers.append(torch.nn.Linear(dim, dim, bias=False, dtype=DTYPE))

    def forward(self, x):
        for layer in self.layers:
            if getattr(layer, "activation_quantization_type", None) == ActivationQuantizationType.STATIC:
                x = quantize_static_quant_activations(x, layer.input_scale, Q_DTYPE)
                x = scale_dequantize(x, layer.input_scale, DTYPE)
            x = layer(x)
        return x

    @classmethod
    def _quantize_and_save_model(cls, model_fp32, q_config, save_path):
        """
        This function quantizes the non-parallel model (i.e. which uses nn.Linear) using pytorch APIs, and stores the
        state dict at save_path in a format that can be loaded into the parallel (NxD) model.

        Note that this is a helper function used only for testing.
        """
        assert not model_fp32.is_parallel
        if q_config["quantization_type"] == QuantizationType.PER_CHANNEL_SYMMETRIC:
            model_fp32_int8 = quantize_pytorch_model_per_channel_symmetric(model_fp32)
        else:
            model_fp32_int8 = quantize_pytorch_model_per_tensor_symmetric(model_fp32)
        state_dict = model_fp32_int8.state_dict()
        convert_qint8_to_int8_state_dict(state_dict=state_dict)
        if q_config["activation_quantization_type"] == ActivationQuantizationType.STATIC:
            for i in range(4):
                state_dict[f"layers.{i}.input_scale"] = torch.randn(1, dtype=DTYPE).abs()

        torch.save(state_dict, save_path)
        return model_fp32_int8, state_dict


def load_quantize_model(q_config: BASE_QCONFIG_DICT_TYPE, model_cls, input_shape):
    model_fp32 = model_cls(is_parallel=False)
    input_fp32 = torch.randn(input_shape)

    # Get Pytorch Quantized Model
    model_fp32.eval()
    model_fp32_int8, state_dict = model_cls._quantize_and_save_model(model_fp32, q_config, save_path=PT_SAVE_PATH)

    # Get NxD version of Quatization Model on CPU
    mapping = {torch.nn.Linear: QuantizedCpuLinear}
    nxd_quantized_cpu_model = convert(model_fp32, q_config=q_config, mapping=mapping)
    convert_qint8_to_int8_state_dict(state_dict=state_dict)
    nxd_quantized_cpu_model.load_state_dict(state_dict, strict=False)

    return model_fp32_int8, input_fp32, model_fp32, nxd_quantized_cpu_model


def load_model(q_config: BASE_QCONFIG_DICT_TYPE, model_cls):
    model_parallel = model_cls(is_parallel=True)
    model_quant = convert(model_parallel, q_config=q_config, inplace=True, mapping=None)
    print(model_quant)
    all_parameters_name = []
    for name, _ in model_quant.named_parameters():
        all_parameters_name.append(name)
    print(all_parameters_name)

    return model_quant


def checkpoint_loader_fn():
    checkpoint = torch.load(PT_SAVE_PATH)
    return {k: v for k, v in checkpoint.items() if v is not None}


def load_traced_model(input_fp32, qconfig, model_cls):
    sample_inputs = input_fp32
    load_model_partial = partial(load_model, qconfig, model_cls)

    builder = ModelBuilder(
            router=None,
            tp_degree=2,
            checkpoint_loader=checkpoint_loader_fn,
            compiler_workdir="compiler_workdir",
        )
    builder.add(
        key="main",
        model_instance=BaseModelInstance(
            module_cls=load_model_partial,
            input_output_aliases={},
        ),
        example_inputs=[(sample_inputs,)],
        compiler_args="--auto-cast=none",
    )
    neuron_model = builder.trace(initialize_model_weights=True)
    return neuron_model


def validate_against_pytorch_quantization(pytorch_quantized_cpu_model, nxd_quantized_cpu_model):
    """
    This checks that the weights when dequantized from a pytorch quantized model to fp32 vs weights when dequantized to fp32 from a
    nxd cpu model logic are exactly same. This would give assurance that the weight transfer was exactly same
    """

    pytorch_quantized_cpu_model_sd = pytorch_quantized_cpu_model.state_dict()
    nxd_quantized_cpu_model_sd = nxd_quantized_cpu_model.state_dict()
    assertion = False
    for key in pytorch_quantized_cpu_model_sd.keys():
        if "_packed_params._packed_params" in key:
            prefix = key.split("_packed_params._packed_params")[0]
            assert torch.allclose(
                pytorch_quantized_cpu_model_sd[key][0].dequantize(),
                nxd_quantized_cpu_model_sd[prefix + "weight"] * \
                nxd_quantized_cpu_model_sd[prefix + "scale"]
            )
            assertion = True
    assert assertion
    print("Test successful for validate_against_pytorch_quantization")


def recreate_sharded_scales(traced_model_sd, scale_name, partition_dim):
    tensors_to_gather = []
    for i in range(2):
        tensors_to_gather.append(traced_model_sd[f"models.{i}.weights.{scale_name}"])
    recreated_scale = torch.cat(tensors_to_gather, axis=partition_dim)
    return recreated_scale


def is_scalar_partitioned(scalar_tensor):
    if scalar_tensor.shape == (1,) or max(scalar_tensor.shape) == dim:
        return False
    return True


def extract_partition_dim(scale_tensor):
    scale_shape = scale_tensor.shape
    for i, shape_dim in enumerate(scale_shape):
        if shape_dim > 1:
            return i
    raise RuntimeError("scale is not really sharded")


def run_quantization_test(q_config, model_cls, input_shape):
    torch.manual_seed(0)
    model_fp32_int8, input_fp32, model_fp32, nxd_quantized_cpu_model = load_quantize_model(
        q_config=q_config, model_cls=model_cls, input_shape=input_shape
    )
    traced_model = load_traced_model(input_fp32=input_fp32, qconfig=q_config, model_cls=model_cls)

    nxd_result = traced_model(input_fp32)

    if q_config["activation_quantization_type"] == ActivationQuantizationType.NONE:
        cpu_result = model_fp32_int8(input_fp32)
        fp_32_result = model_fp32(input_fp32)
        # Validate the CPU version of our de-quant logic matches the pytorch dequant
        validate_against_pytorch_quantization(
            pytorch_quantized_cpu_model=model_fp32_int8, nxd_quantized_cpu_model=nxd_quantized_cpu_model
        )
        # NxD result and Pytorch Quantized Result
        assert torch.allclose(cpu_result, nxd_result, atol=1e-2)

        # FP32 model result and NxD result
        atol = 1e-3 if q_config["quantization_type"] == QuantizationType.PER_CHANNEL_SYMMETRIC else 1e-2
        torch.allclose(fp_32_result, nxd_result, atol=atol)
    print(nxd_quantized_cpu_model(input_fp32), "\n", nxd_result)
    # CPU quantized result and NxD result to be exactly equal if scales are equal
    assert torch.allclose(nxd_quantized_cpu_model(input_fp32), nxd_result)

    print(f"\n Test successful for Quantized Layers with qconfig {q_config}")

    if os.path.exists(PT_SAVE_PATH):
        os.remove(PT_SAVE_PATH)

    if os.path.exists("compiler_workdir") and os.path.isdir("compiler_workdir"):
        shutil.rmtree("compiler_workdir")

if __name__ == "__main__":
    common_args = dict(
        model_cls=Model,
        input_shape=(1, 2, dim),
    )

    q_configs = [get_default_custom_qconfig_dict(), get_default_per_channel_custom_qconfig_dict()]
    qconfig_static = get_default_custom_qconfig_dict()
    qconfig_static["activation_quantization_type"] = ActivationQuantizationType.STATIC
    q_configs.append(qconfig_static)

    for q_config in q_configs:
        try:
            print(f"\n Testing with qconfig {q_config}")
            run_quantization_test(q_config, **common_args)
        except Exception:
            print(traceback.format_exc())
            assert False, f"\n Test failed for qconfig {q_config}"
