import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.quantization.dequantize import scale_dequantize
from neuronx_distributed.quantization.quantization_config import (
    BASE_QCONFIG_DICT_TYPE,
    QuantizationType,
    get_default_custom_qconfig_dict,
    get_default_per_channel_custom_qconfig_dict,
)
from neuronx_distributed.quantization.quantization_utils import (
    convert_qint8_to_int8_state_dict,
    quantize_pytorch_model_per_channel_symmetric,
    quantize_pytorch_model_per_tensor_symmetric,
)
from neuronx_distributed.quantization.quantize import convert

dim = 6
torch.manual_seed(0)


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
        per_channel_axis=0,
    ) -> None:
        super(QuantizedCpuLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8, device=device), requires_grad=False
        )
        if quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=dtype))
        else:
            weight_shape = self.weight.shape
            scale_shape = [1] * len(weight_shape)
            scale_shape[per_channel_axis] = weight_shape[per_channel_axis]
            self.scale = torch.nn.Parameter(torch.ones(scale_shape, device=self.weight.device), requires_grad=False)
        if bias:
            raise NotImplementedError()
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor):
        weight = scale_dequantize(self.weight, scale=self.scale, upcast_dtype=input.dtype)
        return F.linear(input, weight, self.bias)

    @classmethod
    def from_float(
        cls,
        mod,
        q_config,
    ):
        """Create a QuantizedRowParallel from a float module

        Args:
            mod: float module
        """
        assert mod.__class__.__name__ == "Linear", "Linear expected"
        return QuantizedCpuLinear(
            in_features=mod.in_features,
            out_features=mod.out_features,
            bias=mod.bias,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
            quantization_type=q_config["quantization_type"],
        )


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lay1 = torch.nn.Linear(dim, dim, bias=False, dtype=torch.float32)
        self.lay2 = torch.nn.Linear(dim, dim, bias=False, dtype=torch.float32)
        self.lay3 = torch.nn.Linear(dim, dim, bias=False, dtype=torch.float32)
        self.lay4 = torch.nn.Linear(dim, dim, bias=False, dtype=torch.float32)

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        return x


class Model_Parallel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lay1 = ColumnParallelLinear(
            input_size=dim, output_size=dim, bias=False, gather_output=False, dtype=torch.float32
        )
        self.lay2 = RowParallelLinear(
            input_size=dim, output_size=dim, bias=False, input_is_parallel=True, dtype=torch.float32
        )
        self.lay3 = ColumnParallelLinear(
            input_size=dim, output_size=dim, bias=False, gather_output=False, dtype=torch.float32
        )
        self.lay4 = RowParallelLinear(
            input_size=dim, output_size=dim, bias=False, input_is_parallel=True, dtype=torch.float32
        )

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        return x


def load_qunatize_model(q_config: BASE_QCONFIG_DICT_TYPE):
    model_fp32 = Model()
    input_fp32 = torch.randn((2, dim))

    # Get Pytorch Quantized Model
    model_fp32.eval()
    if q_config["quantization_type"] == QuantizationType.PER_CHANNEL_SYMMETRIC:
        model_fp32_int8 = quantize_pytorch_model_per_channel_symmetric(model_fp32)
    else:
        model_fp32_int8 = quantize_pytorch_model_per_tensor_symmetric(model_fp32)

    state_dict = model_fp32_int8.state_dict()
    torch.save(state_dict, "fp32_qint8_model.pt")

    # Get NxD version of Quatization Model on CPU
    mapping = {torch.nn.Linear: QuantizedCpuLinear}
    nxd_quantized_cpu_model = convert(model_fp32, q_config=q_config, mapping=mapping)
    convert_qint8_to_int8_state_dict(state_dict=state_dict)
    nxd_quantized_cpu_model.load_state_dict(state_dict, strict=False)

    return model_fp32_int8, input_fp32, model_fp32, nxd_quantized_cpu_model


def load_model(q_config: BASE_QCONFIG_DICT_TYPE):
    model_parallel = Model_Parallel()
    model_quant = convert(model_parallel, q_config=q_config, inplace=True, mapping=None)
    print(model_quant)
    all_parameters_name = []
    for name, _ in model_quant.named_parameters():
        all_parameters_name.append(name)
    print(all_parameters_name)

    alias = {}

    return model_quant, alias


def checkpoint_loader_fn():
    return torch.load("fp32_qint8_model.pt")


def load_traced_model(input_fp32, qconfig):
    from neuronx_distributed.trace import parallel_model_trace

    sample_inputs = input_fp32
    load_model_partial = partial(load_model, qconfig)
    traced_model = parallel_model_trace(
        load_model_partial,  # This loads the parallel model
        sample_inputs,
        tp_degree=2,
        compiler_workdir="compiler_workdir",  # This is where you will find the hlo & neff
        compiler_args="--auto-cast=none",  # Pass your compiler flags here,
        inline_weights_to_neff=False,
        spmd_mode=True,
        checkpoint_loader_callable=checkpoint_loader_fn,
        force_custom_init_on_device=True,
    )
    return traced_model


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
                scale_dequantize(
                    nxd_quantized_cpu_model_sd[prefix + "weight"],
                    nxd_quantized_cpu_model_sd[prefix + "scale"],
                    torch.float32,
                ),
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


def validate_scales_in_nxd_model(nxd_quantized_cpu_model, traced_model):
    traced_model_sd = traced_model.state_dict()
    traced_model_sd_rank0 = traced_model.models[0].weights.state_dict()
    nxd_quantized_cpu_model_sd = nxd_quantized_cpu_model.state_dict()
    for key, _ in traced_model_sd_rank0.items():
        if "scale" in key:
            cpu_scale = nxd_quantized_cpu_model_sd[key.replace("->", ".")]
            if not is_scalar_partitioned(traced_model_sd_rank0[key]):
                nxd_scale = traced_model_sd_rank0[key]
            else:
                nxd_scale = recreate_sharded_scales(
                    traced_model_sd, key, extract_partition_dim(traced_model_sd_rank0[key])
                )
            assert torch.allclose(cpu_scale, nxd_scale)
    print("scale verification successful")


def main(q_config):
    model_fp32_int8, input_fp32, model_fp32, nxd_quantized_cpu_model = load_qunatize_model(q_config=q_config)
    traced_model = load_traced_model(input_fp32=input_fp32, qconfig=q_config)

    # Validate the CPU version of our de-quant logic matches the pytorch dequant
    validate_against_pytorch_quantization(
        pytorch_quantized_cpu_model=model_fp32_int8, nxd_quantized_cpu_model=nxd_quantized_cpu_model
    )

    # Validate that the scales in NxD model are correct
    validate_scales_in_nxd_model(nxd_quantized_cpu_model, traced_model)

    cpu_result = model_fp32_int8(input_fp32)
    nxd_result = traced_model(input_fp32)
    fp_32_result = model_fp32(input_fp32)

    # CPU quantized result and NxD result to be exactly equal
    assert torch.allclose(nxd_quantized_cpu_model(input_fp32), nxd_result)

    # NxD result and Pytorch Quantized Result
    assert torch.allclose(cpu_result, nxd_result, atol=1e-2)

    # FP32 model result and NxD result
    atol = 1e-3 if q_config["quantization_type"] == QuantizationType.PER_CHANNEL_SYMMETRIC else 1e-2
    torch.allclose(fp_32_result, nxd_result, atol=atol)

    print(f"Test successful for Quantized Layers with qconfig {q_config}")

    if os.path.exists("fp32_qint8_model.pt"):
        os.remove("fp32_qint8_model.pt")

    if os.path.exists("compiler_workdir") and os.path.isdir("compiler_workdir"):
        shutil.rmtree("compiler_workdir")


if __name__ == "__main__":
    try:
        q_config = get_default_custom_qconfig_dict()
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(main, q_config)
            results = future.result()
    except Exception:
        print(traceback.format_exc())

    try:
        q_config = get_default_per_channel_custom_qconfig_dict()
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(main, q_config)
            results = future.result()
    except Exception:
        print(traceback.format_exc())
