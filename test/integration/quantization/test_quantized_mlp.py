import os
import shutil

import torch
import torch.nn.functional as F

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.quantization.dequantize import dequantize
from neuronx_distributed.quantization.quantization_utils import (
    convert_float_model_to_pytorch_int8_model,
    convert_qint8_to_int8_state_dict,
)
from neuronx_distributed.quantization.quantize import convert

dim = 6
torch.manual_seed(0)


class QuantizedCpuLinear(torch.nn.Module):
    """
    CPU version of our Dequant logic . Used for testing.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=None) -> None:
        super(QuantizedCpuLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.int8, device=device), requires_grad=False
        )
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=dtype))
        if bias:
            raise NotImplementedError()
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor):
        weight = dequantize(self.weight, scale=self.scale, upcast_dtype=input.dtype)
        return F.linear(input, weight, self.bias)

    @classmethod
    def from_float(
        cls,
        mod,
        quantization_type,
        quantized_dtype,
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


def load_qunatize_model():
    model_fp32 = Model()
    input_fp32 = torch.randn((2, dim))

    # Get Pytorch Quantized Model
    model_fp32.eval()
    model_fp32_int8 = convert_float_model_to_pytorch_int8_model(model_fp32)
    state_dict = model_fp32_int8.state_dict()
    torch.save(state_dict, "fp32_qint8_model.pt")

    # Get NxD version of Quatization Model on CPU
    mapping = {torch.nn.Linear: QuantizedCpuLinear}
    nxd_quantized_cpu_model = convert(model_fp32, q_config=None, mapping=mapping)
    convert_qint8_to_int8_state_dict(state_dict=state_dict)
    nxd_quantized_cpu_model.load_state_dict(state_dict, strict=False)

    return model_fp32_int8, input_fp32, model_fp32, nxd_quantized_cpu_model


def load_model():
    model_parallel = Model_Parallel()
    model_quant = convert(model_parallel, q_config=None, inplace=True, mapping=None)
    print(model_quant)
    all_parameters_name = []
    for name, _ in model_quant.named_parameters():
        all_parameters_name.append(name)
    print(all_parameters_name)

    alias = {}

    return model_quant, alias


def checkpoint_loader_fn():
    return torch.load("fp32_qint8_model.pt")


def load_traced_model(input_fp32):
    from neuronx_distributed.trace import parallel_model_trace

    sample_inputs = input_fp32
    traced_model = parallel_model_trace(
        load_model,  # This loads the parallel model
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
                dequantize(
                    nxd_quantized_cpu_model_sd[prefix + "weight"],
                    nxd_quantized_cpu_model_sd[prefix + "scale"],
                    torch.float32,
                ),
            )
            assertion = True
    assert assertion
    print("Test successful for validate_against_pytorch_quantization")


def validate_scales_in_nxd_model(nxd_quantized_cpu_model, traced_model):
    traced_model_sd = traced_model.models[0].weights.state_dict()
    nxd_quantized_cpu_model_sd = nxd_quantized_cpu_model.state_dict()
    for key, _ in traced_model_sd.items():
        if "scale" in key:
            cpu_scale = nxd_quantized_cpu_model_sd[key.replace("->", ".")]
            nxd_scale = traced_model_sd[key]
            assert cpu_scale == nxd_scale
    print("scale verification successful")


def main():
    model_fp32_int8, input_fp32, model_fp32, nxd_quantized_cpu_model = load_qunatize_model()
    traced_model = load_traced_model(input_fp32=input_fp32)

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
    torch.allclose(fp_32_result, nxd_result, atol=1e-2)

    print("Test successful for Quantized Layers")

    if os.path.exists("fp32_qint8_model.pt"):
        os.remove("fp32_qint8_model.pt")

    if os.path.exists("compiler_workdir") and os.path.isdir("compiler_workdir"):
        shutil.rmtree("compiler_workdir")


if __name__ == "__main__":
    main()
