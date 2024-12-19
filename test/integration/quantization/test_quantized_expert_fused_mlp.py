import torch
import traceback
from concurrent.futures import ProcessPoolExecutor

from neuronx_distributed.modules.moe.moe_parallel_layers import (
    ExpertFusedColumnParallelLinear,
    ExpertFusedRowParallelLinear,
)
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

from test_quantized_mlp import QuantizedCpuLinear, run_quantization_test


num_experts = 3
intermediate_size = 4
hidden_size = 5
capacity = 2
torch.manual_seed(0)


class ExpertFusedLinear(torch.nn.Module):
    def __init__(self, num_experts, in_features, out_features, dtype):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.experts = torch.nn.ModuleList([
            torch.nn.Linear(in_features, out_features, bias=False, dtype=dtype)
            for e in range(num_experts)
        ])

    def forward(self, input_):
        assert len(input_.shape) == 3 and input_.shape[0] == self.num_experts
        output = torch.stack([self.experts[e](input_[e]) for e in range(num_experts)], dim=0)
        return output


class QuantizedExpertFusedCpuLinear(torch.nn.Module):

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        bias: bool,
        device=None,
        dtype=None,
        quantization_type=QuantizationType.PER_TENSOR_SYMMETRIC,
        per_channel_axis=None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        per_channel_axis = per_channel_axis if per_channel_axis is not None else 0
        self.experts = torch.nn.ModuleList([
            QuantizedCpuLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                device=device,
                dtype=dtype,
                quantization_type=quantization_type,
                per_channel_axis=per_channel_axis,
            )
            for e in range(num_experts)
        ])

    def forward(self, input_):
        assert len(input_.shape) == 3 and input_.shape[0] == self.num_experts
        output = torch.stack([self.experts[e](input_[e]) for e in range(num_experts)], dim=0)
        return output

    @classmethod
    def from_float(
        cls,
        mod,
        q_config,
    ):
        assert mod.__class__.__name__ == "ExpertFusedLinear", "ExpertFusedLinear expected"
        return QuantizedExpertFusedCpuLinear(
            num_experts=mod.num_experts,
            in_features=mod.in_features,
            out_features=mod.out_features,
            bias=mod.bias,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
            quantization_type=q_config["quantization_type"],
            per_channel_axis=q_config.get("quantization_per_channel_axis"),
        )


class ExpertFusedModel(torch.nn.Module):

    NUM_LAYERS = 4

    def __init__(self, is_parallel):
        torch.manual_seed(0)  # to ensure the weight is the same on every initialization
        super().__init__()
        self.num_experts = num_experts
        self.is_parallel = is_parallel

        self.layers = torch.nn.ModuleList()
        assert self.NUM_LAYERS % 2 == 0
        for i in range(self.NUM_LAYERS):
            if is_parallel:
                if i % 2 == 0:
                    self.layers.append(ExpertFusedColumnParallelLinear(
                        num_experts=num_experts,
                        input_size=hidden_size,
                        output_size=intermediate_size,
                        dtype=torch.float32,
                    ))
                else:
                    self.layers.append(ExpertFusedRowParallelLinear(
                        num_experts=num_experts,
                        input_size=intermediate_size,
                        output_size=hidden_size,
                        reduce_output=True,
                        dtype=torch.float32,
                    ))
            else:
                if i % 2 == 0:
                    self.layers.append(ExpertFusedLinear(num_experts, hidden_size, intermediate_size, dtype=torch.float32))
                else:
                    self.layers.append(ExpertFusedLinear(num_experts, intermediate_size, hidden_size, dtype=torch.float32))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @classmethod
    def requantize_weights_across_experts(cls, state_dict, num_experts):
        """
        Re-quantize the expert weights (corresponding to the given prefix) after fusing the
        weights along the expert dimension. This is necessary as each linear layer is quantized individually.
        Whereas in NxD we fuse the weights together across experts, and want a single scale which is independent of the
        expert dimension.
        """
        expert_fused_state_dict = {}

        for layer_idx in range(cls.NUM_LAYERS):
            expert_weights = []
            scales = []
            prefix = f"layers.{layer_idx}"
            for e in range(num_experts):
                # (I, H)
                expert_weights.append(state_dict[f"{prefix}.experts.{e}.weight"])
                # (1, ) for per_tensor or (I, 1) for per_channel
                scales.append(state_dict[f"{prefix}.experts.{e}.scale"])

            # Re-quantize the weights after fusing the weights along the expert dimension

            # (E, H, I)
            weight = torch.stack(expert_weights, dim=0).transpose(1, 2)
            # (E, 1) for per_tensor or (E, I, 1) for per_channel
            scale = torch.stack(scales, dim=0)

            is_per_channel = (len(scale.shape) == 3)
            if is_per_channel:
                scale = scale.transpose(1, 2)

            # combined_scale: (1, ) for per_tensor or (1, 1, I) for per_channel
            combined_scale = torch.max(scale, dim=0, keepdim=is_per_channel).values

            dtype = weight.dtype
            if is_per_channel:
                # (E, H, I) * (E, 1, I) -> (E, H, I)
                dequantized_weight = weight.to(dtype=torch.float32) * scale
            else:
                # (E, H, I) * (E, 1, 1) -> (E, H, I)
                dequantized_weight = weight.to(dtype=torch.float32) * scale.unsqueeze(2)

            # (E, H, I) / (1, ) for per_tensor or (E, H, I) / (1, 1, I) for per_channel -> (E, H, I)
            quantized_weight = (dequantized_weight / combined_scale).to(dtype)

            expert_fused_state_dict[f"{prefix}.weight"] = quantized_weight
            expert_fused_state_dict[f"{prefix}.scale"] = combined_scale

        return expert_fused_state_dict

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
        expert_fused_state_dict = cls.requantize_weights_across_experts(state_dict, model_fp32.num_experts)
        torch.save(expert_fused_state_dict, save_path)
        return model_fp32_int8, state_dict


if __name__ == "__main__":

    common_args = dict(
        model_cls=ExpertFusedModel,
        input_shape=(num_experts, capacity, hidden_size),
        # Skip scales validation since the MoE layer scales are combined across experts
        validate_scales=False,
    )

    try:
        q_config = get_default_custom_qconfig_dict()
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_quantization_test, q_config, **common_args)
            results = future.result()
    except Exception:
        print(traceback.format_exc())

    try:
        q_config = get_default_per_channel_custom_qconfig_dict()
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_quantization_test, q_config, **common_args)
            results = future.result()
    except Exception:
        print(traceback.format_exc())
