import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import torch
from torch.ao.nn.quantized.dynamic.modules.linear import _quantize_weight
from torch.ao.quantization.qconfig import default_dynamic_qconfig

import neuronx_distributed.parallel_layers.parallel_state as p_state
from neuronx_distributed.modules.moe.moe_parallel_layers import (
    ExpertFusedColumnParallelLinear,
    ExpertFusedRowParallelLinear,
)
from neuronx_distributed.quantization.quantize import convert

num_experts = 3
intermediate_size = 4
hidden_size = 5
capacity = 2
torch.manual_seed(0)

TEMP_COMPILER_WORK_DIR = "compiler_workdir"
TEMP_STATE_DICT_NAME = "quantized_state_dict.pt"

SCALE = 0.123


class Model(torch.nn.Module):
    def __init__(self):
        torch.manual_seed(0)  # to ensure the weight is the same on every initialization
        super().__init__()
        self.lay1 = ExpertFusedColumnParallelLinear(
            num_experts=num_experts,
            input_size=hidden_size,
            output_size=intermediate_size,
            dtype=torch.float32,
        )
        self.lay2 = ExpertFusedRowParallelLinear(
            num_experts=num_experts,
            input_size=intermediate_size,
            output_size=hidden_size,
            reduce_output=True,
            dtype=torch.float32,
        )
        self.lay3 = ExpertFusedColumnParallelLinear(
            num_experts=num_experts,
            input_size=hidden_size,
            output_size=intermediate_size,
            dtype=torch.float32,
        )
        self.lay4 = ExpertFusedRowParallelLinear(
            num_experts=num_experts,
            input_size=intermediate_size,
            output_size=hidden_size,
            reduce_output=True,
            dtype=torch.float32,
        )

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        x = self.lay3(x)
        x = self.lay4(x)
        return x


def quantize_weight(float_state_dict):
    int8_state_dict = {}
    for name, weight in float_state_dict.items():
        weight_observer = default_dynamic_qconfig.weight()
        weight_observer(weight)
        qint8_weight = _quantize_weight(weight, weight_observer)
        int8_state_dict[name] = qint8_weight.int_repr()
        int8_state_dict[name.replace("weight", "scale")] = torch.tensor([qint8_weight.q_scale()])
    return int8_state_dict


def get_input():
    return torch.randn((num_experts, capacity, hidden_size))


def init_ditributed_env():
    os.environ["RANK"] = str(0)
    os.environ["WORLD_SIZE"] = str(1)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2024"

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="xla")

    p_state.destroy_model_parallel()
    p_state.initialize_model_parallel(tensor_model_parallel_size=1)


def _prepare_state_dict():
    init_ditributed_env()
    with torch.no_grad():
        model = Model()
        float_sd = model.state_dict()
        q_sd = quantize_weight(float_sd)
    torch.save(q_sd, TEMP_STATE_DICT_NAME)
    p_state.destroy_model_parallel()


def prepare_state_dict():
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_prepare_state_dict)
        future.result()


def load_model():
    model = Model()
    model_quant = convert(model, q_config=None, inplace=True, mapping=None)
    print(model_quant)
    all_parameters_name = []
    for name, _ in model_quant.named_parameters():
        all_parameters_name.append(name)
    print(all_parameters_name)

    alias = {}

    return model_quant, alias


def checkpoint_loader_fn():
    return torch.load(TEMP_STATE_DICT_NAME)


def load_traced_model(input_fp32):
    from neuronx_distributed.trace import parallel_model_trace

    sample_inputs = input_fp32
    traced_model = parallel_model_trace(
        load_model,  # This loads the parallel model
        sample_inputs,
        tp_degree=2,
        compiler_workdir=TEMP_COMPILER_WORK_DIR,  # This is where you will find the hlo & neff
        compiler_args="--auto-cast=none",  # Pass your compiler flags here,
        inline_weights_to_neff=False,
        spmd_mode=True,
        checkpoint_loader_callable=checkpoint_loader_fn,
        force_custom_init_on_device=True,
    )
    return traced_model


def get_output_from_traced_quantized_model(input_fp32):
    prepare_state_dict()
    traced_quantized_model = load_traced_model(input_fp32)
    return traced_quantized_model(input_fp32)


def _get_output_from_cpu_model(input_fp32):
    init_ditributed_env()
    with torch.no_grad():
        model = Model()
        output = model(input_fp32)
    p_state.destroy_model_parallel()
    return output


def get_output_from_cpu_model(input_fp32):
    """Put execution in another process to avoid neuron device not available error"""
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_get_output_from_cpu_model, input_fp32)
        output = future.result()
    return output


def main():
    input_fp32 = get_input()
    cpu_float_result = get_output_from_cpu_model(input_fp32)
    traced_quantized_result = get_output_from_traced_quantized_model(input_fp32)

    print(f"cpu result: {cpu_float_result}")
    print(f"traced quantized result: {traced_quantized_result}")
    assert torch.allclose(cpu_float_result, traced_quantized_result, atol=1e-2)

    print("Test succeeded for Quantized Expert-fused Parallel Linear Layers!")

    if os.path.exists(TEMP_STATE_DICT_NAME):
        os.remove(TEMP_STATE_DICT_NAME)

    if os.path.exists(TEMP_COMPILER_WORK_DIR) and os.path.isdir(TEMP_COMPILER_WORK_DIR):
        shutil.rmtree(TEMP_COMPILER_WORK_DIR)


if __name__ == "__main__":
    main()
