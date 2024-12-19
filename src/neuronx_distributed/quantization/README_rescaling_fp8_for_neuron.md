## Why we need to rescale FP8 weights
The FP8 format neuron supports for E4M3 is the `FP8_EXP4 (IEEE-754)`, which is different than the `OCP FP8 E4M3/e4m3fn`, which is more commonly available data type on GPUs.

One of the main difference is that on Neuron FP8E4M3(FP8_EXP4) datatype, the range is +/-240 while for the OCP FP8 E4M3/e4m3fn, the range is +/-448.
Value outside +/-240 on Neuron devices, for E4M3 might result into NaNs.

This is the reason, we have to `rescale` openly available checkpoints, to fit our range.

One of the way we can achieve this is mentioned below.

```

import torch
from safetensors.torch import load_file, save_file
import os
from neuronx_distributed.quantization.dequantize import scale_dequantize

FP8_SCALING_FACTOR = 448.0 / 240.0
ckpts_paths = ["model-00001-of-00109.safetensors", "model-00005-of-00109.safetensors", 
              "model-00002-of-00109.safetensors", "model-00006-of-00109.safetensors",
              "model-00003-of-00109.safetensors", "model-00007-of-00109.safetensors",
              "model-00004-of-00109.safetensors"]


def find_min_max(path):
    ckpt = load_file(path)
    min_val = 1000.0
    max_val = -1000.0
    fp8_found = False
    for key in ckpt.keys():
        tensor = ckpt[key]
        if tensor.dtype == torch.float8_e4m3fn:
            fp8_found = True
            bf_16_tensor = tensor.bfloat16()
            min_val = min(min_val, bf_16_tensor.min())
            max_val = max(max_val, bf_16_tensor.max())
    if fp8_found:
        print(f"For path: {path}, min_val: {min_val}, max_val: {max_val}")


def rescale(initial_weight, initial_scale):
    initial_weight_bf16 = initial_weight.bfloat16()

    final_weight_bf16 = initial_weight_bf16 / FP8_SCALING_FACTOR
    final_scale = initial_scale * FP8_SCALING_FACTOR
    return final_weight_bf16.to(torch.float8_e4m3fn), final_scale

def verify(initial_bf16, final_bf16, weight_key, scale_key):
    for atol in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        if torch.allclose(final_bf16, initial_bf16, atol=atol):
            print(f"Verified rescaling for {weight_key}/{scale_key} with atol {atol}")
            return
    
    raise RuntimeError(f"Cannot verify rescaling for {weight_key}/{scale_key}")


def rescale_checkpoints(path, new_path):
    ckpt = load_file(path)
    all_keys = [key for key in ckpt.keys() if key.endswith("weight_scale")]
    for key in all_keys:
        scale_key = key
        weight_key = key.replace("weight_scale", "weight")
        
        initial_weight, initial_scale = ckpt[weight_key], ckpt[scale_key]
        final_weight, final_scale = rescale(initial_weight=initial_weight, initial_scale=initial_scale)
        assert initial_weight.dtype == torch.float8_e4m3fn and final_weight.dtype == torch.float8_e4m3fn and initial_scale.dtype == torch.float32 and final_scale.dtype == torch.float32

        initial_bf16 = scale_dequantize(tensor=initial_weight, scale=initial_scale, upcast_dtype=torch.bfloat16)
        final_bf16 = scale_dequantize(tensor=final_weight, scale=final_scale, upcast_dtype=torch.bfloat16)
        
        verify(initial_bf16=initial_bf16, final_bf16=final_bf16, weight_key=weight_key, scale_key=scale_key)

        ckpt[weight_key] = final_weight
        ckpt[scale_key] = final_scale

    save_file(ckpt, new_path)
    print(f"saved {new_path}")



base_path = "[PATH FOR FP8 E4M3FN CHECKPOINTS ]"
new_base_path = "[PATH TO SAVE RESCALED FP8E4M3 CHECKPOINTS]"

# Verify the initial ranges from base path
for file_name in ckpts_paths:
    ckpt_path = os.path.join(base_path, file_name)
    find_min_max(ckpt_path)

# Rescale and save
for file_name in ckpts_paths:
    ckpt_path = os.path.join(base_path, file_name)
    new_ckpt_path = os.path.join(new_base_path, file_name)
    rescale_checkpoints(ckpt_path, new_ckpt_path)

# Verify the final ranges from new_base_path
for file_name in ckpts_paths:
    ckpt_path = os.path.join(new_base_path, file_name)
    find_min_max(ckpt_path)

```
