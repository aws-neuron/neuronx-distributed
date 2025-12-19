import pytest
import logging
import os

import numpy as np
import torch
from torch_xla.core import xla_model as xm
from torch_neuronx.testing import neuron_allclose

from neuronxcc.nki.language import nc

from neuronx_distributed.utils.random import set_random_seed
from neuronx_distributed.utils.model_utils import get_platform_lnc
from neuronx_distributed.kernels.output_proj import cte_output_proj_dequant_kernel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Env var
# To save hlo and neff in curr dir
os.environ['NEURON_FRAMEWORK_DEBUG'] = "1"
# float8_e4m3fn is not supported in neuronxcc. Set UNSAFE_FP8FNCAST to enable e4m3fn -> e4m3 casting
os.environ["UNSAFE_FP8FNCAST"] = "1"
os.environ["NEURON_CC_FLAGS"] = " --disable-dge --internal-hlo2tensorizer-options='--experimental-unsafe-fp8e4m3fn-as-fp8e4m3'"


@pytest.mark.parametrize("B, S, H, n, d, test_bias, compute_dtype, should_dequant, block_size", [
    # Qwen3 MoE CTE
    (1, 640, 4096, 16, 128, False, torch.float16, False, None), # CP, no dequant, FP16 weight
    (1, 640, 4096, 16, 128, False, torch.float16, True, (128,128)), # CP, dequant FP8 to FP16
    (1, 10240, 4096, 16, 128, False, torch.bfloat16, False, None), # no CP, no dequant, BF16 weight
    (1, 10240, 4096, 16, 128, False, torch.bfloat16, True, (128,128)), # CP, dequant FP8 to BF16
    (16, 640, 8192, 16, 128, False, torch.float16, False, None), # BS16, CP, no dequant, FP16 weight
    (16, 640, 8192, 16, 128, False, torch.bfloat16, True, (128,128)), # BS16, CP, dequant FP8 to BF16
    (32, 640, 8192, 16, 128, False, torch.bfloat16, True, (128,128)), # BS32, CP, dequant FP8 to BF16
    (64, 640, 8192, 16, 128, False, torch.bfloat16, True, (128,128)), # BS64, CP, dequant FP8 to BF16
])
def test_cte_output_proj_dequant_kernel(B, S, H, n, d, test_bias, compute_dtype, should_dequant, block_size):
    """
    B: batch size
    S: sequence length (if using context parallel, this would be S//cp_degree)
    H: hidden size
    n: num of attention heads per core (is using tensor parallel, this would be N//tp_degree)
    d: hidden dimension per head
    test_bias: bool, whether to include bias or not
    should_dequant: bool, whether to dequantize fp8 checkpoint or not
    block_size: Tuple[int]=None, needed for blockwise (de)quantization
    """
    set_random_seed(0)
    grid = (nc(get_platform_lnc().value),)

    # Generate random inputs
    # The compute is: out(B, S, H) = attention_output(B, S, n, d) @ out_proj_weight(n * d, H)
    attention_output = torch.randn((B, S, n*d), dtype=compute_dtype) * 0.02
    o_proj_weight = torch.randn((n*d, H), dtype=compute_dtype) * 0.02
    o_proj_bias = None
    if test_bias:
        o_proj_bias = torch.randn((H,), dtype=compute_dtype) * 0.02

    # Test FP8 dequantization
    if should_dequant:
        assert o_proj_bias is None, "FP8 dequantization with bias is not supported yet"

        # View weight tensor in blocks
        blocks = o_proj_weight.view(-1, block_size[0], o_proj_weight.size(1))
        blocks = blocks.view(blocks.size(0), blocks.size(1), -1, block_size[1]) # (n*d//block_size[0], block_size[0], H//block_size[1], block_size[1])
        # Calculate max abs value for each block
        max_abs_values = torch.amax(blocks.abs(), dim=(1, 3))  # (n*d//block_size[0], H//block_size[1])
        # Compute scaling factors (avoid division by zero)
        fp8_max, fp8_min = 240.0, -240.0
        o_proj_weight_scale = max_abs_values / fp8_max
        o_proj_weight_scale_no_repeat = torch.clamp(o_proj_weight_scale, min=1e-5)  # Prevent division by zero
        o_proj_weight_scale_broadcast_tile_dim = o_proj_weight_scale_no_repeat.repeat_interleave(block_size[0], dim=0)
        o_proj_weight_scale = o_proj_weight_scale_broadcast_tile_dim.repeat_interleave(block_size[1], dim=1) # (n*d, H)
        # Scale the values and simulate converting to float8
        o_proj_weight_fp8 = o_proj_weight / o_proj_weight_scale
        o_proj_weight_fp8 = torch.clamp(o_proj_weight_fp8, fp8_min, fp8_max)
        o_proj_weight_fp8 = o_proj_weight_fp8.to(torch.float8_e4m3fn)

    # Get golden output 
    # To generate apple-to-apple golden, dequantize FP8 on CPU
    if should_dequant:
        o_proj_weight_dequantized = o_proj_weight_fp8.to(torch.float32) * o_proj_weight_scale.to(torch.float32)
        o_proj_weight_dequantized = o_proj_weight_dequantized.to(compute_dtype)
        golden_out = attention_output @ o_proj_weight_dequantized
    else:
        golden_out = attention_output @ o_proj_weight
        if test_bias:
            golden_out += o_proj_bias

    # Kernel wants BndS layout for input.
    attention_output = attention_output.reshape(B, S, n, d)
    kernel_attn_in = attention_output.permute(0, 2, 3, 1) # (B, n, d, S)
    
    # Move tensors to device
    device = xm.xla_device()
    kernel_attn_in = kernel_attn_in.to(device)

    # Get kernel output
    if should_dequant:
        o_proj_weight_fp8 = o_proj_weight_fp8.to(device)
        o_proj_weight_scale = o_proj_weight_scale.to(device)
        o_proj_weight_scale_no_repeat = o_proj_weight_scale_no_repeat.to(device)
        o_proj_weight_scale_broadcast_tile_dim = o_proj_weight_scale_broadcast_tile_dim.to(device)

        for i in range(5):
            kernel_out = cte_output_proj_dequant_kernel[grid](
                active=kernel_attn_in,
                weight=o_proj_weight_fp8,
                bias=o_proj_bias, # None in dequant test case
                scale=o_proj_weight_scale_broadcast_tile_dim,
                block_size=block_size,
            )
    else:
        o_proj_weight = o_proj_weight.to(device)
        if test_bias:
            o_proj_bias = o_proj_bias.to(device)

        kernel_out = cte_output_proj_dequant_kernel[grid](
            active=kernel_attn_in,
            weight=o_proj_weight,
            bias=o_proj_bias,
        )

    # Validate accuracy
    allclose_summary = neuron_allclose(golden_out, kernel_out.cpu())
    logger.info(f"allclose_summary: {allclose_summary}")
    assert allclose_summary.allclose, \
           f"Failed allclosing for config B={B}, S={S}, H={H}, n={n}, d={d}, test_bias={test_bias}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])