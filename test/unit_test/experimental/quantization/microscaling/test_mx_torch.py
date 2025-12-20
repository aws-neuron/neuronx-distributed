import pytest

import numpy as np
import torch
import ml_dtypes

from neuronx_distributed.quantization.microscaling.transform_weights import get_mxfp4_tensor_from_uint16, get_mxfp8_tensor_from_uint32
from neuronx_distributed.experimental.quantization.microscaling.mx_torch import quantize_mxfp8, dequantize_mx_tensor, matmul_mx_single_tile, matmul_mx
from neuronx_distributed.experimental.quantization.microscaling.swizzle import swizzle_tensor

# TODO[release] refactor to remove try/except and import from nki testing directly
try:
    from neuronxcc.nki._private.test.mx_util import quantize_mx_golden, nc_matmul_mx_golden
    import neuronxcc.nki._private.private_api as nki_private
    NKI_FP4_X4 = nki_private.float4_e2m1fn_x4
    NKI_FP8_X4 = nki_private.float8_e4m3fn_x4
    MX_IMPORTABLE = True
except Exception:
    MX_IMPORTABLE = False

MIN_MAX_DTYPE_MAP = {
    torch.uint16: (0, 0xFFFF),      # [0, 2**16)
    torch.uint32: (0, 0xFFFFFFFF),  # [0, 2**32)
}

DESCALE_FUNC_MAP = {
    torch.uint16: get_mxfp4_tensor_from_uint16,
    torch.uint32: get_mxfp8_tensor_from_uint32,
}

# scale values correspond to observed values from checkpoint
SCALE_MIN, SCALE_MAX = 119, 123

def _replace_fp8_x4_nans(tensor):
    """
    Helper function to replace any cases of S 1111 111 in fp8_x4 packed into uint32.
    """
    # Convert to int32 for bitwise operations, then back to uint32
    tensor_int32 = tensor.to(torch.int32)
    flat = tensor_int32.flatten()
    result = torch.zeros_like(flat)
    
    for i in range(4):
        shift = i * 8
        val = (flat >> shift) & 0xFF
        # Fix NaNs: if exponent==15 and mantissa==7, keep only sign bit
        is_nan = ((val >> 3) & 0xF == 15) & ((val & 0x7) == 7)
        fixed_val = torch.where(is_nan, val & 0x80, val)
        result |= fixed_val << shift
    
    return result.reshape(tensor.shape).to(torch.uint32)


def _convert_bf16_torch_to_numpy(tensor):
    return tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)


@pytest.mark.parametrize(
    "input_shape,seed", [
        pytest.param((2880, 64), 0, id="t64"),
        pytest.param((2880, 512), 1, id="t512"),
    ]
)
@pytest.mark.parametrize(
    "input_range,atol,rtol", [
        pytest.param((-1e-3, 1e-3), 1.23e-4, 1.25e-1, id="-1e-3_1e-3"),
        pytest.param((-1e-2, 1e-2), 9.77e-4, 1.25e-1, id="-1e-2_1e-2"),
        pytest.param((-1e-1, 1e-1), 0, 0, id="-1e-1_1e-1"),
        pytest.param((-1, 1), 0, 0, id="-4_1"),
        pytest.param((-10, 10), 0, 0, id="-10_10"),
    ]
)
@pytest.mark.parametrize("descaled_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.skipif(not MX_IMPORTABLE, reason="Unable to import NKI private functions")
def test_quantize_mxfp8(input_shape, input_range, descaled_dtype, seed, atol, rtol):
    # Use different manual seeds to get different random inputs
    torch.manual_seed(seed)

    # NOTE: inputs must be bf16 for quantization
    # pre-swizzle input to achieve contiguous [8, 4] blocks for quantization
    max_val, min_val = input_range
    input_bf16_swizzled = swizzle_tensor(torch.rand(input_shape, dtype=torch.bfloat16) * (max_val - min_val) + min_val)

    # calculate golden using compiler golden func
    quant_golden, scale_golden = quantize_mx_golden(_convert_bf16_torch_to_numpy(input_bf16_swizzled), NKI_FP8_X4)

    # quantize to mxfp8 packed into uint32 using torch
    quant_torch, scale_torch = quantize_mxfp8(input_bf16_swizzled, out_x4_dtype=torch.uint32, fp8_dtype=torch.float8_e4m3fn)

    descaled_golden = dequantize_mx_tensor(torch.tensor(quant_golden.view(np.uint32), dtype=torch.uint32), torch.tensor(scale_golden, dtype=torch.uint8), dtype=descaled_dtype, input_is_transposed=True)
    descaled_torch = dequantize_mx_tensor(quant_torch, scale_torch, dtype=descaled_dtype, input_is_transposed=True)

    torch.testing.assert_close(descaled_torch, descaled_golden, atol=atol, rtol=rtol)
    print("Tensors match!")

@pytest.mark.parametrize(
    "input_shape,seed", [
        pytest.param((2880, 64), 0, id="t64"),
        pytest.param((2880, 512), 1, id="t512"),
    ]
)
@pytest.mark.parametrize(
    "input_range", [
        pytest.param((-1e-3, 1e-3), id="-1e-3_1e-3"),
        pytest.param((-1e-2, 1e-2), id="-1e-2_1e-2"),
        pytest.param((-1e-1, 1e-1), id="-1e-1_1e-1"),
        pytest.param((-1, 1), id="-4_1"),
        pytest.param((-10, 10), id="-10_10"),
    ]
)
@pytest.mark.skipif(not MX_IMPORTABLE, reason="Unable to import NKI private functions")
def test_quantize_mxfp8_unbiased_scale(input_shape, input_range, seed):
    # Use different manual seeds to get different random inputs
    torch.manual_seed(seed)

    # NOTE: inputs must be bf16 for quantization
    # pre-swizzle input to achieve contiguous [8, 4] blocks for quantization
    max_val, min_val = input_range
    input_bf16_swizzled = swizzle_tensor(torch.rand(input_shape, dtype=torch.bfloat16) * (max_val - min_val) + min_val)

    # golden scale should be 1 smaller than unbiased scale
    _, scale_golden = quantize_mxfp8(input_bf16_swizzled, out_x4_dtype=torch.uint32, fp8_dtype=torch.float8_e4m3fn)
    _, scale_torch = quantize_mxfp8(input_bf16_swizzled, out_x4_dtype=torch.uint32, fp8_dtype=torch.float8_e4m3fn, use_unbiased_scale=True)

    print(f"{scale_golden=}")
    print(f"{scale_torch+1=}")

    assert torch.equal(scale_golden, scale_torch - 1)
    print("Tensors match!")

@pytest.mark.parametrize(
    "tensor_dtype,tensor_shape,input_is_transposed,output_is_transposed,output_quad_row,seed", [
        # fp4_x4 (uint16) tests - (I, H) shapes
        pytest.param(torch.uint16, (2880, 2880), False, False, False, 0),
        pytest.param(torch.uint16, (2880, 2880), True, False, False, 1),
        pytest.param(torch.uint16, (3072, 3072), False, True, False, 2),
        pytest.param(torch.uint16, (3072, 3072), False, False, True, 3),
        
        # fp8_x4 (uint32) tests - (I, H) shapes  
        pytest.param(torch.uint32, (2880, 2880), True, True, False, 4),
        pytest.param(torch.uint32, (2880, 2880), True, False, True, 5),
        pytest.param(torch.uint32, (3072, 3072), False, False, False, 6),
        pytest.param(torch.uint32, (3072, 3072), False, True, False, 7),
        
        # Multi-dimensional expert shapes (E, I, H) and (E, H, I), using E=4 for speed
        pytest.param(torch.uint16, (4, 2880, 2880), False, False, False, 8),
        pytest.param(torch.uint16, (4, 2880, 2880), True, False, False, 9),
        pytest.param(torch.uint16, (4, 2880, 2880), False, False, True, 10),
        pytest.param(torch.uint16, (4, 3072, 3072), True, True, False, 11),
        pytest.param(torch.uint16, (4, 3072, 3072), False, True, False, 12),
        pytest.param(torch.uint16, (4, 3072, 3072), True, False, True, 13),
    ]
)
def test_dequantize_mx_tensor(tensor_dtype, tensor_shape, input_is_transposed, output_is_transposed, output_quad_row, seed):
    # Use different manual seeds to get different random inputs
    torch.manual_seed(seed)
    
    # Scale shape: (*shape, M, K//8), K dim is block dim
    scale_shape = list(tensor_shape)
    if input_is_transposed:
        scale_shape[-2] = scale_shape[-2] // 8 
    else:
        scale_shape[-1] = scale_shape[-1] // 8
    scale = torch.randint(SCALE_MIN, SCALE_MAX, scale_shape, dtype=torch.uint8)
    tensor = torch.randint(*MIN_MAX_DTYPE_MAP[tensor_dtype], tensor_shape, dtype=tensor_dtype)
    
    # Remove NaNs for fp8_x4
    if tensor_dtype == torch.uint32:
        tensor = _replace_fp8_x4_nans(tensor)
    
    # Golden = base descale func
    target_dtype = torch.float32
    ref_tensor = tensor
    ref_scale = scale
    if input_is_transposed:
        ref_tensor = torch.transpose(tensor, -2, -1)
        ref_scale = torch.transpose(scale, -2, -1)
    
    # Reshape for block processing
    *shape, M, K = ref_tensor.shape
    blocks = ref_tensor.reshape((*shape, M, K//8, 8))

    if tensor_dtype == torch.uint16:
        expected = get_mxfp4_tensor_from_uint16(blocks, ref_scale, dtype=target_dtype, output_quad_row=output_quad_row)
    elif tensor_dtype == torch.uint32:
        expected = get_mxfp8_tensor_from_uint32(blocks, ref_scale, dtype=target_dtype, output_quad_row=output_quad_row)
    
    # Apply output transposes
    if output_quad_row:
        expected = torch.transpose(expected, -3, -2)
    elif output_is_transposed:
        expected = torch.transpose(expected, -2, -1)
    
    # Call wrapper descale func
    result = dequantize_mx_tensor(
        tensor, 
        scale, 
        target_dtype,
        input_is_transposed=input_is_transposed,
        output_is_transposed=output_is_transposed,
        output_quad_row=output_quad_row
    )

    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "tile0_shape,tile1_shape,tile0_dtype,tile1_dtype,seed,atol,rtol", [
        pytest.param((128, 128), (128, 512), torch.uint16, torch.uint16, 0, 0, 0),
        pytest.param((128, 128), (128, 512), torch.uint16, torch.uint32, 1, 4.763e-6, 5.39e-4),
        pytest.param((128, 128), (128, 512), torch.uint32, torch.uint32, 2, 2.9e-6, 1.82e-1),
    ]
)
@pytest.mark.skipif(not MX_IMPORTABLE, reason="Unable to import NKI private functions")
def test_matmul_mx_single_tile(tile0_shape, tile1_shape, tile0_dtype, tile1_dtype, seed, atol, rtol):
    # Use different manual seeds to get different random inputs
    torch.manual_seed(seed)

    DTYPE_MAP = {
        torch.uint16: NKI_FP4_X4,
        torch.uint32: NKI_FP8_X4,
    }

    # max tile shape for fp32 accumulation
    scale0_shape = (tile0_shape[0] // 8,) + tile0_shape[1:]
    scale1_shape = (tile1_shape[0] // 8,) + tile1_shape[1:]
    tile0 = torch.randint(*MIN_MAX_DTYPE_MAP[tile0_dtype], tile0_shape, dtype=tile0_dtype)
    tile1 = torch.randint(*MIN_MAX_DTYPE_MAP[tile1_dtype], tile1_shape, dtype=tile1_dtype)
    scale0 = torch.randint(SCALE_MIN, SCALE_MAX, scale0_shape, dtype=torch.uint8)
    scale1 = torch.randint(SCALE_MIN, SCALE_MAX, scale1_shape, dtype=torch.uint8)

    # remove nans that are generated through randomization on packed dtype
    if tile0_dtype == torch.uint32:
        tile0 = _replace_fp8_x4_nans(tile0)
    if tile1_dtype == torch.uint32:
        tile1 = _replace_fp8_x4_nans(tile1)

    # calculate golden using compiler golden func
    tile0_np = tile0.numpy().view(DTYPE_MAP[tile0_dtype])
    tile1_np = tile1.numpy().view(DTYPE_MAP[tile1_dtype])
    scale0_np = scale0.numpy()
    scale1_np = scale1.numpy()
    res_np = nc_matmul_mx_golden(stationary_x4=tile0_np, moving_x4=tile1_np, stationary_scale=scale0_np, moving_scale=scale1_np)

    # calculate torch res, use default fp32 output type to compare against numpy
    res_torch = matmul_mx_single_tile(tile0, tile1, scale0, scale1)

    torch.testing.assert_close(res_torch, torch.from_numpy(res_np), atol=atol, rtol=rtol)
    print("Tensors match!")


@pytest.mark.parametrize(
    "tensor0_shape,tensor1_shape,seed", [
        pytest.param((2880//4, 64), (2880//4, 2880), 0, id="t64"),
        pytest.param((2880//4, 128), (2880//4, 2880), 1, id="t128"),
        pytest.param((2880//4, 256), (2880//4, 2880), 2, id="t256"),
        pytest.param((2880//4, 512), (2880//4, 2880), 3, id="t512"),
    ]
)
@pytest.mark.parametrize(
    "tensor0_dtype,tensor1_dtype,accumulation_dtype,output_dtype,atol,rtol", [
        # NOTE: skips evaluating accumulate in bf16 + upcast fp32
        # fp32 accumulation + no downcast
        pytest.param(torch.uint16, torch.uint16, torch.float32, torch.float32, 0, 0, id="mxfp4@mxfp4_fp32_fp32"),
        pytest.param(torch.uint32, torch.uint16, torch.float32, torch.float32, 8.59e-6, 9.38e-2, id="mxfp8@mxfp4_fp32_fp32"),
        pytest.param(torch.uint32, torch.uint32, torch.float32, torch.float32, 4.28e-4, 7.5e-1, id="mxfp8@mxfp8_fp32_fp32"),

        # NOTE: we see larger divergence for non fp33+fp32 due to different accumulation dtypes + patterns on CPU + bf16 downcast
        # fp32 accumulation + downcast bf16
        pytest.param(torch.uint16, torch.uint16, torch.float32, torch.bfloat16, 0, 0, id="mxfp4@mxfp4_fp32_bf16"),
        pytest.param(torch.uint32, torch.uint16, torch.float32, torch.bfloat16, 6.25e-2, 9.38e-2, id="mxfp8@mxfp4_fp32_bf16"), # t64 is 1.6e-2, t128/256 are 3.2e-2 atol
        pytest.param(torch.uint32, torch.uint32, torch.float32, torch.bfloat16, 2.0, 7.5e-1, id="mxfp8@mxfp8_fp32_bf16"),

        # bf16 accumulation + no downcast
        pytest.param(torch.uint16, torch.uint16, torch.bfloat16, torch.bfloat16, 7.82e-3, 584.0, id="mxfp4@mxfp4_bf16_bf16"),
        pytest.param(torch.uint32, torch.uint16, torch.bfloat16, torch.bfloat16, 0.25, 2192.0, id="mxfp8@mxfp4_bf16_bf16"),
        pytest.param(torch.uint32, torch.uint32, torch.bfloat16, torch.bfloat16, 16.0, 16384.0, id="mxfp8@mxfp8_bf16_bf16"),
    ]
)
def test_matmul_mx(tensor0_shape, tensor1_shape, tensor0_dtype, tensor1_dtype, accumulation_dtype, output_dtype, seed, atol, rtol):
    # Use different manual seeds to get different random inputs
    torch.manual_seed(seed)
    
    scale0_shape = (tensor0_shape[0] // 8,) + tensor0_shape[1:]
    scale1_shape = (tensor1_shape[0] // 8,) + tensor1_shape[1:]    
    tensor0 = torch.randint(*MIN_MAX_DTYPE_MAP[tensor0_dtype], tensor0_shape, dtype=tensor0_dtype)
    tensor1 = torch.randint(*MIN_MAX_DTYPE_MAP[tensor1_dtype], tensor1_shape, dtype=tensor1_dtype)
    scale0 = torch.randint(SCALE_MIN, SCALE_MAX, scale0_shape, dtype=torch.uint8)
    scale1 = torch.randint(SCALE_MIN, SCALE_MAX, scale1_shape, dtype=torch.uint8)

    # Apply NaN fixing for uint32 tensors
    if tensor0_dtype == torch.uint32:
        tensor0 = _replace_fp8_x4_nans(tensor0)
    if tensor1_dtype == torch.uint32:
        tensor1 = _replace_fp8_x4_nans(tensor1)

    # calculate golden using torch bf16
    tensor0_descaled = dequantize_mx_tensor(tensor0, scale0, accumulation_dtype, input_is_transposed=True)
    tensor1_descaled = dequantize_mx_tensor(tensor1, scale1, accumulation_dtype, input_is_transposed=True, output_is_transposed=True)
    res_bf16 = torch.einsum("th,hi->ti", tensor0_descaled, tensor1_descaled).to(output_dtype)

    # calculate torch res
    res_mx_torch = matmul_mx(tensor0, tensor1, scale0, scale1, accumulation_dtype=accumulation_dtype, output_dtype=output_dtype)

    # TODO: finish validating this, determine tolerances 
    
    torch.testing.assert_close(res_mx_torch, res_bf16, atol=atol, rtol=rtol)
    print("Tensors match!")


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
