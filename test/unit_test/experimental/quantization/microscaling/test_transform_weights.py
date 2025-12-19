import pytest
import math

import torch

from neuronx_distributed.quantization.microscaling.transform_weights import (
    FP4_VALUES, get_mxfp4_tensor, get_mxfp4_tensor_from_uint16, get_mxfp8_tensor_from_uint32, pack_byte_4bit_tensor, pack_fp4_x4_uint16, split_byte_4bit_tensor, split_gate_up, dequant_byte_4bit_tensor
)

UINT8_MIN, UINT8_MAX = 0, 0xFF # 2**8
UINT16_MIN, UINT16_MAX = 0, 0xFFFF # 2**16
UINT32_MIN, UINT32_MAX = 0, 0xFFFFFFFF # 2**32

def test_extract_gate_up_interleaved():
    """
    Validate that two methods of extracting gate/up from interleaved weights match:
    1. MXFP4->BF16 and then extract
    2. extract and then MXFP4->BF16
    """
    torch.manual_seed(0)
    
    GATE_UP_SHAPE = (12, 576, 9, 16)
    SCALE_SHAPE = (12, 576, 9)
    BIAS_SHAPE = (12, 576)
    SCALE_MIN, SCALE_MAX = 119, 123
    blocks = torch.randint(UINT8_MIN, UINT8_MAX, GATE_UP_SHAPE, dtype=torch.uint8)
    scales = torch.randint(SCALE_MIN, SCALE_MAX, SCALE_SHAPE, dtype=torch.uint8)
    bias = torch.rand(BIAS_SHAPE, dtype=torch.bfloat16)

    fused_output = get_mxfp4_tensor(blocks, scales)

    gate_oai = fused_output[:, ::2, :]
    up_oai = fused_output[:, 1::2, :]

    gate_fp4, scale_gate, _, up_fp4, scale_up, _ = split_gate_up(blocks, scales, bias)

    gate_strided = get_mxfp4_tensor(gate_fp4, scale_gate)
    up_strided = get_mxfp4_tensor(up_fp4, scale_up)

    assert torch.equal(gate_oai, gate_strided)
    assert torch.equal(up_oai, up_strided)
    print("Tensors match!")


def test_extract_fp4_x4():
    """
    Validate that packing to uint16+descaling to bf16 matches reference.
    """
    torch.manual_seed(0)

    W_SHAPE = (12, 288, 9, 16)
    SCALE_SHAPE = (12, 288, 9)
    SCALE_MIN, SCALE_MAX = 119, 123
    
    blocks = torch.randint(UINT8_MIN, UINT8_MAX, W_SHAPE, dtype=torch.uint8)
    scales = torch.randint(SCALE_MIN, SCALE_MAX, SCALE_SHAPE, dtype=torch.uint8)

    # Unpack fp4_x2/uint8 and descale to bf16
    descaled_golden = get_mxfp4_tensor(blocks, scales)

    # Pack fp4_x4 into uint16, then unpack and descale to bf16
    blocks_uint16 = pack_fp4_x4_uint16(blocks)
    descaled_from_uint16 = get_mxfp4_tensor_from_uint16(blocks_uint16, scales)

    assert torch.equal(descaled_golden, descaled_from_uint16)

    """Test get_mxfp4_tensor_from_uint16 with fixed input values"""
    # Create small tensor with known uint16 values
    blocks = torch.tensor([
        [[0x1234, 0x5678]],
        [[0x9ABC, 0xDEF0]]
    ], dtype=torch.uint16)
    
    # Fixed scale values
    scales = torch.tensor([
        [120],
        [122]
    ], dtype=torch.uint8)

    # Get output tensor
    result = get_mxfp4_tensor_from_uint16(blocks, scales)

    expected = torch.tensor([
        [0x4, 0x3, 0x2, 0x1, 0x8, 0x7, 0x6, 0x5],
        [0xC, 0xB, 0xA, 0x9, 0x0, 0xF, 0xE, 0xD]
    ], dtype=torch.long)
    lut = torch.tensor(FP4_VALUES, dtype=torch.bfloat16)
    expected = lut[expected] * 2.0**(scales.to(torch.int32)-127)
    expected = expected.to(torch.bfloat16)

    torch.testing.assert_close(result, expected)        
    
    print("Tensors match!")


def test_extract_fp4_x4_quad_row():
    """
    Validate correctness of unpacking fp4_x4 to quad row.
    """
    torch.manual_seed(0)

    W_SHAPE = (12, 288, 9, 8)
    W_SCALE_SHAPE = (12, 288, 9)
    IN_SHAPE = (12, 9, 8)
    IN_SCALE_SHAPE = (12, 9)
    SCALE_MIN, SCALE_MAX = 119, 123
    
    W_blocks_uint16 = torch.randint(UINT16_MIN, UINT16_MAX, W_SHAPE, dtype=torch.uint16)
    W_scales = torch.randint(SCALE_MIN, SCALE_MAX, W_SCALE_SHAPE, dtype=torch.uint8)
    in_blocks_uint16 = torch.randint(UINT16_MIN, UINT16_MAX, IN_SHAPE, dtype=torch.uint16)
    in_scales = torch.randint(SCALE_MIN, SCALE_MAX, IN_SCALE_SHAPE, dtype=torch.uint8)

    # Unpack fp4_x4/uint16 and descale to bf16 + accumulate over H dim
    W_descaled_golden = get_mxfp4_tensor_from_uint16(W_blocks_uint16, W_scales)
    in_descaled_golden = get_mxfp4_tensor_from_uint16(in_blocks_uint16, in_scales)
    res_golden = torch.einsum("eih,th->tei", W_descaled_golden, in_descaled_golden)

    # Unpack fp4_x4/uint16 in quad row format and descale to bf16 + accumulate over H/4 and quad dim 
    W_descaled_quad = get_mxfp4_tensor_from_uint16(W_blocks_uint16, W_scales, output_quad_row=True)
    in_descaled_quad = get_mxfp4_tensor_from_uint16(in_blocks_uint16, in_scales, output_quad_row=True)
    res_quad = torch.einsum("eihq,thq->tei", W_descaled_quad, in_descaled_quad)

    torch.testing.assert_close(res_golden, res_quad)
    print("Tensors match!")


def test_extract_fp8_x4_quad_row():
    """
    Validate correctness of unpacking fp8_x4 to quad row + matmul with mxfp4 weights.
    """
    torch.manual_seed(0)

    W_SHAPE = (12, 288, 9, 8)
    W_SCALE_SHAPE = (12, 288, 9)
    IN_SHAPE = (12, 9, 8)
    IN_SCALE_SHAPE = (12, 9)
    SCALE_MIN, SCALE_MAX = 119, 123
    
    W_blocks_uint16 = torch.randint(UINT16_MIN, UINT16_MAX, W_SHAPE, dtype=torch.uint16)
    W_scales = torch.randint(SCALE_MIN, SCALE_MAX, W_SCALE_SHAPE, dtype=torch.uint8)
    in_blocks_uint32 = torch.randint(UINT32_MIN, UINT32_MAX, IN_SHAPE, dtype=torch.uint32)
    in_scales = torch.randint(SCALE_MIN, SCALE_MAX, IN_SCALE_SHAPE, dtype=torch.uint8)

    # Unpack fp4_x4/fp8_x4 and descale to bf16 + accumulate over H dim
    W_descaled_golden = get_mxfp4_tensor_from_uint16(W_blocks_uint16, W_scales)
    in_descaled_golden = get_mxfp8_tensor_from_uint32(in_blocks_uint32, in_scales, replace_nan_with_zeros=True)
    res_golden = torch.einsum("eih,th->tei", W_descaled_golden, in_descaled_golden)

    # Unpack fp4_x4/uint16 in quad row format and descale to bf16 + accumulate over H/4 and quad dim 
    W_descaled_quad = get_mxfp4_tensor_from_uint16(W_blocks_uint16, W_scales, output_quad_row=True)
    in_descaled_quad = get_mxfp8_tensor_from_uint32(in_blocks_uint32, in_scales, replace_nan_with_zeros=True, output_quad_row=True)
    res_quad = torch.einsum("eihq,thq->tei", W_descaled_quad, in_descaled_quad)

    torch.testing.assert_close(res_golden, res_quad)
    print("Tensors match!") 


class TestSplitByte4bitTensor:

    def test_basic_functionality(self):
        """Test basic tensor splitting functionality"""
        # Create a simple 2D tensor with known values
        tensor = torch.tensor([[0x12, 0x34], [0x56, 0x78]], dtype=torch.uint8)
        
        result = split_byte_4bit_tensor(tensor)
        
        # Expected: split each byte into low and high nibbles
        # 0x12 -> low=0x02, high=0x01
        # 0x34 -> low=0x04, high=0x03
        expected = torch.tensor([
            [0x02, 0x01, 0x04, 0x03],  # First row
            [0x06, 0x05, 0x08, 0x07]   # Second row
        ], dtype=torch.uint8)
        
        assert torch.equal(result, expected)

    def test_single_element_tensor(self):
        """Test with single element tensor"""
        tensor = torch.tensor([[0xAB]], dtype=torch.uint8)
        
        result = split_byte_4bit_tensor(tensor)
        
        # 0xAB -> low=0x0B, high=0x0A
        expected = torch.tensor([[0x0B, 0x0A]], dtype=torch.uint8)
        
        assert torch.equal(result, expected)

    def test_3d_tensor(self):
        """Test with 3D tensor to verify prefix shape handling"""
        tensor = torch.tensor([[[0x12], [0x34]], [[0x56], [0x78]]], dtype=torch.uint8)
        
        result = split_byte_4bit_tensor(tensor)
        
        expected = torch.tensor([
            [[0x02, 0x01], [0x04, 0x03]],
            [[0x06, 0x05], [0x08, 0x07]]
        ], dtype=torch.uint8)
        
        assert torch.equal(result, expected)

    def test_chunking_behavior(self):
        """Test chunking with small rows_per_chunk"""
        # Create tensor larger than chunk size
        tensor = torch.randint(0, 256, (5, 3), dtype=torch.uint8)
        
        # Test with small chunk size
        result_chunked = split_byte_4bit_tensor(tensor, rows_per_chunk=2)
        result_normal = split_byte_4bit_tensor(tensor)
        
        # Results should be identical regardless of chunking
        assert torch.equal(result_chunked, result_normal)
    
    def test_output_shape(self):
        """Test that output shape is correct"""
        batch_size, seq_len, hidden_dim = 3, 4, 8
        tensor = torch.randint(0, 256, (batch_size, seq_len, hidden_dim), dtype=torch.uint8)
        
        result = split_byte_4bit_tensor(tensor)
        
        # Output should double the last dimension
        expected_shape = (batch_size, seq_len, hidden_dim * 2)
        assert result.shape == expected_shape
    
    def test_nibble_extraction(self):
        """Test correct nibble extraction"""
        # Test with known byte values
        tensor = torch.tensor([[0x00, 0x0F, 0xF0, 0xFF]], dtype=torch.uint8)
        
        result = split_byte_4bit_tensor(tensor)
        
        # 0x00 -> low=0x00, high=0x00
        # 0x0F -> low=0x0F, high=0x00  
        # 0xF0 -> low=0x00, high=0x0F
        # 0xFF -> low=0x0F, high=0x0F
        expected = torch.tensor([[0x00, 0x00, 0x0F, 0x00, 0x00, 0x0F, 0x0F, 0x0F]], dtype=torch.uint8)
        
        assert torch.equal(result, expected)
    
    def test_empty_tensor(self):
        """Test with empty tensor"""
        tensor = torch.empty((0, 5), dtype=torch.uint8)
        
        result = split_byte_4bit_tensor(tensor)
        
        assert result.shape == (0, 10)
        assert result.dtype == torch.uint8
    
    @pytest.mark.parametrize("rows_per_chunk", [1, 10, 100, 16384 * 512])
    def test_different_chunk_sizes(self, rows_per_chunk):
        """Test with different chunk sizes"""
        tensor = torch.randint(0, 256, (20, 6), dtype=torch.uint8)
        
        result = split_byte_4bit_tensor(tensor, rows_per_chunk=rows_per_chunk)
        
        # Should always produce same result regardless of chunk size
        expected = split_byte_4bit_tensor(tensor, rows_per_chunk=16384 * 512)
        assert torch.equal(result, expected)    


class TestPackByte4bitTensor:
    
    def test_basic_packing(self):
        """Test basic nibble packing functionality"""
        # Create tensor with known low/high nibble pairs
        tensor = torch.tensor([[[0x01, 0x02], [0x03, 0x04]]], dtype=torch.uint8)
        
        result = pack_byte_4bit_tensor(tensor)
        
        # Expected: (hi << 4) | lo = (0x01 << 4) | 0x02 = 0x12
        expected = torch.tensor([[[0x21], [0x43]]], dtype=torch.uint8)
        
        assert torch.equal(result, expected)

    def test_single_pair(self):
        """Test with single nibble pair"""
        tensor = torch.tensor([[0x0A, 0x0B]], dtype=torch.uint8)
        
        result = pack_byte_4bit_tensor(tensor)
        
        # (0x0A << 4) | 0x0B = 0xAB
        expected = torch.tensor([[0xBA]], dtype=torch.uint8)
        
        assert torch.equal(result, expected)
    
    def test_output_shape(self):
        """Test output shape is half the input last dimension"""
        tensor = torch.randint(0, 16, (2, 3, 8), dtype=torch.uint8)
        
        result = pack_byte_4bit_tensor(tensor)
        
        assert result.shape == (2, 3, 4)
    
    def test_chunking_consistency(self):
        """Test chunking produces identical results"""
        tensor = torch.randint(0, 16, (10, 6), dtype=torch.uint8)
        
        result_chunked = pack_byte_4bit_tensor(tensor, rows_per_chunk=3)
        result_normal = pack_byte_4bit_tensor(tensor)
        
        assert torch.equal(result_chunked, result_normal)
    
    def test_nibble_values(self):
        """Test with specific nibble values"""
        # Test edge cases: 0x00, 0x0F combinations
        tensor = torch.tensor([
            [[0x00, 0x00], [0x00, 0x0F], [0x0F, 0x00], [0x0F, 0x0F]]
        ], dtype=torch.uint8)
        
        result = pack_byte_4bit_tensor(tensor)
        
        expected = torch.tensor([[[0x00], [0xF0], [0x0F], [0xFF]]], dtype=torch.uint8)
        
        assert torch.equal(result, expected)
    
    def test_3d_tensor(self):
        """Test with 3D tensor"""
        tensor = torch.tensor([
            [[0x01, 0x02], [0x03, 0x04]],
            [[0x05, 0x06], [0x07, 0x08]]
        ], dtype=torch.uint8)
        
        result = pack_byte_4bit_tensor(tensor)
        
        expected = torch.tensor([
            [[0x21], [0x43]],
            [[0x65], [0x87]]
        ], dtype=torch.uint8)
        
        assert torch.equal(result, expected)
        assert result.shape == (2, 2, 1)
    
    @pytest.mark.parametrize("rows_per_chunk", [1, 5, 100])
    def test_different_chunk_sizes(self, rows_per_chunk):
        """Test with different chunk sizes"""
        tensor = torch.randint(0, 16, (8, 4), dtype=torch.uint8)
        
        result = pack_byte_4bit_tensor(tensor, rows_per_chunk=rows_per_chunk)
        expected = pack_byte_4bit_tensor(tensor, rows_per_chunk=16384 * 512)
        
        assert torch.equal(result, expected)
    
    def test_empty_tensor(self):
        """Test with empty tensor"""
        tensor = torch.empty((0, 4), dtype=torch.uint8)
        
        result = pack_byte_4bit_tensor(tensor)
        
        assert result.shape == (0, 2)
        assert result.dtype == torch.uint8


def test_split_bytes_pack_bytes_invertible():
    """Test that split_byte and pack_bytes are inverse operations"""
    # Test with random tensor
    original = torch.randint(0, 256, (3, 4, 6), dtype=torch.uint8)
    
    # Split then pack should return original
    split = split_byte_4bit_tensor(original)
    packed = pack_byte_4bit_tensor(split)
    
    assert torch.equal(packed, original)
    
    # Test with specific values
    original = torch.tensor([[[0x12, 0x34], [0xAB, 0xCD]]], dtype=torch.uint8)
    split = split_byte_4bit_tensor(original)
    packed = pack_byte_4bit_tensor(split)
    
    assert torch.equal(packed, original)


def test_split_bytes_get_mxfp4_tensor():
    W_SHAPE = (12, 288, 9, 16)
    SCALE_SHAPE = (12, 288, 9)
    SCALE_MIN, SCALE_MAX = 119, 123
    
    blocks = torch.randint(UINT8_MIN, UINT8_MAX, W_SHAPE, dtype=torch.uint8)
    scales = torch.randint(SCALE_MIN, SCALE_MAX, SCALE_SHAPE, dtype=torch.uint8)

    expected = get_mxfp4_tensor(blocks, scales)

    blocks = split_byte_4bit_tensor(blocks).to(torch.long)
    actual = dequant_byte_4bit_tensor(blocks, scales)

    torch.testing.assert_close(actual, expected)


def test_split_bytes_get_mxfp4_tensor_from_uint16():
    W_SHAPE = (12, 288, 9, 16)
    SCALE_SHAPE = (12, 288, 9)
    SCALE_MIN, SCALE_MAX = 119, 123
    
    blocks = torch.randint(UINT8_MIN, UINT8_MAX, W_SHAPE, dtype=torch.uint8)
    scales = torch.randint(SCALE_MIN, SCALE_MAX, SCALE_SHAPE, dtype=torch.uint8)

    # Unpack fp4_x2/uint8 and descale to bf16
    descaled_golden = get_mxfp4_tensor(blocks, scales)

    # Pack fp4_x4 into uint16, then unpack and descale to bf16
    blocks_uint16 = pack_fp4_x4_uint16(blocks)
    expected_uint16 = get_mxfp4_tensor_from_uint16(blocks_uint16, scales)    

    # Using split_byte
    blocks = split_byte_4bit_tensor(blocks).to(torch.long)
    actual = dequant_byte_4bit_tensor(blocks, scales)

    torch.testing.assert_close(actual, expected_uint16)
    torch.testing.assert_close(actual, descaled_golden)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
