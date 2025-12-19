import pytest
import numpy as np
import math
from typing import List, Optional

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

from neuronx_distributed.kernels.indexed_flatten import indexed_flatten
from torch_neuronx.utils import get_platform_target


@pytest.mark.parametrize("T,E,f_len,flattened_offsets,flattened_offsets_start,output_len,padding_val,expected_p99_lat_us", [
    (4096, 1, 128, [2048,], 0, 8192, -1, 30),
    (4096, 2, 128, [0, 2048,], 0, 8192, -1, 30),
    (4096, 2, 256, [0, 2048,], 0, 8192, -1, 30),
    (4096, 3, 128, [0, 2048, 4096], 0, 10240, -1, 35),
    (10240, 16, 128, [512 * i for i in range(16)], 0, 20480, -1, 60),
    (10240, 17, 128, [512 * i for i in range(17)], 0, 20480, -1, 60),
    (65536, 4, 128, [1024 * i for i in range(4)], 0, 69120, -1, 40),
    (65536, 4, 256, [1024 * i for i in range(4)], 0, 69120, -1, 40),
])
def test_indexed_flatten(
    T: int,
    E: int,
    f_len: int,
    flattened_offsets: List[int],
    flattened_offsets_start: int,
    output_len: int,
    padding_val: int,
    expected_p99_lat_us: int,
):
    assert get_platform_target() != "trn1", "Kernel is not supported on TRN1"
    
    # Prepare input and golden
    input = np.zeros((E, T), dtype=np.int32)
    expected_output = np.full((output_len,), padding_val, dtype=np.int32)
    
    for e in range(E):
        input[e] = flattened_offsets_start + e
        expected_output[flattened_offsets[flattened_offsets_start + e]:flattened_offsets[flattened_offsets_start + e]+T] = flattened_offsets_start + e

    # Get kernel outputs
    actual_output = indexed_flatten[nl.nc(2)](
        input_tensor=input,
        f_len=f_len,
        output_len=output_len,
        row_offsets=(np.array(flattened_offsets) // f_len).astype(np.int32),
        row_offsets_start=np.array([flattened_offsets_start // f_len]).astype(np.int32),
    )

    # Run benchmark
    bench_func = nki.benchmark(warmup=20, iters=100)(indexed_flatten[nl.nc(2)])
    bench_func(
        input_tensor=input,
        f_len=f_len,
        output_len=output_len,
        row_offsets=(np.array(flattened_offsets) // f_len).astype(np.int32),
        row_offsets_start=np.array([flattened_offsets_start // f_len]).astype(np.int32),
        padding_val=padding_val,
    )
    p99_lat = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)

    assert np.allclose(actual_output, expected_output), \
        f"Failed indices match for config {T=}, {E=}, {f_len=}, {flattened_offsets=}, {flattened_offsets_start=}, {output_len=}, {padding_val=}"
    lat_regression_threshold = 1.05
    assert p99_lat < lat_regression_threshold * expected_p99_lat_us, \
        f"Failed latency assertion for config {T=}, {E=}, {f_len=}, {flattened_offsets=}, {flattened_offsets_start=}, {output_len=}, {padding_val=}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
