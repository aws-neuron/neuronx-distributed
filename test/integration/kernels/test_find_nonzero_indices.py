import pytest
import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

from neuronx_distributed.kernels.find_nonzero_indices import find_nonzero_indices
from torch_neuronx.utils import get_platform_target

def golden_find_nonzero_indices(arr):
    indices = []
    for index, value in enumerate(arr):
        if value != 0:
            indices.append(index)
    
    num_pad = len(arr) - len(indices)
    for _ in range(num_pad):
        indices.append(-1)
    
    return np.array(indices)

@pytest.mark.parametrize("T,E,top_k,E_offset,E_local,chunk_size,in_dtype,expected_p99_lat_us", [
    (4096, 128, 8, 0, 4, 4096, np.int32, 70),
    (4096, 128, 8, 0, 4, 4096, np.float32, 70),
    (10240, 128, 4, 16, 16, 10240, np.int32, 150),
    (10240, 128, 4, 16, 16, 10240, np.float32, 150),
    (65536, 128, 4, 32, 16, 16384, np.int32, 750),
    (65536, 128, 4, 32, 16, 16384, np.float32, 750),
])
def test_count_nonzeros_shard_rows(
    T: int,
    E: int,
    top_k: int,
    E_offset: int,
    E_local: int,
    chunk_size: int,
    in_dtype: np.dtype,
    expected_p99_lat_us: int,
):
    assert get_platform_target() != "trn1", "Kernel is not supported on TRN1"
    np.random.seed(42)
    
    # Prepare input
    K = T * top_k // E
    flattened_input = np.random.permutation(np.concatenate([np.ones(K * E), np.zeros((T-K)*E)]))
    input = flattened_input.reshape(E,T).astype(in_dtype)
    
    # Construct goldens
    expected_indices = np.zeros((E_local, T), dtype=in_dtype)
    expected_nonzero_counts = np.zeros((E_local,), dtype=in_dtype)
    for e in range(E_local):
        expected_indices[e] = golden_find_nonzero_indices(input[e + E_offset].tolist())
        expected_nonzero_counts[e] = np.sum(input[e + E_offset])

    # If input is float, set values to a non-integer.
    if in_dtype == np.float32:
        input[input > 0] = 0.1

    # Get kernel outputs
    actual_indices, actual_nonzero_counts = find_nonzero_indices[nl.nc(2)](
        input_tensor=input.T,
        row_start_id=np.array([E_offset]).astype(np.int32),
        n_rows=E_local,
        chunk_size = chunk_size,
        index_dtype=nl.int32,
    )

    # Run benchmark
    bench_func = nki.benchmark(warmup=20, iters=100)(find_nonzero_indices[nl.nc(2)])
    bench_func(
        input_tensor=input.T,
        row_start_id=np.array([E_offset]).astype(np.int32),
        n_rows=E_local,
        chunk_size = chunk_size,
        index_dtype=nl.int32,
    )
    p99_lat = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)

    assert np.allclose(actual_indices, expected_indices), \
        f"Failed indices match for config {T=}, {E=}, {top_k=}, {E_offset=}, {E_local=}, {chunk_size=}, {in_dtype=}"
    assert np.allclose(actual_nonzero_counts, expected_nonzero_counts), \
        f"Failed counts match for config {T=}, {E=}, {top_k=}, {E_offset=}, {E_local=}, {chunk_size=}, {in_dtype=}"
    lat_regression_threshold = 1.05
    assert p99_lat < lat_regression_threshold * expected_p99_lat_us, \
        f"Failed latency assertion for config {T=}, {E=}, {top_k=}, {E_offset=}, {E_local=}, {chunk_size=}, {in_dtype=}"\
        f"expected latency: {expected_p99_lat_us}, actual latency: {p99_lat}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

