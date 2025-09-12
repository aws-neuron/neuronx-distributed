import pytest
import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

from neuronx_distributed.kernels.find_index import find_index_shard_rows
from torch_neuronx.utils import get_platform_target

def routing_setup_random_topk(T, top_k, E):
    '''
    Generates a random expert mask with the given configuration.
    Args:
        T: number of tokens
        top_k: top k experts to route a token to.
        E: number of experts
    Returns:
        expert_mask: (T, E) boolean array. If element (t,e) is 1, it indicates t-th token is assigned to the e-th expert.
        expert_index: (T, top_k) array. For each token, this contains the indices of the chosen expert.
    '''
    expert_mask = np.zeros((T, E), dtype=np.int32)
    expert_index = np.zeros((T, top_k), dtype=np.int32)
    for t in range(T):
        expert_ids = np.random.choice(E, size=top_k, replace=False)
        expert_mask[t, expert_ids] = 1
        expert_index[t] = expert_ids

    return expert_mask, expert_index


def get_capacity_dropping(T, top_k, E, cf):
    '''
    Gets the expert capacity with the given configurations.
    Args:
        T: number of tokens
        top_k: top k experts to route a token to.
        E: number of experts
        cf: capacity factor
    Returns:
        capacity
    '''
    return math.ceil((T * top_k) / E) * cf


def get_golden(mask, n_values_max):
    '''
    Gets the golden index-in-expert to index-in-sequence mapping for dropping.
    Args:
        mask: (E, T) boolean array. If element (e, t) is 1, it indicates t-th token is assigned to the e-th expert.
        n_values_max: maximum number of 1s to find per row in the mask.
    Returns:
        true_indices: (E, n_values_max) array,
            The element at index (e, i) indicates the index of the i-th 1 in the e-th row.
            If a row has fewer than n_values_max 1s, the corresponding positions are filled with -1.
        true_counts: int32 tensor of shape (E,)
    '''
    E, _ = mask.shape
    true_indices = -1 * np.ones((E, n_values_max), dtype=np.int32)
    true_counts = np.zeros((E), dtype=np.int32)
    for e in range(E):
        indices = np.nonzero(mask[e])[0]
        n_indices = min(indices.shape[0], n_values_max)
        true_indices[e, :n_indices] = indices[:n_indices]
        # return the actual number of ones, not the capacity cap.
        true_counts[e] = indices.shape[0]
    return true_indices, true_counts


@pytest.mark.parametrize("T,E,SP,EP,top_k,cf,expected_p99_lat_us", [
    # Dropping
    (2048, 256, 1, 64, 8, 1, 27),
    (8192, 256, 1, 64, 8, 1, 42),
    (16384, 256, 1, 64, 8, 1, 76),
    pytest.param(2048, 128, 1, 64, 1, 1, 0, marks=pytest.mark.xfail(reason="Kernel not generalized to case where values are too few")),
    (8192, 128, 1, 64, 1, 1, 34),
    (16384, 128, 1, 64, 1, 1, 46),
    # Dropless
    (2048, 256, 1, 64, 8, -1, 35),
    (8192, 256, 1, 64, 8, -1, 185),
    (16384, 256, 1, 64, 8, -1, 624),
    (2048, 128, 1, 64, 1, -1, 28),
    (8192, 128, 1, 64, 1, -1, 104),
    (16384, 128, 1, 64, 1, -1, 325),
])
def test_find_index(T,E,SP,EP,top_k,cf,expected_p99_lat_us):
    def nc(x): return x if get_platform_target() == "trn1" else nl.nc(x)

    np.random.seed(42)
    expert_mask, _ = routing_setup_random_topk(T, top_k, E)
    if cf > 0:
        num_tokens_per_expert = get_capacity_dropping(T, top_k, E, cf)
    else:
        num_tokens_per_expert = T
    assert E % EP == 0, f"Number of experts {E} not divisble by EP degree {EP}"
    E_local = E // EP
    T_local = T // SP
    expert_mask = expert_mask[:T_local, :E_local]
    n_programs = 1 if E_local == 1 else 2

    # (E, T)
    expert_mask = expert_mask.T

    # Get golden
    expected_true_indices, expected_true_counts = get_golden(expert_mask, num_tokens_per_expert)

    # Get kernel outputs
    actual_true_indices, actual_true_counts = find_index_shard_rows[(nc(n_programs),)](expert_mask, num_tokens_per_expert, num_tokens_per_expert)

    bench_func = nki.benchmark(warmup=20, iters=100)(find_index_shard_rows[(nc(n_programs),)])
    bench_func(expert_mask, num_tokens_per_expert, num_tokens_per_expert)
    p99_lat = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)

    assert np.allclose(actual_true_indices, expected_true_indices), f"Failed indices match for config T={T}, E={E}, SP={SP}, EP={EP}, top_k={top_k}, cf={cf}"
    assert np.allclose(actual_true_counts, expected_true_counts), f"Failed counts match for config T={T}, E={E}, SP={SP}, EP={EP}, top_k={top_k}, cf={cf}"
    lat_regression_threshold = 1.05
    assert p99_lat < lat_regression_threshold * expected_p99_lat_us, f"Failed p99 latency check for config T={T}, E={E}, SP={SP}, EP={EP}, top_k={top_k}, cf={cf}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

