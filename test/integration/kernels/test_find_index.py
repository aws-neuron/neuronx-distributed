import pytest
import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

from neuronx_distributed.kernels.find_index import find_index_shard_rows, count_nonzeros_shard_rows
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


@pytest.mark.parametrize("T,E,E_local,sliced_mask,bool_mask,expected_p99_lat_us", [
    (2048, 256, 4, True, True, 20),
    (8192, 256, 4, True, True, 30),
    (16384, 256, 4, True, True, 40),
    (2048, 128, 8, True, True, 20),
    (8192, 128, 8, True, True, 30),
    (16384, 128, 8, True, True, 45),
    (2048, 256, 4, False, False, 23),
    (8192, 256, 4, False, False, 35),
    (16384, 256, 4, False, False, 50),
    (2048, 128, 8, False, False, 25),
    (8192, 128, 8, False, False, 35),
    (16384, 128, 8, False, False, 55),
])
def test_count_nonzeros_shard_rows(
    T: int,
    E: int,
    E_local: int,
    sliced_mask: bool,
    bool_mask: bool,
    expected_p99_lat_us: int,
):
    np.random.seed(42)
    expert_mask, _ = routing_setup_random_topk(T, 5, E)
    expert_mask = expert_mask.astype(np.float32)

    expert_mask_sliced = expert_mask[:, :E_local]
    expert_mask_sliced = expert_mask_sliced.T
    expert_mask = expert_mask.T

    # Get golden
    expected_counts = np.sum(expert_mask_sliced, axis=1)

    # Prepare kernel inputs
    kernel_args = {}
    if sliced_mask:
        kernel_args["mask"] = expert_mask_sliced
    else:
        kernel_args["mask"] = expert_mask
        kernel_args["row_start_id"] = np.array([0]).astype(np.int32)
        kernel_args["n_rows"] = E_local
    if not bool_mask:
        # simulate passing expert affinities directly to kernel and have the kernel handle boolean conversion
        kernel_args["mask"][kernel_args["mask"] > 0] = 0.1
        kernel_args["boolean_mask"] = False

    # Get kernel outputs
    actual_counts = count_nonzeros_shard_rows[nl.nc(2)](**kernel_args)

    bench_func = nki.benchmark(warmup=20, iters=100)(count_nonzeros_shard_rows[nl.nc(2)])
    bench_func(**kernel_args)
    p99_lat = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)

    assert np.allclose(actual_counts, expected_counts), f"Failed indices match for config T={T}, E={E}, E_local={E_local}, sliced_mask={sliced_mask}, bool_mask={bool_mask}"
    lat_regression_threshold = 1.05
    assert p99_lat < lat_regression_threshold * expected_p99_lat_us, f"Failed indices match for config T={T}, E={E}, E_local={E_local}, sliced_mask={sliced_mask}, bool_mask={bool_mask}"


@pytest.mark.parametrize("T,E,SP,EP,top_k,cf,sliced_mask,bool_mask,expected_p99_lat_us", [
    # Dropping, sliced boolean expert mask
    (2048, 256, 1, 64, 8, 1, True, True, 27),
    (8192, 256, 1, 64, 8, 1, True, True, 42),
    (16384, 256, 1, 64, 8, 1, True, True, 76),
    pytest.param(2048, 128, 1, 64, 1, 1, True, True, 0, marks=pytest.mark.xfail(reason="Kernel not generalized to case where values are too few")),
    (8192, 128, 1, 64, 1, 1, True, True, 34),
    (16384, 128, 1, 64, 1, 1, True, True, 46),
    # Dropping, full expert affinities, mask conversion in kernel
    (2048, 256, 1, 64, 8, 1, False, False, 36),
    (8192, 256, 1, 64, 8, 1, False, False, 49),
    (16384, 256, 1, 64, 8, 1, False, False, 81),
    pytest.param(2048, 128, 1, 64, 1, 1, False, False, 0, marks=pytest.mark.xfail(reason="Kernel not generalized to case where values are too few")),
    (8192, 128, 1, 64, 1, 1, False, False, 39),
    (16384, 128, 1, 64, 1, 1, False, False, 52),
    # Dropless, sliced boolean expert mask
    (2048, 256, 1, 64, 8, -1, True, True, 35),
    (8192, 256, 1, 64, 8, -1, True, True, 185),
    (16384, 256, 1, 64, 8, -1, True, True, 624),
    (2048, 128, 1, 64, 1, -1, True, True, 28),
    (8192, 128, 1, 64, 1, -1, True, True, 104),
    (16384, 128, 1, 64, 1, -1, True, True, 325),
    # Dropless, full expert affinities, mask conversion in kernel
    (2048, 256, 1, 64, 8, -1, False, False, 44),
    (8192, 256, 1, 64, 8, -1, False, False, 192),
    (16384, 256, 1, 64, 8, -1, False, False, 628),
    (2048, 128, 1, 64, 1, -1, False, False, 33),
    (8192, 128, 1, 64, 1, -1, False, False, 109),
    (16384, 128, 1, 64, 1, -1, False, False, 333),
])
def test_find_index_shard_rows(
    T: int,
    E: int,
    SP: int,
    EP: int,
    top_k: int,
    cf: float,
    sliced_mask: bool,
    bool_mask: bool,
    expected_p99_lat_us: int,
):
    assert SP == 1, "Kernel is expected to run in pure EP as of now"
    def nc(x): return x if get_platform_target() == "trn1" else nl.nc(x)

    np.random.seed(42)
    expert_mask, _ = routing_setup_random_topk(T, top_k, E)
    expert_mask = expert_mask.astype(np.float32)
    if cf > 0:
        num_tokens_per_expert = get_capacity_dropping(T, top_k, E, cf)
    else:
        num_tokens_per_expert = T
    assert E % EP == 0, f"Number of experts {E} not divisble by EP degree {EP}"
    E_local = E // EP
    T_local = T // SP
    n_programs = 1 if E_local == 1 else 2

    expert_mask_sliced = expert_mask[:T_local, :E_local]
    expert_mask_sliced = expert_mask_sliced.T
    expert_mask = expert_mask.T

    # Get golden
    expected_true_indices, expected_true_counts = get_golden(expert_mask_sliced, num_tokens_per_expert)

    # Prepare kernel inputs
    kernel_args = {
        "n_values_max": num_tokens_per_expert,
        "n_values_min": num_tokens_per_expert,
    }
    if sliced_mask:
        kernel_args["mask"] = expert_mask_sliced
    else:
        kernel_args["mask"] = expert_mask
        kernel_args["row_start_id"] = np.array([0]).astype(np.int32)
        kernel_args["n_rows"] = E_local
    if not bool_mask:
        # simulate passing expert affinities directly to kernel and have the kernel handle boolean conversion
        kernel_args["mask"][kernel_args["mask"] > 0] = 0.1
        kernel_args["boolean_mask"] = False

    # Get kernel outputs
    actual_true_indices, actual_true_counts = find_index_shard_rows[(nc(n_programs),)](**kernel_args)

    bench_func = nki.benchmark(warmup=20, iters=100)(find_index_shard_rows[(nc(n_programs),)])
    bench_func(**kernel_args)
    p99_lat = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)

    assert np.allclose(actual_true_indices, expected_true_indices), f"Failed indices match for config T={T}, E={E}, SP={SP}, EP={EP}, top_k={top_k}, cf={cf}, sliced_mask={sliced_mask}, bool_mask={bool_mask}"
    assert np.allclose(actual_true_counts, expected_true_counts), f"Failed counts match for config T={T}, E={E}, SP={SP}, EP={EP}, top_k={top_k}, cf={cf}, sliced_mask={sliced_mask}, bool_mask={bool_mask}"
    lat_regression_threshold = 1.05
    assert p99_lat < lat_regression_threshold * expected_p99_lat_us, f"Failed p99 latency check for config T={T}, E={E}, SP={SP}, EP={EP}, top_k={top_k}, cf={cf}, sliced_mask={sliced_mask}, bool_mask={bool_mask}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

