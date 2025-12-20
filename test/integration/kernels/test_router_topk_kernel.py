import pytest
import numpy as np
import neuronxcc.nki as nki
from neuronxcc.nki.language import nc

from neuronx_distributed.utils.model_utils import get_platform_lnc
from neuronx_distributed.kernels.router_topk_kernel import rmsnorm_router_top_k_with_masks_kernel


def setup_random_router_data(T, H, E):
    """
    Generates random router input data with the given configuration.
    Args:
        T: number of tokens
        H: hidden dimension
        E: number of experts
        dtype_hidden: data type for hidden states
        dtype_weights: data type for router weights
    Returns:
        hidden_states: (T, H) tensor
        router_weights: (E, H) tensor  
        router_bias: (E,) tensor
    """
    hidden_states = np.random.uniform(-0.05, 0.05, (T, H)).astype(np.float16)
    router_weight = np.random.uniform(-0.05, 0.05, (E, H)).astype(np.float32)
    router_bias = np.random.uniform(-0.05, 0.05, (1, E)).astype(np.float16)
    rms_weight = np.random.uniform(-0.05, 0.05, (H)).astype(np.float32)
    residual = np.random.uniform(-0.05, 0.05, (T, H)).astype(np.float16)
    return hidden_states, residual, router_weight, rms_weight, router_bias


def get_golden_router_topk(hidden_states, router_weight, rmsnorm_weight, router_bias, top_k, act_fn, topk_first, eps = 1e-6):
    """
    Gets the golden reference implementation for router top-k computation.
    Args:
        hidden_states: (T, H) tensor
        router_weights: (E, H) tensor
        router_bias: (E,) tensor
        top_k: number of top experts to select
        act_fn: activation function ("sigmoid" or "softmax")
        topk_first: whether to apply top-k before activation
    Returns:
        router_logits: (T, E) tensor
        expert_affinities: (T, top_k) or (T, E) tensor depending on topk_first
        expert_index: (T, top_k) tensor
        expert_mask: (T, E) tensor
        expert_affinities_masked: (T, E) tensor
    """
    residual = hidden_states
    hidden_states = hidden_states.astype(np.float32)
        
    # Compute RMS over the last dimension
    variance = np.mean(hidden_states ** 2, axis=-1, keepdims=True)
    rms_scale = np.reciprocal(np.sqrt(variance + eps))
    
    weighted_hidden_states = hidden_states * rmsnorm_weight.astype(np.float32)

    new_hidden_states = rms_scale * weighted_hidden_states
    hidden_states = new_hidden_states
    # Matrix multiplication
    router_logits = np.matmul(hidden_states.astype(np.float32), router_weight.T.astype(np.float32))
    if router_bias is not None:
        router_logits = router_logits + router_bias.reshape(1, -1).astype(np.float32)

    # Top-k selection
    T, E = router_logits.shape
    expert_index = np.zeros((T, top_k), dtype=np.int32)
    
    for i in range(T):
        # Get indices of top_k values
        expert_index[i] = np.argsort(router_logits[i])[-top_k:][::-1]
    
    if topk_first:
        # Extract top-k values
        router_logits_value = np.zeros((T, top_k), dtype=np.float32)
        for i in range(T):
            for k in range(top_k):
                router_logits_value[i, k] = router_logits[i, expert_index[i, k]]
        
        # Apply activation
        if act_fn == "sigmoid":
            expert_affinities = 1 / (1 + np.exp(-router_logits_value))
        elif act_fn == "softmax":
            exp_values = np.exp(router_logits_value - np.max(router_logits_value, axis=1, keepdims=True))
            expert_affinities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    else:
        # Apply activation to all logits
        if act_fn == "sigmoid":
            expert_affinities = 1 / (1 + np.exp(-router_logits))
        elif act_fn == "softmax":
            exp_values = np.exp(router_logits - np.max(router_logits, axis=1, keepdims=True))
            expert_affinities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # Create expert mask
    expert_mask = np.zeros((T, E), dtype=np.float32)
    for i in range(T):
        for k in range(top_k):
            expert_mask[i, expert_index[i, k]] = 1.0
    
    # Create masked affinities
    if topk_first:
        expert_affinities_masked = np.zeros((T, E), dtype=np.float32)
        for i in range(T):
            for k in range(top_k):
                expert_affinities_masked[i, expert_index[i, k]] = expert_affinities[i, k]
    else:
        expert_affinities_masked = expert_affinities * expert_mask
    
    return residual, new_hidden_states, router_logits, expert_affinities, expert_index, expert_mask, expert_affinities_masked


@pytest.mark.parametrize("T, H, E, top_k, act_fn, bias, topk_first, expected_p99_lat_us", [
    # topk_first=True cases
    (32, 5120, 16, 1, "sigmoid", True, True, 50),
    (256, 5120, 16, 4, "sigmoid", True, True, 150),
    (256, 3072, 128, 4, "softmax", False, True, 150),
    (1024, 5120, 16, 1, "softmax", True, True, 600),
    (1024, 3072, 128, 4, "sigmoid", True, True, 600),

    # topk_first=False cases
    (128, 5120, 16, 1, "sigmoid", True, False, 100),
    (128, 3072, 16, 4, "softmax", False, False, 100),
    (1024, 5120, 16, 1, "sigmoid", True, False, 600),
    (1024, 3072, 16, 4, "softmax", False, False, 600),
])
def test_router_topk_kernel(T, H, E, top_k, act_fn, bias, topk_first, expected_p99_lat_us):
    """Test router top-k kernel with various configurations."""
    np.random.seed(0)
    grid = (nc(get_platform_lnc().value),)
    hidden_states, residual, router_weight, rmsnorm_weight, router_bias = setup_random_router_data(T, H, E)
    if not bias:
        router_bias = None

    # Get golden reference
    residual_golden, new_hidden_states_golden, router_logits_golden, expert_affinities_golden, expert_index_golden, expert_mask_golden, expert_affinities_masked_golden = get_golden_router_topk(
        hidden_states=hidden_states+residual, 
        router_weight=router_weight, 
        rmsnorm_weight=rmsnorm_weight, 
        router_bias=router_bias, 
        top_k=top_k, 
        act_fn=act_fn, 
        topk_first=topk_first)

    # Run kernel
    residual, new_hidden_states, router_logits, expert_affinities, expert_index, expert_affinities_masked = rmsnorm_router_top_k_with_masks_kernel[grid](
        hidden_states=hidden_states,
        residual=residual,
        router_weight=router_weight, 
        rmsnorm_weight=rmsnorm_weight,
        router_bias=router_bias, 
        top_k=top_k, 
        act_fn=act_fn, 
        topk_first=topk_first)

    bench_func = nki.benchmark(warmup=20, iters=100)(rmsnorm_router_top_k_with_masks_kernel[grid])
    bench_func(
        hidden_states=hidden_states,
        residual=residual,
        router_weight=router_weight, 
        rmsnorm_weight=rmsnorm_weight,
        router_bias=router_bias, 
        top_k=top_k, 
        act_fn=act_fn, 
        topk_first=topk_first)
    
    p99_lat = bench_func.benchmark_result.nc_latency.get_latency_percentile(99)

    # Assertions
    assert np.allclose(residual, residual_golden, atol=1e-4, rtol=1e-2), \
        f"Failed residual match for config T={T}, H={H}, E={E}, top_k={top_k}, act_fn={act_fn}"
    
    assert np.allclose(new_hidden_states, new_hidden_states_golden, atol=1e-4, rtol=1e-2), \
        f"Failed residual match for config T={T}, H={H}, E={E}, top_k={top_k}, act_fn={act_fn}"
    
    assert np.allclose(router_logits, router_logits_golden, atol=1e-4, rtol=1e-2), \
        f"Failed router_logits match for config T={T}, H={H}, E={E}, top_k={top_k}, act_fn={act_fn}"

    assert np.allclose(expert_affinities, expert_affinities_golden, atol=1e-4, rtol=1e-2), \
        f"Failed expert_affinities match for config T={T}, H={H}, E={E}, top_k={top_k}, act_fn={act_fn}"

    assert np.allclose(expert_affinities_masked, expert_affinities_masked_golden, atol=1e-4, rtol=1e-2), \
        f"Failed expert_affinities_masked match for config T={T}, H={H}, E={E}, top_k={top_k}, act_fn={act_fn}"
    
    assert np.array_equal(expert_index, expert_index_golden), \
        f"Failed expert_index match for config T={T}, H={H}, E={E}, top_k={top_k}, act_fn={act_fn}"

    # Performance regression check
    lat_regression_threshold = 1.05
    assert p99_lat < lat_regression_threshold * expected_p99_lat_us, \
        f"Failed p99 latency check for config T={T}, H={H}, E={E}, top_k={top_k}, act_fn={act_fn}. " \
        f"Expected: {expected_p99_lat_us}us, Got: {p99_lat}us"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
