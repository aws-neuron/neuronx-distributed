import pytest
import numpy as np
import math

import torch
import nki
import nki.language as nl

from nkilib.core.subkernels.find_nonzero_indices import find_nonzero_indices
from torch_neuronx.utils import get_platform_target

NP_TO_TORCH_DTYPE = {
    np.int32: torch.int32,
    np.float32: torch.float32,
}

def golden_find_nonzero_indices(arr):
    indices = []
    for index, value in enumerate(arr):
        if value != 0:
            indices.append(index)
    
    num_pad = len(arr) - len(indices)
    for _ in range(num_pad):
        indices.append(-1)
    
    return np.array(indices)

@pytest.mark.parametrize("T,E,top_k,E_offset,E_local,chunk_size,in_dtype", [
    (4096, 128, 8, 0, 4, 4096, np.int32),
    (4096, 128, 8, 0, 4, 4096, np.float32),
    (10240, 128, 4, 16, 16, 10240, np.int32),
    (10240, 128, 4, 16, 16, 10240, np.float32),
    (65536, 128, 4, 32, 16, 16384, np.int32),
    (65536, 128, 4, 32, 16, 16384, np.float32),
    (10240, 128, 4, 120, 8, 10240, np.float32),  # rank 15: E_offset=120
])
def test_count_nonzeros_shard_rows(
    T: int,
    E: int,
    top_k: int,
    E_offset: int,
    E_local: int,
    chunk_size: int,
    in_dtype: np.dtype,
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

    # Convert inputs to torch tensors on XLA device for NKI backend
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    torch_dtype = NP_TO_TORCH_DTYPE[in_dtype]
    input_tensor = torch.from_numpy(input.T.copy()).to(torch_dtype).to(device)
    col_start_id = torch.tensor([E_offset], dtype=torch.int32).to(device)

    # Get kernel outputs (use nki.language dtype, not numpy)
    actual_indices_t, actual_nonzero_counts_t = find_nonzero_indices[2](
        input_tensor=input_tensor,
        col_start_id=col_start_id,
        n_cols=E_local,
        chunk_size=chunk_size,
        index_dtype=nl.int32,
    )
    actual_indices = actual_indices_t.cpu().numpy() if torch.is_tensor(actual_indices_t) else np.asarray(actual_indices_t)
    actual_nonzero_counts = actual_nonzero_counts_t.cpu().numpy() if torch.is_tensor(actual_nonzero_counts_t) else np.asarray(actual_nonzero_counts_t)

    assert np.allclose(actual_indices, expected_indices), \
        f"Failed indices match for config {T=}, {E=}, {top_k=}, {E_offset=}, {E_local=}, {chunk_size=}, {in_dtype=}"
    assert np.allclose(actual_nonzero_counts, expected_nonzero_counts), \
        f"Failed counts match for config {T=}, {E=}, {top_k=}, {E_offset=}, {E_local=}, {chunk_size=}, {in_dtype=}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
