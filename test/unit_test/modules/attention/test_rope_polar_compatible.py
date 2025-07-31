import logging

import pytest
import torch
from transformers.models.llama4.modeling_llama4 import apply_rotary_emb
from neuronx_distributed.utils.random import set_random_seed
from neuronx_distributed.modules.attention.utils import precompute_freqs_cis, apply_rotary_polar_compatible

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
set_random_seed(0)

@pytest.mark.parametrize(
    "dtype, num_chunks, seq_len, n_local_heads, head_dim",
    [
        pytest.param(
            dtype, num_chunks, seq_len, n_local_heads, head_dim,
            id=f"dtype_{str(dtype).split('.')[-1]}_xshape_{num_chunks}_{seq_len}_{n_local_heads}_{head_dim}",
        )
        for dtype in [torch.float32, torch.float16, torch.bfloat16]
        for num_chunks, seq_len, n_local_heads, head_dim in [
            (8, 577, 16, 88),
            (5, 1024, 16, 80)
        ]
    ],
)
def test_rotary_polar_compatible(dtype, num_chunks, seq_len, n_local_heads, head_dim):
    # inputs
    xq = torch.randn(
        [
            num_chunks,
            seq_len,
            n_local_heads,
            head_dim,
        ],
        dtype=dtype,
    )  # (bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = torch.randn_like(xq)  # (bsz, seqlen, self.n_local_heads, self.head_dim)

    neuron_freqs = precompute_freqs_cis(head_dim, seq_len, theta=10000.0).unsqueeze(0)
    original_freq_cis = torch.view_as_complex(
        torch.stack([torch.cos(neuron_freqs), torch.sin(neuron_freqs)], dim=-1)
    )

    logger.info(
        f"Got original_freq_cis {original_freq_cis.shape} {original_freq_cis}"
    )  # torch.Size([1, 577, 44])
    logger.info(f"Got neuron_freqs {neuron_freqs.shape} {neuron_freqs}")  # torch.Size([1, 577, 44])
    expected_xq, _ = apply_rotary_emb(xq, xk, freqs_cis=original_freq_cis)

    neuron_xq, _ = apply_rotary_polar_compatible(xq, xk, neuron_freqs)

    torch.testing.assert_close(neuron_xq, expected_xq)
    logger.info("Golden and Neuron outputs match!")
    