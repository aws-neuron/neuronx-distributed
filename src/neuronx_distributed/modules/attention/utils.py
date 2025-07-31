"""Utility functions for neuronx_distributed.modules.attention

This module contains helper functions for the attention mechanisms,
particularly the RoPE (Rotary Position Embeddings) implementation
with frequency scaling techniques. These utilities enable extension
of effective context length beyond the original training window.
"""

import math
import torch

# Values obtained from grid search, specifically for Llama3.2 MM PyTorch Implementation
ROPE_DEFAULTS = {
    'factor': 8,
    'low_freq_factor': 1,
    'high_freq_factor': 4,
    'original_max_position_embeddings': 8192
}

def apply_scaling(freqs: torch.Tensor, **kwargs):
    
    params = {**ROPE_DEFAULTS, **kwargs}

    low_freq_wavelen = params['original_max_position_embeddings'] / params['low_freq_factor']
    high_freq_wavelen = params['original_max_position_embeddings'] / params['high_freq_factor']
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / params['factor'])
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (params['original_max_position_embeddings'] / wavelen - params['low_freq_factor']) / (
                params['high_freq_factor'] - params['low_freq_factor']
            )
            new_freqs.append((1 - smooth) * freq / params['factor'] + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False, device=None, **kwargs):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs, **kwargs)
    freqs = torch.outer(t, freqs)
    return freqs

def apply_rotary_polar_compatible(query, key, freqs_cis):
    # Ensure freqs_cis is in FP32 for accuracy
    if freqs_cis.dtype != torch.float32:
        raise ValueError(
            f"Expect freqs_cis.dtype == torch.float32 to ensure accuracy, got {freqs_cis.dtype}"
        )

    freqs_cis_real = freqs_cis.cos().unsqueeze(2)
    freqs_cis_imag = freqs_cis.sin().unsqueeze(2)

    def rotate(input):
        real = input[..., ::2]
        imag = input[..., 1::2]

        # For complex multiplication
        # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)

        # ac - bd
        rot_real = (real * freqs_cis_real) - (imag * freqs_cis_imag)

        # ad + bc
        rot_imag = (real * freqs_cis_imag) + (freqs_cis_real * imag)

        return torch.cat([rot_real.unsqueeze(-1), rot_imag.unsqueeze(-1)], dim=-1).reshape(
            input.shape
        )

    query_rot = rotate(query)
    key_rot = rotate(key)

    return query_rot.type_as(query), key_rot.type_as(key)
