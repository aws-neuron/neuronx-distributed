"""Utility script to create a Mamba2Config given the size of the model."""

from typing import Dict

from mamba2 import Mamba2Config
from dataclasses import dataclass, astuple


@dataclass
class ConfParams:
    d_model: int
    n_layers: int
    head_dim: int = 128


CONFIGS_KWARGS: Dict[str, ConfParams] = {
    'Mamba130M': ConfParams(d_model=768, n_layers=24),
    'Mamba370M': ConfParams(d_model=1024, n_layers=48),
    'Mamba780M': ConfParams(d_model=1536, n_layers=48),
    'Mamba1B': ConfParams(d_model=2048, n_layers=48),
    'Mamba3B': ConfParams(d_model=2560, n_layers=64),
    'Mamba7B': ConfParams(d_model=4096, n_layers=64),
}


def get_config(name: str, vocab_size, rmsnorm_within_groups=True, n_groups=8):
    d_model, n_layers, head_dim = astuple(CONFIGS_KWARGS[name])
    config = Mamba2Config(
        vocab_size=vocab_size,
        hidden_size=d_model,
        head_dim=head_dim,
        num_heads=(d_model * 2) // head_dim,
        num_hidden_layers=n_layers,
        tie_word_embeddings=True,
        use_cache=False,
        n_groups=n_groups,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        rmsnorm_within_groups=rmsnorm_within_groups,
    )
    return config

