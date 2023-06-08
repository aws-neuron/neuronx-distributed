import torch
from torch import Size
from typing import Union, List


def _set_sequence_parallel_enabled(
    param: torch.Tensor,
    sequence_parallel_enabled: bool,
) -> None:
    setattr(param, "sequence_parallel_enabled", sequence_parallel_enabled)

_shape_t = Union[int, List[int], Size]

class LayerNorm(torch.nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
        sequence_parallel_enabled: bool = False,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.elementwise_affine:
            _set_sequence_parallel_enabled(self.weight, self.sequence_parallel_enabled)
            _set_sequence_parallel_enabled(self.bias, self.sequence_parallel_enabled)
