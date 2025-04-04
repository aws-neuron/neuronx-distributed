from typing import List, Union
import os
import torch
from torch import Size


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
        bias: bool = True,
        device=None,
        dtype=None,
        sequence_parallel_enabled: bool = False,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.elementwise_affine:
            _set_sequence_parallel_enabled(self.weight, self.sequence_parallel_enabled)
            if bias:
                _set_sequence_parallel_enabled(self.bias, self.sequence_parallel_enabled)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        original_input_dtype = input.dtype
        if os.environ.get("XLA_DOWNCAST_BF16") == "1":
            input = input.to(torch.double)
        output = super().forward(input)
        output.to(original_input_dtype)
        return output
