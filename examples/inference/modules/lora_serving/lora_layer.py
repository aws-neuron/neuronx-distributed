import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiLoraLinear(nn.Module):
    def __init__(
        self,
        max_loras: int,
        input_size: int,
        output_size: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.max_loras = max_loras
        self.weight_shape = (max_loras, output_size, input_size)
        self.weight = nn.Parameter(torch.zeros(*self.weight_shape, dtype=dtype), requires_grad=False)

    def get_weight(self, adapter_ids) -> torch.Tensor:
        if adapter_ids is None:
            adapter_ids = 0
        return self.weight[adapter_ids].squeeze()

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor = None) -> torch.Tensor:
        weight = self.get_weight(adapter_ids)
        return F.linear(x, weight)


class MultiLoraConv2d(nn.Conv2d):
    def __init__(
        self,
        max_loras: int,
        input_size: int,
        output_size: int,
        kernel_size,
        stride,
        padding,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.max_loras = max_loras
        super().__init__(input_size, output_size, kernel_size, stride, padding, bias=False, dtype=dtype)
        self.weight = nn.Parameter(torch.empty(max_loras, input_size, output_size // self.groups, *self.kernel_size, dtype=dtype), requires_grad=False)

    def get_weight(self, adapter_ids) -> torch.Tensor:
        if adapter_ids is None:
            adapter_ids = 0
        return self.weight[adapter_ids].squeeze()

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor = None) -> torch.Tensor:
        weight = self.get_weight(adapter_ids)
        return self._conv_forward(x, weight, None)


class MultiLoraEmbedding(nn.Module):
    def __init__(
        self,
        max_loras: int,
        input_size: int,
        output_size: int,
        padding_idx: Optional[int],
        max_norm: Optional[float],
        norm_type: float,
        scale_grad_by_freq: bool,
        sparse: bool,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.max_loras = max_loras
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.weight_shape = (max_loras, output_size, input_size)
        self.weight = nn.Parameter(torch.zeros(*self.weight_shape, dtype=dtype), requires_grad=False)


    def get_weight(self, adapter_ids: torch.Tensor) -> torch.Tensor:
        if adapter_ids is None:
            adapter_ids = 0
        return self.weight[adapter_ids].squeeze()


    def _embed(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            x,
            weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor = None) -> torch.Tensor:
        weight = self.get_weight(adapter_ids)
        return self._embed(x, weight.T)
