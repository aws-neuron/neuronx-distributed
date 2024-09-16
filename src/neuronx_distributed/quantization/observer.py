"""
This module implements observers which are used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).
"""

from typing import Any, Dict, List, Tuple

import torch
from torch.ao.quantization.observer import UniformQuantizationObserverBase


class PerChannelAbsMaxObserver(UniformQuantizationObserverBase):
    r"""Observer module for computing the quantization parameters based on the
    running per channel abs max values.

    This observer uses the tensor abs max statistics to compute the per channel
    quantization parameters. The module records the running abs maximum
    of incoming tensors, and uses this statistic to compute the quantization
    parameters.

    Args:
        ch_axis: Channel axis
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
        quant_min: Minimum quantization value. If unspecified, it will follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will follow the 8-bit setup.
        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.

    The quantization parameters are computed the same way as in
    :class:`~torch.ao.quantization.observer.MinMaxObserver`, with the difference
    that the running abs max values are stored per channel.
    Scales and zero points are thus computed per channel as well.

    """
    max_val: torch.Tensor

    def __init__(
        self,
        ch_axis=0,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False,
        quant_min=None,
        quant_max=None,
        factory_kwargs=None,
        eps=torch.finfo(torch.float32).eps,
    ) -> None:

        # Currently only suppport torch.quint8
        assert dtype == torch.qint8, "Only torch.qint8 is supported"

        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            reduce_range=reduce_range,
            quant_min=quant_min,
            quant_max=quant_max,
            factory_kwargs=factory_kwargs,
            eps=eps,
        )
        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
        self.ch_axis = ch_axis
        self.register_buffer("max_val", torch.tensor([], **factory_kwargs))

    def forward(self, x_orig):
        return self._forward(x_orig)

    def _forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        max_val = self.max_val
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.max_val.dtype)
        y = torch.flatten(y, start_dim=1)

        if max_val.numel() == 0:
            max_val = torch.amax(torch.abs(y), dim=1, keepdim=True)
        else:
            max_val_cur = torch.amax(torch.abs(y), dim=1, keepdim=True)
            max_val = torch.max(max_val_cur, max_val)
        self.max_val.resize_(max_val.shape)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters, given max
        value tensors

        Returns:
            scales: Scales tensor of shape (#channels,)
            zero_points: Zero points tensor of shape (#channels,)
        """

        quant_max = self.quant_max
        # After profiling remove the assertion if taking too much time
        assert torch.all(self.max_val >= 0)
        max_val_pos = self.max_val # self.max_val already is the absolute maximum

        device = max_val_pos.device
        scale = torch.ones(max_val_pos.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(max_val_pos.size(), dtype=torch.int64, device=device)

        if self.qscheme == torch.per_tensor_symmetric or self.qscheme == torch.per_channel_symmetric:
            scale = max_val_pos / (float(quant_max))
            scale = torch.max(scale, self.eps)
        else:
            raise ValueError(f"Only support {torch.per_tensor_symmetric} and {torch.per_channel_symmetric}")

        # For scalar values, cast them to Tensors of size 1 to keep the shape
        # consistent with default values in FakeQuantize.
        if len(scale.shape) == 0:
            # TODO: switch to scale.item() after adding JIT support
            scale = torch.tensor([float(scale)], dtype=scale.dtype, device=device)
        if len(zero_point.shape) == 0:
            # TODO: switch to zero_point.item() after adding JIT support
            zero_point = torch.tensor([int(zero_point)], dtype=zero_point.dtype, device=device)

        return scale.squeeze(1), zero_point.squeeze(1)

    def extra_repr(self):
        return f"abs_max_val={self.max_val}"

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        raise NotImplementedError()

    def _load_from_state_dict_script(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, torch.Tensor],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        raise NotImplementedError()

    @torch.jit.export
    def reset_min_max_vals(self):
        """Resets the min/max values."""
        # This used to be torch.ones but that does not work because
        # JIT compiler can optimize it via common subexpression elimination
        # in which case both min_val and max_val point to the same tensor.
        self.max_val = torch.rand(
            0,
        )
