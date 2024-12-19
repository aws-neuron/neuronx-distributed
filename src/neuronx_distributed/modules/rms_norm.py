import numbers
import os

import torch
from torch.nn import init
from torch.nn.parameter import Parameter


# Reference implementation from Huggingface
def manual_rms_norm(input, normalized_shape, weight, eps):
    # layer norm should always be calculated in fp32
    # Keep all the original dtypes for down casting after RMS Norm
    # Cast the input to fp32 = double
    original_input_dtype = input.dtype
    if os.environ.get("XLA_DOWNCAST_BF16", None) == 1:
        input = input.to(torch.double)
    else:
        input = input.to(torch.float32)

    dims = tuple(i for i in range(-1, -len(normalized_shape)-1, -1))
    variance = input.pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)

    if weight is None:
        # Downcast the output/input back to its original dtype
        return input.to(original_input_dtype)

    output = weight * input

    # Downcast the output to the original dtype
    output = output.to(original_input_dtype)

    return output


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        sequence_parallel_enabled: bool = False,
        **kwargs,
    ):
        super().__init__()
        if "elementwise_affine" in kwargs:
            import warnings

            warnings.warn("RMSNorm does not support `elementwise_affine` argument")
            elementwise_affine = kwargs.pop("elementwise_affine")
            if not elementwise_affine:
                raise RuntimeError("RMSNorm does not support `elementwise_affine = False`")
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = True
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
            setattr(self.weight, "sequence_parallel_enabled", sequence_parallel_enabled)
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input):
        return manual_rms_norm(input, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)
