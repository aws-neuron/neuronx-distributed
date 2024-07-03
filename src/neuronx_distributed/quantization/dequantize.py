import torch


def dequantize(tensor: torch.Tensor, scale: torch.Tensor, upcast_dtype: torch.dtype) -> torch.Tensor:
    """
    A utility function to dequantize a tensor from lower dtype to upcast dtype based on its corresponding scale
    Note: It will not convert back the tensor to its existing dtype

    Args:
        tensor (torch.Tensor): tensor to be dequantized
        scale (torch.Tensor): scale to be used for dequantization

    Returns:
        torch.Tensor: upcasted tensor with the same dtype as the input tensor
        torch.Tensor: the scale used to dequantize the input tensor
    """
    upcast_tensor = tensor.to(upcast_dtype)
    upcast_tensor *= scale
    return upcast_tensor
