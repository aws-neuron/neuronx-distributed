import torch

def direct_cast_dequantize(tensor: torch.Tensor, upcast_dtype: torch.dtype) -> torch.Tensor:
    """
    A utility function to dequantize a tensor from lower dtype to upcast dtype without any scaling factor

    Args:
        tensor (torch.Tensor): tensor to be dequantized
        upcast_dtype (torch.dtype): upcast dtype

    Returns:
        torch.Tensor: upcasted tensor
    """
    upcast_tensor = tensor.to(upcast_dtype)
    return upcast_tensor

def scale_dequantize(tensor: torch.Tensor, scale: torch.Tensor, upcast_dtype: torch.dtype) -> torch.Tensor:
    """
    A utility function to dequantize a tensor from lower dtype to upcast dtype based on its corresponding scale
    Note: It will not convert back the tensor to its existing dtype

    Args:
        tensor (torch.Tensor): tensor to be dequantized
        scale (torch.Tensor): scale to be used for dequantization

    Returns:
        torch.Tensor: upcasted tensor multiplied with scale
    """
    upcast_tensor = tensor.to(torch.float32)
    upcast_tensor *= scale
    upcast_tensor = upcast_tensor.to(upcast_dtype)
    return upcast_tensor
