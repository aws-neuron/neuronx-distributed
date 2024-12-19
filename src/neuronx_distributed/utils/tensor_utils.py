import torch


def cumsum(tensor, dim=0, tril_size=2048):
    """Implements cumulative sum operation using matrix multiplication.

    Currently, only summing along the first dimension of the given tensor is supported.

    Note that the input tensor is up-cast to float-64 for precision of the operations.
    The returned tensor is cast back to the same dtype as the input tensor, therefore the user must be careful
    with the type of the input tensor to avoid overflows and precision issues.

    Arguments:
        tensor: The 2D tensor to perform the cumsum operation on.
        dim: The dimension along which to perform the cumsum operation. Currently only dim=0 is supported.
        tril_size: The size of the lower triangular matrix used for performing the cumsum. This determines the size of
                   each matrix multiplication performed, and therefore the number of shards in which the input tensor
                   is split into for processing. Defaults to 2048.

    Returns:
        cumsum_tensor: 2D output tensor containing the output of the cumsum operation.
    """

    if len(tensor.shape) != 2:
        raise ValueError(f"Expected 2D input tensor, unsupported shape: {str(tensor.shape)}")

    if dim != 0:
        raise NotImplementedError(f"Only cumsum along dimension-0 is currently supported, unexpected dim={dim}")

    dtype = tensor.dtype    

    num_tokens = tensor.shape[0]
    if num_tokens % tril_size == 0:
        num_iters = num_tokens // tril_size
        last_iter_tokens = tril_size
    else:
        num_iters = num_tokens // tril_size + 1
        last_iter_tokens = num_tokens % tril_size

    # For precision, perform matmul in float64 (to prevent auto-downcasting to bf16)
    results = []
    rolling_sum = torch.zeros(1, tensor.shape[1], dtype=torch.float64, device=tensor.device)
    for i in range(num_iters):
        # There may be less than tril_size tokens in the last iteration
        iter_tokens = tril_size if i < num_iters - 1 else last_iter_tokens
        tril = torch.tril(torch.ones(iter_tokens, iter_tokens, device=tensor.device)).to(dtype=torch.float64)

        input_slice = tensor.narrow(0, i * tril_size, iter_tokens).to(dtype=torch.float64)
        output_slice = rolling_sum + torch.matmul(tril, input_slice)
        results.append(output_slice)
        if i < num_iters - 1:
            rolling_sum = output_slice.narrow(0, -1, 1)

    # Concatenate results, and cast back to original dtype
    return torch.cat(results, dim=0).to(dtype)
