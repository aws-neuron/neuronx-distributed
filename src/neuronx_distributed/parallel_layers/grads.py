import torch
from torch._six import inf
import torch_xla.core.xla_model as xm
from .layers import param_is_not_tensor_parallel_duplicate
from .parallel_state import get_tensor_model_parallel_group


def param_is_not_shared(param):
    return not hasattr(param, 'shared') or not param.shared


def clip_grad_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    device = xm.xla_device()

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    grads = []
    grads_for_norm = []
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none:
            grad = param.grad.detach()
            grads.append(grad)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            grads_for_norm.append(grad)

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = torch.FloatTensor([float(0.0)]).to(device)

    # Calculate norm.
    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm = torch.FloatTensor([float(total_norm)]).to(device)
        torch.distributed.all_reduce(total_norm,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_tensor_model_parallel_group())
        total_norm = total_norm[0].item()

    else:
        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm**norm_type

        torch.distributed.all_reduce(total_norm,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_tensor_model_parallel_group())
        total_norm = torch.pow(total_norm, 1.0 / norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)

    for g in grads:
        g.data.mul_(
            torch.where(clip_coeff < 1, clip_coeff,
                        torch.tensor(1., device=device)))
    return total_norm