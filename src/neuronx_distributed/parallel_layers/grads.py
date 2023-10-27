import os

import torch
import torch_xla.core.xla_model as xm

from ..utils.logger import get_logger
from .layers import param_is_not_tensor_parallel_duplicate
from .parallel_state import (
    get_data_parallel_group,
    get_data_parallel_size,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_size,
    rmsg,
)

logger = get_logger()

# allreduce bucket buffer size
_ALLREDUCE_BUCKET_CAP_MB = 512  # MB


def param_is_not_shared(param):
    return not hasattr(param, "shared") or not param.shared


def clip_grad_norm(parameters, max_norm, norm_type=2, zero1_optimizer=False, force_spmd=True):
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
        zero1_optimizer(bool): Whether zero1 optimizer is used, if so we will collect
            grad norm from the world group
        force_spmd(bool): Whether to force spmd when calculating global norm. If set to
            True we will sum the tp duplicated paramter grads on all ranks and divide them
            by tp size. Warning: If the grads are too small the division might result in
            incorrect results.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if torch.isinf(torch.tensor(norm_type)):
        # Always use spmd for inf norm since it is using MAX operation
        force_spmd = True

    def _allreduce_norm_across_model_parallel_groups(total_norm, reduce_op):
        if get_tensor_model_parallel_size() > 1:
            torch.distributed.all_reduce(
                total_norm,
                op=reduce_op,
                group=get_tensor_model_parallel_group(),
            )
        if get_pipeline_model_parallel_size() > 1:
            torch.distributed.all_reduce(
                total_norm,
                op=reduce_op,
                group=get_pipeline_model_parallel_group(),
            )

    def _allreduce_norm_across_world_group(total_norm, reduce_op):
        torch.distributed.all_reduce(total_norm, op=reduce_op)

    device = xm.xla_device()

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    grads = []
    grads_for_norm = []
    grads_for_norm_tp_duplicated = []
    for param in parameters:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        is_tp_param = hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
        if grad_not_none:
            grad = param.grad.detach()
            grads.append(grad)
        if grad_not_none and is_not_shared:
            if is_tp_param or (is_not_tp_duplicate and not force_spmd):
                # TP parallelized parameters
                # (not force_spmd) only tp rank 0 will add non-tp paramaters
                grads_for_norm.append(grad)
            elif force_spmd:
                # non-tp paramaters
                grads_for_norm_tp_duplicated.append(grad)

    # Norm parameters.
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = torch.FloatTensor([float(0.0)]).to(device)

    # Calculate norm.
    if torch.isinf(torch.tensor(norm_type)):
        total_norm_tp_duplicated = max(grad.abs().max() for grad in grads_for_norm_tp_duplicated)
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm = max(total_norm_tp_duplicated, total_norm)
        total_norm = torch.FloatTensor([float(total_norm)]).to(device)
        if not zero1_optimizer:
            _allreduce_norm_across_model_parallel_groups(total_norm, torch.distributed.ReduceOp.MAX)
        else:
            _allreduce_norm_across_world_group(total_norm, torch.distributed.ReduceOp.MAX)
        total_norm = total_norm[0].item()

    else:
        if force_spmd:
            # sum the non-tp grad norm and scale by the tp_size
            for grad in grads_for_norm_tp_duplicated:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm**norm_type
            total_norm /= get_tensor_model_parallel_size()

        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type)
            total_norm += grad_norm**norm_type

        if not zero1_optimizer:
            _allreduce_norm_across_model_parallel_groups(total_norm, torch.distributed.ReduceOp.SUM)
        else:
            _allreduce_norm_across_world_group(total_norm, torch.distributed.ReduceOp.SUM)
        total_norm = torch.pow(total_norm, 1.0 / norm_type)

    # Scale.
    clip_coeff = max_norm / (total_norm + 1.0e-6)

    for g in grads:
        g.data.mul_(torch.where(clip_coeff < 1, clip_coeff, torch.tensor(1.0, device=device)))
    return total_norm


def bucket_allreduce_gradients(grads_list):
    """
    All reduce bucket gradients for data parallelism.
    Referred from https://code.amazon.com/packages/Neuron-Nemo-Megatron/blobs/899fc918ffa82e4bea46750ff6dfe5b909d144a9/--/nemo/nemo/collections/nlp/models/language_modeling/megatron_base_model.py#L57 # noqa: E501
    """
    bucket_cap = int(os.getenv("ALLREDUCE_BUCKET_CAP_MB", _ALLREDUCE_BUCKET_CAP_MB)) * 1024 * 1024
    # Reverse the gradients list so that we start allreduce from the last layer
    # onwards. This allows allreduce to trigger as soon as the bucket fills up and
    # overlap with backward pass.
    gradients = reversed(grads_list)
    total = 0
    tensor_bucket = []
    groups = get_data_parallel_group()._mesh

    for grad in gradients:
        grad.data /= get_data_parallel_size()
        grad_bytes = grad.numel() * grad.element_size()

        # Gradient is larger than bucket_cap, don't bucketize
        if grad_bytes > bucket_cap:
            # Flush out previous buckets even if they don't fill up
            # This maintains the strict reverse ordering
            if len(tensor_bucket):
                xm.all_reduce("sum", tensor_bucket, groups=groups)
                total = 0
                tensor_bucket = []
            xm.all_reduce("sum", [grad], groups=groups)
            continue

        # Bucketize till the total spills over
        total += grad_bytes
        if total > bucket_cap:
            logger.debug(rmsg(f"all_reduce for total {total} bytes with groups {groups}"))
            xm.all_reduce("sum", tensor_bucket, groups=groups)
            total = grad_bytes
            tensor_bucket = []
        tensor_bucket.append(grad)

    # Flush the last remaining bucket
    if len(tensor_bucket):
        logger.debug(rmsg(f"all_reduce last bucket of {len(tensor_bucket)} tensors with groups {groups}"))
        xm.all_reduce("sum", tensor_bucket, groups=groups)
