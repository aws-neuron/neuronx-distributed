import os

from typing import List, Union, Optional, Iterable

import torch
import torch_xla.core.xla_model as xm

from ..utils.logger import get_logger
from .layers import param_is_not_tensor_parallel_duplicate
from .mappings import reduce_from_tensor_model_parallel_region
from .parallel_state import (
    get_data_parallel_group,
    get_data_parallel_size,
    get_expert_data_parallel_replica_groups,
    get_expert_data_parallel_group,
    get_expert_data_parallel_size,
    get_expert_model_parallel_replica_groups,
    get_expert_model_parallel_group,
    get_expert_model_parallel_size,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_size,
    rmsg,
)
from .comm import all_reduce

logger = get_logger()

# allreduce bucket buffer size
_ALLREDUCE_BUCKET_CAP_MB = 512  # MB


def param_is_not_shared(param):
    return not hasattr(param, "shared") or not param.shared


def get_grad_norm(
    parameters: Union[Iterable[torch.Tensor], torch.Tensor],
    norm_type: Union[float, int] = 2,
    zero1_optimizer: bool = False,
    zero1_optimizer_groups: Optional[List[List[int]]] = None,
    force_spmd: bool = True,
):
    """Get gradient norm of an iterable of parameters.

    This can handle model parallel parameters.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
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

    if not zero1_optimizer and zero1_optimizer_groups is not None:
        raise ValueError(
            "Getting zero1_optimizer_groups while zero1_optimizer is False. When using zero-1 optimizer grad clipping is handled by optimizer."
        )  # noqa

    def _allreduce_norm_across_parallel_groups(total_norm, ep_total_norm, reduce_op):
        """
        - zero1 without groups: allreduce across world groups
        - otherwise allreduce across each parallel group
        """
        if zero1_optimizer and zero1_optimizer_groups is None:
            torch.distributed.all_reduce(total_norm, op=reduce_op)
            if get_expert_model_parallel_size() > 1:
                torch.distributed.all_reduce(ep_total_norm, op=reduce_op)
                total_norm += ep_total_norm

        else:
            if get_expert_model_parallel_size() > 1:
                torch.distributed.all_reduce(
                    ep_total_norm,
                    op=reduce_op,
                    group=get_expert_model_parallel_group(),
                )
            if reduce_op == torch.distributed.ReduceOp.MAX:
                total_norm = max(total_norm, ep_total_norm)
            else:
                # reduce_op will be SUM for Lp-norms
                total_norm += ep_total_norm
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
            if zero1_optimizer_groups is not None:
                all_reduce(
                    xm.REDUCE_SUM,
                    [total_norm],
                    groups=zero1_optimizer_groups,
                    pin_layout=True,
                )
        return total_norm

    def _is_ep_param(obj):
        return hasattr(obj, "expert_model_parallel") and obj.expert_model_parallel

    def _is_ep_grad(obj):
        return hasattr(obj, "expert_model_parallel") and obj.expert_model_parallel

    params: List[torch.Tensor] = [parameters] if not isinstance(parameters, List) else parameters
    device = params[0].device
    dtype = params[0].dtype

    grads = []
    grads_for_norm = []
    grads_for_norm_tp_duplicated = []
    for param in params:
        grad_not_none = param.grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
        is_tp_param = hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
        is_ep_param = _is_ep_param(param)
        if grad_not_none:
            grad = param.grad.detach()
            grads.append(grad)
        if grad_not_none and is_not_shared:
            if is_ep_param:
                grad.expert_model_parallel = True
            if is_tp_param or (is_not_tp_duplicate and not force_spmd):
                # TP parallelized parameters
                # (not force_spmd) only tp rank 0 will add non-tp paramaters
                grads_for_norm.append(grad)
            elif force_spmd:
                # non-tp paramaters
                grads_for_norm_tp_duplicated.append(grad)

    # Norm parameters.
    norm_type_float = float(norm_type)
    total_norm = torch.tensor([float(0.0)], dtype=dtype, device=device)
    ep_total_norm = torch.tensor([float(0.0)], dtype=dtype, device=device)

    # Calculate norm.
    if torch.isinf(torch.tensor(norm_type_float)):
        total_norm_tp_duplicated = max(grad.abs().max() for grad in grads_for_norm_tp_duplicated)
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm = max(total_norm_tp_duplicated, total_norm)
        total_norm = torch.tensor([float(total_norm)], dtype=dtype, device=device)
        _allreduce_norm_across_parallel_groups(total_norm, total_norm, torch.distributed.ReduceOp.MAX)
        total_norm = total_norm[0].item()
    else:
        if force_spmd:
            # sum the non-tp grad norm and scale by the tp_size
            for grad in grads_for_norm_tp_duplicated:
                grad_norm = torch.norm(grad, norm_type_float)
                if _is_ep_grad(grad):
                    ep_total_norm += grad_norm**norm_type_float
                else:
                    total_norm += grad_norm**norm_type_float
            tp_size = get_tensor_model_parallel_size()
            total_norm /= tp_size
            ep_total_norm /= tp_size

        for grad in grads_for_norm:
            grad_norm = torch.norm(grad, norm_type_float)
            if _is_ep_grad(grad):
                ep_total_norm += grad_norm**norm_type_float
            else:
                total_norm += grad_norm**norm_type_float

        total_norm = _allreduce_norm_across_parallel_groups(total_norm, ep_total_norm, torch.distributed.ReduceOp.SUM)
        total_norm = torch.pow(total_norm, 1.0 / norm_type_float)

    return total_norm


def clip_grad_norm(
    parameters: Union[Iterable[torch.Tensor], torch.Tensor],
    max_norm: Union[float, int],
    norm_type: Union[float, int] = 2,
    zero1_optimizer: bool = False,
    zero1_optimizer_groups: Optional[List[List[int]]] = None,
    force_spmd: bool = True,
):
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

    params = [parameters] if not isinstance(parameters, List) else parameters

    # Get norm
    total_norm = get_grad_norm(
        params,
        norm_type=norm_type,
        zero1_optimizer=zero1_optimizer,
        zero1_optimizer_groups=zero1_optimizer_groups,
        force_spmd=force_spmd,
    )

    clip_grads_with_norm(params, total_norm, max_norm)
    return total_norm


def clip_grads_with_norm(parameters: List[torch.Tensor], total_norm, max_norm):
    assert len(parameters) > 0, "Parameters should be a non-empty list for gradient clipping"
    device = parameters[0].device
    grads = []
    for param in parameters:
        if param.grad is not None:
            grad = param.grad.detach()
            grads.append(grad)

    clip_coeff = max_norm / (total_norm + 1.0e-6)
    for g in grads:
        g.data.mul_(
            torch.where(
                clip_coeff < 1,
                clip_coeff.to(dtype=total_norm.dtype),
                torch.tensor(1.0, dtype=total_norm.dtype, device=device),
            )
        )
    return total_norm


def bucket_allreduce_gradients(grads_list, reduce_over_ep_group=False):
    """
    All reduce bucket gradients for data parallelism.
    Referred from https://github.com/aws-neuron/neuronx-nemo-megatron/blob/main/nemo/nemo/collections/nlp/models/language_modeling/megatron_base_model.py#L58 # noqa: E501
    """
    bucket_cap = int(os.getenv("ALLREDUCE_BUCKET_CAP_MB", _ALLREDUCE_BUCKET_CAP_MB)) * 1024 * 1024

    dtype_groups = {}
    for grad in grads_list:
        tp = grad.dtype
        if tp not in dtype_groups:
            dtype_groups[tp] = []
        dtype_groups[tp].append(grad)
    logger.debug(rmsg(f"reduce grads dtype_groups counts {[(tp, len(group)) for tp, group in dtype_groups.items()]}"))

    for tp in dtype_groups:
        grads = dtype_groups[tp]

        # Reverse the gradients list so that we start allreduce from the last layer
        # onwards. This allows allreduce to trigger as soon as the bucket fills up and
        # overlap with backward pass.
        gradients = reversed(grads)
        total = 0
        tensor_bucket: List[torch.Tensor] = []

        # if reduce_over_ep_group == False, the assumption is that we are allreducing
        # all gradients over the expert-data-parallel group. if there is no ep, this
        # is the only allreduce that we do, since data-parallel-group == expert-data-parallel-group.
        # otherwise, non-expert-parallel gradients will go through an additional allreduce
        # with reduce_over_ep_group == True, so that they are reduced over the full dp group.
        groups = (
            get_expert_model_parallel_replica_groups()
            if reduce_over_ep_group
            else get_expert_data_parallel_replica_groups()
        )

        # the assumption is that if we are reducing over ep group, then we
        # will also separately reduce over expert data parallel group, and the
        # normalization has already happened.
        size = get_data_parallel_size() if not reduce_over_ep_group else 1.0

        for grad in gradients:
            grad.data /= size
            grad_bytes = grad.numel() * grad.element_size()

            # Gradient is larger than bucket_cap, don't bucketize
            if grad_bytes > bucket_cap:
                # Flush out previous buckets even if they don't fill up
                # This maintains the strict reverse ordering
                if len(tensor_bucket):
                    all_reduce("sum", tensor_bucket, groups=groups)
                    total = 0
                    tensor_bucket = []
                all_reduce("sum", [grad], groups=groups)
                continue

            # Bucketize till the total spills over
            total += grad_bytes
            if total > bucket_cap:
                logger.debug(rmsg(f"all_reduce for total {total} bytes with groups {groups}"))
                all_reduce("sum", tensor_bucket, groups=groups)
                total = grad_bytes
                tensor_bucket = []
            tensor_bucket.append(grad)

        # Flush the last remaining bucket
        if len(tensor_bucket):
            logger.debug(rmsg(f"all_reduce last bucket of {len(tensor_bucket)} tensors with groups {groups}"))
            all_reduce("sum", tensor_bucket, groups=groups)


def allreduce_sequence_parallel_gradients(optimizer):
    """All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
    Modified from megatron-lm:
    https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
    """
    grads = []
    for param_group in optimizer.__getstate__()["param_groups"]:
        for group, params in param_group.items():
            if group == "params":
                for p in params:
                    if isinstance(p, torch.Tensor) and getattr(p, "sequence_parallel_enabled", False):
                        if p.grad is not None:
                            grads.append(p.grad.data)
                        elif hasattr(p, "main_grad"):
                            grads.append(p.main_grad.data)
    for grad in grads:
        reduce_from_tensor_model_parallel_region(grad)
