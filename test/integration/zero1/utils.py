import copy
import random
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import numpy as np

import torch
from torch import Tensor
from torch.optim import Optimizer

from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_replica_groups,
    get_data_parallel_rank,
    get_data_parallel_size,
    get_pipeline_model_parallel_replica_groups,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_replica_groups,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
    get_expert_model_parallel_replica_groups,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_size,
    get_expert_data_parallel_replica_groups,
    get_expert_data_parallel_rank,
    get_expert_data_parallel_size,
)

PP_GROUP_PG_GLOO = None
TP_GROUP_PG_GLOO = None
DP_GROUP_PG_GLOO = None
EMP_GROUP_PG_GLOO = None
EDP_GROUP_PG_GLOO = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_test_params(use_ep=False, dtype=None, device=None):
    seed = int(
        "11{}{}".format(
            get_pipeline_model_parallel_rank(),
            get_tensor_model_parallel_rank(),
        )
    )
    set_seed(seed)
    params = [torch.randn(32, 32) for _ in range(12)]
    seed = int(
        "1{}{}{}".format(
            get_expert_model_parallel_rank(),
            get_pipeline_model_parallel_rank(),
            get_tensor_model_parallel_rank(),
        )
    )
    set_seed(seed)
    params = params + [torch.randn(32, 32) for _ in range(12)]

    if dtype is not None:
        params = [p.to(dtype=dtype) for p in params]
    if device is not None:
        params = [p.to(device=device) for p in params]
    for idx, p in enumerate(params):
        p.tensor_model_parallel = True
        p.requires_grad = True
        if idx >= 12:
            p.expert_model_parallel = True
    return params


def initialize_gloo_groups():
    rank = torch.distributed.get_rank()

    global PP_GROUP_PG_GLOO
    assert PP_GROUP_PG_GLOO is None, "pp gloo groups are already initialized!"
    pp_group_spmd = get_pipeline_model_parallel_replica_groups()
    for pp_group in pp_group_spmd:
        pg = torch.distributed.new_group(ranks=pp_group, backend="gloo")
        if rank in pp_group:
            PP_GROUP_PG_GLOO = pg

    global TP_GROUP_PG_GLOO
    assert TP_GROUP_PG_GLOO is None, "tp gloo groups are already initialized!"
    tp_group_spmd = get_tensor_model_parallel_replica_groups()
    for tp_group in tp_group_spmd:
        pg = torch.distributed.new_group(ranks=tp_group, backend="gloo")
        if rank in tp_group:
            TP_GROUP_PG_GLOO = pg

    global DP_GROUP_PG_GLOO
    assert DP_GROUP_PG_GLOO is None, "dp gloo groups are already initialized!"
    dp_group_spmd = get_data_parallel_replica_groups()
    for dp_group in dp_group_spmd:
        pg = torch.distributed.new_group(ranks=dp_group, backend="gloo")
        if rank in dp_group:
            DP_GROUP_PG_GLOO = pg

    global EMP_GROUP_PG_GLOO
    assert EMP_GROUP_PG_GLOO is None, "emp gloo groups are already initialized!"
    emp_group_spmd = get_expert_model_parallel_replica_groups()
    for emp_group in emp_group_spmd:
        pg = torch.distributed.new_group(ranks=emp_group, backend="gloo")
        if rank in emp_group:
            EMP_GROUP_PG_GLOO = pg

    global EDP_GROUP_PG_GLOO
    assert EDP_GROUP_PG_GLOO is None, "edp gloo groups are already initialized!"
    edp_group_spmd = get_expert_data_parallel_replica_groups()
    for edp_group in edp_group_spmd:
        pg = torch.distributed.new_group(ranks=edp_group, backend="gloo")
        if rank in edp_group:
            EDP_GROUP_PG_GLOO = pg


def get_pp_gloo_group():
    global PP_GROUP_PG_GLOO
    assert PP_GROUP_PG_GLOO is not None, "pp gloo groups are not initialized!"
    return PP_GROUP_PG_GLOO


def get_tp_gloo_group():
    global TP_GROUP_PG_GLOO
    assert TP_GROUP_PG_GLOO is not None, "tp gloo groups are not initialized!"
    return TP_GROUP_PG_GLOO


def get_dp_gloo_group():
    global DP_GROUP_PG_GLOO
    assert DP_GROUP_PG_GLOO is not None, "dp gloo groups are not initialized!"
    return DP_GROUP_PG_GLOO


def get_emp_gloo_group():
    global EMP_GROUP_PG_GLOO
    assert EMP_GROUP_PG_GLOO is not None, "emp gloo groups are not initialized!"
    return EMP_GROUP_PG_GLOO


def get_edp_gloo_group():
    global EDP_GROUP_PG_GLOO
    assert EDP_GROUP_PG_GLOO is not None, "edp gloo groups are not initialized!"
    return EDP_GROUP_PG_GLOO


def destroy_gloo_groups():
    global PP_GROUP_PG_GLOO
    global TP_GROUP_PG_GLOO
    global DP_GROUP_PG_GLOO
    global EMP_GROUP_PG_GLOO
    global EDP_GROUP_PG_GLOO
    PP_GROUP_PG_GLOO = None
    TP_GROUP_PG_GLOO = None
    DP_GROUP_PG_GLOO = None
    EMP_GROUP_PG_GLOO = None
    EDP_GROUP_PG_GLOO = None


class RefOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterator[Tensor],
        optimizer_class: Type[Optimizer],
        optimizer_dtype: Optional[Any] = None,
        grad_clipping: bool = True,
        max_norm: Optional[float] = None,
        pin_layout: bool = True,
        sharding_groups: Optional[Any] = None,
        grad_norm_groups: Optional[Any] = None,
        lazy_init: bool = False,
        coalesce_cc: bool = False,
        **defaults: Any,
    ):
        super().__init__(params, defaults)

        self.optimizer_class = optimizer_class
        self.defaults = defaults
        self.optimizer_dtype = optimizer_dtype if optimizer_dtype is not None else torch.float32
        self.grad_clipping = grad_clipping
        self.max_norm = max_norm if max_norm is not None else 1.0

        self._grad_norm = None

        self.inited = False
        if not lazy_init:
            self.init_zero()

    def init_zero(self):
        # Copied parameters for use in optimizer
        copied_param_groups = self._copy_parameters()
        # Optimizer initialization
        self.base_optimizer = self.optimizer_class(copied_param_groups, **self.defaults)
        self._sync_param_groups(self.param_groups, self.base_optimizer.param_groups)
        self.inited = True

    @property
    def grad_norm(self):
        return self._grad_norm

    def _copy_parameters(self):
        self.device = None
        all_params = []
        for param_group in self.param_groups:
            for param in param_group["params"]:
                all_params.append(param)
                if self.device is None:
                    self.device = param.device
                else:
                    assert self.device == param.device, "Params should on the same device."
        assert self.device.type == "cpu"

        copied_params_groups = []
        for param_group in self.param_groups:
            copied_params = []
            for param in param_group["params"]:
                copied_data = param.data
                if copied_data.dtype != self.optimizer_dtype:
                    copied_data = copied_data.to(dtype=self.optimizer_dtype)
                copied_param = torch.nn.Parameter(copied_data, requires_grad=param.requires_grad)
                copied_params.append(copied_param)
            copied_params_group = copy.copy(param_group)
            copied_params_group["params"] = copied_params
            copied_params_groups.append(copied_params_group)

        return copied_params_groups

    @staticmethod
    def _sync_param_groups(
        src_param_groups: List[Dict[Any, Any]],
        dst_param_groups: List[Dict[Any, Any]],
    ) -> None:
        assert len(src_param_groups) == len(
            dst_param_groups
        ), "Mismatch between number of source and destination parameter groups"
        for src_param_group, dst_param_group in zip(src_param_groups, dst_param_groups):
            # Sync all attributes except the parameters
            for attr in filter(lambda x: x != "params", src_param_group.keys()):
                dst_param_group[attr] = src_param_group[attr]

    @torch.no_grad()
    def _get_grad_norm(
        self,
        norm_type: Union[float, int] = 2.0,
    ) -> torch.Tensor:
        # calculate norm's square sum
        dtype = torch.float32
        total_norm = torch.tensor([float(0.0)], dtype=dtype)
        ep_total_norm = torch.tensor([float(0.0)], dtype=dtype)
        for param_group, copied_param_group in zip(self.param_groups, self.base_optimizer.param_groups):
            for param, copied in zip(param_group["params"], copied_param_group["params"]):
                if param.grad is not None:
                    grad_norm = torch.norm(copied.grad, norm_type)
                    if hasattr(param, "expert_model_parallel") and param.expert_model_parallel:
                        ep_total_norm += grad_norm**norm_type
                    else:
                        total_norm += grad_norm**norm_type

        # all-reduce
        if get_expert_model_parallel_size() > 1:
            torch.distributed.all_reduce(
                ep_total_norm,
                op=torch.distributed.ReduceOp.SUM,
                group=get_emp_gloo_group(),
            )
        total_norm += ep_total_norm
        if get_tensor_model_parallel_size() > 1:
            torch.distributed.all_reduce(
                total_norm,
                op=torch.distributed.ReduceOp.SUM,
                group=get_tp_gloo_group(),
            )
        if get_pipeline_model_parallel_size() > 1:
            torch.distributed.all_reduce(
                total_norm,
                op=torch.distributed.ReduceOp.SUM,
                group=get_pp_gloo_group(),
            )

        # calculate final norm value
        total_norm = torch.pow(total_norm, 1.0 / norm_type)
        return total_norm

    @torch.no_grad()
    def _clip_grad_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
    ) -> torch.Tensor:
        total_norm = self._get_grad_norm(norm_type=norm_type)
        self._grad_norm = total_norm.to(self.optimizer_dtype)

        grads = []
        for param_group, copied_param_group in zip(self.param_groups, self.base_optimizer.param_groups):
            for param, copied in zip(param_group["params"], copied_param_group["params"]):
                if param.grad is not None:
                    grads.append(copied.grad.detach())

        clip_coeff = max_norm / (total_norm + 1.0e-6)
        for g in grads:
            g.data.mul_(
                torch.where(
                    clip_coeff < 1,
                    clip_coeff.to(dtype=total_norm.dtype),
                    torch.tensor(1.0, dtype=total_norm.dtype),
                )
            )
        return self._grad_norm

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert self.inited, "must call init_zero() first"

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # sync to base optimizer
        self._sync_param_groups(self.param_groups, self.base_optimizer.param_groups)

        # do master copy and all-reduce
        for param_group, copied_param_group in zip(self.param_groups, self.base_optimizer.param_groups):
            for param, copied in zip(param_group["params"], copied_param_group["params"]):
                if param.grad is not None:
                    padded_grad = param.grad.clone()
                    grad_copied = padded_grad

                    # reduce grads
                    # cast to fp32 and cast back as gloo not support bf16
                    original_dtype = grad_copied.dtype
                    grad_copied = grad_copied.to(dtype=torch.float32)
                    if hasattr(param, "expert_model_parallel") and param.expert_model_parallel:
                        torch.distributed.all_reduce(
                            grad_copied,
                            op=torch.distributed.ReduceOp.SUM,
                            group=get_edp_gloo_group(),
                        )
                        grad_copied /= get_expert_data_parallel_size()
                        grad_copied /= get_expert_model_parallel_size()
                    else:
                        torch.distributed.all_reduce(
                            grad_copied,
                            op=torch.distributed.ReduceOp.SUM,
                            group=get_edp_gloo_group(),
                        )
                        grad_copied /= get_expert_data_parallel_size()
                        torch.distributed.all_reduce(
                            grad_copied,
                            op=torch.distributed.ReduceOp.SUM,
                            group=get_emp_gloo_group(),
                        )
                        grad_copied /= get_expert_model_parallel_size()
                    grad_copied = grad_copied.to(dtype=original_dtype)

                    if grad_copied.dtype != self.optimizer_dtype:
                        grad_copied = grad_copied.to(dtype=self.optimizer_dtype)
                    copied.grad = grad_copied

        # grad clipping
        if self.grad_clipping:
            # Update unscale/clip with sub partitions
            self._clip_grad_norm(max_norm=self.max_norm)

        # Step the wrapped optimizer
        # Closure already executed, pass none here
        self.base_optimizer.step(closure=None, **kwargs)
        # Remove copieds' grads
        self.base_optimizer.zero_grad(set_to_none=True)

        # copy back params
        for param_group, copied_param_group in zip(self.param_groups, self.base_optimizer.param_groups):
            for param, copied in zip(param_group["params"], copied_param_group["params"]):
                if param.grad is not None:
                    copied_data = copied.data
                    if param.dtype != self.optimizer_dtype:
                        copied_data = copied_data.to(dtype=param.dtype)
                    param.data.copy_(copied_data)

        # sync back
        self._sync_param_groups(self.base_optimizer.param_groups, self.param_groups)

        return loss

    def state_dict(self):
        state_dict = super().state_dict()
        base_state = self.base_optimizer.state_dict()["state"]
        state_dict["base_state"] = base_state
        return state_dict
