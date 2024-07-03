import copy
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_size,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_size,
)

PP_GROUP_PG_GLOO = None
TP_GROUP_PG_GLOO = None
DP_GROUP_PG_GLOO = None


def get_test_params(dtype=None, device=None):
    params = [torch.randn(512, 512) for _ in range(24)]
    if dtype is not None:
        params = [p.to(dtype=dtype) for p in params]

    if device is not None:
        params = [p.to(device=device) for p in params]
    for p in params:
        p.tensor_model_parallel = True
        p.requires_grad = True
    return params


def initialize_gloo_groups():
    global PP_GROUP_PG_GLOO
    assert PP_GROUP_PG_GLOO is None, "pp gloo groups are already initialized!"
    pp_group_spmd = get_pipeline_model_parallel_group(as_list=True)
    rank = torch.distributed.get_rank()
    for pp_group in pp_group_spmd:
        pg = torch.distributed.new_group(ranks=pp_group, backend="gloo")
        if rank in pp_group:
            PP_GROUP_PG_GLOO = pg

    global TP_GROUP_PG_GLOO
    assert TP_GROUP_PG_GLOO is None, "tp gloo groups are already initialized!"
    tp_group_spmd = get_tensor_model_parallel_group(as_list=True)
    rank = torch.distributed.get_rank()
    for tp_group in tp_group_spmd:
        pg = torch.distributed.new_group(ranks=tp_group, backend="gloo")
        if rank in tp_group:
            TP_GROUP_PG_GLOO = pg

    global DP_GROUP_PG_GLOO
    assert DP_GROUP_PG_GLOO is None, "dp gloo groups are already initialized!"
    dp_group_spmd = get_data_parallel_group(as_list=True)
    rank = torch.distributed.get_rank()
    for dp_group in dp_group_spmd:
        pg = torch.distributed.new_group(ranks=dp_group, backend="gloo")
        if rank in dp_group:
            DP_GROUP_PG_GLOO = pg


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


def destroy_gloo_groups():
    global PP_GROUP_PG_GLOO
    global TP_GROUP_PG_GLOO
    global DP_GROUP_PG_GLOO
    PP_GROUP_PG_GLOO = None
    TP_GROUP_PG_GLOO = None
    DP_GROUP_PG_GLOO = None


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
    def _clip_grad_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
    ) -> torch.Tensor:
        all_parameters = []
        for param_group, copied_param_group in zip(self.param_groups, self.base_optimizer.param_groups):
            for param, copied in zip(param_group["params"], copied_param_group["params"]):
                if param.grad is not None:
                    all_parameters.append(copied)

        total_norm = torch.nn.utils.clip_grad_norm_(
            all_parameters,
            max_norm=max_norm,
            norm_type=norm_type,
        )
        total_norm = total_norm.double()  # gloo not support bfloat16
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
        self._grad_norm = total_norm.to(self.optimizer_dtype)

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert self.inited, "must call init_zero() first"

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # sync to base optimizer
        self._sync_param_groups(self.param_groups, self.base_optimizer.param_groups)

        # do master copy
        for param_group, copied_param_group in zip(self.param_groups, self.base_optimizer.param_groups):
            for param, copied in zip(param_group["params"], copied_param_group["params"]):
                if param.grad is not None:
                    padded_grad = param.grad.clone()
                    grad_copied = padded_grad

                    # reduce grads
                    # cast to fp32 and cast back as gloo not support bf16
                    original_dtype = grad_copied.dtype
                    grad_copied = grad_copied.to(dtype=torch.float32)
                    grad_copied /= get_data_parallel_size()
                    torch.distributed.all_reduce(
                        grad_copied,
                        op=torch.distributed.ReduceOp.SUM,
                        group=get_dp_gloo_group(),
                    )
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
