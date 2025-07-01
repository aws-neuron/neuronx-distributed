import gc
import math
import os
from typing import Union, Optional, Callable, List, Any, Dict, Tuple

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer

from ..parallel_layers.checkpointing import ensure_directory_exists
from ..parallel_layers.grads import get_grad_norm, clip_grads_with_norm
from ..utils.model_utils import recursive_filter
from ..parallel_layers.parallel_state import (
    get_data_parallel_replica_groups,
    get_data_parallel_rank,
    get_expert_data_parallel_replica_groups,
    get_expert_data_parallel_size,
    get_expert_model_parallel_size,
    get_expert_model_parallel_replica_groups,
    get_tensor_model_parallel_rank,
    model_parallel_is_initialized,
)
from ..parallel_layers.utils import get_local_world_size, move_all_tensor_to_cpu
from ..utils.logger import get_logger

logger = get_logger()


class NeuronZero1Optimizer(ZeroRedundancyOptimizer):
    def __init__(self, *args, **kwargs):
        if not model_parallel_is_initialized():
            raise RuntimeError("initialize_model_parallel need to be called before creating NeuronZero1Optimizer")

        # Default to use DP groups for sharding
        if "sharding_groups" not in kwargs or kwargs["sharding_groups"] is None:
            kwargs["sharding_groups"] = get_data_parallel_replica_groups()

        # hard-code since use_world_for_grad_norm = True does not work with TP
        self._use_world_for_grad_norm = False

        super().__init__(*args, **kwargs)
        if kwargs.get("lazy_init"):
            from neuronx_distributed.trainer import hooks
            from neuronx_distributed.trainer.trainer import (
                filter_to_local_parameter_group,
            )

            hooks.register_post_partition_hook(filter_to_local_parameter_group, [self])
            hooks.register_post_partition_hook(self.init_zero)

    def _set_grad_norm(self, grad_norm):
        self._grad_norm = grad_norm.detach().clone()

    @property
    def grad_norm(self) -> Optional[torch.Tensor]:
        return self._grad_norm

    def _shard_parameters(self):
        try:
            return super()._shard_parameters()
        except Exception as e:
            if isinstance(e, AssertionError):
                raise RuntimeError(
                    "To use NeuronZero1Optimizer, all parameters passed in should on the same XLA device."
                    "If you are using pipeline parallel, use `model.local_parameters` or `model.local_named_parameters`."
                )
            else:
                raise e

    def _get_params_and_grad_norm(self, norm_type):
        all_parameters = []
        for param_group, sharded_param_group in zip(self.param_groups, self.base_optimizer.param_groups):
            for param, shard in zip(param_group["params"], sharded_param_group["params"]):
                if shard.grad is not None:
                    if hasattr(param, "shared"):
                        shard.shared = param.shared
                    if hasattr(param, "tensor_model_parallel"):
                        shard.tensor_model_parallel = param.tensor_model_parallel
                    if hasattr(param, "expert_model_parallel"):
                        shard.expert_model_parallel = True
                    all_parameters.append(shard)

        zero1_optimizer_groups = None if self._use_world_for_grad_norm else self._sharding_groups

        # Get norm
        grad_norm = get_grad_norm(
            all_parameters,
            norm_type=norm_type,
            zero1_optimizer=True,
            zero1_optimizer_groups=zero1_optimizer_groups,
        )
        return all_parameters, grad_norm

    @torch.no_grad()
    def _clip_grad_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
    ) -> None:

        all_parameters, grad_norm = self._get_params_and_grad_norm(norm_type)
        self._set_grad_norm(grad_norm)
        clip_grads_with_norm(all_parameters, grad_norm, max_norm)

    # [TODO] Remove this method
    def save_sharded_state_dict(self, output_dir: str, num_workers_per_step: int = 8) -> None:
        """Save a model checkpoint."""
        logger.info(
            "`NeuronZero1Optimizer.save_sharded_state_dict` is deprecated, please use `nxd.save_checkpoint` instead."
        )

        logger.debug("optimizer.saving checkpoint to %s", output_dir)

        state_dict = self.state_dict()
        state_dict["dp_rank"] = get_data_parallel_rank()
        state_dict["tp_rank"] = get_tensor_model_parallel_rank()

        chkpt_path = output_dir
        chkpt_path = os.path.join(
            chkpt_path,
            "optim.dp_rank_{:02d}.tp_rank_{:02d}".format(state_dict["dp_rank"], state_dict["tp_rank"]),
        )
        ensure_directory_exists(chkpt_path)

        local_rank = xr.local_ordinal()
        for worker in range(math.ceil(get_local_world_size() / num_workers_per_step)):
            if local_rank // num_workers_per_step == worker:
                logger.debug("optimizer.worker %d saving checkpoint %s", local_rank, chkpt_path)
                cpu_data = move_all_tensor_to_cpu(state_dict)
                torch.save(cpu_data, chkpt_path)
                del cpu_data
                gc.collect()
            xm.rendezvous("optimizer.save_checkpoint" + str(worker))

        xm.rendezvous("optimizer checkpoint done")

    # [TODO] Remove this method
    def load_sharded_state_dict(self, output_dir: str, num_workers_per_step: int = 8) -> None:
        logger.info(
            "`NeuronZero1Optimizer.load_sharded_state_dict` is deprecated, please use `nxd.load_checkpoint` instead."
        )

        chkpt_path = output_dir
        chkpt_path = os.path.join(
            chkpt_path,
            "optim.dp_rank_{:02d}.tp_rank_{:02d}".format(get_data_parallel_rank(), get_tensor_model_parallel_rank()),
        )

        logger.debug("optimizer.loading checkpoint from %s", chkpt_path)

        local_rank = xr.local_ordinal()
        for worker in range(math.ceil(get_local_world_size() / num_workers_per_step)):
            if local_rank // num_workers_per_step == worker:
                logger.debug("optimizer.worker %d resuming from checkpoint %s", local_rank, chkpt_path)
                check_point = torch.load(chkpt_path, map_location="cpu")
                self.load_state_dict(check_point)
                del check_point
                gc.collect()
            xm.rendezvous("optimizer.load_checkpoint" + str(worker))


class NeuronEPZero1Optimizer(NeuronZero1Optimizer):
    def __init__(self, *args, **kwargs):
        parameters = args[0] if len(args) > 0 else kwargs["params"]
        ep_parameters = recursive_filter(parameters, self._is_ep_param)
        non_ep_parameters = recursive_filter(parameters, lambda x: not self._is_ep_param(x))

        # Default to use DP groups for sharding
        if "sharding_groups" not in kwargs or kwargs["sharding_groups"] is None:
            kwargs["sharding_groups"] = get_data_parallel_replica_groups()
        elif kwargs["sharding_groups"] != get_data_parallel_replica_groups():
            raise ValueError("Custom sharding group for Zero-1 with expert parallelism is not supported.")

        if "params" in kwargs:
            kwargs.pop("params")

        self.non_ep_zero_optimizer = NeuronZero1Optimizer(non_ep_parameters, *args[1:], **kwargs)

        kwargs["sharding_groups"] = get_expert_data_parallel_replica_groups()
        self.ep_zero_optimizer = NeuronZero1Optimizer(ep_parameters, *args[1:], **kwargs)

        # Avoid this optimizer to create hooks for grad accumulation
        kwargs["use_grad_acc_hook"] = False

        super(NeuronEPZero1Optimizer, self).__init__(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Reset the gradients of the two optimizers and hooked grad accumulators
        when GPU-compatible precision is enabled."""
        self.ep_zero_optimizer.zero_grad(set_to_none=set_to_none)
        self.non_ep_zero_optimizer.zero_grad(set_to_none=set_to_none)

    def _is_ep_param(self, param):
        return hasattr(param, "expert_model_parallel") and param.expert_model_parallel

    def _filter_param_groups(
        self, groups: List[Dict[str, List[Any]]], predicate: Callable[..., List[Any]]
    ) -> List[Dict[str, List[Any]]]:
        filtered = [
            {k: v if k != "params" else [p for p in filter(predicate, v)]} for group in groups for k, v in group.items()
        ]
        return filtered

    @property
    def sharding_groups(self):
        return self.non_ep_zero_optimizer._sharding_groups

    @sharding_groups.setter
    def sharding_groups(self, new_sharding_groups):
        assert not self.inited, "already inited, cannot change sharding_groups"
        self.non_ep_zero_optimizer._sharding_groups = new_sharding_groups

    def _combine_grad_norms(self, norms, norm_type):
        if torch.isinf(torch.tensor(norm_type)):
            return max(norms)

        norm_squares = [torch.pow(n, norm_type) for n in norms]
        return torch.pow(sum(norm_squares), 1.0 / norm_type)

    @torch.no_grad()
    def _clip_grad_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
    ) -> None:

        non_ep_parameters, non_ep_grad_norm = self.non_ep_zero_optimizer._get_params_and_grad_norm(norm_type)

        # break the graph between the two grad norm calls to avoid runtime error
        # TODO remove this
        xm.mark_step()

        ep_parameters, ep_grad_norm = self.ep_zero_optimizer._get_params_and_grad_norm(norm_type)

        grad_norm = self._combine_grad_norms([non_ep_grad_norm, ep_grad_norm], norm_type)
        self._set_grad_norm(grad_norm)

        clip_grads_with_norm(non_ep_parameters + ep_parameters, grad_norm, max_norm)

    def _get_sharding_schemes(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # sequentially reduce over expert-data-parallel and expert-model-parallel groups
        edp_groups = get_expert_data_parallel_replica_groups()
        edp_size = get_expert_data_parallel_size()
        emp_size = get_expert_model_parallel_size()
        non_ep_sharding_scheme = [
            {
                "sharding_group": edp_groups,
                "group_size": edp_size,
                "scale_factor": 1.0,
            },
            {
                "sharding_group": get_expert_model_parallel_replica_groups(),
                "group_size": emp_size,
                "scale_factor": 1.0,
            },
        ]

        ep_sharding_scheme = [
            {
                "sharding_group": edp_groups,
                "group_size": edp_size,
                # EP grads further need to be scaled down by EP degree
                "scale_factor": 1.0 / emp_size,
            },
        ]

        return non_ep_sharding_scheme, ep_sharding_scheme

    def _reduce_gradients(self, **kwargs) -> None:

        non_ep_sharding_scheme, ep_sharding_scheme = self._get_sharding_schemes()

        # LR scheduler will modify some training parameters like learning rate
        # of this optimizer, so we need to propagate these parameters to the
        # base optimizers.
        self._sync_param_groups(self.param_groups, self.non_ep_zero_optimizer.param_groups)
        self.non_ep_zero_optimizer._reduce_gradients(sharding_scheme=non_ep_sharding_scheme, **kwargs)  # noqa: W0212

        self._sync_param_groups(self.param_groups, self.ep_zero_optimizer.param_groups)
        self.ep_zero_optimizer._reduce_gradients(sharding_scheme=ep_sharding_scheme, **kwargs)  # noqa: W0212

    def _update_parameters(self, **kwargs) -> None:
        non_ep_sharding_scheme, ep_sharding_scheme = self._get_sharding_schemes()

        self.non_ep_zero_optimizer._update_parameters(sharding_scheme=non_ep_sharding_scheme, **kwargs)  # noqa: W0212
        self.ep_zero_optimizer._update_parameters(sharding_scheme=ep_sharding_scheme, **kwargs)  # noqa: W0212

    def _get_offset(self, dct: Dict[int, int]) -> int:
        if len(dct) > 0:
            return max(k for k in dct) + 1
        return 1

    def state_dict(self) -> Dict[str, Any]:
        """Combine the state_dicts of the two base optimizers"""

        non_ep_state_dict = self.non_ep_zero_optimizer.state_dict()
        ep_state_dict = self.ep_zero_optimizer.state_dict()
        ep_param_id_offset = self._get_offset(non_ep_state_dict["state"])
        ep_base_state_offset = self._get_offset(non_ep_state_dict["base_state"])
        ep_shape_info_offset = self._get_offset(non_ep_state_dict["shape_info"])
        ep_param_group_offset = len(non_ep_state_dict["param_groups"])

        state_dict: Dict[str, Any] = {
            "ep_param_id_offset": ep_param_id_offset,
            "ep_param_group_offset": ep_param_group_offset,
            "ep_base_state_offset": ep_base_state_offset,
            "ep_shape_info_offset": ep_shape_info_offset,
        }

        # combine param_groups
        state_dict["param_groups"] = non_ep_state_dict["param_groups"] + ep_state_dict["param_groups"]

        # combine state
        state: Dict[int, int] = non_ep_state_dict["state"]
        state_dict["state"] = state
        base_state: Dict[int, int] = non_ep_state_dict["base_state"]
        state_dict["base_state"] = base_state
        shape_info: Dict[int, int] = non_ep_state_dict["shape_info"]
        state_dict["shape_info"] = shape_info
        for k, v in ep_state_dict["state"].items():
            state[ep_param_id_offset + k] = v

        for k, v in ep_state_dict["base_state"].items():
            base_state[ep_base_state_offset + k] = v

        for k, v in ep_state_dict["shape_info"].items():
            shape_info[ep_shape_info_offset + k] = v

        return state_dict

    def load_state_dict(self, state_dict):
        """Split the state_dict for the two base optimizers and load them individually"""

        if (
            "ep_param_id_offset" not in state_dict
            or "ep_param_group_offset" not in state_dict
            or "ep_base_state_offset" not in state_dict
            or "ep_shape_info_offset" not in state_dict
        ):
            raise ValueError("state_dict is not compatible with expert parallelism and Zero-1.")

        ep_param_id_offset = state_dict["ep_param_id_offset"]
        ep_param_group_offset = state_dict["ep_param_group_offset"]
        ep_base_state_offset = state_dict["ep_base_state_offset"]
        ep_shape_info_offset = state_dict["ep_shape_info_offset"]

        # split param_groups
        non_ep_param_groups = state_dict["param_groups"][:ep_param_group_offset]
        ep_param_groups = state_dict["param_groups"][ep_param_group_offset:]

        def _split_states(state_key, offset):
            non_ep_states, ep_states = {}, {}
            for k, v in state_dict[state_key].items():
                if k < offset:
                    non_ep_states[k] = v
                else:
                    ep_states[k - offset] = v
            return non_ep_states, ep_states

        non_ep_state, ep_state = _split_states("state", ep_param_id_offset)
        non_ep_base_state, ep_base_state = _split_states("base_state", ep_base_state_offset)
        non_ep_shape_info, ep_shape_info = _split_states("shape_info", ep_shape_info_offset)

        non_ep_state_dict = {
            "state": non_ep_state,
            "base_state": non_ep_base_state,
            "shape_info": non_ep_shape_info,
            "param_groups": non_ep_param_groups,
        }

        ep_state_dict = {
            "state": ep_state,
            "base_state": ep_base_state,
            "shape_info": ep_shape_info,
            "param_groups": ep_param_groups,
        }

        self.non_ep_zero_optimizer.load_state_dict(non_ep_state_dict)
        self.ep_zero_optimizer.load_state_dict(ep_state_dict)
