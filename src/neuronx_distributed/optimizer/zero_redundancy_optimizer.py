import gc
import math
import os
from typing import Union

import torch
import torch_xla.core.xla_model as xm
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer

from ..parallel_layers.checkpointing import ensure_directory_exists
from ..parallel_layers.grads import clip_grad_norm
from ..parallel_layers.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
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
            kwargs["sharding_groups"] = get_data_parallel_group(as_list=True)

        # If use dp groups for sharding, calculate the grad norm with world group
        if kwargs["sharding_groups"] == get_data_parallel_group(as_list=True):
            self._use_world_for_grad_norm = True
        else:
            self._use_world_for_grad_norm = False

        super().__init__(*args, **kwargs)

    @property
    def grad_norm(self):
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

    @torch.no_grad()
    def _clip_grad_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
    ) -> torch.Tensor:
        all_parameters = []
        for param_group, sharded_param_group in zip(self.param_groups, self.base_optimizer.param_groups):
            for param, shard in zip(param_group["params"], sharded_param_group["params"]):
                if shard.grad is not None:
                    if hasattr(param, "shared"):
                        shard.shared = param.shared
                    if hasattr(param, "tensor_model_parallel"):
                        shard.tensor_model_parallel = param.tensor_model_parallel
                    all_parameters.append(shard)

        zero1_optimizer_groups = None if self._use_world_for_grad_norm else self._sharding_groups
        self._grad_norm = clip_grad_norm(
            all_parameters,
            max_norm=max_norm,
            norm_type=norm_type,
            zero1_optimizer=True,
            zero1_optimizer_groups=zero1_optimizer_groups,
        )

    # [TODO] Remove this method
    def save_sharded_state_dict(self, output_dir: str, num_workers_per_step: int = 8) -> None:
        """Save a model checkpoint."""
        logger.info(
            "`NeuronZero1Optimizer.save_sharded_state_dict` is deprecated, please use `nxd.save_checkpoint` instead."
        )

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                logger.debug("optimizer.saving checkpoint to {}".format(output_dir))
        else:
            logger.debug("optimizer.saving checkpoint to {}".format(output_dir))

        state_dict = self.state_dict()
        state_dict["dp_rank"] = get_data_parallel_rank()
        state_dict["tp_rank"] = get_tensor_model_parallel_rank()

        chkpt_path = output_dir
        chkpt_path = os.path.join(
            chkpt_path,
            "optim.dp_rank_{:02d}.tp_rank_{:02d}".format(get_data_parallel_rank(), get_tensor_model_parallel_rank()),
        )
        ensure_directory_exists(chkpt_path)

        local_rank = xm.get_local_ordinal()
        for worker in range(math.ceil(get_local_world_size() / num_workers_per_step)):
            if local_rank // num_workers_per_step == worker:
                logger.debug(f"optimizer.worker {local_rank} saving checkpoint {chkpt_path}")
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

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                logger.debug(f"optimizer.loading checkpoint from {chkpt_path}")
        else:
            logger.debug(f"optimizer.loading checkpoint from {chkpt_path}")

        local_rank = xm.get_local_ordinal()
        for worker in range(math.ceil(get_local_world_size() / num_workers_per_step)):
            if local_rank // num_workers_per_step == worker:
                logger.debug(f"optimizer.worker {local_rank} resuming from checkpoint {chkpt_path}")
                check_point = torch.load(chkpt_path, map_location="cpu")
                self.load_state_dict(check_point)
                del check_point
                gc.collect()
            xm.rendezvous("optimizer.load_checkpoint" + str(worker))
