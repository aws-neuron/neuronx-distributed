import gc
import math
import os
from typing import Union

import torch
import torch_xla.core.xla_model as xm
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer

from ..parallel_layers.checkpointing import ensure_directory_exists
from ..parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_data_parallel_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)
from ..parallel_layers.utils import add_barrier, get_local_world_size
from ..utils.logger import get_logger

logger = get_logger()


class NeuronZero1Optimizer(ZeroRedundancyOptimizer):
    @torch.no_grad()
    def _calc_grad_norm(
        self,
        norm_type: Union[float, int] = 2.0,
    ) -> torch.Tensor:
        grads_for_norm = []
        duplicate_grads_for_norm = []
        # use a trick to become spmd here
        for param_group, sharded_param_group in zip(self.param_groups, self.base_optimizer.param_groups):
            for param, shard in zip(param_group["params"], sharded_param_group["params"]):
                if param.grad is not None:
                    is_not_shared = not hasattr(param, "shared") or not param.shared
                    is_not_tp_duplicate = hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
                    if is_not_shared:
                        if is_not_tp_duplicate:
                            grads_for_norm.append(shard.grad.detach())
                        else:
                            duplicate_grads_for_norm.append(shard.grad.detach())
        # Norm parameters.
        if norm_type != 2.0:
            raise RuntimeError(f"only norm type 2 is supported, getting {norm_type}")
        total_norm = torch.zeros([], dtype=self.optimizer_dtype, device=self.device)
        for grad in duplicate_grads_for_norm:
            grad_norm = (grad * grad).sum()
            total_norm += grad_norm
        total_norm /= get_tensor_model_parallel_size()
        for grad in grads_for_norm:
            grad_norm = (grad * grad).sum()
            total_norm += grad_norm
        # All-reduce across data parallel groups
        xm.all_reduce(
            xm.REDUCE_SUM,
            [total_norm],
            groups=self._sharding_groups,
            pin_layout=self.pin_layout,
        )
        # All-reduce across other parallel groups, usually model parallel groups
        if self._grad_norm_groups is not None:
            xm.all_reduce(
                xm.REDUCE_SUM,
                [total_norm],
                groups=self._grad_norm_groups,
                pin_layout=self.pin_layout,
            )
        total_norm = torch.pow(total_norm, 1.0 / norm_type)
        return total_norm

    def save_sharded_state_dict(
        self,
        output_dir: str,
        save_serially: bool = True,
    ) -> None:
        """Save a model checkpoint."""

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                logger.debug("optimizer.saving checkpoint to {}".format(output_dir))
        else:
            logger.debug("optimizer.saving checkpoint to {}".format(output_dir))

        state_dict = self.state_dict()
        state_dict["dp_rank"] = get_data_parallel_rank()

        chkpt_path = output_dir
        chkpt_path = os.path.join(chkpt_path, "optim.dp_rank_{:02d}".format(get_data_parallel_rank()))

        if get_tensor_model_parallel_rank() == 0:
            ensure_directory_exists(chkpt_path)
        if save_serially:
            cpu_data = xm._maybe_convert_to_cpu(state_dict, convert=(get_tensor_model_parallel_rank() == 0))
            for dp_rank in range(0, get_data_parallel_size()):
                # Staggering save checkpoints
                if get_tensor_model_parallel_rank() == 0 and get_data_parallel_rank() == dp_rank:
                    torch.save(cpu_data, chkpt_path)
                add_barrier(f"optimizer.ckpt-save-{dp_rank}")
        else:
            cpu_data = xm._maybe_convert_to_cpu(state_dict, convert=(get_tensor_model_parallel_rank() == 0))
            torch.save(cpu_data, chkpt_path)

        add_barrier("optimizer checkpoint done")

    def load_sharded_state_dict(self, output_dir: str, num_workers_per_step=8) -> None:
        chkpt_path = output_dir
        chkpt_path = os.path.join(chkpt_path, "optim.dp_rank_{:02d}".format(get_data_parallel_rank()))

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
            add_barrier("optimizer.load_checkpoint" + str(worker))
