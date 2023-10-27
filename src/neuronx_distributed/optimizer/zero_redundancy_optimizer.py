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
    get_data_parallel_rank,
    get_data_parallel_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)
from ..parallel_layers.utils import (
    get_local_world_size,
    move_all_tensor_to_cpu,
)
from ..utils.logger import get_logger

logger = get_logger()


class NeuronZero1Optimizer(ZeroRedundancyOptimizer):
    @torch.no_grad()
    def _clip_grad_norm(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int] = 2.0,
    ) -> torch.Tensor:
        all_parameters = []
        for param_group, sharded_param_group in zip(self.param_groups, self.base_optimizer.param_groups):
            for param, shard in zip(param_group["params"], sharded_param_group["params"]):
                if param.grad is not None:
                    if hasattr(param, "shared"):
                        shard.shared = param.shared
                    if hasattr(param, "tensor_model_parallel"):
                        shard.tensor_model_parallel = param.tensor_model_parallel
                    all_parameters.append(shard)

        # [TODO] Find a way to expose global_norm to user
        global_norm = clip_grad_norm(all_parameters, max_norm=max_norm, norm_type=norm_type, zero1_optimizer=True)

    # [TODO] Use the save/load function from parallel_layers/checkpointing.py
    def save_sharded_state_dict(self, output_dir: str, num_workers_per_step: int = 8) -> None:
        """Save a model checkpoint."""

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

    # [TODO] Use the save/load function from parallel_layers/checkpointing.py
    def load_sharded_state_dict(self, output_dir: str, num_workers_per_step: int = 8) -> None:
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

