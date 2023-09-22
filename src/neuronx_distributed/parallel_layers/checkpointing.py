import os
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm

from ..utils.logger import get_logger
from .layers import create_local_weight_cpu
from .parallel_state import (
    get_data_parallel_rank,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)
from .utils import add_barrier, cast_all, get_local_world_size

logger = get_logger()


def ensure_directory_exists(filename: str) -> None:
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_sharded_model_dict(model: torch.nn.Module, model_state_dict: dict) -> dict:
    from ..pipeline.model import NxDPPModel

    tp_size = get_tensor_model_parallel_size()
    if isinstance(model, NxDPPModel):
        model = model.original_torch_module
    # Use state_dict to keep the shared parameters
    for name, param in model.state_dict(keep_vars=True).items():
        if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
            if param.partition_dim not in [0, 1]:
                raise Exception(f"Partiton value of 0,1 are supported, found {param.partition_dim}.")
            per_partition_size = model_state_dict[name].shape[param.partition_dim] // tp_size
            full_weight = model_state_dict[name]
            model_state_dict[name] = create_local_weight_cpu(
                full_weight, param.partition_dim, per_partition_size, param.partition_stride
            )
    return model_state_dict


def save(
    checkpoint: dict,
    output_dir: str,
    save_serially: bool = True,
    down_cast_bf16: bool = False,
) -> None:
    """Save a model checkpoint."""

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        if rank == 0:
            logger.info("saving checkpoint to {}".format(output_dir))
    else:
        logger.info("saving checkpoint to {}".format(output_dir))
        rank = 0

    chkpt_path = output_dir
    chkpt_path = os.path.join(
        chkpt_path,
        "tp_rank_{:02d}_pp_rank{:02d}.pt".format(get_tensor_model_parallel_rank(), get_pipeline_model_parallel_rank()),
    )

    if down_cast_bf16:
        checkpoint = cast_all(checkpoint, from_dtype=torch.float32, to_dtype=torch.bfloat16)
    if rank % get_local_world_size() == 0:
        ensure_directory_exists(chkpt_path)
    add_barrier("Ensure directory exists Done")

    if save_serially:
        # TODO: optmization to save multiple ranks together
        for tp_rank in range(0, get_tensor_model_parallel_size()):
            for pp_rank in range(0, get_pipeline_model_parallel_size()):
                # Staggering save checkpoints
                if (
                    get_data_parallel_rank() == 0
                    and get_tensor_model_parallel_rank() == tp_rank
                    and get_pipeline_model_parallel_rank() == pp_rank
                ):
                    cpu_data = xm._maybe_convert_to_cpu(checkpoint)
                    torch.save(cpu_data, chkpt_path)
                add_barrier(f"ckpt-save-{tp_rank}")
    else:
        cpu_data = xm._maybe_convert_to_cpu(checkpoint, convert=(get_data_parallel_rank() == 0))
        if get_data_parallel_rank() == 0:
            torch.save(cpu_data, chkpt_path)

    should_chkpt = get_data_parallel_rank() == 0
    cpu_data = xm._maybe_convert_to_cpu(state_dict, convert=should_chkpt)
    if should_chkpt:
        ensure_directory_exists(chkpt_path)
        torch.save(cpu_data, chkpt_path)

    xm.rendezvous("Checkpoint Done")


def load(
    chkpt_path: str,
    model: torch.nn.Module = None,
    model_key: Optional[str] = "model",
    sharded: bool = True,
) -> dict:
    """Load a checkpoint and return. In case the model object is
    provided, it will load the model weights. For large models, to avoid
    host OOM, it is expected to pass the model object.
    """

    if not sharded:
        assert (
            model is not None
        ), "When checkpoint is not shareded, model object needs to be passed"

    if model is not None:
        from ..pipeline.model import NxDPPModel

        if (isinstance(model, NxDPPModel) and model.model_moved_to_device) or list(model.parameters())[
            0
        ].device == xm.xla_device():
            logger.warning(
                f"[Warning] It is recommended to call load \
                           before moving model to device to reduce redundant graphs."
            )

    # Checkpoint.
    if sharded:
        checkpoint_name = (
            os.path.join(
                chkpt_path,
                "tp_rank_{:02d}_pp_rank{:02d}.pt".format(
                    get_tensor_model_parallel_rank(), get_pipeline_model_parallel_rank()
                ),
            )
            if sharded
            else chkpt_path
        )
    else:
        checkpoint_name = chkpt_path

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.debug(f" loading checkpoint from {chkpt_path}")
    else:
        logger.debug(f" loading checkpoint from {chkpt_path}")

    world_size = get_tensor_model_parallel_size()
    rank = get_tensor_model_parallel_rank()
    for worker_start in range(0, world_size):
        if rank == worker_start:
            logger.debug(f"Worker {rank} resuming from checkpoint {checkpoint_name}")
            check_point = torch.load(checkpoint_name, map_location="cpu")
            if model:
                if model_key is not None:
                    model_state_dict = check_point[model_key]
                else:
                    model_state_dict = check_point
                if not sharded:
                    model_state_dict = get_sharded_model_dict(model, model_state_dict)
                model.load_state_dict(model_state_dict, strict=True)
                if model_key is not None:
                    del check_point[model_key]
                else:
                    check_point = None
            gc.collect()
        xm.rendezvous("neuron.load_checkpoint" + str(worker_start))

    return check_point
