import os
import gc
import logging
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from .parallel_state import (
    get_data_parallel_rank,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def ensure_directory_exists(filename: str) -> None:
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_sharded_model_dict(model: torch.nn.Module, model_state_dict: dict) -> dict:
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_size()
    for name, param in model.named_parameters():
        if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
            per_partition_size = (
                model_state_dict[name].shape[param.partition_dim] // tp_size
            )
            if param.partition_dim == 0:
                model_state_dict[name] = model_state_dict[name][
                    per_partition_size * tp_rank : per_partition_size * (tp_rank + 1)
                ]
            elif param.partition_dim == 1:
                model_state_dict[name] = model_state_dict[name][
                    :, per_partition_size * tp_rank : per_partition_size * (tp_rank + 1)
                ]
            else:
                raise Exception(
                    f"Partiton value of 0,1 are supported, found {param.partition_dim}."
                )
    return model_state_dict


def save(state_dict: dict, output_dir: str) -> None:
    """Save a model checkpoint."""

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.debug("saving checkpoint to {}".format(output_dir))
    else:
        logger.debug("saving checkpoint to {}".format(output_dir))

    state_dict["tp_rank"] = get_tensor_model_parallel_rank()

    chkpt_path = output_dir
    chkpt_path = os.path.join(
        chkpt_path, "tp_rank_{:02d}".format(get_tensor_model_parallel_rank())
    )

    if down_cast_bf16:
        state_dict = cast_all(state_dict, from_dtype=torch.float32, to_dtype=torch.bfloat16)
    if get_data_parallel_rank() == 0:
        ensure_directory_exists(chkpt_path)
    if save_serially:
        cpu_data = xm._maybe_convert_to_cpu(state_dict, convert=(get_data_parallel_rank() == 0))
        for tp_rank in range(0, get_tensor_model_parallel_size()):
            # Staggering save checkpoints
            if get_data_parallel_rank() == 0 and get_tensor_model_parallel_rank() == tp_rank:
                torch.save(cpu_data, chkpt_path)
            add_barrier(f"ckpt-save-{tp_rank}")
    else:
        cpu_data = xm._maybe_convert_to_cpu(state_dict, convert=(get_data_parallel_rank() == 0))
        torch.save(cpu_data, chkpt_path)

    should_chkpt = get_data_parallel_rank() == 0
    cpu_data = xm._maybe_convert_to_cpu(state_dict, convert=should_chkpt)
    if should_chkpt:
        ensure_directory_exists(chkpt_path)
        torch.save(cpu_data, chkpt_path)

    xm.rendezvous("Checkpoint Done")


def load(
    output_dir: str,
    model: torch.nn.Module = None,
    model_key: str = "model",
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

    # Checkpoint.
    chkpt_path = output_dir
    checkpoint_name = (
        os.path.join(
            chkpt_path, "tp_rank_{:02d}".format(get_tensor_model_parallel_rank())
        )
        if sharded
        else chkpt_path
    )

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
                model_state_dict = check_point[model_key]
                if not sharded:
                    model_state_dict = get_sharded_model_dict(model, model_state_dict)
                model.load_state_dict(model_state_dict, strict=True)
                del check_point[model_key]
            gc.collect()
        xm.rendezvous("neuron.load_checkpoint" + str(worker_start))

    return check_point
