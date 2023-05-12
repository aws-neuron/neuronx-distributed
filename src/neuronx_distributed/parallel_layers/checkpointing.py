import os
import gc
import logging
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from .parallel_state import (get_data_parallel_rank,
                             get_tensor_model_parallel_rank,
                             get_tensor_model_parallel_size)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def ensure_directory_exists(filename: str) -> None:
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save(state_dict: dict, output_dir: str) -> None:
    """Save a model checkpoint."""

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.debug('saving checkpoint to {}'.format(output_dir))
    else:
        logger.debug('saving checkpoint to {}'.format(output_dir))

    state_dict['tp_rank'] = get_tensor_model_parallel_rank()

    chkpt_path = output_dir
    chkpt_path = os.path.join(
        chkpt_path, 'tp_rank_{:02d}'.format(get_tensor_model_parallel_rank()))

    if get_data_parallel_rank() == 0:
        ensure_directory_exists(chkpt_path)

    should_chkpt = get_data_parallel_rank() ==0 
    cpu_data = xm._maybe_convert_to_cpu(state_dict, convert=should_chkpt)
    if should_chkpt:
        ensure_directory_exists(chkpt_path)
        torch.save(cpu_data, chkpt_path)

    xm.rendezvous('Checkpoint Done')


def load(output_dir: str, model: torch.nn.Module =None, model_key: str ='model') -> dict:
    """Load a checkpoint and return. In case the model object is 
    provided, it will load the model weights. For large models, to avoid
    host OOM, it is expected to pass the model object.
    """

    # Checkpoint.
    chkpt_path = output_dir
    checkpoint_name = os.path.join(
        chkpt_path, 'tp_rank_{:02d}'.format(get_tensor_model_parallel_rank()))

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.debug(f' loading checkpoint from {chkpt_path}')
    else:
        logger.debug(f' loading checkpoint from {chkpt_path}')

    world_size = get_tensor_model_parallel_size()
    rank = get_tensor_model_parallel_rank()
    for worker_start in range(0, world_size):
        if rank == worker_start:
            logger.debug(
                f'Worker {rank} resuming from checkpoint {checkpoint_name}')
            check_point = torch.load(checkpoint_name, map_location='cpu')
            if model:
                model.load_state_dict(check_point[model_key], strict=True)
                del check_point[model_key]
            gc.collect()
        xm.rendezvous('neuron.load_checkpoint' + str(worker_start))

    return check_point