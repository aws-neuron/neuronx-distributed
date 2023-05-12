import os
import gc
import logging
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
from .parallel_state import (get_data_parallel_rank,
                             get_tensor_model_parallel_rank,
                             get_tensor_model_parallel_world_size)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save(data, file_or_path):
    should_chkpt = get_data_parallel_rank() ==0 
    cpu_data = xm._maybe_convert_to_cpu(data, convert=should_chkpt)
    if should_chkpt:
        logger.debug('Save path:{}'.format(file_or_path))
        ensure_directory_exists(file_or_path)
        torch.save(cpu_data, file_or_path)


def save_checkpoint(step, epoch, model, optimizer, lr_scheduler, output_dir, minimal_ckpt=None):
    """Save a model checkpoint."""

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.debug('saving checkpoint at step {:7d} to {}'.format(
                step, output_dir))
    else:
        logger.debug('saving checkpoint at step {:7d} to {}'.format(
            step, output_dir))

    state_dict = {}

    state_dict['step'] = step
    state_dict['epoch'] = epoch
    state_dict['model'] = model.state_dict()
    state_dict['tp_rank'] = get_tensor_model_parallel_rank()

    if not minimal_ckpt:
        if optimizer is not None:
            state_dict['optimizer'] = optimizer.state_dict()
        if lr_scheduler is not None:
            state_dict['lr_scheduler'] = lr_scheduler.state_dict()

    chkpt_path = output_dir
    checkpoint_name = os.path.join(
        chkpt_path, 'mp_rank_{:02d}_step_{:d}'.format(
            get_tensor_model_parallel_rank(), step))

    if get_data_parallel_rank() == 0:
        ensure_directory_exists(checkpoint_name)

    save(state_dict, checkpoint_name)

    xm.rendezvous('Checkpoint Done')


def load_checkpoint(model, optimizer, step, output_dir):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """

    # Checkpoint.
    chkpt_path = output_dir
    checkpoint_name = os.path.join(
        chkpt_path, 'mp_rank_{:02d}_step_{:d}'.format(
            get_tensor_model_parallel_rank(), step))

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.debug(f' loading checkpoint from {chkpt_path} at step {step}')
    else:
        logger.debug(f' loading checkpoint from {chkpt_path} at step {step}')

    world_size = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    for worker_start in range(0, world_size):
        if rank == worker_start:
            logger.debug(
                f'Worker {rank} resuming from checkpoint {checkpoint_name} at step {step}')
            check_point = torch.load(checkpoint_name, map_location='cpu')
            model.load_state_dict(check_point['model'], strict=True)
            if 'optimizer' in check_point:
                optimizer.load_state_dict(check_point['optimizer'])
            if 'lr_scheduler' in check_point:
                scheduler_state_dict = check_point.pop('lr_scheduler')

            epoch = check_point.get('epoch', 0)
            del check_point
            gc.collect()
        xm.rendezvous('neuron.load_checkpoint' + str(worker_start))

    return step, epoch, scheduler_state_dict