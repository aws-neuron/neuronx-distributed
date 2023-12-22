import gc
import math
import os
import shutil

import torch
import torch_xla.core.xla_model as xm
import torch_xla.utils.serialization as xser

from neuronx_distributed.optimizer import NeuronZero1Optimizer
from neuronx_distributed.parallel_layers.checkpointing import _xser_load
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
)
from neuronx_distributed.parallel_layers.utils import (
    get_local_world_size,
    move_all_tensor_to_cpu,
)
from neuronx_distributed.pipeline import NxDPPModel
from neuronx_distributed.trainer.optimizer import NxDOptimizer
from neuronx_distributed.utils.logger import get_logger

logger = get_logger()


def _get_path(prefix, tp=True, pp=True, dp=False):
    path = ""
    path += "_dp_rank_{:02d}".format(get_data_parallel_rank() if dp else 0)
    path += "_tp_rank_{:02d}".format(get_tensor_model_parallel_rank() if tp else 0)
    path += "_pp_rank_{:02d}".format(get_pipeline_model_parallel_rank() if pp else 0)
    if path != "":
        path = path[1:]
    path += ".pt"
    return "{}/{}".format(prefix, path)


def _save(ckpt, path, dp=False, num_workers=8, use_xser=False):
    # quick path when use xser
    if use_xser:
        num_workers = get_local_world_size()

    local_rank = xm.get_local_ordinal()
    for worker in range(math.ceil(get_local_world_size() / num_workers)):
        if dp or (not dp and get_data_parallel_rank() == 0):
            if local_rank // num_workers == worker:
                logger.debug(f"worker {local_rank} saving checkpoint {path}")
                if use_xser:
                    ref_data = xser._rewrite_data(xser._get_tensors_folder(path), ckpt, True)
                    torch.save(ref_data, path)
                    del ref_data
                else:
                    cpu_data = move_all_tensor_to_cpu(ckpt)
                    torch.save(cpu_data, path)
                    del cpu_data
                gc.collect()
        xm.rendezvous(f"worker-{worker}: checkpoint saved")
    xm.rendezvous("save checkpoint done")


def _load(obj, path, num_workers=8, strict=True, use_xser=False):
    # quick path when use xser
    if use_xser:
        num_workers = get_local_world_size()

    local_rank = xm.get_local_ordinal()
    for worker in range(math.ceil(get_local_world_size() / num_workers)):
        if local_rank // num_workers == worker:
            logger.debug(f"worker {local_rank} loading checkpoint {path}")
            if use_xser:
                ckpt = _xser_load(path)
            else:
                ckpt = torch.load(path, map_location="cpu")
            if isinstance(obj, torch.nn.Module):
                obj.load_state_dict(ckpt, strict=strict)
            else:
                obj.load_state_dict(ckpt)
            del ckpt
            gc.collect()
        xm.rendezvous(f"worker-{worker}: checkpoint loaded")
    xm.rendezvous("load checkpoint done")


def save_checkpoint(
    path,
    tag,
    model=None,
    optimizer=None,
    scheduler=None,
    user_content=None,
    num_workers=8,
    use_xser=False,
    num_kept_ckpts=None,
):
    """
    Method to save checkpoint, return ``None``.

    In ``use_xser`` is ``True``, the file structure looks like:
    - output_dir:
      - tag:
        - model or optim:
          - dp_rank_xx_tp_rank_xx_pp_rank_xx.pt (ref_data file)
          - dp_rank_xx_tp_rank_xx_pp_rank_xx.pt.tensors:
            - tensor_x.pt
        - scheduler.pt
        - user_content.pt
      - newest

    Otherwise, the file structure looks like:
    - output_dir:
      - tag:
        - model or optim:
          - dp_rank_xx_tp_rank_xx_pp_rank_xx.pt
        - scheduler.pt
        - user_content.pt
      - newest

    Parameters:
        path (str):
            path to save the checkpoints.
        tag (str):
            tag to save the checkpoints.
        model (torch.nn.Module):
            model to save, optinal.
        optimizer (torch.optim.Optimizer):
            optimizer to save, optinal.
        scheduler:
            scheduler to save, optinal.
        user_content:
            user contents to save, optinal.
        num_workers (int):
            num of workers to save the checkpoints on the same time, range: 1-32.
        use_xser (bool):
            whether to use torch-xla serialization. When enabled, ``num_workers`` will be ignored
            and maximum num of workers will be used. Default: ``False``.
        num_kept_ckpts (int):
            number of checkpoints to keep on disk, optional. Default: ``None``.
    """
    # TODO: Use distributed checkpoint
    assert torch.distributed.is_initialized(), "Only support distributed training mode."

    ckpt_path = path
    os.makedirs(ckpt_path, exist_ok=True)

    if torch.distributed.get_rank() == 0:
        newest_file = os.path.join(path, "newest")
        if os.path.isfile(newest_file):
            with open(newest_file, "r") as fd:
                existing_files = fd.readlines()
        else:
            existing_files = []
        with open(newest_file, "w") as fd:
            # Remove the oldest checkpoints until existing_files meets num_kept_ckpts-1
            if num_kept_ckpts is not None:
                while len(existing_files) >= num_kept_ckpts:
                    oldest_dir = os.path.join(path, existing_files[0][:-1])
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                    existing_files = existing_files[1:]
            existing_files.append(f"{tag}\n")
            fd.writelines(existing_files)

    ckpt_path = os.path.join(ckpt_path, str(tag))

    if torch.distributed.get_rank() == 0:
        logger.info("saving checkpoint to {}".format(ckpt_path))

    xm.rendezvous("save all checkpoints start")

    # save model
    if model is not None:
        os.makedirs(os.path.join(ckpt_path, "model"), exist_ok=True)
        model_path = os.path.join(ckpt_path, _get_path("model"))
        if isinstance(model, NxDPPModel):
            ckpt = model.local_state_dict()
        else:
            ckpt = model.state_dict()
        _save(ckpt, model_path, num_workers=num_workers, use_xser=use_xser)

    # save optimizer
    if optimizer is not None:
        os.makedirs(os.path.join(ckpt_path, "optim"), exist_ok=True)
        if isinstance(optimizer, NxDOptimizer):
            zero1_enabled = optimizer.nxd_config["optimizer_config"]["zero_one_enabled"]
        else:
            zero1_enabled = isinstance(optimizer, NeuronZero1Optimizer)
        optimizer_path = os.path.join(ckpt_path, _get_path("optim", dp=zero1_enabled))
        _save(optimizer.state_dict(), optimizer_path, dp=zero1_enabled, num_workers=num_workers, use_xser=use_xser)

    # save scheduler
    if scheduler is not None:
        if torch.distributed.get_rank() == 0:
            torch.save(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

    # save user content
    if user_content is not None:
        if torch.distributed.get_rank() == 0:
            torch.save(user_content, os.path.join(ckpt_path, "user_content.pt"))

    xm.rendezvous("save all checkpoints done")


def load_checkpoint(
    path,
    tag=None,
    model=None,
    optimizer=None,
    scheduler=None,
    num_workers=8,
    strict=True,
):
    """
    Method to load checkpoint, return user contents if exists otherwise ``None``.
    If ``tag`` not provided, will try to use the newest tag tracked by ``save_checkpoint``.

    Parameters:
        path (str):
            path to load the checkpoints.
        tag (str):
            tag to load the checkpoints.
        model (torch.nn.Module):
            model to load, optinal.
        optimizer (torch.optim.Optimizer):
            optimizer to load, optinal.
        scheduler:
            scheduler to load, optinal.
        num_workers (int):
            num of workers to load the checkpoints on the same time, range: 1-32.
        strict (bool):
            whether to use strict mode when loading model checkpoint. Default: ``True``.
    """
    assert torch.distributed.is_initialized(), "Only support distributed training mode."

    if tag is None:
        newest_path = os.path.join(path, "newest")
        assert os.path.exists(newest_path) and os.path.isfile(newest_path)
        with open(newest_path, "r") as fd:
            tag = fd.readlines()[-1][:-1]

    ckpt_path = os.path.join(path, str(tag))
    assert os.path.exists(ckpt_path) and os.path.isdir(ckpt_path)

    use_xser = False
    for x in os.listdir(ckpt_path):
        inner_path = os.path.join(ckpt_path, x)
        if os.path.isdir(inner_path):
            for y in os.listdir(inner_path):
                if y.endswith(".tensors"):
                    use_xser = True
                    break
        if use_xser:
            break

    if torch.distributed.get_rank() == 0:
        logger.info("loading checkpoint from {}".format(ckpt_path))

    # load model
    if model is not None:
        model_path = os.path.join(ckpt_path, _get_path("model"))
        _load(model, model_path, num_workers=num_workers, strict=strict, use_xser=use_xser)

    # load optimizer
    if optimizer is not None:
        if isinstance(optimizer, NxDOptimizer):
            zero1_enabled = optimizer.nxd_config["optimizer_config"]["zero_one_enabled"]
        else:
            zero1_enabled = isinstance(optimizer, NeuronZero1Optimizer)
        optimizer_path = os.path.join(ckpt_path, _get_path("optim", dp=zero1_enabled))
        _load(optimizer, optimizer_path, num_workers=num_workers, use_xser=use_xser)

    # load scheduler
    if scheduler is not None:
        ckpt = torch.load(os.path.join(ckpt_path, "scheduler.pt"), map_location="cpu")
        scheduler.load_state_dict(ckpt)

    # load user content
    user_content = None
    user_content_path = os.path.join(ckpt_path, "user_content.pt")
    if os.path.exists(user_content_path):
        user_content = torch.load(user_content_path, map_location="cpu")

    xm.rendezvous("load all checkpoints done")
    return user_content
