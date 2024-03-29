import gc
import math
import os
from datetime import datetime
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.serialization as xser
import concurrent.futures
from typing import List, Tuple
from neuronx_distributed.optimizer import NeuronZero1Optimizer
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_group,
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
from .checkpoint_storage import BaseCheckpointStorage, create_checkpoint_storage

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


def _determine_remove_paths(checkpoint_dir: BaseCheckpointStorage, num_kept: int):
    '''
    deteremine checkpoint directories to be removed to satisfy num_kept
    return value: relative path to checkpoint_dir
    '''
    if (num_kept is None) or (num_kept <= 0):
        # num_kept being None or -1 means to keep every checkpoints
        return []

    tags = checkpoint_dir.list_checkpoint_tags()
    if len(tags) <= num_kept:
        return []

    return tags[0: len(tags) - num_kept]


def _bulk_save(checkpoint_dir: BaseCheckpointStorage, save_items: List[Tuple[object, str]]):
    for obj, filename in save_items:
        checkpoint_dir.save_object(obj, filename)


class CheckpointIOState:
    '''
    class to store state of asynchronous checkpoint saving
    '''

    def __init__(self, async_save: bool =False):
        '''
        async_save : whether to use asynchronous checkpoint saving. Default no
       '''
        self._async_save = async_save
        self._current_tag = None

        if self._async_save:
            self._checkpoint_dir = None
            self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
            self._save_items : list[(torch.Tensor, src)] = list()
            self._save_task: concurrent.futurex = None
            self._remove_path : str = None
            self._remove_task : concurrent.future = None

    def begin(self, tag: str):
        if torch.distributed.get_rank() == 0:
            method = "async" if self._async_save else "synced"
            logger.info(f"{method} saving of checkpoint {tag} began")

        if not self._async_save:
            self._current_tag = tag
            return

        if self._current_tag is not None:
            self.wait_save()

        self._current_tag = tag

    def add_save_task(self, checkpoint_dir: BaseCheckpointStorage, obj: object, filename: str):
        self._checkpoint_dir = checkpoint_dir
        if self._async_save:
            self._save_items.append( (obj, filename) )
        else:
            self._checkpoint_dir.save_object(obj, filename)

    def end(self, num_kept: int):
        if self._async_save:
            self._num_kept = num_kept
            if len(self._save_items) > 0:
                self._save_task = self._executor.submit(_bulk_save, self._checkpoint_dir, self._save_items)
            if torch.distributed.get_rank() == 0:
                logger.info(f"async saving of checkpoint {self._current_tag} requested")
        else:
            xm.rendezvous("saving checkpoint done")
            if torch.distributed.get_rank() == 0:
                logger.info(f"synced saving of checkpoint {self._current_tag} completed")
                self._checkpoint_dir.save_text("1", os.path.join(self._current_tag, "done"))

            if torch.distributed.get_rank() == 0:
                self.submit_remove(num_kept)

    def wait_save(self):
        if not self._async_save:
            return

        # first wait for save to finish
        tasks = []
        if self._save_task:
            done, _ = concurrent.futures.wait([self._save_task])
            for f in done:
                if f.exception():
                    raise f.exception()

        xm.rendezvous(f"async saving checkpoint done")

        if self._save_task:
            self._save_task = None
            self._save_items = []
            if torch.distributed.get_rank()  == 0:
                self._checkpoint_dir.save_text("1", os.path.join(self._current_tag, "done"))

        if torch.distributed.get_rank() == 0:
            logger.info(f"async saving of checkpoint {self._current_tag} completed")

        # remove checkpoint if necessary.
        if torch.distributed.get_rank() == 0:
            # first wait previous round of removal to finish
            self.wait_remove()
            self.submit_remove(self._num_kept)

    def submit_remove(self, num_kept: int):
        remove_paths = _determine_remove_paths(self._checkpoint_dir, num_kept)
        if len(remove_paths) == 0:
            logger.info(f"no checkpoints to remove.")
        elif len(remove_paths) > 1:
            logger.error(f"There are more than 1 checkpoint under {self._checkpoint_dir.dirname()} to remove, which is not expected. Are you using the same checkpoint directory for 2 different runs?")
            raise RuntimeError("too many checkpoints to remove")
        else:
            logger.info(f"removing previous checkpoint in {remove_paths[0]}")
            # remove the done file first to avoid the situation
            # the deletion was interrupted by hardware failure,
            # leaving a partially deleted checkpoint with the "done" tag.
            # That can be picked up by resumed training job and cause failure
            done_file = os.path.join(remove_paths[0], "done")
            self._checkpoint_dir.remove_file(done_file)
            logger.info(f"done tag in {remove_paths[0]} cleared")
            if self._async_save:
                self._remove_path = remove_paths[0]
                self._remove_task = self._executor.submit(self._checkpoint_dir.remove_dir, self._remove_path)
                logger.info(f"async removal of {self._remove_path} requested.")
            else:
                self._checkpoint_dir.remove_dir(remove_paths[0])
                logger.info(f"previous checkpoint in {remove_paths[0]} successfully removed")

    def wait_remove(self):
        if not self._async_save:
            return

        if self._remove_task:
            done, _ = concurrent.futures.wait([self._remove_task])
            for f in done:
                if f.exception():
                    raise f.exception()

            logger.info(f"async removal of {self._remove_path} completed")
            self._remove_path = None
            self._remove_task = None

    def wait_all(self):
        # sequence is important here because wait_save may submit request to remove checkpoint
        self.wait_save()
        self.wait_remove()


def _get_my_group_info(groups: List[List[int]]):
    global_rank = torch.distributed.get_rank()
    for group in groups:
        if global_rank in group:
            return group.index(global_rank), len(group)

    raise RuntimeError(f"Error: global rank {global_rank} is not in groups")


def _xser_load_data(checkpoint_dir: BaseCheckpointStorage, path: str, groups: List[List[int]] = None):
    '''
    load tensors saved in path into a state_dict.
    Parameters:
    groups: a list of groups. Each group is a list of ranks whose path are the same. groups being None means every rank's data is unique.
           When groups is provided, 1 rank in a group will load data from path, then broadcast result to other ranks.
    '''
    ref_data = checkpoint_dir.load_object(path)
    # check the existance of info.pt file because older version (<=0.6.0) does not generate this file.
    # checkpoint generated using older version still need to be supported.
    ref_info = checkpoint_dir.load_object(path + ".info.pt") if checkpoint_dir.file_exists(path + ".info.pt") else None

    tensor_folder = path + ".tensors"

    if groups is not None:
        my_rank_in_group, my_group_size = _get_my_group_info(groups)

    def convert_fn(tensors):
        rewritten_tensors = []

        for t in tensors:
            tensor_file = os.path.join(tensor_folder, "tensor_{}.pt".format(t.tid))
            if (ref_info is not None) and (groups is not None):
                # When there is redundency (groups is not None) and we know the tensor's shape and dtype (ref_info is not None)
                # we use the following optimization:
                #    among workers that has same tensor (in same group), only 1 worker read tensor from disk
                #    other workers will get the tensor from network broadcasting
                #
                # we used round robin to select which worker will read from disk to evenly
                # distribute the load tasks.
                if (t.tid  % my_group_size) == my_rank_in_group:
                    loaded = checkpoint_dir.load_object(tensor_file).to(xm.xla_device())
                else:
                    dtype = ref_info[t.tid]["dtype"]
                    shape = ref_info[t.tid]["shape"]
                    loaded = torch.zeros(shape, dtype=dtype, device=xm.xla_device())
                # we use all_reduce to implement broadcast because xla does not have native broadcast support.
                xm.all_reduce(xm.REDUCE_SUM, [loaded], groups=groups)
            else:
                # when dtype and shape are not available or there is no redundency, all workers load tensor from disk
                loaded = checkpoint_dir.load_object(tensor_file).to(xm.xla_device())

            rewritten_tensors.append(loaded)

        if groups is not None:
            xm.mark_step()
        return rewritten_tensors

    def select_fn(v):
         return type(v) == xser.TensorReference

    return xm.ToXlaTensorArena(convert_fn, select_fn).transform(ref_data)


class _InternalTensorReference:

    def __init__(self, tid, shape, dtype):
        self.tid = tid
        self.shape = shape
        self.dtype = dtype

def _assign_tensors_to_bins(tensors, bin_count) -> List[List[int]]:
    '''
    assign a list of tensors into multiple bins, such that each bin's
    total tensor size are similar.
    Args:
    tensors: a list of tensors
    bin_count: number of bins
    Return
    a list of list, each sublist contain indices of tensors.
    '''

    bin_tidxs = [ [] for i in range(bin_count)]
    bin_sizes = [ 0 for i in range(bin_count)]

    tensor_sizes = []
    for i,tensor in enumerate(tensors):
        tensor_sizes.append( (i, torch.numel(tensor) * tensor.element_size()))

    # we use Karmarkar–Karp bin packing algorithm to yield most evenly distributed bin
    # total size. It goes like the following:
    # First, sort tensor by size.
    # Then loop over all tensors.
    # For each tensor find the bin with smallest total size, and assign the tensor
    # to the bin.
    tensor_sizes = sorted(tensor_sizes, key=lambda a: a[1])

    for tidx, tensor_size in tensor_sizes:
        bid = bin_sizes.index(min(bin_sizes))
        bin_tidxs[bid].append(tidx)
        bin_sizes[bid] += tensor_size
    return bin_tidxs


def _xser_save_data(checkpoint_dir: BaseCheckpointStorage, path: str, state_dict, iostate, groups: List[List[int]] = None):
    '''
    This function save the tensors in a state_dict into a directory.
    Each tensor will be saved as a separate file.
    Args:
      path: a directory that tensors will be written to
      state_dict: a state dict
     iostate: an object of CheckpointIOState
     groups: a list of list. Each sub-list is a list of worker's ranks whose state_dict are the same. groups being None means every rank's state_dic is unique.
             When groups is provided, save task are evenly split between workers in same group
    '''
    if groups is not None:
        my_rank_in_group, my_group_size = _get_my_group_info(groups)

    def convert_fn(tensors):
        torch_xla._XLAC._xla_sync_multi(
            tensors, devices=[], wait=True, sync_xla_data=True)

        if groups is None:
            my_tensors = None
        else:
            my_tensors = _assign_tensors_to_bins(tensors, my_group_size)[my_rank_in_group]

        rewritten_tensors = []
        for i, t in enumerate(tensors):
            if (my_tensors is None) or (i in my_tensors):
                t0 = datetime.now()
                cpu_data = t.cpu()
                t1 = datetime.now()
                iostate.add_save_task(checkpoint_dir, cpu_data, xser._get_tensor_file(path, i))
                if torch.distributed.get_rank() == 0:
                    logger.debug(f"    transfer tensor {i} to cpu elapsed: {(t1 - t0).total_seconds()} seconds")
            rewritten_tensors.append(_InternalTensorReference(i, t.shape, t.dtype))
        return rewritten_tensors

    def select_fn(v):
        return type(v) == torch.Tensor and xm.is_xla_tensor(v)

    if groups is None or my_rank_in_group == 0:
        checkpoint_dir.create_dir(path)

    if groups is not None:
        # this rendezvous point makes other workers in the group wait
        # for "path" to be created.
        xm.rendezvous("create tensor directory")
    return xm.ToXlaTensorArena(convert_fn, select_fn).transform(state_dict)


def _extract_tensor_info_and_update_state_dict(state_dict: dict, tensor_info: dict):
    '''
    for a given state_dict, replace _InternalTensorReference with XserTensorReference,
    and put the dtype and shape in a separate accout.
    '''
    for k, v in state_dict.items():
        if type(v) == _InternalTensorReference:
            tensor_info[v.tid] = {"dtype": v.dtype, "shape": v.shape}
            state_dict[k] = xser.TensorReference(v.tid)
        if type(v) == dict:
            _extract_tensor_info_and_update_state_dict(v, tensor_info)


def _save(ckpt, checkpoint_dir: BaseCheckpointStorage, path: str, groups=None, num_workers=8, use_xser=False, iostate=None):
    if groups is not None:
        my_rank_in_group, my_group_size = _get_my_group_info(groups)

    # quick path when use xser
    if use_xser:
        state_dict = _xser_save_data(checkpoint_dir, xser._get_tensors_folder(path), ckpt, iostate, groups)
        if (groups is None) or (my_rank_in_group == 0):
            tensor_info = {}
            # to make sure path can be loaded using xser.load(), we must update
            # state_dict such that it does not have _InternalTensorReference
            _extract_tensor_info_and_update_state_dict(state_dict, tensor_info)
            iostate.add_save_task(checkpoint_dir, state_dict, path)
            # the info.pt file is used by broadcast based loading (see _xser_load_data)
            iostate.add_save_task(checkpoint_dir, tensor_info, path + ".info.pt")
        return

    local_rank = xm.get_local_ordinal()
    for worker in range(math.ceil(get_local_world_size() / num_workers)):
        if groups is None or my_rank_in_group == 0:
            if local_rank // num_workers == worker:
                logger.debug(f"worker {local_rank} saving checkpoint {path}")
                cpu_data = move_all_tensor_to_cpu(ckpt)
                iostate.add_save_task(checkpoint_dir, cpu_data, path)


def _load_obj_from_state_dict(obj, state_dict, strict):
    if isinstance(obj, torch.nn.Module):
        obj.load_state_dict(state_dict, strict=strict)
    else:
        obj.load_state_dict(state_dict)
    del state_dict
    gc.collect()


def _load(obj, checkpoint_dir: BaseCheckpointStorage, path: str, groups: List[List[int]]=None, num_workers: int=8, strict: bool=True, use_xser: bool=False):
    '''
    Load object the save as path.

    Parameters

    path: self explained

    groups: a list of list that represents the replica status of path. Each sublist is a list of worker's ranks.
            Workers whose rank in the same sublist has the same path. When groups is None, no worker has the same path.
    '''
    # quick path when use xser
    if use_xser:
        ckpt = _xser_load_data(checkpoint_dir, path, groups)
        _load_obj_from_state_dict(obj, ckpt, strict)
        return

    local_rank = xm.get_local_ordinal()
    for worker in range(math.ceil(get_local_world_size() / num_workers)):
        if local_rank // num_workers == worker:
            logger.debug(f"worker {local_rank} loading checkpoint {path}")
            ckpt = checkpoint_dir.load_object(path, map_location="cpu")
            _load_obj_from_state_dict(obj, ckpt, strict)
        xm.rendezvous(f"worker-{worker}: checkpoint loaded")
    xm.rendezvous("load checkpoint done")


def has_checkpoint(
    checkpoint_dir_str: str
):
    checkpoint_dir = create_checkpoint_storage(checkpoint_dir_str)
    return len(checkpoint_dir.list_checkpoint_tags()) > 0

g_iostate = None

def save_checkpoint(
    checkpoint_dir_str,
    tag,
    model=None,
    optimizer=None,
    scheduler=None,
    user_content=None,
    num_workers=8,
    use_xser=False,
    num_kept_ckpts=None,
    async_save=False
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
        async_save (bool):
            whether to use asynchronous saving method
    """
    # TODO: Use distributed checkpoint
    assert torch.distributed.is_initialized(), "Only support distributed training mode."

    checkpoint_dir = create_checkpoint_storage(checkpoint_dir_str)
    checkpoint_dir.create_dir(".")

    global g_iostate
    if g_iostate is None:
        g_iostate = CheckpointIOState(async_save)
        import atexit
        atexit.register(CheckpointIOState.wait_all, g_iostate)

    g_iostate.begin(tag)
    ckpt_path = str(tag)

    # save model
    if model is not None:
        checkpoint_dir.create_dir(os.path.join(ckpt_path, "model"), exist_ok=True)
        model_path = os.path.join(ckpt_path, _get_path("model"))
        if isinstance(model, NxDPPModel):
            ckpt = model.local_state_dict()
        else:
            ckpt = model.state_dict()
        groups = get_data_parallel_group(as_list=True)
        _save(ckpt, checkpoint_dir, model_path, groups=groups, num_workers=num_workers, use_xser=use_xser, iostate=g_iostate)

    # save optimizer
    if optimizer is not None:
        checkpoint_dir.create_dir(os.path.join(ckpt_path, "optim"), exist_ok=True)
        if isinstance(optimizer, NxDOptimizer):
            zero1_enabled = optimizer.nxd_config["optimizer_config"]["zero_one_enabled"]
        else:
            zero1_enabled = isinstance(optimizer, NeuronZero1Optimizer)
        optimizer_path = os.path.join(ckpt_path, _get_path("optim", dp=zero1_enabled))
        groups = None if zero1_enabled else get_data_parallel_group(as_list=True)
        _save(optimizer.state_dict(), checkpoint_dir, optimizer_path, groups=groups, num_workers=num_workers, use_xser=use_xser, iostate=g_iostate)

    # save scheduler
    if scheduler is not None:
        if torch.distributed.get_rank() == 0:
            g_iostate.add_save_task(checkpoint_dir, scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

    # save user content
    if user_content is not None:
        if torch.distributed.get_rank() == 0:
            g_iostate.add_save_task(checkpoint_dir, user_content, os.path.join(ckpt_path, "user_content.pt"))

    g_iostate.end(num_kept_ckpts)

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

    checkpoint_dir = create_checkpoint_storage(path)

    if tag is None:
        tags = checkpoint_dir.list_checkpoint_tags()
        if len(tags)==0:
            raise RuntimeError("Error: no checkpoint under directory {checkpoint_dir}")
        tag = tags[-1]

    ckpt_path = str(tag)

    use_xser = checkpoint_dir.is_checkpoint_xser(tag)

    if torch.distributed.get_rank() == 0:
        logger.info("loading checkpoint from {}".format(ckpt_path))

    # load model
    if model is not None:
        model_path = os.path.join(ckpt_path, _get_path("model"))
        groups =  get_data_parallel_group(as_list=True)
        _load(model, checkpoint_dir, model_path, groups=groups, num_workers=num_workers, strict=strict, use_xser=use_xser)

    # load optimizer
    if optimizer is not None:
        if isinstance(optimizer, NxDOptimizer):
            zero1_enabled = optimizer.nxd_config["optimizer_config"]["zero_one_enabled"]
        else:
            zero1_enabled = isinstance(optimizer, NeuronZero1Optimizer)
        groups = None if zero1_enabled else get_data_parallel_group(as_list=True)
        optimizer_path = os.path.join(ckpt_path, _get_path("optim", dp=zero1_enabled))
        _load(optimizer, checkpoint_dir, optimizer_path, groups=groups, num_workers=num_workers, use_xser=use_xser)

    # load scheduler
    if scheduler is not None:
        ckpt = checkpoint_dir.load_object(os.path.join(ckpt_path, "scheduler.pt"), map_location="cpu")
        scheduler.load_state_dict(ckpt)

    # load user content
    user_content = None
    user_content_path = os.path.join(ckpt_path, "user_content.pt")
    if checkpoint_dir.file_exists(user_content_path):
        user_content = checkpoint_dir.load_object(user_content_path, map_location="cpu")

    if torch.distributed.get_rank() == 0:
        logger.info("loading checkpoint done")

    xm.rendezvous("load all checkpoints done")
    return user_content

