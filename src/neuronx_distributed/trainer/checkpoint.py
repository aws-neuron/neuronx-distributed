import concurrent.futures
import gc
import math
import os
import re
from datetime import datetime
from packaging import version
from typing import List, Tuple, Optional, Any, Dict

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.utils.serialization as xser

from neuronx_distributed.optimizer import NeuronZero1Optimizer

if version.parse(torch.__version__) >= version.parse("2.1"):
    HAVE_DCP_SUPPORT = True
    import neuronx_distributed.optimizer.zero_dcp_utils as dcp_utils
else:
    HAVE_DCP_SUPPORT = False
    dcp_utils = None

from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_expert_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_expert_data_parallel_size,
    get_expert_model_parallel_size,
    get_expert_data_parallel_group,
    get_expert_model_parallel_group,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    model_parallel_is_initialized,
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


def _get_path(prefix: str, tp: bool = True, pp: bool = True, dp: bool = False, ep: bool = False) -> str:
    path = ""
    path += "_dp_rank_{:02d}".format(get_data_parallel_rank() if dp else 0)
    path += "_ep_rank_{:02d}".format(get_expert_model_parallel_rank()) if ep else ""
    path += "_tp_rank_{:02d}".format(get_tensor_model_parallel_rank() if tp else 0)
    path += "_pp_rank_{:02d}".format(get_pipeline_model_parallel_rank() if pp else 0)
    if path != "":
        path = path[1:]
    path += ".pt"
    return f"{prefix}/{path}"


def _determine_remove_tags(checkpoint_dir: BaseCheckpointStorage, num_kept: int) -> List[str]:
    """
    deteremine checkpoint tags to be removed to satisfy num_kept
    return value: a list of tags
    """
    tags = checkpoint_dir.list_checkpoint_tags()

    corrupted_tags = []
    completed_tags = []
    for tag in tags:
        if checkpoint_dir.file_exists(os.path.join(tag, "done")):
            completed_tags.append(tag)
        else:
            # corrupted checkpoint can be from interrupted deletion or interrupted save.
            # we only want to remove the corrupted checkpoints from interruped deletion,
            # because corrupted checkpoint from interrupted save will be overwritten by
            # resumed training.
            # corrupted checkpoint from interrupted deletion can be identified by that
            # they are followed by a completed checkpoint. therefore we stop record them
            # when there is a completed tag.
            if len(completed_tags) == 0:
                corrupted_tags.append(tag)

    remove_tags = corrupted_tags
    if num_kept is not None and num_kept != -1 and len(completed_tags) > num_kept:
        remove_tags += completed_tags[0 : len(completed_tags) - num_kept]

    return remove_tags

# Global ThreadPoolExecutor to avoid reinitialization
_executor = None

def _bulk_save(checkpoint_dir: BaseCheckpointStorage, save_items: List[Tuple[object, str]]) -> None:
    for obj, filename in save_items:
        checkpoint_dir.save_object(obj, filename)


class CheckpointIOState:
    """
    class to store state of asynchronous checkpoint saving
    """

    def __init__(self, async_save: bool = False):
        """
        async_save : whether to use asynchronous checkpoint saving. Default no
        """
        self._async_save = async_save
        self._current_tag = None
        self._relative_filenames: set[set] = set()

        if self._async_save:
            self._checkpoint_dir = None
            global _executor
            _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            self._save_items: List[Tuple[torch.Tensor, str]] = []
            self._save_task: Optional[concurrent.futures.Future] = None
            self._dcp_save_items: List[Tuple[Any]] = []
            self._dcp_save_task: Optional[concurrent.futures.Future] = None
            self._remove_tags: Optional[List[str]] = None
            self._remove_task: Optional[concurrent.futures.Future] = None
            self._num_kept: Optional[int]

    def begin(self, checkpoint_dir: BaseCheckpointStorage, tag: str) -> None:
        self._checkpoint_dir = checkpoint_dir

        if self._async_save and self._current_tag is not None:
            self.wait_save(async_remove=True)

        self._current_tag = tag
        xm.rendezvous("create ckpt dir after every neuron-core has filepath")
        if torch.distributed.get_rank() == 0:
            method = "async" if self._async_save else "synced"
            logger.info("%s saving of checkpoint %s began", method, tag)
            self._checkpoint_dir.create_dir(self._current_tag)
            # create a "checkpoint" tag to mark the directory as checkpoint directory
            # this is to distinguish checkpoint from users' own data directory under output directory
            self._checkpoint_dir.save_text("1", os.path.join(self._current_tag, "checkpoint"))

    def add_save_task(self, obj: Any, filename: str) -> None:
        assert filename.startswith(self._current_tag + "/")
        relative_filename = filename[len(self._current_tag) + 1 :]
        self._relative_filenames.add(relative_filename)
        if self._async_save:
            self._save_items.append((obj, filename))
        else:
            assert self._checkpoint_dir
            self._checkpoint_dir.save_object(obj, filename)

    def add_dcp_save_task(self, checkpoint_dir: BaseCheckpointStorage, state_dict: dict, optimizer, model, ckpt_path):
        path = os.path.join(checkpoint_dir.dirname(), ckpt_path, "optim")
        aux_infos = dcp_utils.get_dcp_aux_infos(model, optimizer)
        state_dict_cpu = move_all_tensor_to_cpu(state_dict)
        if self._async_save:
            self._dcp_save_items.append((path, state_dict_cpu, aux_infos))
        else:
            dcp_utils.save_optim_state_dict(path, state_dict_cpu, aux_infos)

    def _dealloc_tensor_host_memory_callback(self, future):
        """Future callback to asynchronous deallocate the tensor host memory
        """
        self._save_items = []
        gc.collect()

    def end(self, num_kept: int) -> None:
        if self._async_save:
            self._num_kept = num_kept
            if len(self._save_items) > 0:
                self._save_task = _executor.submit(_bulk_save, self._checkpoint_dir, self._save_items)
                # After save async thread is finished, use callback to async dealloc the tensor host memory
                self._save_task.add_done_callback(self._dealloc_tensor_host_memory_callback)
            if len(self._dcp_save_items) > 0:
                self._dcp_save_task = _executor.submit(dcp_utils.save_optim_state_dict, *(self._dcp_save_items[0]))
                # After save async thread is finished, use callback to async dealloc the tensor host memory
                self._dcp_save_task.add_done_callback(self._dealloc_tensor_host_memory_callback)
            logger.info("async saving of checkpoint %s requested", self._current_tag)
        else:
            xm.rendezvous("saving checkpoint done")
            if torch.distributed.get_rank() == 0:
                logger.info("synced saving of checkpoint %s completed", self._current_tag)
                self._checkpoint_dir.save_text("1", os.path.join(self._current_tag, "done"))
            xm.rendezvous("mark checkpoint as done")
            self.submit_remove(num_kept, async_remove=False)

    def wait_save(self, async_remove: bool) -> None:
        if not self._async_save:
            return

        # first wait for save to finish
        if self._save_task:
            done, _ = concurrent.futures.wait([self._save_task])
            for f in done:
                if f.exception():
                    raise f.exception()
        if self._dcp_save_task:
            done, _ = concurrent.futures.wait([self._dcp_save_task])
            for f in done:
                if f.exception():
                    raise f.exception()

        xm.rendezvous("async saving checkpoint done")

        if self._save_task:
            self._save_task = None
            # This is already asynchronously invoked in _dealloc_tensor_host_memory_callback
            # However, this needs to be kept for the sync checkpointing case
            self._save_items = []
        if self._dcp_save_task:
            self._dcp_save_task = None
            self._dcp_save_items = []
        if torch.distributed.get_rank() == 0:
            self._checkpoint_dir.save_text("1", os.path.join(self._current_tag, "done"))

        xm.rendezvous("mark checkpoint as done")

        logger.info("async saving of checkpoint %s completed", self._current_tag)

        # remove checkpoint if necessary.
        self.wait_remove()
        self.submit_remove(self._num_kept, async_remove=async_remove)

    def submit_remove(self, num_kept: int, async_remove: bool, remove_tags: Optional[List[str]] = None) -> None:
        remove_tags = remove_tags or []
        remove_tags = remove_tags if len(remove_tags) else _determine_remove_tags(self._checkpoint_dir, num_kept)
        xm.rendezvous("determine remove tags done")
        if len(remove_tags) == 0:
            logger.info("no checkpoints to remove.")
            return

        if torch.distributed.get_rank() == 0:
            logger.info("removing previous checkpoint in %s", remove_tags)
            # remove the done file first to avoid the situation
            # the deletion was interrupted by faults, leaving
            # a corrupted checkpoint_dir with the "done" tag.
            completed_tags = []
            for remove_tag in remove_tags:
                done_file = os.path.join(remove_tag, "done")
                if self._checkpoint_dir.file_exists(done_file):
                    completed_tags.append(remove_tag)
                    self._checkpoint_dir.remove_file(done_file)

            logger.info("done tags in %s cleared", completed_tags)

        remove_filenames = []
        for remove_tag in remove_tags:
            for relative_filename in self._relative_filenames:
                remove_filenames.append(os.path.join(remove_tag, relative_filename))

        if async_remove:
            self._remove_tags = remove_tags
            self._remove_task = _executor.submit(self._checkpoint_dir.remove_files, remove_filenames)
            logger.info("async removal of %s requested.", self._remove_tags)
        else:
            self._checkpoint_dir.remove_files(remove_filenames)
            xm.rendezvous("remove files done")
            # wait until everyone deleted the files they wrote, then rank 0 delete what were left
            if torch.distributed.get_rank() == 0:
                self._checkpoint_dir.remove_dirs(remove_tags)
                logger.info("previous checkpoint in %s successfully removed", remove_tags)

    def wait_remove(self) -> None:
        if self._remove_task:
            done, _ = concurrent.futures.wait([self._remove_task])
            for f in done:
                if f.exception():
                    raise f.exception()

            xm.rendezvous("remove files done")
            if torch.distributed.get_rank() == 0:
                self._checkpoint_dir.remove_dirs(self._remove_tags)
                logger.info("async removal of %s completed", self._remove_tags)
            self._remove_tags = None
            self._remove_task = None
        # This rendevous is necessary, since it avoids the race condition that
        # can occur when each worker is trying to find the next set of files to
        # delete. Its a corner case, where worker 0 is still deleting, and worker
        # 1 has moved on to the submit_remove task. It tries to find_files, and
        # in the process runs into a race condition, resulting in file not found
        # error.
        xm.rendezvous("Wait for all workers to come from deletion")

    def wait_all(self) -> None:
        # when this function is called, ThreadPool may have been shutdown.
        # there fore we must use synced remove
        self.wait_save(async_remove=False)


def _get_my_group_info(groups: List[List[int]]) -> Tuple[int, int]:
    global_rank = torch.distributed.get_rank()
    for group in groups:
        if global_rank in group:
            return group.index(global_rank), len(group)

    raise RuntimeError(f"Error: global rank {global_rank} is not in groups")


def _get_ep_group_info() -> Tuple[int, int, int, int, int, int]:
    emp_rank = get_expert_model_parallel_rank()
    edp_rank = get_expert_data_parallel_rank()
    emp_size = get_expert_model_parallel_size()
    edp_size = get_expert_data_parallel_size()
    emp_group = get_expert_model_parallel_group(as_list=True)
    edp_group = get_expert_data_parallel_group(as_list=True)

    return emp_rank, edp_rank, emp_size, edp_size, emp_group, edp_group


def _xser_load_data(checkpoint_dir: BaseCheckpointStorage, path: str, groups: Optional[List[List[int]]] = None, ep_only: bool = False):
    """
    load tensors saved in path into a state_dict.
    Parameters:
    groups: a list of groups. Each group is a list of ranks whose path are the same. groups being None means every rank's data is unique.
           When groups is provided, 1 rank in a group will load data from path, then broadcast result to other ranks.
    """
    ref_data = checkpoint_dir.load_object(path)
    # check the existance of info.pt file because older version (<=0.6.0) does not generate this file.
    # checkpoint generated using older version still need to be supported.
    ref_info = checkpoint_dir.load_object(path + ".info.pt") if checkpoint_dir.file_exists(path + ".info.pt") else None

    ep_tensor_folder = path + ".tensors"

    if groups is not None:
        my_rank_in_group, my_group_size = _get_my_group_info(groups)
        # for non-ep tensors all ranks load from ep_rank_00 folder, round-robin
        non_ep_tensor_folder = re.sub("ep_rank_\d{2}", "ep_rank_00", path) + ".tensors"

    if model_parallel_is_initialized():
        emp_rank, edp_rank, emp_size, edp_size, emp_group, edp_group = _get_ep_group_info()

    def convert_fn(tensors):
        rewritten_tensors = [None for t in tensors]

        def _is_ep(dct):
            return "expert_model_parallel" in dct and dct["expert_model_parallel"]

        ep_tensors = [(i, t) for i, t in enumerate(tensors) if _is_ep(ref_info[t.tid])]
        non_ep_tensors = [(i, t) for i, t in enumerate(tensors) if not _is_ep(ref_info[t.tid])]

        def _load_tensors(tensor_list, _group, _rank, _group_size, _tensor_folder):
            for idx, (original_idx, t) in enumerate(tensor_list):
                tensor_file = os.path.join(_tensor_folder, "tensor_{}.pt".format(t.tid))

                if (ref_info is not None) and (groups is not None):
                    # When there is redundency (groups is not None) and we know the tensor's shape and dtype (ref_info is not None)
                    # we use the following optimization:
                    #    among workers that has same tensor (in same group), only 1 worker read tensor from disk
                    #    other workers will get the tensor from network broadcasting
                    #
                    # we used round robin to select which worker will read from disk to evenly
                    # distribute the load tasks.

                    if (idx % _group_size) == _rank:
                        loaded = checkpoint_dir.load_object(tensor_file).to(xm.xla_device())
                    else:
                        dtype = ref_info[t.tid]["dtype"]
                        shape = ref_info[t.tid]["shape"]
                        loaded = torch.zeros(shape, dtype=dtype, device=xm.xla_device())

                    # we use all_reduce to implement broadcast because xla does not have native broadcast support.
                    xm.all_reduce(xm.REDUCE_SUM, [loaded], groups=_group)
                else:
                    # when dtype and shape are not available or there is no redundency, all workers load tensor from disk
                    loaded = checkpoint_dir.load_object(tensor_file).to(xm.xla_device())

                rewritten_tensors[original_idx] = loaded

        if groups is not None:
            _load_tensors(ep_tensors, edp_group, edp_rank, edp_size, ep_tensor_folder)
            _load_tensors(non_ep_tensors, groups, my_rank_in_group, my_group_size, non_ep_tensor_folder)
        elif ep_only:
            _load_tensors(ep_tensors, None, None, None, ep_tensor_folder)
        else:
            _load_tensors(ep_tensors + non_ep_tensors, None, None, None, ep_tensor_folder)

        if groups is not None:
            xm.mark_step()
        return rewritten_tensors

    def select_fn(v):
        return isinstance(v, xser.TensorReference)

    return xm.ToXlaTensorArena(convert_fn, select_fn).transform(ref_data)


class _InternalTensorReference:
    def __init__(self, tid, shape, dtype, expert_model_parallel):
        self.tid = tid
        self.shape = shape
        self.dtype = dtype
        self.expert_model_parallel = expert_model_parallel


def _assign_tensors_to_bins(tensors: List[torch.Tensor], bin_count: int) -> List[List[int]]:
    """
    assign a list of tensors into multiple bins, such that each bin's
    total tensor size are similar.
    Args:
    tensors: a list of tensors
    bin_count: number of bins
    Return
    a list of list, each sublist contain indices of tensors.
    """

    bin_tidxs = [[] for i in range(bin_count)]
    bin_sizes = [0 for i in range(bin_count)]

    tensor_sizes = []
    for i, tensor in enumerate(tensors):
        tensor_sizes.append((i, torch.numel(tensor) * tensor.element_size()))

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


def _xser_save_data(
    checkpoint_dir: BaseCheckpointStorage, path: str, state_dict, iostate: CheckpointIOState, groups: Optional[List[List[int]]] = None
) -> Any:
    """
    This function save the tensors in a state_dict into a directory.
    Each tensor will be saved as a separate file.
    Args:
      path: a directory that tensors will be written to
      state_dict: a state dict
     iostate: an object of CheckpointIOState
     groups: a list of list. Each sub-list is a list of worker's ranks whose state_dict are the same. groups being None means every rank's state_dic is unique.
             When groups is provided, save task are evenly split between workers in same group
    """
    if groups is not None:
        my_rank_in_group, my_group_size = _get_my_group_info(groups)

    emp_rank, edp_rank, emp_size, edp_size, emp_group, edp_group = _get_ep_group_info()

    def convert_fn(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        torch_xla._XLAC._xla_sync_multi(tensors, devices=[], wait=True, sync_xla_data=True)

        if groups is None:
            my_tensors = None
        else:
            my_tensors = _assign_tensors_to_bins(tensors, edp_size)[edp_rank]

        rewritten_tensors = []
        for i, t in enumerate(tensors):
            is_expert_parallel = hasattr(t, "expert_model_parallel") and t.expert_model_parallel
            if (my_tensors is None) or (i in my_tensors):
                t0 = datetime.now()
                cpu_data = t.cpu()
                t1 = datetime.now()
                # if the below condition is not satisfied, someone else will store the same data
                if my_tensors is None or (is_expert_parallel or emp_rank == 0):
                    iostate.add_save_task(cpu_data, xser._get_tensor_file(path, i))
                logger.debug("    transfer tensor %d to cpu elapsed: %d seconds", i, (t1 - t0).total_seconds())
            rewritten_tensors.append(_InternalTensorReference(i, t.shape, t.dtype, is_expert_parallel))
        return rewritten_tensors

    def select_fn(v):
        return isinstance(v, torch.Tensor)  # and xm.is_xla_tensor(v)

    checkpoint_dir.create_shared_dir(path)
    return xm.ToXlaTensorArena(convert_fn, select_fn).transform(state_dict)


def _extract_tensor_info_and_update_state_dict(state_dict: Dict[str, Any], tensor_info: Dict[int, Dict[str, Any]]) -> None:
    """
    for a given state_dict, replace _InternalTensorReference with XserTensorReference,
    and put the dtype and shape in a separate accout.
    """
    for k, v in state_dict.items():
        if isinstance(v, _InternalTensorReference):
            tensor_info[v.tid] = {"dtype": v.dtype, "shape": v.shape, "expert_model_parallel": v.expert_model_parallel}
            state_dict[k] = xser.TensorReference(v.tid)
        if isinstance(v, dict):
            _extract_tensor_info_and_update_state_dict(v, tensor_info)


def _save(
    ckpt: Any, checkpoint_dir: BaseCheckpointStorage, path: str, groups: Optional[List[List[int]]] = None, num_workers: int = 8, use_xser: bool = False, iostate: Optional[CheckpointIOState] = None, optimizer: bool = False
) -> None:
    if groups is not None:
        my_rank_in_group, my_group_size = _get_my_group_info(groups)
    emp_rank, edp_rank, emp_size, edp_size, emp_group, edp_group = _get_ep_group_info()

    # quick path when use xser
    if use_xser:
        state_dict = _xser_save_data(checkpoint_dir, xser._get_tensors_folder(path), ckpt, iostate, groups)
        if (groups is None) or (edp_rank == 0):
            tensor_info = {}
            # to make sure path can be loaded using xser.load(), we must update
            # state_dict such that it does not have _InternalTensorReference
            _extract_tensor_info_and_update_state_dict(state_dict, tensor_info)
            iostate.add_save_task(state_dict, path)
            # the info.pt file is used by broadcast based loading (see _xser_load_data)
            iostate.add_save_task(tensor_info, path + ".info.pt")
        return

    local_rank = xm.get_local_ordinal()
    for worker in range(math.ceil(get_local_world_size() / num_workers)):
        if groups is None or my_rank_in_group == 0:
            if local_rank // num_workers == worker:
                logger.debug(f"worker {local_rank} saving checkpoint {path}")
                cpu_data = move_all_tensor_to_cpu(ckpt)
                iostate.add_save_task(cpu_data, path)


def _load_obj_from_state_dict(obj: Any, state_dict: Dict[str, Any], strict: bool) -> None:
    if isinstance(obj, torch.nn.Module):
        obj.load_state_dict(state_dict, strict=strict)
    elif isinstance(obj, dict):
        for k in state_dict:
            obj[k] = state_dict[k]
    else:
        obj.load_state_dict(state_dict)
    del state_dict
    gc.collect()


def _load(
    obj: Any,
    checkpoint_dir: BaseCheckpointStorage,
    path: str,
    groups: Optional[List[List[int]]] = None,
    num_workers: int = 8,
    strict: bool = True,
    use_xser: bool = False,
) -> None:
    """
    Load object the save as path.

    Parameters

    path: self explained

    groups: a list of list that represents the replica status of path. Each sublist is a list of worker's ranks.
            Workers whose rank in the same sublist has the same path. When groups is None, no worker has the same path.
    """
    # quick path when use xser
    if use_xser:
        ckpt = _xser_load_data(checkpoint_dir, path, groups)
        ckpt = _move_step_to_cpu(ckpt)
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


def has_checkpoint(checkpoint_dir_str: str) -> bool:
    checkpoint_dir = create_checkpoint_storage(checkpoint_dir_str)
    return len(checkpoint_dir.list_completed_checkpoint_tags()) > 0


g_iostate = None


def save_checkpoint(
    checkpoint_dir_str: str,
    tag: str,
    model: Any = None,
    optimizer: Any = None,
    scheduler=None,
    user_content=None,
    num_workers=8,
    use_xser=False,
    num_kept_ckpts=None,
    async_save=False,
    zero1_optimizer=False,
    use_zero1_dcp=False,
) -> None:
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
        checkpoint_dir_str (str):
            path to save the checkpoints.
        tag (str):
            tag to save the checkpoints.
        model (torch.nn.Module or dict):
            model to save, optional.
        optimizer (torch.optim.Optimizer or dict):
            optimizer to save, optional.
        scheduler:
            scheduler to save, optional.
        user_content:
            user contents to save, optional.
        num_workers (int):
            num of workers to save the checkpoints on the same time, range: 1-32.
        use_xser (bool):
            whether to use torch-xla serialization. When enabled, ``num_workers`` will be ignored
            and maximum num of workers will be used. Default: ``False``.
        num_kept_ckpts (int):
            number of checkpoints to keep on disk, optional. Default: ``None``.
        async_save (bool):
            whether to use asynchronous saving method.
        zero1_optimizer (bool):
            whether the optimizer state is from a zero1 optimizer, used when optimizer is a dict.
        use_zero1_dcp (bool):
            whether to use Distributed Checkpoint for ZeRO-1 optimizer.
    """
    # TODO: Use distributed checkpoint
    assert torch.distributed.is_initialized(), "Only support distributed training mode."

    checkpoint_dir = create_checkpoint_storage(checkpoint_dir_str)
    if torch.distributed.get_rank() == 0:
        checkpoint_dir.create_dir(".")

    global g_iostate
    if g_iostate is None:
        g_iostate = CheckpointIOState(async_save)
        import atexit

        atexit.register(CheckpointIOState.wait_all, g_iostate)

    g_iostate.begin(checkpoint_dir, tag)
    ckpt_path = str(tag)

    ep_enabled = get_expert_model_parallel_size() > 1

    # save model
    if model is not None:
        if torch.distributed.get_rank() == 0:
            checkpoint_dir.create_dir(os.path.join(ckpt_path, "model"), exist_ok=True)
        model_path = os.path.join(ckpt_path, _get_path("model", ep=ep_enabled))

        if isinstance(model, NxDPPModel):
            ckpt = model.local_state_dict()
        elif isinstance(model, dict):
            ckpt = model
        else:
            ckpt = model.state_dict()

        groups = get_expert_data_parallel_group(as_list=True)
        _save(
            ckpt,
            checkpoint_dir,
            model_path,
            groups=groups,
            num_workers=num_workers,
            use_xser=use_xser,
            iostate=g_iostate,
        )

    # save optimizer
    if optimizer is not None:
        if torch.distributed.get_rank() == 0:
            checkpoint_dir.create_dir(os.path.join(ckpt_path, "optim"), exist_ok=True)
        if isinstance(optimizer, NxDOptimizer):
            zero1_enabled = optimizer.nxd_config["optimizer_config"]["zero_one_enabled"]
            optimizer_state_dict = optimizer.state_dict()
        elif isinstance(optimizer, dict):
            zero1_enabled = zero1_optimizer
            optimizer_state_dict = optimizer
        else:
            zero1_enabled = isinstance(optimizer, NeuronZero1Optimizer)
            optimizer_state_dict = optimizer.state_dict()

        if zero1_enabled:
            par_args = {"dp": True}
        elif ep_enabled:
            par_args = {"ep": True}
        else:
            par_args = {}

        optimizer_path = os.path.join(ckpt_path, _get_path("optim", **par_args))
        groups = None if zero1_enabled else get_data_parallel_group(as_list=True)
        if zero1_enabled and use_zero1_dcp and HAVE_DCP_SUPPORT:
            assert async_save, "Now we only use DCP for async checkpoint saving."
            g_iostate.add_dcp_save_task(checkpoint_dir, optimizer_state_dict, optimizer, model, ckpt_path)
        else:
            _save(
                optimizer_state_dict,
                checkpoint_dir,
                optimizer_path,
                groups=groups,
                num_workers=num_workers,
                use_xser=use_xser,
                iostate=g_iostate,
                optimizer=True,
            )

    # save scheduler
    if scheduler is not None:
        if torch.distributed.get_rank() == 0:
            g_iostate.add_save_task(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

    # save user content
    if user_content is not None:
        if torch.distributed.get_rank() == 0:
            g_iostate.add_save_task(user_content, os.path.join(ckpt_path, "user_content.pt"))

    g_iostate.end(num_kept_ckpts)


def _move_step_to_cpu(ckpt: Any, key: Optional[str] = None) -> Any:
    if key == "step":
        return ckpt.cpu()

    if isinstance(ckpt, dict):
        return {k: _move_step_to_cpu(v, k) for k, v in ckpt.items()}

    return ckpt


def load_checkpoint(
    path: str,
    tag: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    num_workers: int = 8,
    strict: bool = True,
    use_zero1_dcp: bool = False,
) -> Any:
    """
    Method to load checkpoint, return user contents if exists otherwise ``None``.
    If ``tag`` not provided, will try to use the newest tag tracked by ``save_checkpoint``.

    Parameters:
        path (str):
            path to load the checkpoints.
        tag (str):
            tag to load the checkpoints, optional.
        model (torch.nn.Module):
            model to load, optional.
        optimizer (torch.optim.Optimizer):
            optimizer to load, optional.
        scheduler:
            scheduler to load, optional.
        num_workers (int):
            num of workers to load the checkpoints on the same time, range: 1-32.
        strict (bool):
            whether to use strict mode when loading model checkpoint. Default: ``True``.
        use_zero1_dcp (bool):
            whether to use Distributed Checkpoint for ZeRO-1 optimizer.
    """
    assert torch.distributed.is_initialized(), "Only support distributed training mode."

    checkpoint_dir = create_checkpoint_storage(path)
    ep_enabled = get_expert_model_parallel_size() > 1

    if ep_enabled and strict:
        print("Strict checkpoint loading is not supported with expert parallelism.")

    if tag is None:
        tags = checkpoint_dir.list_completed_checkpoint_tags()
        if len(tags) == 0:
            raise RuntimeError("Error: no checkpoint under directory {checkpoint_dir}")
        tag = tags[-1]

    ckpt_path = str(tag)

    use_xser = checkpoint_dir.is_checkpoint_xser(tag)

    logger.info("loading checkpoint from %s", ckpt_path)

    # load model
    if model is not None:
        model_path = os.path.join(ckpt_path, _get_path("model", ep=ep_enabled))
        groups = get_data_parallel_group(as_list=True)
        _load(
            model, checkpoint_dir, model_path, groups=groups, num_workers=num_workers, strict=strict, use_xser=use_xser
        )

    # load optimizer
    if optimizer is not None:
        if isinstance(optimizer, NxDOptimizer):
            zero1_enabled = optimizer.nxd_config["optimizer_config"]["zero_one_enabled"]
        elif isinstance(optimizer, dict):
            # zero1 optimizer spread optimizer states across dp group, and each process has unique optimizer state and will save to disk
            # therefore, if there exists a file named dp_rank_01_tp_rank_00_pp_rank_00.pt, then the checkpoint was generated by zero1 optimizer.
            zero1_optimizer_specific_file = os.path.join(ckpt_path, "optim", "dp_rank_01_tp_rank_00_pp_rank_00.pt")
            zero1_enabled = checkpoint_dir.file_exists(zero1_optimizer_specific_file)
        elif isinstance(optimizer, NeuronZero1Optimizer):
            zero1_enabled = True
        else:
            raise RuntimeError(
                f"Error: invalid type for the argument optimizer for load_checkpoint, expecting a dict or NxDOptimizer, or NeuronZero1Optimizer, got {type(optimizer)}"
            )

        ep_enabled = get_expert_model_parallel_size() > 1

        if zero1_enabled:
            par_args = {"dp": True}
            groups = None
        elif ep_enabled:
            par_args = {"ep": True}
            groups = get_expert_data_parallel_group(as_list=True)
        else:
            par_args = {}
            groups = get_data_parallel_group(as_list=True)

        if zero1_enabled and use_zero1_dcp and HAVE_DCP_SUPPORT:
            aux_infos = dcp_utils.get_dcp_aux_infos(model, optimizer)
            dcp_utils.load_optim_state_dict(os.path.join(checkpoint_dir.dirname(), ckpt_path, "optim"), optimizer, aux_infos)
        else:
            optimizer_path = os.path.join(ckpt_path, _get_path("optim", **par_args))
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

    logger.info("loading checkpoint done")

    xm.rendezvous("load all checkpoints done")
    return user_content


def finalize_checkpoint() -> None:
    if g_iostate:
        g_iostate.wait_all()
