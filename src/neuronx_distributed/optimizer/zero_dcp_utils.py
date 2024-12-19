import copy
import functools
import itertools
import logging
import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.nn.functional as F

from torch.distributed.checkpoint.default_planner import DefaultSavePlanner, DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    Metadata,
    TensorProperties as MetadataTensorProperties,
)
from torch.distributed.checkpoint.planner import LoadPlan, SavePlan
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)
import torch_xla.core.xla_model as xm

from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_replica_groups,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_size,
    get_pipeline_model_parallel_replica_groups,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_replica_groups,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
    rmsg,
)

# avoid to log out `_dedup_tensors`, it's just 'step's and it's too long
logging.getLogger("torch.distributed.checkpoint._dedup_tensors").setLevel(logging.WARNING)

MAX_RETRY = 100


def _alloc_tensor(props: MetadataTensorProperties, size: Sequence[int]) -> torch.Tensor:
    return torch.empty(
        size=size,
        dtype=props.dtype,
        layout=props.layout,
        requires_grad=props.requires_grad,
        pin_memory=props.pin_memory,
        device="cpu",
    )


def _get_optim_pid_to_params(optim: torch.optim.Optimizer) -> Dict[int, torch.nn.Parameter]:
    ret = {pid: param for param_group in optim.param_groups for pid, param in enumerate(param_group["params"])}
    return ret


def _get_param_to_param_names(model: torch.nn.Module) -> Dict[torch.nn.Parameter, str]:
    ret = {param: name for name, param in model.named_parameters()}
    return ret


def _get_optim_pid_to_param_names(model: torch.nn.Module, optim: torch.optim.Optimizer) -> Dict[int, str]:
    optim_pid_to_params = _get_optim_pid_to_params(optim)
    param_to_param_names = _get_param_to_param_names(model)
    ret = {k: param_to_param_names[v] for k, v in optim_pid_to_params.items()}
    return ret


def _tensor_to_sharded_tensor(
    tensor: torch.Tensor,
    param_shape: Sequence[int],
    dp_rank: Optional[int] = None,
) -> Union[torch.Tensor, ShardedTensor]:
    # quick path for scalars
    if tensor.dim() == 0:
        return tensor

    dp_size = get_data_parallel_size()
    if dp_rank is None:
        dp_rank = get_data_parallel_rank()

    padded_shape = (tensor.shape[0] * dp_size,) + tensor.shape[1:]
    if dp_rank == dp_size - 1 and padded_shape[0] != param_shape[0]:
        # unpad
        tensor = tensor[: tensor.shape[0] - padded_shape[0] + param_shape[0]].clone()

    offsets = [0] * len(padded_shape)
    offsets[0] = (padded_shape[0] // dp_size) * dp_rank
    local_shards = [Shard.from_tensor_and_offsets(tensor, offsets, dp_rank)]

    # Create a ShardedTensor without invoking communication.
    chunk_sizes = []
    for i in range(dp_size):
        if i == dp_size - 1 and padded_shape[0] != param_shape[0]:
            shape = (tensor.shape[0] - (padded_shape[0] - param_shape[0]),) + tensor.shape[1:]
            chunk_sizes.append(shape)
        else:
            chunk_sizes.append(tensor.shape)
    dim0_offsets = [0] + list(itertools.accumulate([chunk_size[0] for chunk_size in chunk_sizes]))[:-1]
    offsets = [0] * (len(chunk_sizes[0]) - 1)
    chunk_offsets = [[d0] + offsets for d0 in dim0_offsets]
    placements = [f"rank:{r}/cpu" for r in range(len(chunk_sizes))]
    assert len(chunk_sizes) == len(chunk_offsets) == len(placements)
    shards_metadata = [
        ShardMetadata(offset, list(size), placement)
        for offset, size, placement in zip(chunk_offsets, chunk_sizes, placements)
    ]
    sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=shards_metadata,
        size=torch.Size(param_shape),
        tensor_properties=TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        ),
    )
    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards,
        sharded_tensor_metadata=sharded_tensor_metadata,
        process_group=get_data_parallel_group(),
    )


def _sharded_tensor_to_tensor(
    tensor: Union[torch.Tensor, ShardedTensor],
    param_shape: Sequence[int],
) -> torch.Tensor:
    # quick path for scalars
    if not isinstance(tensor, ShardedTensor):
        return tensor

    dp_rank = get_data_parallel_rank()
    dp_size = get_data_parallel_size()

    tensor = tensor.local_shards()[0].tensor
    padded_dim0 = ((param_shape[0] + dp_size - 1) // dp_size) * dp_size
    if dp_rank == dp_size - 1 and padded_dim0 != param_shape[0]:
        # pad
        pad_size = padded_dim0 - param_shape[0]
        tensor = F.pad(tensor, [0, 0] * (tensor.dim() - 1) + [0, pad_size])
    return tensor


def _wrap_optim_state_dict(
    state_dict: Dict[str, Any],
    aux_infos: Dict[str, Any],
    dedup: bool = False,
    pp_rank: Optional[int] = None,
    tp_rank: Optional[int] = None,
    dp_rank: Optional[int] = None,
) -> Dict[str, Any]:
    pp_rank = get_pipeline_model_parallel_rank() if pp_rank is None else pp_rank
    tp_rank = get_tensor_model_parallel_rank() if tp_rank is None else tp_rank

    optim_pid_to_params = aux_infos["optim_pid_to_params"]
    optim_pid_to_pnames = aux_infos["optim_pid_to_pnames"]
    pnames_to_optim_pids = {v: k for k, v in optim_pid_to_pnames.items()}

    # replace pid with pname
    new_state_dict = copy.copy(state_dict)
    new_state_dict["base_state"] = {optim_pid_to_pnames[k]: v for k, v in state_dict["base_state"].items()}
    shape_info = new_state_dict["shape_info"]

    # flatten state dict
    new_state_dict, mappings = flatten_state_dict(new_state_dict)
    for k, v in new_state_dict.items():
        if isinstance(v, torch.Tensor):
            # tensor to sharded tensor
            pname = mappings[k][1]
            pid = pnames_to_optim_pids[pname]
            param_shape = shape_info[pid]
            new_state_dict[k] = _tensor_to_sharded_tensor(v, param_shape, dp_rank)
            # add tp and pp info
            if v.dim() > 0:  # TODO: merge constant scalars
                fqn = [str(mappings[k][-1])]
                if get_pipeline_model_parallel_size() > 1:
                    # TODO: deduplicate shared params
                    fqn.append("|pp-{:04d}".format(pp_rank))
                if get_tensor_model_parallel_size() > 1:
                    param = optim_pid_to_params[pid]
                    if not dedup or hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
                        fqn.append("|tp-{:04d}".format(tp_rank))
                new_path = [*mappings[k][:-1], "".join(fqn)]
                mappings[k] = tuple(new_path)

    new_state_dict = unflatten_state_dict(new_state_dict, mappings)
    return new_state_dict


def _unwrap_optim_state_dict(
    state_dict: Dict[str, Any],
    aux_infos: Dict[str, Any],
) -> Dict[str, Any]:
    optim_pid_to_pnames = aux_infos["optim_pid_to_pnames"]
    pnames_to_optim_pids = {v: k for k, v in optim_pid_to_pnames.items()}

    # flatten state dict
    state_dict, mappings = flatten_state_dict(state_dict)
    shape_info = state_dict["shape_info"]
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            # sharded tensor to tensor
            pname = mappings[k][1]
            param_shape = shape_info[pnames_to_optim_pids[pname]]
            state_dict[k] = _sharded_tensor_to_tensor(v, param_shape)
            # remove tp and pp info
            fqn = str(mappings[k][-1])
            origin_key = fqn[: fqn.index("|")] if "|" in fqn else fqn
            new_path = [*mappings[k][:-1], origin_key]
            mappings[k] = tuple(new_path)
    state_dict = unflatten_state_dict(state_dict, mappings)

    # replace pname with pid
    state_dict["base_state"] = {pnames_to_optim_pids[k]: v for k, v in state_dict["base_state"].items()}
    return state_dict


def _prepare_optim_state_dict(
    metadata: Metadata,
    aux_infos: Dict[str, Any],
    dedup: bool = False,
    pp_rank: Optional[int] = None,
    tp_rank: Optional[int] = None,
    dp_rank: Optional[int] = None,
) -> Dict[str, Any]:
    pp_rank = get_pipeline_model_parallel_rank() if pp_rank is None else pp_rank
    tp_rank = get_tensor_model_parallel_rank() if tp_rank is None else tp_rank

    optim_pid_to_pnames = aux_infos["optim_pid_to_pnames"]
    pnames_to_optim_pids = {v: k for k, v in optim_pid_to_pnames.items()}

    shape_info = aux_infos["shape_info"]

    new_state_dict: Dict[str, Any] = {}
    for fqn, value in metadata.state_dict_metadata.items():
        if isinstance(value, BytesStorageMetadata):
            new_state_dict[fqn] = "<bytes_io>"
            continue
        # value: TensorStorageMetadata
        if value.size.numel() == 1:
            new_state_dict[fqn] = _alloc_tensor(value.properties, value.size)
        else:
            pp_rank_from_fqn = 0
            if get_pipeline_model_parallel_size() > 1:
                pp_rank_match = re.search(r"\|pp-(\d{4})", fqn)
                assert pp_rank_match
                pp_rank_from_fqn = int(pp_rank_match.group(1))
            tp_rank_from_fqn = 0
            if get_tensor_model_parallel_size() > 1:
                tp_rank_match = re.search(r"\|tp-(\d{4})", fqn)
                assert tp_rank_match
                tp_rank_from_fqn = int(tp_rank_match.group(1))
            if pp_rank_from_fqn == pp_rank and tp_rank_from_fqn == tp_rank:
                origin_key = fqn[: fqn.index("|")] if "|" in fqn else fqn
                assert origin_key.startswith("base_state.")
                pname = origin_key[11:]  # cut "base_state."
                pname = pname[: pname.rindex(".")]
                pid = pnames_to_optim_pids[pname]
                param_shape = shape_info[pid]
                dp_size = get_data_parallel_size()
                shard_shape = tuple(
                    (param_shape[0] + dp_size - 1) // dp_size,
                    *param_shape[1:],
                )
                new_state_dict[fqn] = _tensor_to_sharded_tensor(
                    _alloc_tensor(value.properties, shard_shape), param_shape, dp_rank=dp_rank
                )

    return new_state_dict


def _generate_all_local_save_plans(
    state_dict: Dict[str, Any],
    aux_infos: Dict[str, Any],
    dedup: bool = False,
) -> List[SavePlan]:
    def _generate_one_local_save_plan(global_rank):
        # calc pp rank
        pp_rank = None
        for group in get_pipeline_model_parallel_replica_groups():
            for i, g in enumerate(group):
                if g == global_rank:
                    pp_rank = i
                    break
        # calc tp rank
        tp_rank = None
        for group in get_tensor_model_parallel_replica_groups():
            for i, g in enumerate(group):
                if g == global_rank:
                    tp_rank = i
                    break
        # calc dp rank
        dp_rank = None
        for group in get_data_parallel_replica_groups():
            for i, g in enumerate(group):
                if g == global_rank:
                    dp_rank = i
                    break

        wrapped_state_dict = _wrap_optim_state_dict(
            state_dict, aux_infos, dedup=dedup, pp_rank=pp_rank, tp_rank=tp_rank, dp_rank=dp_rank
        )
        planner = DefaultSavePlanner()
        planner.set_up_planner(wrapped_state_dict, global_rank == 0)
        local_plan = planner.create_local_plan()
        return local_plan

    all_plans = [_generate_one_local_save_plan(i) for i in range(dist.get_world_size())]
    return all_plans


def _generate_all_local_load_plans(
    state_dict: Dict[str, Any],
    aux_infos: Dict[str, Any],
    metadata: Metadata,
    dedup: bool = False,
) -> List[LoadPlan]:
    def _generate_one_local_load_plan(global_rank):
        # calc pp rank
        pp_rank = None
        for group in get_pipeline_model_parallel_replica_groups():
            for i, g in enumerate(group):
                if g == global_rank:
                    pp_rank = i
                    break
        # calc tp rank
        tp_rank = None
        for group in get_tensor_model_parallel_replica_groups():
            for i, g in enumerate(group):
                if g == global_rank:
                    tp_rank = i
                    break
        # calc dp rank
        dp_rank = None
        for group in get_data_parallel_replica_groups():
            for i, g in enumerate(group):
                if g == global_rank:
                    dp_rank = i
                    break

        wrapped_state_dict = _prepare_optim_state_dict(
            metadata, aux_infos, dedup=dedup, pp_rank=pp_rank, tp_rank=tp_rank, dp_rank=dp_rank
        )
        wrapped_state_dict = unflatten_state_dict(wrapped_state_dict, metadata.planner_data)
        planner = DefaultLoadPlanner()
        planner.set_up_planner(wrapped_state_dict, metadata, global_rank == 0)
        local_plan = planner.create_local_plan()
        return local_plan

    all_plans = [_generate_one_local_load_plan(i) for i in range(dist.get_world_size())]
    return all_plans


@functools.lru_cache(maxsize=None)  # equal to `@functools.cache`
def get_dcp_aux_infos(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
) -> Dict[str, Any]:
    return {
        "optim_pid_to_params": _get_optim_pid_to_params(optim),
        "optim_pid_to_pnames": _get_optim_pid_to_param_names(model, optim),
        "shape_info": optim.state_dict()["shape_info"],
    }


def save_optim_state_dict(
    path: str,
    state_dict: Dict[str, Any],
    aux_infos: Dict[str, Any],
    dedup: bool = False,
) -> Metadata:
    """
    Method to wrap optimizer state dict, make it become a DCP friendly format.

    Parameters:
        path (str):
            save path.
        state_dict (dict):
            optimizer state dict.
        aux_infos (dict):
            auxiliary infomation extracted from model and optimizer.
        dedup (bool):
            if deduplicate tensor parallel parameters.

    Returns:
        Metadata or None.
    """
    wrapped_state_dict = _wrap_optim_state_dict(state_dict, aux_infos, dedup=dedup)

    storage_writer = dist_cp.FileSystemWriter(path)
    is_coordinator = dist.get_rank() == 0
    planner = DefaultSavePlanner()

    global_metatadata = None

    # get SavePlans
    planner.set_up_planner(wrapped_state_dict, is_coordinator=is_coordinator)
    storage_writer.set_up_storage_writer(is_coordinator)
    local_plan = planner.create_local_plan()
    local_plan = storage_writer.prepare_local_plan(local_plan)

    all_local_plans = _generate_all_local_save_plans(state_dict, aux_infos, dedup=dedup)

    all_local_plans, global_metatadata = planner.create_global_plan(all_local_plans)
    all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
    central_plan = all_local_plans[dist.get_rank()]

    # write data and global metadata
    final_local_plan = planner.finish_plan(central_plan)
    all_writes = storage_writer.write_data(final_local_plan, planner)
    all_writes.wait()
    write_results = all_writes.value()

    # save + rename
    with open(os.path.join(path, "write_results.tmp.tmp.{}".format(dist.get_rank())), "wb") as f:
        pickle.dump(write_results, f)
        os.fsync(f.fileno())
    os.rename(
        os.path.join(path, "write_results.tmp.tmp.{}".format(dist.get_rank())),
        os.path.join(path, "write_results.tmp.{}".format(dist.get_rank())),
    )

    if dist.get_rank() == 0:
        file_paths = [os.path.join(path, "write_results.tmp.{}".format(i)) for i in range(dist.get_world_size())]
        count = 0
        success = False
        while count < MAX_RETRY and not success:
            try:
                if all(os.path.exists(file_path) for file_path in file_paths):
                    success = True
                else:
                    time.sleep(1)  # Wait for specified interval before retrying
                    count += 1
            except Exception as e:
                logging.debug(f"An error occurred: {e}")
                count += 1
        logging.debug("count: {}, rank: {}".format(count, dist.get_rank()))
        if not success:
            raise RuntimeError(rmsg("Failed to check write_results files exist."))

        all_write_results = []
        for global_rank in range(dist.get_world_size()):
            tmp_path = os.path.join(path, "write_results.tmp.{}".format(global_rank))
            with open(tmp_path, "rb") as f:
                all_write_results.append(pickle.load(f))
            os.remove(tmp_path)

        storage_writer.finish(metadata=global_metatadata, results=all_write_results)

    return global_metatadata


def load_optim_state_dict(
    path: str,
    optimizer: torch.optim.Optimizer,
    aux_infos: Dict[str, Any],
    dedup: bool = False,
) -> None:
    """
    Method to wrap optimizer state dict, make it become a DCP friendly format.

    Parameters:
        path (str):
            save path.
        state_dict (dict):
            optimizer state dict.
        aux_infos (dict):
            auxiliary infomation extracted from model and optimizer.
        dedup (bool):
            if deduplicate tensor parallel parameters.

    Returns:
        None.
    """
    storage_reader = dist_cp.FileSystemReader(path)
    is_coordinator = dist.get_rank() == 0
    planner = DefaultLoadPlanner()

    metadata = storage_reader.read_metadata()

    wrapped_state_dict = _prepare_optim_state_dict(metadata, aux_infos, dedup=dedup)
    wrapped_state_dict = unflatten_state_dict(wrapped_state_dict, metadata.planner_data)

    planner.set_up_planner(wrapped_state_dict, metadata, is_coordinator)
    storage_reader.set_up_storage_reader(metadata, is_coordinator)
    local_plan = planner.create_local_plan()
    local_plan = storage_reader.prepare_local_plan(local_plan)

    all_local_plans = _generate_all_local_load_plans(wrapped_state_dict, aux_infos, metadata, dedup=dedup)

    all_local_plans = planner.create_global_plan(all_local_plans)
    all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
    central_plan = all_local_plans[dist.get_rank()]

    final_local_plan = planner.finish_plan(central_plan)
    all_reads = storage_reader.read_data(final_local_plan, planner)
    all_reads.wait()
    xm.rendezvous("neuronx_distributed.optimizer.zero_dcp_utils.load_optim_state_dict")

    state_dict = _unwrap_optim_state_dict(wrapped_state_dict, aux_infos)
    optimizer.load_state_dict(state_dict)
