import collections
import os
from typing import List, Sequence

import torch
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm

from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_rank
)
from neuronx_distributed.utils.logger import get_logger

import numpy as np

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}

logger = get_logger()


class EmbeddingUtility:
    """Split the vocabulary into `world_size` chunks and return the
    first and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [fist, last)"""

    @staticmethod
    def range_from_per_partition_vocab_size(per_partition_vocab_size: int, rank, world_size: int) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) -> Sequence[int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return EmbeddingUtility.range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size)


def param_is_not_tensor_parallel_duplicate(param: torch.Tensor) -> bool:
    return (hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel) or (
        get_tensor_model_parallel_rank() == 0
    )


def set_tensor_model_parallel_attributes(tensor: torch.Tensor, is_parallel: bool, dim: int, stride: int = 1) -> None:
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(
    tensor: torch.Tensor,
) -> None:
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor: torch.Tensor, source_tensor: torch.Tensor) -> None:
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def is_torch_version_greater_than_2():
    return torch.__version__.startswith("2")


def is_pjrt_device():
    return os.environ.get("PJRT_DEVICE", None) == "NEURON"


def cast_tensor(tensor, from_dtype=torch.float32, to_dtype=torch.bfloat16):
    return tensor.to(dtype=to_dtype) if tensor.dtype == from_dtype else tensor


def cast_all(state, from_dtype=torch.float32, to_dtype=torch.bfloat16):
    if isinstance(state, torch.Tensor):
        return cast_tensor(state, from_dtype=from_dtype, to_dtype=to_dtype)
    else:
        if isinstance(state, dict):
            new_dict = {}
            for k in state.keys():
                new_dict[k] = cast_all(state[k], from_dtype=from_dtype, to_dtype=to_dtype)
            return new_dict
        elif isinstance(state, (list, tuple)):
            return type(state)(cast_all(x, from_dtype=from_dtype, to_dtype=to_dtype) for x in state)
        else:
            # We only cast Tensor, list, tuple or dict of tensors.
            return state

# Refering to https://github.com/NVIDIA/apex/blob/master/apex/_autocast_utils.py#L22
def cast_if_autocast_enabled(*args):
    if not torch.is_autocast_enabled():
        return args
    else:
        return _cast(args, torch.get_autocast_gpu_dtype())

# Modifying from https://github.com/pytorch/pytorch/blob/main/torch/cuda/amp/autocast_mode.py#L57, removed check for cuda device
def _cast(value, dtype):
    if isinstance(value, torch.Tensor):
        is_eligible = (
            value.is_floating_point()
            and (value.dtype is not torch.float64)
        )
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, (str, bytes)):
        return value
    elif isinstance(value, np.ndarray):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in value.items()}
    elif isinstance(value, collections.abc.Iterable):
        iterable = (_cast(v, dtype) for v in value)
        if isinstance(value, (list, tuple)):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value

def move_all_tensor_to_cpu(data, convert=True):
    def is_xla_tensor(tensor):
        return tensor.device.type == "xla"

    def convert_fn(tensors):
        torch_xla._XLAC._xla_sync_multi(tensors, devices=[], wait=True, sync_xla_data=True)
        if not convert:
            return tensors
        return [tensor.to("cpu") for tensor in tensors]

    def select_fn(v):
        return type(v) == torch.Tensor and is_xla_tensor(v)

    return xm.ToXlaTensorArena(convert_fn, select_fn).transform(data)


def get_local_world_size():
    if is_torch_version_greater_than_2():
        # With pjrt this only works after init_process_group()
        import torch_xla.experimental.pjrt as pjrt

        return pjrt.local_process_count()
    else:
        return xm.xrt_world_size() // int(os.environ[xenv.HOST_WORLD_SIZE])


def move_model_to_device(model: torch.nn.Module, device: torch.device) -> None:
    logger.warn(
        "parallel_layers.move_model_to_device method is deprecated, \
        please use neuronx_distributed.utils.model_utils.move_model_to_device"
    )
    from neuronx_distributed.utils import model_utils

    model_utils.move_model_to_device(model, device)

def verify_casted_dtype(value):
    """Veryfy whether the input values have been casted correctly"""
    if not torch.is_autocast_enabled():
        return
    else:
        if isinstance(value, torch.Tensor):
            assert value.dtype == torch.get_autocast_gpu_dtype(), f"Datatype of tensor is expected to be {torch.get_autocast_gpu_dtype()}, got {value.dtype} instead"
        elif isinstance(value, collections.abc.Mapping):
            for k, v in value.items():
                verify_casted_dtype(k)
                verify_casted_dtype(v)
        elif isinstance(value, collections.abc.Iterable):
            for v in value:
                verify_casted_dtype(v)
        else:
            return
