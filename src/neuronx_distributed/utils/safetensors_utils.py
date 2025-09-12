# Copyright 2025 HuggingFace Inc. Safetensors team. All rights reserved.
#
# This code is derived from HuggingFace Safetensors' torch.py implementation
# (specifically the duplicate tensor removal functionality during serialization).
# https://github.com/huggingface/safetensors/blob/main/bindings/python/py_src/safetensors/torch.py
# Slightly modified for compatibility with NeuronX Distributed Interface (NxDI).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from collections import defaultdict
from typing import Set

import logging

_float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
_float8_e5m2 = getattr(torch, "float8_e5m2", None)

_SIZE = {
    torch.int64: 8,
    torch.float32: 4,
    torch.int32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int16: 2,
    torch.uint8: 1,
    torch.int8: 1,
    torch.bool: 1,
    torch.float64: 8,
    _float8_e4m3fn: 1,
    _float8_e5m2: 1,
} 

def _storage_ptr(tensor):
    try:
        return tensor.untyped_storage().data_ptr()
    except Exception:
        # Fallback for torch==1.10
        try:
            return tensor.storage().data_ptr()
        except NotImplementedError:
            # Fallback for meta storage
            return 0

def _end_ptr(tensor):
    if tensor.nelement():
        stop = tensor.view(-1)[-1].data_ptr() + _SIZE[tensor.dtype]
    else:
        stop = tensor.data_ptr()
    return stop

def _storage_size(tensor):
    try:
        return tensor.untyped_storage().nbytes()
    except AttributeError:
        # Fallback for torch==1.10
        try:
            return tensor.storage().size() * _SIZE[tensor.dtype]
        except NotImplementedError:
            # Fallback for meta storage
            # On torch >=2.0 this is the tensor size
            return tensor.nelement() * _SIZE[tensor.dtype]

def _filter_shared_not_shared(tensors, state_dict):
    filtered_tensors = []
    for shared in tensors:
        if len(shared) < 2:
            filtered_tensors.append(shared)
            continue

        areas = []
        for name in shared:
            tensor = state_dict[name]
            areas.append((tensor.data_ptr(), _end_ptr(tensor), name))
        areas.sort()

        _, last_stop, last_name = areas[0]
        filtered_tensors.append({last_name})
        for start, stop, name in areas[1:]:
            if start >= last_stop:
                filtered_tensors.append({name})
            else:
                filtered_tensors[-1].add(name)
            last_stop = stop

    return filtered_tensors

def _find_shared_tensors(state_dict):
    tensors = defaultdict(set)
    for k, v in state_dict.items():
        if v.device != torch.device("meta") and _storage_ptr(v) != 0 and _storage_size(v) != 0:
            tensors[(v.device, _storage_ptr(v), _storage_size(v))].add(k)
    tensors = list(sorted(tensors.values()))
    tensors = _filter_shared_not_shared(tensors, state_dict)
    return tensors

def _is_complete(tensor):
    return tensor.data_ptr() == _storage_ptr(tensor) and tensor.nelement() * _SIZE[tensor.dtype] == _storage_size(tensor)

def _remove_duplicate_names(
    state_dict,
    *,
    preferred_names = None,
    discard_names = None,
    remove_duplicate_tensors = False,
):
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set([name for name in shared if _is_complete(state_dict[name])])
        if not complete_names and not remove_duplicate_tensors:
            logging.warning(f"Found shared tensors {shared}, this can cause checkpoint saving to fail. This warning can be safely ignored when using shard-on-load.")
            continue
        if not complete_names and remove_duplicate_tensors:
            raise RuntimeError(
                "Error while trying to find names to remove to save state dict, but found no suitable name to keep"
                f" for saving amongst: {shared}. None is covering the entire storage. Refusing to save/load the model"
                " since you could be storing much more memory than needed."
            )

        keep_name = sorted(list(complete_names))[0]

        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove

def  check_for_duplicate_tensors(checkpoint, remove_duplicate_tensors=False):
    """
    Removes duplicate tensor entries from a checkpoint dictionary to ensure compatibility with safetensors format.
    
    SafeTensors format does not support tensor sharing, so this function identifies and removes
    duplicate tensor references, keeping only one instance of each tensor.

    Args:
        checkpoint: Dictionary containing tensor names as keys and their corresponding tensor values
        remove_duplicate_tensors: If set, this will allow removing duplicate tensors from the checkpoint, if they exist.

    Returns:
        dict: Modified checkpoint dictionary with duplicate tensors removed if remove_duplicate_tensors is set,
        otherwise returns an unmodified checkpoint.
    """
    to_removes = _remove_duplicate_names(checkpoint, remove_duplicate_tensors=remove_duplicate_tensors)

    if not remove_duplicate_tensors:
        return checkpoint

    for _, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            del checkpoint[to_remove]

    return checkpoint