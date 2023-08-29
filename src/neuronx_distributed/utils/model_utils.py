from contextlib import contextmanager
from typing import Optional, Set

import torch

from ..parallel_layers.parallel_state import rmsg
from ..parallel_layers.utils import set_tensor_model_parallel_attributes
from ..utils.logger import get_logger

logger = get_logger()

_TRANSFORMERS_AVAIL = True
try:
    from transformers import PreTrainedModel
except ImportError:
    _TRANSFORMERS_AVAIL = False

_TORCHDISTX_AVAIL = True
try:
    from torchdistx import deferred_init, fake
except ImportError:
    _TORCHDISTX_AVAIL = False


def analyze_shared_parameters(module, shared_parameters=None, prefix=""):
    """
    Find the shared parameters names for a certain module
    [TODO] for PT 2.x we can use remove_duplicate=False from parameters/named_parameters
    """
    if shared_parameters is None:
        shared_parameters = {}
    for name, param in module._parameters.items():
        param_prefix = prefix + ("." if prefix else "") + name
        if param is None:
            continue
        if param not in shared_parameters:
            shared_parameters[param] = []
        shared_parameters[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        analyze_shared_parameters(m, shared_parameters, submodule_prefix)
    return [x for x in shared_parameters.values() if len(x) > 1]


def retie_shared_weights(module, shared_weight_names):
    """
    Iterate module by module to retie the shared weights
    referred from: https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/utilities/parameter_tying.py#L47  # noqa: E501
    """
    for shared_param in shared_weight_names:
        ref = _get_module_by_path(module, shared_param[0])
        for path in shared_param[1:]:
            _set_module_by_path(module, path, ref)


def _get_module_by_path(module: torch.nn.Module, path: str):
    path = path.split(".")
    for name in path:
        module = getattr(module, name)
    return module


def _set_module_by_path(module: torch.nn.Module, path: str, value: torch.nn.Module):
    path = path.split(".")
    for name in path[:-1]:
        module = getattr(module, name)
    setattr(module, path[-1], value)


def is_hf_pretrained_model(model):
    return _TRANSFORMERS_AVAIL and isinstance(model, PreTrainedModel)


def is_hf_transformers_available():
    return _TRANSFORMERS_AVAIL


@contextmanager
def preserve_shared_weights(model: torch.nn.Module, ignore_hf=False) -> None:
    """
    Retie the shared weights after exiting the context manager
    """
    if not is_hf_pretrained_model(model) or ignore_hf:
        shared_parameter_names = analyze_shared_parameters(model)
        logger.debug(rmsg(f"Find shared weights {shared_parameter_names}"))
    try:
        yield
    finally:
        if is_hf_pretrained_model(model) and not ignore_hf:
            model.tie_weights()
        else:
            retie_shared_weights(model, shared_parameter_names)


@contextmanager
def preserve_parallel_attributes(model: torch.nn.Module) -> None:
    """
    Preserve the following 3 attributes for the model parameters
        - tensor_model_parallel
        - sequence_parallel_enabled
        - shared
    """
    tp_params = {}
    seq_parallel_params = {}
    shared_parameters = {}
    for name, param in model.named_parameters():
        if hasattr(param, "tensor_model_parallel"):
            tp_params[name] = {
                "is_parallel": param.tensor_model_parallel,
                "partition_dim": param.partition_dim,
                "stride": param.partition_stride,
            }
        if hasattr(param, "sequence_parallel_enabled"):
            seq_parallel_params[name] = param.sequence_parallel_enabled
        if hasattr(param, "shared"):
            shared_parameters[name] = param.shared
    try:
        yield
    finally:
        for name, param in model.named_parameters():
            if name in tp_params and not hasattr(param, "tensor_model_parallel"):
                set_tensor_model_parallel_attributes(param, *tp_params[name].values())
            if name in seq_parallel_params and not hasattr(param, "sequence_parallel_enabled"):
                setattr(param, "sequence_parallel_enabled", seq_parallel_params[name])
            if name in shared_parameters and not hasattr(param, "shared"):
                setattr(param, "shared", shared_parameters[name])


def move_model_to_device(model: torch.nn.Module, device: torch.device) -> None:
    with preserve_parallel_attributes(model):
        with preserve_shared_weights(model):
            model.to(device)


def maybe_materalize_model(model: torch.nn.Module) -> None:
    if has_fake_tensors(model):
        with preserve_parallel_attributes(model):
            deferred_init.materialize_module(model)


def has_fake_tensors(
    model: torch.nn.Module,
    ignored_params: Optional[Set[torch.nn.Parameter]] = {},
) -> bool:
    if not _TORCHDISTX_AVAIL:
        return False
    for param in model.parameters():
        if param not in ignored_params and fake.is_fake(param):
            return True
