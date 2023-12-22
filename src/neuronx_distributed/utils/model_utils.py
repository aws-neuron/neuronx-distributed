import math
from contextlib import contextmanager
from typing import Optional, Set

import torch
import torch_xla.core.xla_model as xm
from torch import nn

from ..parallel_layers.parallel_state import rmsg
from ..parallel_layers.utils import (
    get_local_world_size,
    is_torch_version_greater_than_2,
    set_tensor_model_parallel_attributes,
)
from ..utils.logger import get_logger

logger = get_logger()

_TRANSFORMERS_AVAIL = True
try:
    from transformers import PreTrainedModel
except ImportError:
    _TRANSFORMERS_AVAIL = False

_Accelerate_AVAIL = True
try:
    from accelerate import init_on_device as hf_init_on_device
except ImportError:
    _Accelerate_AVAIL = False

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


def is_hf_accelerate_available():
    return _Accelerate_AVAIL


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


def _set_module_param_to_empty(module: torch.nn.Module, device: torch.device, recurse: bool = False):
    """
    Set all parameters for input module to empty like tensors on provided device
    """
    for key, param in module._parameters.items():
        if param is None:
            continue
        with torch.no_grad():
            assert isinstance(param, torch.nn.parameter.Parameter)
            assert param.is_leaf
            t = torch.empty_like(param, device=device)
            out_param = torch.nn.parameter.Parameter(t, param.requires_grad)
            module._parameters[key] = out_param


def reinit_model(model: torch.nn.Module, device: torch.device, param_init_fn):
    """
    Re-initialize model with the param_init_fn on provided device
    """
    with preserve_parallel_attributes(model):
        with preserve_shared_weights(model):
            with torch.no_grad():
                for module in model.modules():
                    _set_module_param_to_empty(module, device)
                    param_init_fn(module)


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


@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = False):
    """
    A context manager under which models are initialized with all parameters on the specified device.
    Referred from: https://github.com/huggingface/accelerate/blob/main/src/accelerate/big_modeling.py#L82
    """
    # Directly use accelerate implementation if available
    if is_hf_accelerate_available():
        with hf_init_on_device(device, include_buffers=include_buffers):
            yield
        return

    if is_torch_version_greater_than_2() and include_buffers:
        with device:
            yield
        return

    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def get_model_sequential(model, device, sequential_move_factor=11, param_init_fn=None):
    from ..pipeline import NxDPPModel

    local_rank = xm.get_local_ordinal()
    local_world_size = get_local_world_size()
    for worker in range(math.ceil(local_world_size / sequential_move_factor)):
        if local_rank // sequential_move_factor == worker:
            if isinstance(model, NxDPPModel):
                model.move_model_to_device()
            else:
                maybe_materalize_model(model)
                if param_init_fn is not None:
                    reinit_model(model, torch.device("cpu"), param_init_fn)
                move_model_to_device(model, device)
        xm.rendezvous("get_model_sequential" + str(worker))
    return model
