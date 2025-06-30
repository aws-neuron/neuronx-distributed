import math
from contextlib import contextmanager
from enum import IntEnum
from typing import Dict, List, Optional, Set, Union, Callable, Any, Iterable, Iterator

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch import nn
from torch_neuronx.utils import get_platform_target

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

_NXDT_AVAIL = True
try:
    from neuronx_distributed_training.models.megatron.module import MegatronModule
except ImportError:
    _NXDT_AVAIL = False


def analyze_shared_parameters(
    module: torch.nn.Module,
    shared_parameters: Optional[Dict[torch.nn.Parameter, List[str]]] = None,
    prefix: str = "",
) -> List[List[str]]:
    """
    Find the shared parameters names for a certain module
    [TODO] for PT 2.x we can use remove_duplicate=False from parameters/named_parameters
    """
    sp: Dict[torch.nn.Parameter, List[str]] = shared_parameters or {}
    for name, param in module._parameters.items():
        if param is None:
            continue
        param_prefix = prefix + ("." if prefix else "") + name
        if param not in sp:
            sp[param] = []
        sp[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        analyze_shared_parameters(m, sp, submodule_prefix)
    return [x for x in sp.values() if len(x) > 1]


def retie_shared_weights(module: torch.nn.Module, shared_weight_names: List[List[str]]) -> None:
    """
    Iterate module by module to retie the shared weights
    referred from: https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/utilities/parameter_tying.py#L47  # noqa: E501
    """
    for shared_param in shared_weight_names:
        ref = _get_module_by_path(module, shared_param[0])
        for path in shared_param[1:]:
            _set_module_by_path(module, path, ref)


def _get_module_by_path(module: torch.nn.Module, path: str) -> torch.nn.Module:
    path_parts: List[str] = path.split(".")
    for name in path_parts:
        module = getattr(module, name)
    return module


def _set_module_by_path(module: torch.nn.Module, path: str, value: torch.nn.Module) -> None:
    path_parts: List[str] = path.split(".")
    for name in path_parts[:-1]:
        module = getattr(module, name)
    setattr(module, path_parts[-1], value)


def is_hf_pretrained_model(model: torch.nn.Module) -> bool:
    return _TRANSFORMERS_AVAIL and isinstance(model, PreTrainedModel)


def is_hf_transformers_available() -> bool:
    return _TRANSFORMERS_AVAIL


def is_hf_accelerate_available() -> bool:
    return _Accelerate_AVAIL


def is_nxdt_pretrained_model(model: torch.nn.Module) -> bool:
    return _NXDT_AVAIL and isinstance(model, MegatronModule)


def is_nxdt_available() -> bool:
    return _NXDT_AVAIL


def recursive_filter(item, predicate):
    """Filter a structure containing tensors based on the given predicate"""

    def _is_tensor_or_parameter(obj):
        return isinstance(obj, (torch.Tensor, nn.Parameter))

    def _augmented_predicate(obj):
        return predicate(obj) if _is_tensor_or_parameter(obj) else True

    out: Union[dict, List, set, tuple, None]
    if isinstance(item, Dict):
        out = {}
        for k, v in item.items():
            if _augmented_predicate(v):
                out[k] = recursive_filter(v, predicate)
    elif isinstance(item, (List, tuple, set)):
        out_tmp: List = []
        for x in item:
            if _augmented_predicate(x):
                out_tmp.append(recursive_filter(x, predicate))
        out = type(item)(out_tmp)
    else:
        # under normal circumstances this should not return None, unless
        # there is an unexpected data structure involved
        out = item if _augmented_predicate(item) else None

    return out


@contextmanager
def preserve_shared_weights(model: torch.nn.Module, ignore_hf: bool = False) -> Iterator[None]:
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
def preserve_parallel_attributes(model: torch.nn.Module) -> Iterator[None]:
    """
    Preserve the following 3 attributes for the model parameters
        - tensor_model_parallel
        - expert_model_parallel
        - sequence_parallel_enabled
        - shared
    """
    tp_params = {}
    ep_params = {}
    seq_parallel_params = {}
    shared_parameters = {}
    for name, param in model.named_parameters():
        if hasattr(param, "tensor_model_parallel"):
            tp_params[name] = {
                "is_parallel": param.tensor_model_parallel,
                "partition_dim": getattr(param, "partition_dim"),
                "stride": getattr(param, "partition_stride"),
                "num_partitions": getattr(param, "num_partitions"),
            }
        if hasattr(param, "expert_model_parallel"):
            ep_params[name] = param.expert_model_parallel
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
            if name in ep_params and not hasattr(param, "expert_model_parallel"):
                setattr(param, "expert_model_parallel", ep_params[name])
            if name in seq_parallel_params and not hasattr(param, "sequence_parallel_enabled"):
                setattr(param, "sequence_parallel_enabled", seq_parallel_params[name])
            if name in shared_parameters and not hasattr(param, "shared"):
                setattr(param, "shared", shared_parameters[name])


def _set_module_param_to_empty(module: torch.nn.Module, device: torch.device, recurse: bool = False) -> None:
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


def reinit_model(model: torch.nn.Module, device: torch.device, param_init_fn: Callable[[torch.nn.Module, torch.device], Any]) -> None:
    """
    Re-initialize model with the param_init_fn on provided device
    """
    with preserve_parallel_attributes(model):
        with preserve_shared_weights(model):
            with torch.no_grad():
                for module in model.modules():
                    _set_module_param_to_empty(module, device)
                    param_init_fn(module, device)


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
    ignored_params: Optional[Set[torch.nn.Parameter]] = None,
) -> bool:
    if not _TORCHDISTX_AVAIL:
        return False
    for param in model.parameters():
        if (not ignored_params or param not in ignored_params) and fake.is_fake(param):
            return True
    return False


@contextmanager
def init_on_device(
    device: torch.device, include_buffers: bool = False, force_custom_init_on_device: bool = False
) -> Iterator[None]:
    """
    A context manager under which models are initialized with all parameters on the specified device.
    Referred from: https://github.com/huggingface/accelerate/blob/main/src/accelerate/big_modeling.py#L82

    NOTE: force_custom_init_on_device is set to force the custom implementation as there is a bug in the Huggingface implementation
    where if the parameter requires_grad =False, it will set to True when it reinitializes the parameter as 'module._parameters[name].__dict__'
    Does not have an attribute requires_grad when its set to False.
    """
    # Directly use accelerate implementation if available
    if is_hf_accelerate_available() and not force_custom_init_on_device:
        with hf_init_on_device(device, include_buffers=include_buffers):
            yield
        return

    if is_torch_version_greater_than_2() and include_buffers and not force_custom_init_on_device:
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
            kwargs["requires_grad"] = module._parameters[name].requires_grad
            # When we have a case of tensor2 = tensor1, it would call the set_attr
            # of param, which in turn would call the register_parameter API.
            # In this case, the new param is already on meta-device, since it was moved
            # previously when it was initialized. Hence, when resetting, you can
            # directly assign that tensor instead of re-init. If you re-init you would
            # lose the relationship.
            module._parameters[name] = (
                param if param.device == device else param_cls(module._parameters[name].to(device), **kwargs)
            )

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full", "arange", "zeros_like", "ones_like"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        setattr(nn.Module, "register_parameter", register_empty_parameter)
        if include_buffers:
            setattr(nn.Module, "register_buffer", register_empty_buffer)
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        setattr(nn.Module, "register_parameter", old_register_parameter)
        if include_buffers:
            setattr(nn.Module, "register_buffer", old_register_buffer)
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)


def get_model_sequential(
    model: torch.nn.Module,
    device: torch.device,
    sequential_move_factor: int = 11,
    param_init_fn: Optional[Callable[[torch.nn.Module, torch.device], Any]] = None,
) -> torch.nn.Module:
    from ..pipeline import NxDPPModel

    local_rank = xr.local_ordinal()
    local_world_size = get_local_world_size()
    for worker in range(math.ceil(local_world_size / sequential_move_factor)):
        if local_rank // sequential_move_factor == worker:
            if isinstance(model, NxDPPModel):
                model.maybe_materialize_local_module()
                model.move_model_to_device()
            else:
                maybe_materalize_model(model)
                if param_init_fn is not None:
                    reinit_model(model, torch.device("cpu"), param_init_fn)
                move_model_to_device(model, device)
        xm.rendezvous("get_model_sequential" + str(worker))
    return model


def check_delay_tracing(nxd_config) -> bool:
    # Temporarily disabling delayed tracing while we investigate some issues
    # TODO re-enable once the issues with delayed tracing are resolved
    return False


def get_delay_tracing(arg) -> bool:
    # Temporarily disabling delayed tracing while we investigate some issues
    # TODO re-enable once the issues with delayed tracing are resolved
    return False

class LogicalNCConfig(IntEnum):
    LNC_1 = 1
    LNC_2 = 2

def get_platform_lnc():
    """
    Get the Logical NeuronCore Configuration (LNC) for the current platform.
    """
    return LogicalNCConfig.LNC_2 if get_platform_target() == "trn2" else LogicalNCConfig.LNC_1
