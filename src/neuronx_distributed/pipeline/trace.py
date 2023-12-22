import inspect
import math
from abc import ABC
from collections import defaultdict
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch_xla.core.xla_model as xm
from torch import nn

try:
    # In case this is removed in the future
    from torch.fx._symbolic_trace import _create_wrapped_func  # noqa
except ImportError:
    _create_wrapped_func = None

import neuronx_distributed
from neuronx_distributed import parallel_layers
from neuronx_distributed.parallel_layers import PARALLEL_FUNCTIONS, PARALLEL_MODULES
from neuronx_distributed.parallel_layers.parallel_state import rmsg
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.utils.model_utils import (
    is_hf_pretrained_model,
    is_hf_transformers_available,
)

logger = get_logger()


class NxDTracer(ABC):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        is_leaf = False
        if any(t in type(m).__name__ for t in self.leaf_modules) or any(
            t == module_qualified_name for t in self.leaf_modules
        ):
            is_leaf = True
        else:
            is_leaf = super().is_leaf_module(m, module_qualified_name)
        return is_leaf


if is_hf_transformers_available():
    from transformers.utils.fx import HFTracer

    class HFTracerWrapper(NxDTracer, HFTracer):
        def __init__(self, **config) -> None:
            super().__init__(
                autowrap_modules=config["autowrap_modules"],
                autowrap_functions=config["autowrap_functions"],
            )
            self.leaf_modules = config.get("leaf_modules", ())
            self.name = "HF"


class TorchTracerWrapper(NxDTracer, torch.fx.Tracer):
    def __init__(self, **config) -> None:
        """
        Setting param_shapes_constant to True,
        from https://github.com/pytorch/pytorch/blob/main/torch/fx/_symbolic_trace.py#L263C13-L267C31
        param_shapes_constant (bool): When this flag is set,  calls to shape,
            size and a few other shape like attributes of a module's parameter
            will be evaluated directly, rather than returning a new Proxy value
            for an attribute access. Backward compatibility for this parameter
            is guaranteed.
        """
        super().__init__(
            autowrap_modules=config["autowrap_modules"],
            autowrap_functions=config["autowrap_functions"],
            param_shapes_constant=True,
        )
        self.leaf_modules = config.get("leaf_modules", [])
        self.name = "pytorch"


def get_concrete_args(model: nn.Module, input_names: List[str]):
    sig = inspect.signature(model.forward)

    if not (set(input_names) <= set(sig.parameters.keys())):
        formatted_input_names = input_names[0] if len(input_names) == 1 else ", ".join(input_names)
        formatted_allowed_input_names = ", ".join(sig.parameters.keys())
        raise ValueError(
            f"The model does not have input(s) named: {formatted_input_names}, expected a subset of the following:"
            f" {formatted_allowed_input_names}"
        )

    return {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}


def get_tracer_class(model, tracer_cls=None):
    # Get the right tracer
    if tracer_cls is None:
        if is_hf_pretrained_model(model):
            tracer_cls = HFTracerWrapper
        else:
            tracer_cls = TorchTracerWrapper
    else:
        if isinstance(tracer_cls, str):
            if tracer_cls == "torch":
                tracer_cls = TorchTracerWrapper
            elif tracer_cls == "hf" and is_hf_transformers_available():
                tracer_cls = HFTracerWrapper
            else:
                raise ValueError(f"Unsupported tracer_cls {tracer_cls}")
    return tracer_cls


@contextmanager
def patch_obj_method(autowrap_obj_methods):
    """
    Patch the methods from a certain object, so FX's symbolic trace won't trace inside the method when it is called
    """
    enabled = _create_wrapped_func is not None and autowrap_obj_methods is not None
    if enabled:
        original_method = defaultdict(dict)
        for obj, methods in autowrap_obj_methods.items():
            assert isinstance(
                methods, list
            ), f"Expect autowrap_obj_methods has list as value but getting {type(methods)}"
            for method in methods:
                if not hasattr(obj, method):
                    raise ValueError(f"Inside autowrap_obj_methods obj type {type(obj)} does not have method {method}")
                original_method[obj][method] = getattr(obj, method)
                setattr(obj, method, _create_wrapped_func(original_method[obj][method]))
    try:
        yield
    finally:
        if enabled:
            for obj, methods in original_method.items():
                for name, original_method in methods.items():
                    setattr(obj, name, original_method)


def trace_model(
    model: nn.Module,
    input_names: Optional[List[str]] = None,
    tracer_cls: Union[Any, str] = None,
    leaf_modules: Optional[List[Any]] = None,
    autowrap_functions: Optional[List[Callable]] = None,
    autowrap_modules: Optional[List[ModuleType]] = None,
    autowrap_obj_methods: Optional[Dict[Any, List[Callable]]] = None,
):
    if _create_wrapped_func is None and autowrap_obj_methods is not None:
        logger.warning(
            f"Can not import _create_wrapped_func from torch.fx.__symbolic_trace, autowrap_obj_method will be ignored"
        )

    tracer_cls = get_tracer_class(model, tracer_cls=tracer_cls)

    if input_names is None:
        logger.warning(f"Getting input_names None. It is recommending to set up input names for tracing.")
        if is_hf_pretrained_model(model):
            input_names = model.dummy_inputs.keys()
        else:
            input_names = []

    input_names = list(input_names)
    concrete_args = get_concrete_args(model, input_names)

    # User specified leaf modules to skip
    if leaf_modules is None:
        leaf_modules = []
    leaf_modules.extend([module.__name__ for module in PARALLEL_MODULES])
    leaf_modules = tuple(set(leaf_modules))
    # Leaf functions to skip
    if autowrap_functions is None:
        autowrap_functions = []
    autowrap_functions.extend(PARALLEL_FUNCTIONS)
    autowrap_functions = tuple(set(autowrap_functions))
    # Everything from these modules will be skipped for tracing
    if autowrap_modules is None:
        autowrap_modules = []
    autowrap_modules.extend([math, neuronx_distributed, parallel_layers, xm])
    autowrap_modules = tuple(set(autowrap_modules))

    logger.debug(rmsg(f"leaf_modules {leaf_modules}"))
    logger.debug(rmsg(f"autowrap_functions {autowrap_functions}"))
    logger.debug(rmsg(f"autowrap_modules {autowrap_modules}"))
    # Tracing.
    tracer = tracer_cls(
        leaf_modules=leaf_modules,
        autowrap_modules=autowrap_modules,
        autowrap_functions=autowrap_functions,
    )
    with patch_obj_method(autowrap_obj_methods):
        traced_graph = tracer.trace(model, concrete_args=concrete_args)
        traced_model = torch.fx.GraphModule(model, traced_graph)

    if is_hf_pretrained_model(model):
        traced_model.config = model.config
        # The model class must be stored as an attribute to allow model deserialization, which uses trace, and thus
        # _generate_dummy_input, where the model class is needed.
        traced_model.class_for_deserialization = model.__class__
        traced_model.device = model.device

    # Some clean up
    traced_model.graph.eliminate_dead_code()
    traced_model.delete_all_unused_submodules()
    traced_model.graph.lint()
    traced_model.recompile()

    # Remove the meta tensors created by HF tracer
    for name, buffer in dict(traced_model.named_buffers()).items():
        if "_tensor_constant" in name and hasattr(traced_model, name) and buffer.device == torch.device("meta"):
            traced_model.__delattr__(name)  # pylint: disable=unnecessary-dunder-call
    return traced_model
