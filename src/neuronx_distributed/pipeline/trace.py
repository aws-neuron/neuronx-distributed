import math
from abc import ABC
from types import ModuleType
from typing import Any, Callable, List, Optional, Union

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils.fx import HFTracer, get_concrete_args

import neuronx_distributed
from neuronx_distributed import parallel_layers
from neuronx_distributed.parallel_layers import PARALLEL_FUNCTIONS, PARALLEL_MODULES
from neuronx_distributed.parallel_layers.parallel_state import rmsg
from neuronx_distributed.utils.logger import get_logger

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


def get_tracer_class(model, tracer_cls=None):
    # Get the right tracer
    if tracer_cls is None:
        if isinstance(model, PreTrainedModel):
            tracer_cls = HFTracerWrapper
        else:
            tracer_cls = TorchTracerWrapper
    else:
        if isinstance(tracer_cls, str):
            if tracer_cls == "torch":
                tracer_cls = TorchTracerWrapper
            elif tracer_cls == "hf":
                tracer_cls = HFTracerWrapper
            else:
                raise ValueError(f"Unsupported tracer_cls {tracer_cls}")
        else:
            return tracer_cls


def trace_model(
    model: nn.Module,
    input_names: Optional[List[str]] = None,
    tracer_cls: Union[torch.fx.Tracer, str] = None,
    leaf_modules: Optional[List[Any]] = None,
    autowrap_functions: Optional[List[Callable]] = None,
    autowrap_modules: Optional[List[ModuleType]] = None,
):
    tracer_cls = get_tracer_class(model, tracer_cls=tracer_cls)

    if input_names is None:
        logger.warning(f"Getting input_names None. It is recommending to set up input names for tracing.")
        if isinstance(model, PreTrainedModel):
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
    autowrap_modules.extend([math, neuronx_distributed, parallel_layers])
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
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced_model = torch.fx.GraphModule(model, traced_graph)

    if isinstance(model, PreTrainedModel):
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
    for name in dict(traced_model.named_buffers()):
        if "tensor_constant" in name and hasattr(traced_model, name):
            traced_model.__delattr__(name)  # pylint: disable=unnecessary-dunder-call
    return traced_model
