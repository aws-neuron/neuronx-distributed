from typing import List
import torch
import torch.nn as nn
import types
from .registry import RuntimeRegister
from neuronx_distributed.utils.logger import get_logger
from functools import wraps
import inspect

# Get the logger
logger = get_logger()


def patch_forward_with_additional_args(model: nn.Module, module_superset):
    original_forward = model.forward
    sig = inspect.signature(original_forward)

    # Capture original arg names (excluding self)
    original_params = list(sig.parameters.values())
    param_names = [p.name for p in original_params if p.name != 'self']
    num_original_args = len(param_names)

    @wraps(original_forward)
    def patched_forward(self, *args):
        # Split out the original args and tr_keys
        original_args = args[:num_original_args]
        tr_args = args[num_original_args:]

        # Number of modules we expect TF for (order defined by register)
        order = module_superset
        k = len(order)

        if len(tr_args) != 2 * k:
            raise ValueError(
                f"[Tensor replacement] modules: expected {2*k} trailing replacement args "
                f"(tensors[{k}] + masks[{k}]), got {len(tr_args)}."
            )

        tr_list   = list(tr_args[:k])
        mask_list = list(tr_args[k:2*k])

        RuntimeRegister.register_runtime_args(
            tr_args=tr_list,
            mask_args=mask_list
        )

        out = original_forward(*original_args)

        # Drop references immediately after the forward so memory can be recycled
        RuntimeRegister.clear_runtime_args()
        return out

    model.forward = types.MethodType(patched_forward, model)
    return model



def modify_model_for_tensor_replacement(model: nn.Module):
    model = patch_forward_with_additional_args(model, RuntimeRegister.module_superset)
    targets = RuntimeRegister.module_superset
    hooks = {}
    def make_hook(name):
        def hook(module: torch.nn.Module, input, output):
            tf = RuntimeRegister._tr_runtime_list.get(name)
            m  = RuntimeRegister._tr_mask_list.get(name)

            if tf is None or m is None:
                raise ValueError(f"Need tensor and mask for tensor replacement. Got {tf} and {m}")

            return torch.where(m, tf, output)  # Replace model output
        return hook
    

    for name, module in model.named_modules():
        if name in targets:
            hooks[name] = (module.register_forward_hook(make_hook(name)))

    return model, hooks

