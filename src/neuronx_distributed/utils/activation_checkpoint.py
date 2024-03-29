from functools import partial
from packaging import version

import torch
from torch_xla.utils.checkpoint import checkpoint as torch_xla_utils_checkpoint
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_PREFIX,
    CheckpointImpl,
    CheckpointWrapper,
)

from neuronx_distributed.parallel_layers.parallel_state import rmsg
from neuronx_distributed.pipeline import NxDPPModel
from neuronx_distributed.trainer.model import NxDModel
from neuronx_distributed.utils.logger import get_logger

logger = get_logger()


class NxDCheckpointWrapper(CheckpointWrapper):
    def named_modules(self, *args, **kwargs):
        for module_name, module in super().named_modules(*args, **kwargs):
            # torch>=2.0 compatibility
            if _CHECKPOINT_PREFIX.endswith("."):
                updated_name = module_name.replace(_CHECKPOINT_PREFIX, "")
            else:
                updated_name = module_name.replace(f"{_CHECKPOINT_PREFIX}.", "")
            yield updated_name, module


def checkpoint_wrapper(
    module: torch.nn.Module,
    checkpoint_impl: CheckpointImpl = CheckpointImpl.NO_REENTRANT,
    checkpoint_fn=None,
    **checkpoint_fn_kwargs,
) -> torch.nn.Module:
    if checkpoint_fn is None and version.parse(torch.__version__) >= version.parse("2.1"):
        checkpoint_fn = partial(
            torch_xla_utils_checkpoint,
            # PTXLA2.1: XLA currently does not support use_reentrant==False
            use_reentrant=True
        )
        # We have to explicitly set the impl to REENTRANT for >=2.1 because
        # the torch=2.1 first parses the implementation value before even checking 
        # the checkpoint_fn
        checkpoint_impl = CheckpointImpl.REENTRANT
    return NxDCheckpointWrapper(
        module,
        checkpoint_impl,
        checkpoint_fn,
        **checkpoint_fn_kwargs,
    )


def apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=lambda _: True,
):
    from torch.distributed.fsdp.wrap import _recursive_wrap, lambda_auto_wrap_policy

    if isinstance(model, NxDPPModel):
        logger.info(
            rmsg("Using NxDPPModel as input, `check_fn` will be ignored. Only transformer layer will be wrapped.")
        )
        activation_checkpoint_module = model.transformer_layer_cls
        check_fn = lambda m: isinstance(m, activation_checkpoint_module)
        model = model.local_module
    elif isinstance(model, NxDModel):
        model = model.local_module()
    _recursive_wrap(
        module=model,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=check_fn),
        wrapper_cls=checkpoint_wrapper_fn,
        ignored_modules=set(),
        ignored_params=set(),
        only_wrap_children=True,
    )
