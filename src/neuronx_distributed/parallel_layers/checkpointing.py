import gc
import os
from typing import Any, Callable, Optional, Dict

import torch
import torch_xla.core.xla_model as xm
import torch_xla.utils.serialization as xser

from ..utils.logger import get_logger
from .layers import create_local_weight
from .parallel_state import (
    get_data_parallel_rank,
    get_data_parallel_size,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)
from .utils import cast_all, get_local_world_size, move_all_tensor_to_cpu

logger = get_logger()

NXD_SKIP_RENDEZVOUS = "NXD_SKIP_RENDEZVOUS"


def ensure_directory_exists(filename: str) -> None:
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)


PreShardHookFn = Callable[[torch.nn.Module, dict, str], bool]


def _invoke_preshard_hook(module: torch.nn.Module, model_state_dict: Dict[str, Any], prefix: str = "") -> None:
    if module is None:
        return

    # This is temporary until we formailze the preshard_hook in src
    if hasattr(module, "preshard_hook"):
        module.preshard_hook(model_state_dict, prefix + "weight")
        return

    for name, child in module.named_children():
        _invoke_preshard_hook(child, model_state_dict, prefix + name + ".")


def get_sharded_model_dict(model: torch.nn.Module, model_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    from ..pipeline.model import NxDPPModel

    tp_size = get_tensor_model_parallel_size()
    if isinstance(model, NxDPPModel):
        model = model.original_torch_module

    _invoke_preshard_hook(model, model_state_dict)

    # Use state_dict to keep the shared parameters
    for name, param in model.state_dict(keep_vars=True).items():
        if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
            if param.partition_dim not in [0, 1]:
                raise Exception(f"Partiton value of 0,1 are supported, found {param.partition_dim}.")
            per_partition_size = model_state_dict[name].shape[param.partition_dim] // tp_size
            full_weight = model_state_dict[name]
            model_state_dict[name] = create_local_weight(
                full_weight, param.partition_dim, per_partition_size, param.partition_stride
            )
    return model_state_dict


def save(
    checkpoint: dict,
    output_dir: str,
    save_serially: bool = True,
    save_xser: bool = False,
    down_cast_bf16: bool = False,
    master_dp_only: bool = True,
) -> None:
    """Save a model/optimizer checkpoint.

    In save_xser case the file structure looks like:
    - output_dir:
      - tp_rank_xx_pp_rank_xx   (ref_data file)
      - tp_rank_xx_pp_rank_xx.tensors:
        - tensor_x.pt

    In non xser case the file structure looks like:
    - output_dir:
      - tp_rank_xx_pp_rank_xx:
        - checkpoint.pt

    """
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    logger.info("saving checkpoint to %s", output_dir)

    chkpt_path = os.path.join(
        output_dir,
        "tp_rank_{:02d}_pp_rank_{:02d}".format(get_tensor_model_parallel_rank(), get_pipeline_model_parallel_rank()),
    )
    if not master_dp_only:
        chkpt_path = chkpt_path + "_dp_rank_{:02d}".format(get_data_parallel_rank())

    if not save_xser:
        chkpt_path = os.path.join(chkpt_path, "checkpoint.pt")

    if down_cast_bf16:
        checkpoint = cast_all(checkpoint, from_dtype=torch.float32, to_dtype=torch.bfloat16)
    if (not save_xser) or rank % get_local_world_size() == 0:
        ensure_directory_exists(chkpt_path)
    xm.rendezvous("Ensure directory exists Done")

    if save_xser:
        master_only = get_data_parallel_rank() == 0 or not master_dp_only
        xser.save(checkpoint, chkpt_path, (not master_only), global_master=True)
    elif save_serially:
        # TODO: optmization to save multiple ranks together
        for tp_rank in range(0, get_tensor_model_parallel_size()):
            for pp_rank in range(0, get_pipeline_model_parallel_size()):
                # Staggering save checkpoints
                if not master_dp_only:
                    for dp_rank in range(0, get_data_parallel_size()):
                        cpu_data = move_all_tensor_to_cpu(checkpoint)
                        torch.save(cpu_data, chkpt_path)
                        del cpu_data
                        gc.collect()
                if (
                    get_data_parallel_rank() == 0
                    and get_tensor_model_parallel_rank() == tp_rank
                    and get_pipeline_model_parallel_rank() == pp_rank
                ):
                    cpu_data = move_all_tensor_to_cpu(checkpoint)
                    torch.save(cpu_data, chkpt_path)
                    del cpu_data
                    gc.collect()
                xm.rendezvous(f"ckpt-save-{tp_rank}")
    else:
        cpu_data = move_all_tensor_to_cpu(checkpoint, convert=(get_data_parallel_rank() == 0))
        if get_data_parallel_rank() == 0 or not master_dp_only:
            torch.save(cpu_data, chkpt_path)
            del cpu_data
            gc.collect()

    xm.rendezvous("Checkpoint Done")


def load(
    chkpt_path: str,
    model: Optional[torch.nn.Module] = None,
    model_or_optimizer: Any = None,
    model_key: Optional[str] = "model",
    load_xser: bool = False,
    sharded: bool = True,
    strict: bool = True,
    master_dp_only: bool = True,
    weights_only: bool = False,
) -> dict:
    """Load a checkpoint (model or optimizer) and return. In case the model/optimizer object is
    provided, it will load the model weights/optimizer stats. For large models/optimizers, to avoid
    host OOM, it is expected to pass the model/optimizer object.
    """

    if model is not None:
        logger.info(
            "`load` kwarg `model` is deprecated, please use `model_or_optimizer` instead as we are supporting to use `load` with optimizer as well"
        )  # noqa: E501
        model_or_optimizer = model

    if not sharded:
        assert (
            model_or_optimizer is not None
        ), "When checkpoint is not shareded, kwarg `model_or_optimizer` needs to be passed"  # noqa: E501

    logger.info("loading checkpoint from %s", chkpt_path)

    skip_rendezvous = os.environ.get(NXD_SKIP_RENDEZVOUS, None) == "1"

    if model_or_optimizer is not None:
        from ..pipeline.model import NxDPPModel

        if isinstance(model_or_optimizer, NxDPPModel) and not load_xser:
            logger.warning(
                "[Warning] It's recommended to use save_xser \
                    to save NxDPPModel to reduce saving time and redundant graphss"
            )

        model_moved_to_device = (
            isinstance(model_or_optimizer, NxDPPModel) and model_or_optimizer.model_moved_to_device
        ) or (
            isinstance(model_or_optimizer, torch.nn.Module)
            and list(model_or_optimizer.parameters())[0].device == xm.xla_device()
        )

        if isinstance(model_or_optimizer, torch.nn.Module) and not model_moved_to_device and load_xser:
            logger.warning(
                "[Warning] For save_xser case it is recommended to call load \
                    after moving model to device to reduce redundant graphs."
            )

    # Checkpoint.
    if sharded:
        checkpoint_name = os.path.join(
            chkpt_path,
            "tp_rank_{:02d}_pp_rank_{:02d}".format(
                get_tensor_model_parallel_rank(), get_pipeline_model_parallel_rank()
            ),
        )
        if not master_dp_only:
            checkpoint_name = checkpoint_name + "_dp_rank_{:02d}".format(get_data_parallel_rank())
        if not load_xser:
            checkpoint_name = os.path.join(checkpoint_name, "checkpoint.pt")
    else:
        checkpoint_name = chkpt_path

    logger.debug(" loading checkpoint from %s", chkpt_path)

    tp_size = get_tensor_model_parallel_size()
    tp_rank = get_tensor_model_parallel_rank()

    for worker_start in range(0, tp_size):
        if tp_rank == worker_start:
            logger.debug(f"Worker {tp_rank} resuming from checkpoint {checkpoint_name}")
            if load_xser:
                check_point = _xser_load(checkpoint_name, weights_only=weights_only)
            else:
                check_point = torch.load(checkpoint_name, weights_only=weights_only, map_location="cpu")
            if model_or_optimizer:
                if model_key is not None:
                    model_state_dict = check_point[model_key]
                else:
                    model_state_dict = check_point
                if not sharded:
                    model_state_dict = get_sharded_model_dict(model_or_optimizer, model_state_dict)
                if isinstance(model_or_optimizer, torch.optim.Optimizer):
                    model_or_optimizer.load_state_dict(model_state_dict)
                else:
                    model_or_optimizer.load_state_dict(model_state_dict, strict=strict)
                if model_key is not None:
                    del check_point[model_key]
                else:
                    check_point = None
            gc.collect()

        # Loading serially
        if not load_xser and not skip_rendezvous:
            xm.rendezvous("neuron.load_checkpoint" + str(worker_start))

    if load_xser and not skip_rendezvous:
        xm.rendezvous("neuron.load_checkpoint")

    return check_point


def _xser_load(path: torch.serialization.FILE_LIKE, weights_only: bool = False) -> Any:
    """
    Modify from xla serialization load https://github.com/pytorch/xla/blob/master/torch_xla/utils/serialization.py#L79-L100,
    with casting tensors to xla device to prevent OOM
    """
    ref_data = torch.load(path, weights_only=weights_only)

    def convert_fn(tensors):
        rewritten_tensors = []
        for t in tensors:
            tensor_path = os.path.join(path + ".tensors", "tensor_{}.pt".format(t.tid))
            rewritten_tensors.append(torch.load(tensor_path, weights_only=weights_only).to(device=xm.xla_device()))
        return rewritten_tensors

    def select_fn(v):
        return isinstance(v, xser.TensorReference)

    return xm.ToXlaTensorArena(convert_fn, select_fn).transform(ref_data)
