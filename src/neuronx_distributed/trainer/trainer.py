from pprint import pformat

import torch
import torch_xla.core.xla_model as xm

from neuronx_distributed.optimizer import NeuronZero1Optimizer
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.pad import pad_model
from neuronx_distributed.parallel_layers.parallel_state import rmsg
from neuronx_distributed.pipeline import NxDPPModel
from neuronx_distributed.trainer.model import NxDModel
from neuronx_distributed.trainer.optimizer import NxDOptimizer
from neuronx_distributed.utils.activation_checkpoint import (
    apply_activation_checkpointing,
)
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.utils.model_utils import (
    get_model_sequential,
    init_on_device,
    is_hf_pretrained_model,
)

logger = get_logger()


def neuronx_distributed_config(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    pipeline_config=None,
    optimizer_config=None,
    activation_checkpoint_config=None,
    pad_model=False,
    sequence_parallel=False,
    model_init_config=None,
):
    if optimizer_config is None:
        optimizer_config = {"zero_one_enabled": False, "grad_clipping": True, "max_grad_norm": 1.0}
    else:
        assert isinstance(optimizer_config, dict), "optimizer_config must be a dict."
        if "zero_one_enabled" not in optimizer_config:
            logger.warning(rmsg("zero_one_enabled is not set, automatically set it to False."))
            optimizer_config.update({"zero_one_enabled": False})
        if "grad_clipping" not in optimizer_config:
            logger.warning(rmsg("grad_clipping is not set, automatically set it to True."))
            optimizer_config.update({"grad_clipping": True})
        if optimizer_config["grad_clipping"]:
            if "max_grad_norm" not in optimizer_config:
                logger.warning(rmsg("max_grad_norm is not set, automatically set it to one."))
                optimizer_config.update({"max_grad_norm": 1.0})

    if model_init_config is None:
        model_init_config = {
            "sequential_move_factor": 11,  # randomly chosen number, work for 20B size model
            "meta_device_init": False,
            "param_init_fn": None,
        }
    else:
        assert isinstance(model_init_config, dict), "model_init_config must be a dict."
        if "sequential_move_factor" not in model_init_config:
            logger.warning(
                rmsg(
                    "sequential_move_factor is not set, automatically set it to 11, this number works for 20B size model."
                )
            )
            model_init_config.update({"sequential_move_factor": 11})  # randomly chosen number, work for 20B size model
        if "meta_device_init" not in model_init_config:
            logger.warning(rmsg("meta_device_init is not set, automatically set it to False."))
            model_init_config.update({"meta_device_init": False})
        if model_init_config["meta_device_init"]:
            if "param_init_fn" not in model_init_config:
                raise ValueError("param_init_fn must be provided when meta_device_init is True")

    config = {
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "pipeline_config": pipeline_config,
        "optimizer_config": optimizer_config,
        "activation_checkpoint_config": activation_checkpoint_config,
        "pad_model": pad_model,
        "sequence_parallel": sequence_parallel,
        "model_init_config": model_init_config,
    }

    if torch.distributed.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=config["tensor_parallel_size"],
            pipeline_model_parallel_size=config["pipeline_parallel_size"],
        )

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        logger.info("NxD config: \n{}".format(pformat(config)))
    return config


def initialize_parallel_model(nxd_config, model_fn, *model_args, **model_kwargs):
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=nxd_config["tensor_parallel_size"],
            pipeline_model_parallel_size=nxd_config["pipeline_parallel_size"],
        )

    # Phase 1: get model and wrap with NxDPPModel
    # TODO: deferred init by config
    meta_device_init = nxd_config["model_init_config"]["meta_device_init"]
    param_init_fn = nxd_config["model_init_config"].get("param_init_fn", None)

    if meta_device_init:
        with init_on_device(device=torch.device("meta")):
            model = model_fn(*model_args, **model_kwargs)
    else:
        model = model_fn(*model_args, **model_kwargs)
    pp_enabled = nxd_config["pipeline_parallel_size"] > 1
    if pp_enabled:
        nxd_config["pipeline_config"].update({"param_init_fn": param_init_fn})
        model = NxDPPModel(model, **nxd_config["pipeline_config"])

    # Phase 2: materialize model and move to device
    sequential_move_factor = nxd_config["model_init_config"]["sequential_move_factor"]

    device = xm.xla_device()
    model = get_model_sequential(model, device, sequential_move_factor, param_init_fn)

    # Phase 3: apply optimization what will change model structure
    # Currently we have:
    #   - pad model
    # TODO: pad model support pp
    if nxd_config["pad_model"]:
        assert is_hf_pretrained_model(model), "only support pad huggingface model now."
        model = pad_model(model, parallel_state.get_tensor_model_parallel_size(), model.config.num_attention_heads)

    # Phase 4: apply optimization what will not change model structure
    # Currently we have:
    #   - activation checkpoint
    nxd_model = NxDModel(model, nxd_config)
    if nxd_config["activation_checkpoint_config"] is not None:
        if nxd_config["activation_checkpoint_config"] == "full":
            if pp_enabled:
                activation_checkpoint_classes = (model.transformer_layer_cls,)
            else:
                assert is_hf_pretrained_model(
                    model
                ), '`activation_checkpoint_config` is "full" only support huggingface transformers model'
                from transformers.trainer_pt_utils import get_module_class_from_name

                activation_checkpoint_classes = []
                for name in model._no_split_modules:
                    activation_checkpoint_classes.append(get_module_class_from_name(nxd_model, name))
        else:
            activation_checkpoint_classes = nxd_config["activation_checkpoint_config"]
            if not isinstance(activation_checkpoint_classes, (list, tuple)):
                activation_checkpoint_classes = (activation_checkpoint_classes,)
        activation_checkpoint_classes = tuple(activation_checkpoint_classes)
        assert len(activation_checkpoint_classes) > 0
        assert all(issubclass(c, torch.nn.Module) for c in activation_checkpoint_classes)
        apply_activation_checkpointing(
            nxd_model,
            check_fn=lambda m: isinstance(m, activation_checkpoint_classes),
        )

    return nxd_model


def initialize_parallel_optimizer(nxd_config, optimizer_class, parameters, **defaults):
    optimizer_config = nxd_config["optimizer_config"]
    if optimizer_config["zero_one_enabled"]:
        optimizer = NeuronZero1Optimizer(
            parameters,
            optimizer_class,
            grad_clipping=optimizer_config["grad_clipping"],
            pin_layout=False,
            sharding_groups=parallel_state.get_data_parallel_group(as_list=True),
            grad_norm_groups=parallel_state.get_tensor_model_parallel_group(as_list=True),
            **defaults,
        )
    else:
        optimizer = optimizer_class(parameters, **defaults)

    return NxDOptimizer(optimizer, nxd_config)
