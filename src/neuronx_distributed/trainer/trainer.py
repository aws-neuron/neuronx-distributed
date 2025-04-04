import os
from pprint import pformat
from typing import Optional

import torch
import torch_xla.core.xla_model as xm
from packaging import version

from neuronx_distributed.optimizer import NeuronZero1Optimizer, NeuronEPZero1Optimizer
from neuronx_distributed.modules.lora import LoraConfig, LoraModel
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.pad import pad_model
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
    is_nxdt_pretrained_model,
    check_delay_tracing,
    get_delay_tracing,
)

logger = get_logger()


def neuronx_distributed_config(
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    expert_parallel_size=1,
    context_parallel_size=1,
    pipeline_config=None,
    optimizer_config=None,
    activation_checkpoint_config=None,
    pad_model=False,
    sequence_parallel=False,
    model_init_config=None,
    lora_config: Optional[LoraConfig] = None,
    mixed_precision_config=None,
    sequential_move_factor=11,
    lnc_size=1,
):
    if optimizer_config is None:
        optimizer_config = {"zero_one_enabled": False, "grad_clipping": True, "max_grad_norm": 1.0}
    else:
        assert isinstance(optimizer_config, dict), "optimizer_config must be a dict."
        if "zero_one_enabled" not in optimizer_config:
            if parallel_state.is_global_rank_zero():
                logger.warning("zero_one_enabled is not set, automatically set it to False.")
            optimizer_config.update({"zero_one_enabled": False})
        if "grad_clipping" not in optimizer_config:
            if parallel_state.is_global_rank_zero():
                logger.warning("grad_clipping is not set, automatically set it to True.")
            optimizer_config.update({"grad_clipping": True})
        if optimizer_config["grad_clipping"]:
            if "max_grad_norm" not in optimizer_config:
                if parallel_state.is_global_rank_zero():
                    logger.warning("max_grad_norm is not set, automatically set it to one.")
                optimizer_config.update({"max_grad_norm": 1.0})

    if mixed_precision_config is None:
        mixed_precision_config = {
            "use_master_weights": optimizer_config["zero_one_enabled"],
            "use_fp32_grad_acc": optimizer_config["zero_one_enabled"],
            "use_master_weights_in_ckpt": False,
        }
    else:
        assert isinstance(mixed_precision_config, dict), "mixed_precision_config must be a dict."
        if "use_master_weights" not in mixed_precision_config:
            if parallel_state.is_global_rank_zero():
                logger.warning("use_master_weights is not set, automatically set it.")
            mixed_precision_config.update(
                {
                    "use_master_weights": optimizer_config["zero_one_enabled"],
                }
            )
        if "use_fp32_grad_acc" not in mixed_precision_config:
            if parallel_state.is_global_rank_zero():
                logger.warning("use_fp32_grad_acc is not set, automatically set it.")
            mixed_precision_config.update(
                {
                    "use_fp32_grad_acc": optimizer_config["zero_one_enabled"],
                }
            )
        if "use_master_weights_in_ckpt" not in mixed_precision_config:
            if parallel_state.is_global_rank_zero():
                logger.warning("use_master_weights_in_ckpt is not set, automatically set it to False.")
            mixed_precision_config.update({"use_master_weights_in_ckpt": False})

    if model_init_config is None:
        model_init_config = {
            "sequential_move_factor": sequential_move_factor,  # randomly chosen number, work for 20B size model
            "meta_device_init": False,
            "param_init_fn": None,
        }
    else:
        assert isinstance(model_init_config, dict), "model_init_config must be a dict."
        if "sequential_move_factor" not in model_init_config:
            if parallel_state.is_global_rank_zero():
                logger.warning(
                    "sequential_move_factor is not set, automatically set it to 11, this number works for 20B size model."
                )
            model_init_config.update({"sequential_move_factor": sequential_move_factor})  # randomly chosen number, work for 20B size model
        if "meta_device_init" not in model_init_config:
            if parallel_state.is_global_rank_zero():
                logger.warning("meta_device_init is not set, automatically set it to False.")
            model_init_config.update({"meta_device_init": False})
        if model_init_config["meta_device_init"]:
            if "param_init_fn" not in model_init_config:
                raise ValueError("param_init_fn must be provided when meta_device_init is True")

    config = {
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "expert_parallel_size": expert_parallel_size,
        "context_parallel_size": context_parallel_size,
        "pipeline_config": pipeline_config,
        "optimizer_config": optimizer_config,
        "activation_checkpoint_config": activation_checkpoint_config,
        "pad_model": pad_model,
        "sequence_parallel": sequence_parallel,
        "model_init_config": model_init_config,
        "lora_config": lora_config,
        "mixed_precision_config": mixed_precision_config,
        "lnc_size": lnc_size,
    }

    if torch.distributed.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=config["tensor_parallel_size"],
            pipeline_model_parallel_size=config["pipeline_parallel_size"],
            expert_model_parallel_size=config["expert_parallel_size"],
            lnc_size=lnc_size,
            context_parallel_size=config["context_parallel_size"]
        )

    if torch.distributed.is_initialized() and parallel_state.is_global_rank_zero():
        logger.info("NxD config: \n{}".format(pformat(config)))
    return config


def initialize_parallel_model(nxd_config, model_fn, include_buffers: bool = False, *model_args, **model_kwargs):
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=nxd_config["tensor_parallel_size"],
            pipeline_model_parallel_size=nxd_config["pipeline_parallel_size"],
            expert_model_parallel_size=nxd_config["expert_parallel_size"],
            lnc_size=nxd_config["lnc_size"],
            context_parallel_size=nxd_config["context_parallel_size"],
        )

    # Phase 1: get the base model
    # TODO: deferred init by config
    meta_device_init = nxd_config["model_init_config"]["meta_device_init"]
    param_init_fn = nxd_config["model_init_config"].get("param_init_fn", None)

    if meta_device_init:
        # We force the init to use custom function, since there is a bug in accelerate's
        # API, as it doesn't handle weight sharing. TODO: Add a fix to accelerate
        with init_on_device(device=torch.device("meta"), include_buffers=include_buffers, force_custom_init_on_device=True):
            model = model_fn(*model_args, **model_kwargs)
    else:
        model = model_fn(*model_args, **model_kwargs)
    base_model = model

    # Phase 2: wrap with NxDPPModel
    pp_enabled = nxd_config["pipeline_parallel_size"] > 1
    if pp_enabled:
        nxd_config["pipeline_config"].update({"param_init_fn": param_init_fn})
        nxd_config["pipeline_config"].update({"use_model_wrapper": True})
        nxd_config["pipeline_config"]["_delay_tracing"] = check_delay_tracing(nxd_config)
        model = NxDPPModel(model, **nxd_config["pipeline_config"])

    # Phase 3: materialize model and move to device
    sequential_move_factor = nxd_config["model_init_config"]["sequential_move_factor"]

    device = xm.xla_device()
    model = get_model_sequential(model, device, sequential_move_factor, param_init_fn)

    # Phase 4: wrap the model with LoraModel
    lora_config = nxd_config.get("lora_config", None)
    if lora_config is not None:
        model = LoraModel(model, lora_config)

    # Phase 5: apply optimization what will change model structure
    # Currently we have:
    #   - pad model
    # TODO: pad model support pp
    if nxd_config["pad_model"]:
        assert is_hf_pretrained_model(base_model), "only support pad huggingface model now."
        model = pad_model(model, parallel_state.get_tensor_model_parallel_size(), model.config.num_attention_heads)

    # Phase 6: apply optimization what will not change model structure
    # Currently we have:
    #   - activation checkpoint
    nxd_model = NxDModel(model, nxd_config)
    if nxd_config["activation_checkpoint_config"] is not None:
        if nxd_config["activation_checkpoint_config"] == "full":
            if pp_enabled:
                activation_checkpoint_classes = [model.transformer_layer_cls]
            elif is_hf_pretrained_model(base_model):
                activation_checkpoint_classes = []
                from transformers.trainer_pt_utils import get_module_class_from_name

                for name in base_model._no_split_modules:
                    activation_checkpoint_classes.append(get_module_class_from_name(nxd_model, name))
            elif is_nxdt_pretrained_model(base_model):
                # NxDT transformer layer will always be this type
                from neuronx_distributed_training.models.megatron.transformer import ParallelTransformerLayer

                activation_checkpoint_classes = [ParallelTransformerLayer]
            else:
                raise RuntimeError(
                    '`activation_checkpoint_config` "full" is only supported for huggingface transformers or nxdt models.'
                )

        else:
            activation_checkpoint_classes = nxd_config["activation_checkpoint_config"]
            if not isinstance(activation_checkpoint_classes, (list, tuple)):
                activation_checkpoint_classes = [activation_checkpoint_classes]
        activation_checkpoint_classes_tuple = tuple(activation_checkpoint_classes)
        assert len(activation_checkpoint_classes_tuple) > 0
        assert all(issubclass(c, torch.nn.Module) for c in activation_checkpoint_classes_tuple)
        apply_activation_checkpointing(
            nxd_model,
            check_fn=lambda m: isinstance(m, activation_checkpoint_classes_tuple),
        )

    return nxd_model


def initialize_parallel_optimizer(nxd_config, optimizer_class, parameters, **defaults):
    optimizer = initialize_optimizer_from_class(nxd_config, optimizer_class, parameters, **defaults)
    nxd_optim = NxDOptimizer(optimizer, nxd_config)
    return nxd_optim


def initialize_optimizer_from_class(nxd_config, optimizer_class, parameters, model=None, **defaults):
    optimizer_config = nxd_config["optimizer_config"]
    mixed_precision_config = nxd_config["mixed_precision_config"]
    if optimizer_config["zero_one_enabled"]:
        ep_enabled = parallel_state.get_expert_model_parallel_size() > 1
        zero1_optimizer_cls = NeuronEPZero1Optimizer if ep_enabled else NeuronZero1Optimizer
        sharding_groups = parallel_state.get_zero1_sharding_groups() if parallel_state.get_context_model_parallel_size() > 1 else parallel_state.get_data_parallel_replica_groups()
        zero1_configs = {
            "grad_clipping": optimizer_config["grad_clipping"],
            "pin_layout": False,
            "sharding_groups": sharding_groups,
            "grad_norm_groups": parallel_state.get_tensor_model_parallel_replica_groups(),
        }
        if version.parse(torch.__version__) == version.parse("2.1"):
            zero1_configs.update({"coalesce_cc": True})
        if version.parse(torch.__version__) >= version.parse("2.2"):
            # P148368176: Bucket cap >140MB causes NaN at step 3 (known issue)
            _ALL_GATHER_REDUCE_SCATTER_BUCKET_CAP_MB = 130
            bucket_cap = int(
                os.getenv("ALL_GATHER_REDUCE_SCATTER_BUCKET_CAP_MB", _ALL_GATHER_REDUCE_SCATTER_BUCKET_CAP_MB)
            )
            reduce_scatter_bucket_cap = bucket_cap
            all_gather_bucket_cap = max(1, bucket_cap // parallel_state.get_data_parallel_size())
            zero1_configs.update({"bucket_cap_mb_all_gather": all_gather_bucket_cap})
            zero1_configs.update({"bucket_cap_mb_reduce_scatter": reduce_scatter_bucket_cap})
        if mixed_precision_config["use_master_weights"]:
            if "XLA_DOWNCAST_BF16" in os.environ and os.environ["XLA_DOWNCAST_BF16"] == "1":
                defaults.update({"optimizer_dtype": torch.double})
        if mixed_precision_config["use_fp32_grad_acc"]:
            zero1_configs.update({"use_grad_acc_hook": True, "higher_cc_precision": True})
        zero1_configs.update(
            {"save_master_weights": True if mixed_precision_config["use_master_weights_in_ckpt"] else False}
        )
        if get_delay_tracing(nxd_config):
            defaults["lazy_init"] = True
        logger.info("printing defaults here %s", defaults)
        logger.info("printing zero1 config here %s", zero1_configs)
        optimizer = zero1_optimizer_cls(
            parameters,
            optimizer_class,
            **zero1_configs,
            **defaults,
        )

    else:
        if mixed_precision_config["use_master_weights"]:
            raise RuntimeError("ZeRO-1 optimizer is not enabled, while `use_master_weights` is True.")
        if mixed_precision_config["use_fp32_grad_acc"] or mixed_precision_config["use_master_weights_in_ckpt"]:
            raise RuntimeError(
                "Non Zero-1 optimizer does not support `use_fp32_grad_acc` of `use_master_weights_in_ckpt`."
            )
        optimizer = optimizer_class(parameters, **defaults)
        if get_delay_tracing(nxd_config):
            from neuronx_distributed.trainer import hooks

            hooks.register_post_partition_hook(filter_to_local_parameter_group, [optimizer])
    return optimizer


"""
During the delayed tracing and partition flow, the optimizer gets initialized with all parameters, as the model has not
yet been moved to the device. When the model is moved to device on xla new tensors are created on device.
We filter the parameters to those in model.local_parameters() once the model is moved onto device.
"""


def filter_to_local_parameter_group(optimizer, model):
    parameters = optimizer.param_groups
    for param_group in parameters:
        filtered_param_list = []
        for meta_param in param_group["params"]:
            if meta_param in model.meta_device_parameter_map:
                meta_or_xla_param = model.meta_device_parameter_map[meta_param]
                if meta_or_xla_param.device.type == "xla":
                    filtered_param_list.append(meta_or_xla_param)
        param_group["params"] = filtered_param_list
    return