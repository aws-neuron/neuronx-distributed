# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Modifications Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
from functools import partial

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch  # noqa: E402
import torch_xla.core.xla_model as xm  # noqa: E402
from data_module import NeuronLightningDataModule  # noqa: E402
from modeling_dbrx_moe_nxd import (  # noqa: E402
    CoreAttention,  # noqa: E402
    DbrxBlock,  # noqa: E402
    DbrxForCausalLM,  # noqa: E402
    init_weights,  # noqa: E402
)  # noqa: E402
from module_dbrx import NeuronDbrxLTModule, NeuronDbrxPPLTModule  # noqa: E402
from lightning.pytorch.trainer.trainer import Trainer  # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary  # noqa: E402
from training_utils import (  # noqa: E402
    create_dbrx_pretraining_dataset,  # noqa: E402
    get_learning_rate_scheduler,  # noqa: E402
    get_mixed_precision_config,  # noqa: E402
)  # noqa: E402
from transformers import AdamW, DbrxConfig, set_seed  # noqa: E402
import neuronx_distributed as nxd  # noqa: E402
from neuronx_distributed.modules.moe.model import MoE  # noqa: E402
from neuronx_distributed.lightning import (  # noqa: E402
    NeuronTensorBoardLogger,  # noqa: E402
    NeuronTQDMProgressBar,  # noqa: E402
    NeuronXLAPrecisionPlugin,  # noqa: E402
    NeuronXLAStrategy,  # noqa: E402
)  # noqa: E402
from neuronx_distributed.modules.moe.loss_function import load_balancing_loss_func  # noqa: E402
from neuronx_distributed.parallel_layers import mappings  # noqa: E402
from neuronx_distributed.utils.adamw_fp32_optim_params import AdamW_FP32OptimParams  # noqa: E402
from neuronx_distributed.parallel_layers.layer_norm import LayerNorm  # noqa: E402

# For PT autocast.
torch.cuda.is_bf16_supported = lambda: True

# Workaround for NaNs seen with transformers version >= 4.21.0
# https://github.com/aws-neuron/aws-neuron-sdk/issues/593
import transformers.modeling_utils as modeling_utils  # noqa: E402

if os.environ.get("XLA_USE_BF16") or os.environ.get("XLA_DOWNCAST_BF16"):
    modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16


def train_dbrx(flags):
    xm.master_print(f"Namespace: {flags}")
    set_seed(flags.seed)

    model_config = _setup_model_config(flags)
    xm.master_print(model_config)
    nxd_config = _setup_nxd_config(flags, model_config)
    model = _initialize_neuron_model(model_config, nxd_config, flags)

    dm = NeuronLightningDataModule(
        create_dbrx_pretraining_dataset,
        flags.data_dir,
        flags.batch_size,
        data_args=(flags.seed,),
    )

    strategy = NeuronXLAStrategy(
        nxd_config=nxd_config,
        save_load_xser=flags.save_load_xser,
    )

    plugins = []

    plugins.append(NeuronXLAPrecisionPlugin())

    callbacks = []
    callbacks.append(NeuronTQDMProgressBar())
    callbacks.append(ModelSummary(max_depth=5))
    if flags.save_checkpoint:
        callbacks.append(
            ModelCheckpoint(
                save_top_k=flags.num_kept_checkpoint,
                monitor="global_step",
                mode="max",
                every_n_train_steps=flags.checkpoint_freq,
                dirpath=flags.checkpoint_dir,
            )
        )

    trainer = Trainer(
        strategy=strategy,
        max_steps=flags.steps_this_run,
        plugins=plugins,
        enable_checkpointing=flags.save_checkpoint,
        logger=NeuronTensorBoardLogger(save_dir=flags.log_dir),
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    if flags.resume_ckpt:
        ckpt_path = os.path.join(
            flags.checkpoint_dir,
            f"epoch={flags.load_epoch}-step={flags.load_step}.ckpt",
        )
        print(f"resume path is {ckpt_path}")
        trainer.fit(model=model, datamodule=dm, ckpt_path=ckpt_path)
    else:
        trainer.fit(model=model, datamodule=dm)


def _setup_model_config(flags):
    model_config = DbrxConfig.from_pretrained(flags.model_path)
    # capacity_factor = None corresponds to full capacity (no token dropping)
    model_config.capacity_factor = float(flags.capacity_factor) if flags.capacity_factor is not None else None
    model_config.sequence_parallel_enabled = flags.sequence_parallel_enabled > 0
    model_config.qkv_linear = flags.qkv_linear > 0
    model_config.selective_checkpoint_enabled = flags.selective_checkpoint_enabled > 0
    model_config.kv_shared_group_size = flags.kv_replicator
    model_config.i_tp_round_factor = flags.i_tp_round_factor
    model_config.max_position_embeddings = max(model_config.max_position_embeddings, args.seq_len)
    model_config.attn_config.use_flash_attention = flags.use_flash_attention
    if flags.num_layers != -1:
        model_config.num_hidden_layers = flags.num_layers
    if flags.vocab_size != -1:
        model_config.vocab_size = flags.vocab_size
    return model_config


def _setup_nxd_config(flags, model_config):
    model_init_config = (
        None
        if not flags.use_meta_device_init
        else {
            "meta_device_init": True,
            "param_init_fn": partial(init_weights, std=model_config.initializer_range),
            "sequential_move_factor": 11,
        }
    )

    mixed_precision_config = get_mixed_precision_config(args.use_gpu_compatible_precision > 0)

    pipeline_config = (
        None
        if flags.pipeline_parallel_size <= 1
        else {
            "transformer_layer_cls": DbrxBlock,
            "num_microbatches": flags.num_microbatches,
            "output_loss_value_spec": (True, False, False, False),
            "input_names": ["input_ids", "labels"],
            "auto_partition": True,
            "trace_file_path": None,
            "param_init_fn": None,
            "leaf_module_cls": [LayerNorm.__name__],
            "autowrap_modules": [mappings],
            "autowrap_functions": [load_balancing_loss_func],
            "use_zero1_optimizer": flags.use_zero_1,
            "use_optimizer_wrapper": True,
            "broadcast_and_average_loss": False,
        }
    )

    return nxd.neuronx_distributed_config(
        tensor_parallel_size=flags.tensor_parallel_size,
        pipeline_parallel_size=flags.pipeline_parallel_size,
        expert_parallel_size=flags.expert_parallel_size,
        pipeline_config=pipeline_config,
        optimizer_config={"zero_one_enabled": flags.use_zero_1, "grad_clipping": True, "max_grad_norm": 1.0},
        sequence_parallel=flags.sequence_parallel_enabled,
        activation_checkpoint_config=(CoreAttention, MoE) if flags.selective_checkpoint_enabled else "full",
        model_init_config=model_init_config,
        mixed_precision_config=mixed_precision_config,
    )


def _initialize_neuron_model(model_config, nxd_config, flags):
    optimizer_cls = AdamW_FP32OptimParams if flags.use_mix_precision else AdamW
    model_cls = NeuronDbrxPPLTModule if flags.pipeline_parallel_size > 1 else NeuronDbrxLTModule

    return model_cls(
        model_fn=DbrxForCausalLM,
        nxd_config=nxd_config,
        model_args=(model_config,),
        opt_cls=optimizer_cls,
        scheduler_cls=get_learning_rate_scheduler,
        opt_kwargs={
            "lr": flags.lr,
            "betas": (flags.beta1, flags.beta2),
            "weight_decay": flags.weight_decay,
            "eps": flags.eps,
        },
        scheduler_args=(flags.max_steps, flags.min_lr, flags.warmup_steps, flags.constant_steps),
        grad_accum_steps=flags.grad_accum_usteps,
        train_batch_size=flags.batch_size,
        logging_interval=flags.logging_interval,
        manual_opt=True,
    )


def _mp_fn(index, flags):
    torch.set_default_tensor_type("torch.FloatTensor")
    train_dbrx(flags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Model weight and config path.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Pre-tokenized dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Worker batch size.")
    parser.add_argument(
        "--max_steps",
        type=int,
        help="Maximum total accumulation-steps to run.",
    )
    parser.add_argument(
        "--steps_this_run",
        type=int,
        help="Exit early at <value> steps and not go to max_steps. -1 to mean no early exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12349,
        help="Random seed. Worker seed is this value + worker rank.",
    )
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="Number of warmup accumulation-steps for learning rate .",
    )
    parser.add_argument(
        "--constant_steps",
        type=int,
        default=0,
        help="Constant staps used in CosineAnnealing scheduler.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help="Minimum LR used in CosineAnnealing scheduler.",
    )
    parser.add_argument(
        "--grad_accum_usteps",
        type=int,
        default=1,
        help="Gradient accumulation micro-steps (an accumulation-step has <value> micro-steps.",
    )
    parser.add_argument("--load_step", type=int, default=0, help="step to load checkpoint from")
    parser.add_argument("--load_epoch", type=int, default=0, help="epoch to load checkpoint from")
    parser.add_argument("--log_dir", type=str, default=os.getcwd() + "/dbrx-logs", help="Directory for log files")
    parser.add_argument("--save_checkpoint", action="store_true", help="Save checkpoints")
    parser.add_argument(
        "--num_kept_checkpoint",
        type=int,
        default=10000,
        help="number of checkpoints kept, old checkpoint will get deleted",
    )
    parser.add_argument("--checkpoint_freq", type=int, default=100000, help="save checkpoint freq")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume_ckpt", action="store_true", help="Resume from checkpoint at resume_step.")
    parser.add_argument("--save_load_xser", action="store_true", help="save/load with xla serialization")

    parser.add_argument("--tensor_parallel_size", default=2, type=int, help="Tensor parallel size")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="PP size")
    parser.add_argument("--expert_parallel_size", type=int, default=1, help="EP size")
    parser.add_argument("--num_microbatches", type=int, default=8, help="num_microbatches used for PP")
    parser.add_argument("--seq_len", default=2048, type=int, help="Sequence length")
    parser.add_argument("--use_mix_precision", action="store_true", help="Use mix precision.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta1 parameter for Adam optimizer")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta2 parameter for Adam optimizer")
    parser.add_argument("--eps", default=1e-8, type=float, help="eps parameter for Adam optimizer")
    parser.add_argument("--use_zero_1", action="store_true", help="Use ZeRO-1.")
    parser.add_argument("--logging_interval", type=int, default=1, help="number of warmup_steps")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=-1,
        help="Override number of layers for this DBRX model",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=-1,
        help="Override vocab size for this DBRX model",
    )
    parser.add_argument(
        "--sequence_parallel_enabled",
        default=False,
        action="store_true",
        help="Enable sequence parallel",
    )
    parser.add_argument(
        "--selective_checkpoint_enabled",
        default=False,
        action="store_true",
        help="Enable selective checkpoint",
    )
    parser.add_argument(
        "--qkv_linear",
        default=0,
        type=int,
        help="Use QKV Linear module",
    )
    parser.add_argument(
        "--kv_replicator",
        default=1,
        type=int,
        help="KV replication number",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use neuron kernel",
    )
    parser.add_argument(
        "--i_tp_round_factor",
        default=1,
        type=int,
        help="The number by which I_TP must be divisible",
    )
    parser.add_argument(
        "--capacity_factor",
        default=None,
        help="MoE capacity factor",
    )
    parser.add_argument(
        "--use_meta_device_init", default=False, action="store_true", help="Enable meta device initialization"
    )
    parser.add_argument(
        "--use_gpu_compatible_precision",
        default=1,
        type=int,
        help="Use gpu compatible precision",
    )

    args = parser.parse_args(sys.argv[1:])

    if args.steps_this_run < 0:
        args.steps_this_run = args.max_steps

    os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0" if args.use_gpu_compatible_precision > 0 else "1"
    if args.use_mix_precision:
        print("Using BF16 Downcast!")
        os.environ["XLA_DOWNCAST_BF16"] = "1"
    else:
        print("Using Pure BF16!")
        os.environ["XLA_USE_BF16"] = "1"

    # WORLD_SIZE is set by torchrun

    _mp_fn(0, args)
