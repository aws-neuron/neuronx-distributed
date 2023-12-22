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
import inspect
import json
import os
import queue
import sys
import time

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from modeling_llama_nxd import CoreAttention, LlamaForCausalLM
from neuronx_distributed.utils.adamw_fp32_optim_params import AdamW_FP32OptimParams
from transformers import AdamW, LlamaConfig, set_seed
from transformers.optimization import get_linear_schedule_with_warmup

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import neuronx_distributed as nxd
from neuronx_distributed.parallel_layers import checkpointing, parallel_state
from neuronx_distributed.parallel_layers.utils import is_pjrt_device

from neuronx_distributed.lightning import (
    NeuronXLAStrategy,
    NeuronXLAPrecisionPlugin,
    NeuronTensorBoardLogger,
    NeuronTQDMProgressBar,
)

from module_llama import NeuronLlamaLTModule
from data_module import NeuronLightningDataModule

from training_utils import (
    create_llama_pretraining_dataset,
)
# For PT autocast.
torch.cuda.is_bf16_supported = lambda: True

# Workaround for NaNs seen with transformers version >= 4.21.0
# https://github.com/aws-neuron/aws-neuron-sdk/issues/593
import transformers.modeling_utils as modeling_utils

if os.environ.get("XLA_USE_BF16") or os.environ.get("XLA_DOWNCAST_BF16"):
    modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16


def train_llama(flags):
    print(f"Namespace: {flags}")
    set_seed(flags.seed)


    nxd_config = nxd.neuronx_distributed_config(
        tensor_parallel_size=flags.tensor_parallel_size,
        optimizer_config={"zero_one_enabled": flags.use_zero_1, "grad_clipping": True, "max_grad_norm": 1.0},
        sequence_parallel=flags.sequence_parallel_enabled,
        activation_checkpoint_config=CoreAttention if flags.selective_checkpoint_enabled else "full",
    )

    model_config = LlamaConfig.from_pretrained(flags.model_path)
    model_config.use_cache = False
    model_config.kv_shared_group_size = args.kv_replicator
    model_config.qkv_linear = args.qkv_linear
    model_config.max_position_embeddings = max(model_config.max_position_embeddings, flags.seq_len)
    if flags.num_layers > 0:
        model_config.num_hidden_layers = flags.num_layers
    if flags.sequence_parallel_enabled:
        model_config.sequence_parallel_enabled = True
    if flags.selective_checkpoint_enabled:
        model_config.selective_checkpoint_enabled = True
    xm.master_print(model_config)

    if flags.use_mix_precision:
        optimizer_cls = AdamW_FP32OptimParams
    else:
        optimizer_cls = AdamW

    def configure_scheduler(optimizer, warmup_steps, max_steps): # PTLTODO: check loading scheduler state dict here
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            last_epoch=-1,
        )


    model = NeuronLlamaLTModule(
        model_fn = LlamaForCausalLM,
        nxd_config = nxd_config,
        model_args = (model_config,),
        opt_cls = optimizer_cls,
        scheduler_cls = configure_scheduler,
        opt_kwargs = {
            "lr": flags.lr,
        },
        scheduler_args = (flags.warmup_steps, flags.max_steps),
        grad_accum_steps = flags.grad_accum_usteps,
        manual_opt = True, 
    )

    dm = NeuronLightningDataModule(
        create_llama_pretraining_dataset,
        flags.data_dir,
        flags.batch_size,
        data_args = (flags.seed,),
    )

    strategy = NeuronXLAStrategy(
        nxd_config = nxd_config,
        save_load_xser = flags.save_load_xser,
    )

    plugins = []

    plugins.append(NeuronXLAPrecisionPlugin())

    callbacks = []
    callbacks.append(NeuronTQDMProgressBar())
    if flags.save_checkpoint:
        callbacks.append(
            ModelCheckpoint(
                save_top_k = flags.num_kept_checkpoint,
                monitor="global_step",
                mode="max",
                every_n_train_steps = flags.checkpoint_freq,
                dirpath = flags.checkpoint_dir,
            )
        )
    
    trainer = Trainer(
        strategy = strategy, 
        max_steps = flags.steps_this_run,
        plugins = plugins,
        enable_checkpointing = flags.save_checkpoint,
        logger = NeuronTensorBoardLogger(save_dir=flags.log_dir),
        log_every_n_steps = 1,
        callbacks = callbacks,
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

    print(f"Training finished!")


def _mp_fn(index, flags):
    torch.set_default_tensor_type("torch.FloatTensor")
    train_llama(flags)


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
    parser.add_argument("--batch_size", type=int, default=8, help="Worker batch size.")
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
        "--grad_accum_usteps",
        type=int,
        default=64,
        help="Gradient accumulation micro-steps (an accumulation-step has <value> micro-steps.",
    )
    parser.add_argument("--load_step", type=int, default=0, help="step to load checkpoint from")
    parser.add_argument("--load_epoch", type=int, default=0, help="epoch to load checkpoint from")
    parser.add_argument("--log_dir", type=str, default=os.getcwd()+"/llama7B-logs", help="Directory for log files")
    parser.add_argument("--save_checkpoint", action="store_true", help="Save checkpoints")
    parser.add_argument("--num_kept_checkpoint", type=int, default=10000, help="number of checkpoints kept, old checkpoint will get deleted")
    parser.add_argument("--checkpoint_freq", type=int, default=100000, help="save checkpoint freq")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume_ckpt", action="store_true", help="Resume from checkpoint at resume_step.")
    parser.add_argument("--save_load_xser", action="store_true", help="save/load with xla serialization")

    parser.add_argument("--tensor_parallel_size", default=2, type=int, help="Tensor parallel size")
    parser.add_argument("--seq_len", default=2048, type=int, help="Sequence length")
    parser.add_argument("--use_mix_precision", action="store_true", help="Use mix precision.")
    parser.add_argument("--use_zero_1", action="store_true", help="Use ZeRO-1.")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=-1,
        help="Override number of layers for this LLaMA model",
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


    args = parser.parse_args(sys.argv[1:])

    if args.steps_this_run < 0:
        args.steps_this_run = args.max_steps

    os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "1"
    if args.use_mix_precision:
        os.environ["XLA_DOWNCAST_BF16"] = "1"
    else:
        os.environ["XLA_USE_BF16"] = "1"

    # WORLD_SIZE is set by torchrun
  
    _mp_fn(0, args)