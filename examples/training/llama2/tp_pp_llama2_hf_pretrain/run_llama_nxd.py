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
import math
import random
import time
import queue

import numpy as np
import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

import neuronx_distributed as nxd
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_data_parallel_size,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    initialize_model_parallel,
)
from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
from neuronx_distributed.parallel_layers.grads import clip_grad_norm
from neuronx_distributed.pipeline import NxDPPModel
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.optimizer import NeuronZero1Optimizer
from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.parallel_layers.checkpointing import save, load
from neuronx_distributed.parallel_layers.utils import is_pjrt_device
from neuronx_distributed.utils import model_utils
from transformers import LlamaConfig
import transformers.modeling_utils as modeling_utils
# For delayed parameter inititalization
# Check https://pytorch.org/torchdistx/latest/deferred_init.html
try:
    from torchdistx import deferred_init
except ImportError:
    deferred_init = None

from modeling_llama_nxd import (
    CoreAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaRMSNorm,
    init_weights,
)

from neuronx_distributed.utils.adamw_fp32_optim_params import AdamW_FP32OptimParams
from activation_checkpoint import apply_checkpoint
from training_utils import (
    get_param_groups_by_weight_decay, 
    get_learning_rate_scheduler, 
    create_llama_pretraining_dataset, 
    create_partition,
    get_sin_cos_matrix
)
from logger import Logger


def allreduce_sequence_parallel_gradients(optimizer):
    """ All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
        Modified from megatron-lm:
        https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
    """
    from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
    grads = []
    for param_group in optimizer.__getstate__()['param_groups']:
        for group, params in param_group.items():
            if group == 'params':
                for p in params:
                    if isinstance(p, torch.Tensor) and p.grad is not None:
                        sequence_parallel_param = getattr(p, 'sequence_parallel_enabled', False)
                        if sequence_parallel_param:
                            grads.append(p.grad.data)
    xm.master_print("# sequence parallel parameters = ", len(grads))
    for grad in grads:
        # sum v.s. average: sum
        reduce_from_tensor_model_parallel_region(grad)

class Throughput:
    def __init__(
        self, batch_size, world_size, grad_accum_usteps, moving_avg_window_size=10, logging_interval=1
    ):
        self.seqs_per_iteration = batch_size * world_size * grad_accum_usteps*logging_interval
        self.moving_avg_window_size = math.ceil(moving_avg_window_size/logging_interval)
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()

    def get_throughput(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        throughput = window_size * self.seqs_per_iteration / self.window_time
        return throughput

def train_llama(args):

    if dist.get_rank() == 0:
        print(f"args {args}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set up Llama config
    config = LlamaConfig.from_pretrained(args.training_config)
    config.use_cache = False
    config.return_dict = False
    config.sequence_parallel_enabled = args.use_sequence_parallel > 0
    config.qkv_linear = args.qkv_linear > 0
    config.selective_checkpoint_enabled = args.use_selective_checkpoint > 0
    config.kv_shared_group_size = args.kv_replicator
    config.max_position_embeddings = max(config.max_position_embeddings, args.seq_len)
    if args.num_layer != -1:
        config.num_hidden_layers = args.num_layer
    if args.hidden_size != -1:
        config.hidden_size = args.hidden_size
    
    pipeline_cuts = create_partition(config.num_hidden_layers, args.pipeline_parallel_size)
    if torch.distributed.get_rank() == 0:
        print(f"pipeline_cuts {pipeline_cuts}")

    # Create model with different options
    # Either deferred_init or meta device initialization will be required to avoid host OOM for 70B model
    if args.use_meta_device_init > 0:
        model_init_config = {
            "meta_device_init": True,
            "param_init_fn": init_weights,
        }
    else:
        model_init_config = None
    nxd_config = nxd.neuronx_distributed_config(
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        pipeline_config={
            "transformer_layer_cls": LlamaDecoderLayer,
            "num_microbatches": args.num_microbatches,
            "output_loss_value_spec": (True, False),
            "input_names": ["input_ids", "attention_mask", "labels"],
            "pipeline_cuts": pipeline_cuts,
            "trace_file_path": args.trace_file_path,
            "param_init_fn": None,
            "leaf_module_cls": [LlamaRMSNorm.__name__],
            "autowrap_modules": [mappings],
            "use_zero1_optimizer": args.use_zero1_optimizer > 0,
            "use_optimizer_wrapper": True,
        },
        optimizer_config={
            "zero_one_enabled": args.use_zero1_optimizer > 0,
            "grad_clipping": True,
            "max_grad_norm": 1.0,
        },
        sequence_parallel=args.use_sequence_parallel,
        activation_checkpoint_config=CoreAttention if args.use_selective_checkpoint > 0 else "full",
        model_init_config=model_init_config,
    )
    
    def get_model(config):
        if args.use_deferred_init > 0 and deferred_init is not None:
            model = deferred_init.deferred_init(LlamaForCausalLM, config)
        else:
            model = LlamaForCausalLM(config)
        # Here we make sure we use the same sine and cosine matrices for all layers.
        # Making use of same tensors would make the CSE algorithm eliminate the lookup call
        # from layers, keeping only lookup from first layer.
        with torch.no_grad():
            cos, sin = get_sin_cos_matrix(config)
            for layer in model.model.layers:
                layer.self_attn.rotary_emb.cos_cached = cos
                layer.self_attn.rotary_emb.sin_cached = sin
        num_params = sum([np.prod(p.size()) for p in model.parameters()])
        if dist.get_rank() == 0:
            print(f"# total parameters: {num_params}")
            print(f"model config {config}")
        return model

   
    # Create NxD model
    model = nxd.initialize_parallel_model(nxd_config, get_model, config)

    param_groups = get_param_groups_by_weight_decay(model)

    opt_cls = AdamW_FP32OptimParams if args.use_fp32_optimizer > 0 else torch.optim.AdamW
    optimizer = nxd.initialize_parallel_optimizer(
        nxd_config, opt_cls, param_groups, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay
    )

    dp_rank = get_data_parallel_rank()
    dp_size = get_data_parallel_size()
    tp_rank = get_tensor_model_parallel_rank()
    pp_rank = get_pipeline_model_parallel_rank()

    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    train_dataloader = create_llama_pretraining_dataset(
        args.training_dir, args.train_batch_size, dp_size, dp_rank, args.seed
    )
    
    print("Creating sample dataloader finised")

    # Only print/logging on the last PP rank of the first PP group
    # Since loss is only in the last PP rank
    should_print = (
        pp_rank == args.pipeline_parallel_size - 1 and dp_rank == 0 and tp_rank == 0
    )
    
    logger = Logger(args, should_print)

    total_steps = 0
    resume_batch_idx = None
    if args.loading_step != -1:
        user_content = nxd.load_checkpoint(
            args.checkpoint_dir,
            tag=f"step_{args.loading_step}",
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
        )
        if user_content is not None:
            resume_batch_idx = user_content["batch_idx"]
            total_steps = user_content["total_steps"]

    epoch = 0
    throughput = Throughput(
        args.train_batch_size, dp_size, 1, 10, args.logging_interval
    )
    while True:
        if torch.distributed.get_rank() == 0:
            print(f"Epoch {epoch}")
        for batch_idx, batch in enumerate(train_dataloader):
            if resume_batch_idx is not None and batch_idx <= resume_batch_idx:
                if torch.distributed.get_rank() == 0:
                    print(f"skipping batch {batch_idx}")
                continue
            start = time.time()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            # Enavle auto-mix-precision if needed
            with torch.autocast(enabled=args.use_amp > 0, dtype=torch.bfloat16, device_type="cuda"):
                # Calling model.run_train instead of model forward to use the PP runtime
                loss = model.run_train(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            total_steps += 1
            optimizer.step()
            global_norm = optimizer.grad_norm # Global norm before clipping
            optimizer.zero_grad()
            lr_scheduler.step()
            if should_print:
                if total_steps % args.logging_interval == 0:
                    xm.add_step_closure(logger.log, (total_steps, loss.detach(), global_norm, lr_scheduler.get_lr()[0], input_ids.detach(), throughput, start))
            xm.mark_step()
            # Saving checkpoints
            if (args.checkpoint_freq > 0) and (total_steps % args.checkpoint_freq == 0):
                nxd.save_checkpoint(
                    args.checkpoint_dir,
                    tag=f"step_{total_steps}",
                    model=model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    user_content={"total_steps": total_steps, "batch_idx": batch_idx, "cli_args": args.__dict__},
                    use_xser=True,
                    num_kept_ckpts=args.num_kept_checkpoint,
                )
            if total_steps >= args.max_steps:
                break

        if total_steps >= args.max_steps:
            break
        epoch += 1

    print("Training finished successfully")

def _mp_fn(index, args):
    train_llama(args)
    xm.rendezvous("_mp_fn finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_microbatches", type=int, default=8, help="num_microbatches")
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="tensor_parallel_size")
    parser.add_argument("--num_layer", type=int, default=-1, help="override model number of layers")
    parser.add_argument("--hidden_size", type=int, default=-1, help="override model model hidden size")
    parser.add_argument("--train_batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1, help="PP size")
    parser.add_argument("--kv_replicator", type=int, default=1, help="KV replication size")
    parser.add_argument("--seq_len", type=int, default=4096, help="PP size")
    parser.add_argument("--training_dir", type=str, default=None)
    parser.add_argument("--training_config", type=str, default=None)
    parser.add_argument("--trace_file_path", type=str, default=None)
    parser.add_argument("--tb_dir", type=str, default="")
    parser.add_argument("--max_steps", type=int, default=100, help="max steps")
    parser.add_argument("--checkpoint_freq", type=int, default=100000, help="save checkpoint freq")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--loading_step", type=int, default=-1, help="load from step, -1 means no load")
    parser.add_argument("--num_kept_checkpoint", type=int, default=-1, help="number of checkpoints kept, old checkpoint will get deleted")
    parser.add_argument("--save_load_xser", type=int, default=1, help="save/load with xla serialization")
    parser.add_argument("--pretrained_weight_dir", type=str, default=None, help="Load dir of pretrained weight")

    # optimization
    opt_grp = parser.add_argument_group(title="optimization", description="arguments for optimization")
    opt_grp.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    opt_grp.add_argument("--beta1", default=0.9, type=float, help="beta1 parameter for Adam optimizer")
    opt_grp.add_argument("--beta2", default=0.95, type=float, help="beta2 parameter for Adam optimizer")
    opt_grp.add_argument("--use_fp32_optimizer", default=0, type=int, help="use_fp32_optimizer")
    opt_grp.add_argument("--use_zero1_optimizer", default=0, type=int, help="use_zero1_optimizer")
    opt_grp.add_argument("--seed", default=1234, type=int, help="random seed")
    opt_grp.add_argument("--use_amp", default=0, type=int, help="use amp data")
    opt_grp.add_argument("--use_deferred_init", default=0, type=int, help="use torchdistx deferred initialization")
    opt_grp.add_argument("--use_meta_device_init", default=0, type=int, help="use meta device initialization")
    opt_grp.add_argument("--use_selective_checkpoint", default=0, type=int, help="enable selective activation checkpointing")
    opt_grp.add_argument("--use_sequence_parallel", default=1, type=int, help="enable sequence parallelism")
    opt_grp.add_argument("--qkv_linear", default=0, type=int, help="Use QKV Linear module")

    # learning rate
    lr_grp = parser.add_argument_group(title="lr", description="arguments for learning rate schedule")
    lr_grp.add_argument("--lr", type=float, default=None, help="Initial learning rate.")
    lr_grp.add_argument("--warmup_steps",type=int,default=None,help="number of warmup_steps")
    lr_grp.add_argument("--constant_steps",type=int,default=None,help="number of warmup_steps")
    lr_grp.add_argument("--min_lr",type=float,default=None,help="Minumum value for learning rate. The scheduler" "clip values below this threshold.")
    lr_grp.add_argument("--logging_interval",type=int,default=1,help="number of warmup_steps")

    args, _ = parser.parse_known_args()
    # Workaround for NaNs seen with transformers version >= 4.21.0
    # https://github.com/aws-neuron/aws-neuron-sdk/issues/593   
    if os.environ.get("XLA_USE_BF16") or os.environ.get("XLA_DOWNCAST_BF16") or args.use_amp > 0:
        modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16

    if os.environ.get("WORLD_SIZE"):
        dist.init_process_group("xla")
        _mp_fn(0, args)
    else:
        xmp.spawn(_mp_fn, args=(args,))