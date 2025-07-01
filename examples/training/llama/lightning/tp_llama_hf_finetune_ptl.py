# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
import json
import os
import sys

import numpy as np
import torch
import torch_xla.core.xla_model as xm
from data_module import NeuronLightningDataModule
from modeling_llama_nxd import CoreAttention, LlamaForCausalLM
from module_llama import NeuronLlamaLTModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.text.rouge import ROUGEScore
from training_utils import create_instruction_based_dataset, get_mixed_precision_config
from transformers import AdamW, LlamaConfig, LlamaTokenizer, set_seed, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
import neuronx_distributed as nxd
from neuronx_distributed.lightning import (
    NeuronTensorBoardLogger,
    NeuronTQDMProgressBar,
    NeuronXLAPrecisionPlugin,
    NeuronXLAStrategy,
)
from neuronx_distributed.modules.lora import LoraConfig
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.utils.adamw_fp32_optim_params import AdamW_FP32OptimParams

# Workaround for NaNs seen with transformers version >= 4.21.0
# https://github.com/aws-neuron/aws-neuron-sdk/issues/593
import transformers.modeling_utils as modeling_utils

# For PT autocast.
torch.cuda.is_bf16_supported = lambda: True

if os.environ.get("XLA_USE_BF16") or os.environ.get("XLA_DOWNCAST_BF16"):
    modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16


def train_llama(flags):
    print(f"Namespace: {flags}")
    set_seed(flags.seed)

    lora_config = None
    if flags.enable_lora:
        target_modules = ["q_proj", "v_proj", "k_proj"] if flags.qkv_linear == 0 else ["qkv_proj"]
        lora_config = LoraConfig(
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            lora_verbose=True,
            target_modules=target_modules,
        )

    mixed_precision_config = get_mixed_precision_config(flags.use_gpu_compatible_precision > 0)

    nxd_config = nxd.neuronx_distributed_config(
        tensor_parallel_size=flags.tensor_parallel_size,
        optimizer_config={"zero_one_enabled": flags.use_zero_1, "grad_clipping": True, "max_grad_norm": 1.0},
        sequence_parallel=flags.sequence_parallel_enabled,
        activation_checkpoint_config=CoreAttention if flags.selective_checkpoint_enabled else "full",
        lora_config=lora_config,
        mixed_precision_config=mixed_precision_config,
    )

    model_config = LlamaConfig.from_pretrained(flags.model_path)
    model_config.pretrained_ckpt = flags.pretrained_ckpt
    model_config.use_cache = False
    model_config.fuse_qkv = flags.fuse_qkv > 0
    model_config.transpose_nki_inputs = flags.transpose_nki_inputs > 0
    model_config.kv_shared_group_size = flags.kv_replicator
    model_config.qkv_linear = flags.qkv_linear
    model_config.max_position_embeddings = max(model_config.max_position_embeddings, flags.seq_len)
    model_config.use_flash_attention = flags.use_flash_attention > 0
    if flags.num_layers > 0:
        model_config.num_hidden_layers = flags.num_layers
    if flags.sequence_parallel_enabled:
        model_config.sequence_parallel_enabled = True
    if flags.selective_checkpoint_enabled:
        model_config.selective_checkpoint_enabled = True
    model_config.head_dim = int(model_config.hidden_size / model_config.num_attention_heads)
    xm.master_print(model_config)

    if flags.use_mix_precision:
        optimizer_cls = AdamW_FP32OptimParams
    else:
        optimizer_cls = AdamW

    def configure_scheduler(optimizer, warmup_steps, max_steps):  # PTLTODO: check loading scheduler state dict here
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            last_epoch=-1,
        )

    tokenizer = AutoTokenizer.from_pretrained(flags.model_name, token=flags.hf_token)
    model = NeuronLlamaLTModule(
        tokenizer=tokenizer,
        model_fn=LlamaForCausalLM,
        nxd_config=nxd_config,
        model_args=(model_config,),
        opt_cls=optimizer_cls,
        scheduler_cls=configure_scheduler,
        opt_kwargs={
            "lr": flags.lr,
        },
        scheduler_args=(flags.warmup_steps, flags.max_steps),
        grad_accum_steps=flags.grad_accum_usteps,
        manual_opt=True,
    )

    dm = NeuronLightningDataModule(
        create_instruction_based_dataset,
        flags.data_dir,
        flags.batch_size,
        data_args=(flags.seed,),
        data_kwargs={"tokenizer": tokenizer, "task": flags.task},
    )

    strategy = NeuronXLAStrategy(
        nxd_config=nxd_config,
        save_load_xser=flags.save_load_xser,
    )

    plugins = []

    plugins.append(NeuronXLAPrecisionPlugin())

    callbacks = []
    callbacks.append(NeuronTQDMProgressBar())
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
        max_epochs=flags.num_train_epochs,
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

    xm.master_print("Training finished!")

    if flags.enable_lora:
        nxd.save_checkpoint(checkpoint_dir_str="lora_adapter", tag="lora", model=model)

    if flags.do_eval and not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
        evaluate(model, tokenizer, dm.test_dataloader(), args.golden_rouge_score_path)
        xm.master_print("Evaluation Finished!")


def evaluate(model, tokenizer, test_loader, golden_rouge_score_path):
    # Need to run
    # python3 -c "import nltk; nltk.download('punkt')"
    # before this run
    rouge = ROUGEScore(compute_on_cpu=True, sync_on_compute=False)
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            input_ids = batch["input_ids"]
            input_ids = input_ids.to("xla")
            labels = batch["labels"]
            input_length = input_ids.shape[-1]
            output_sequences = model.generate(
                input_ids=input_ids,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                top_p=0.9,
                num_beams=4,
            )
            prompt = tokenizer.decode(input_ids[0], clean_up_tokenization_spaces=True)
            predicted_text = tokenizer.decode(
                output_sequences[0][input_length:].cpu(), clean_up_tokenization_spaces=True
            )
            label_text = tokenizer.decode(labels[0].cpu(), clean_up_tokenization_spaces=True)
            rouge.update(predicted_text, label_text)
            if parallel_state.get_tensor_model_parallel_rank() == 0:
                print("=== PROMPT ===")
                print(prompt)
                print("=== GENERATED SEQUENCE ===")
                print(predicted_text)
                print("=== LABEL ===")
                print(f"{label_text}\n")

    rouge_scores = rouge.compute()
    aggregated_rouge_scores = {}
    for key, value in rouge_scores.items():
        aggregated_val = xm.mesh_reduce(key, value.item(), np.mean)
        aggregated_rouge_scores[key] = aggregated_val
    xm.master_print("=== Evaluation Rouge Scores ===")
    xm.master_print(aggregated_rouge_scores)
    if golden_rouge_score_path is not None and xm.is_master_ordinal(local=False):
        tol = 1.0
        xm.master_print(
            f"Comparing eval rouge scores to golden file {golden_rouge_score_path} with abs tolerance {tol}."
        )
        with open(golden_rouge_score_path, "r") as f:
            golden_rouge_scores = json.load(f)
        for rouge_key in golden_rouge_scores.keys():
            assert rouge_key in aggregated_rouge_scores
            assert abs(golden_rouge_scores[rouge_key] - aggregated_rouge_scores[rouge_key]) <= tol
        xm.master_print("Evaluation rouge scores matched goldens!")


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
        "--model_name",
        type=str,
        default='meta-llama/Meta-Llama-3-8B',
        help="Base model name.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Huggingface token to access base model and tokenizer.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Dataset name or directory.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The downstream task type that the model is fine-tuned with.",
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
        "--num_train_epochs",
        type=int,
        help="Maximum numer of epochs to run.",
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
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=os.getcwd() + "/llama7B-pretrained",
        help="Directory for pretrained weights",
    )
    parser.add_argument("--log_dir", type=str, default=os.getcwd() + "/llama7B-logs", help="Directory for log files")
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
    parser.add_argument("--fuse_qkv", type=int, default=1, help="Whether to enable fused qkv")
    parser.add_argument(
        "--transpose_nki_inputs", 
        type=int, 
        default=1, 
        help="Whether to transpose inputs to nki kernel for better perf when using FlashAttention"
    )
    parser.add_argument(
        "--use_flash_attention", 
        type=int, 
        default=0, 
        help="Whether to use NKI FlashAttention"
    )
    parser.add_argument("--do_eval", action="store_true", help="Do evaluation after fine-tuning.")
    parser.add_argument(
        "--golden_rouge_score_path", default=None, type=str, help="Path to golden eval rouge score file."
    )
    parser.add_argument(
        "--enable_lora",
        default=False,
        action="store_true",
        help="Enable LoRA for model fine-tuning.",
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
        os.environ["XLA_DOWNCAST_BF16"] = "1"
    else:
        os.environ["XLA_USE_BF16"] = "1"

    # WORLD_SIZE is set by torchrun

    _mp_fn(0, args)
