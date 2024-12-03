import os

import torch
import torch.distributed as dist
import time

from transformers import AdamW

import neuronx_distributed as nxd
import neuronx_distributed.parallel_layers.parallel_state as ps
import torch_xla.distributed.parallel_loader as pl
from neuronx_distributed.utils.adamw_fp32_optim_params import AdamW_FP32OptimParams
from transformers.optimization import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from mamba2 import Mamba2ForCausalLM
from training_utils import create_llama_pretraining_dataset
from predefined_configs import get_config

import transformers.modeling_utils as modeling_utils

# For PT autocast.
torch.cuda.is_bf16_supported = lambda: True

# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
WORLD_RANK = int(os.environ['RANK'])


def get_mixed_precision_config(args):
    if args.use_zero_1:
        return {
            "use_master_weights": True,
            "use_fp32_grad_acc": True,
            "use_master_weights_in_ckpt": False,
        }
    else:
        return {}


def run(args, backend):
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = xm.xla_device()

    config = get_config(args.model, args.vocab_size, rmsnorm_within_groups=(not args.rmsnorm_across_groups),
                        n_groups=args.tp)

    def get_model():
        model = Mamba2ForCausalLM(config).to(dtype=args.dtype)
        model.train()

        # check that weight tying worked
        if model.config.tie_word_embeddings:
            assert model.backbone.embeddings.weight is model.lm_head.weight
        return model

    nxd_config = nxd.neuronx_distributed_config(
        tensor_parallel_size=args.tp,
        optimizer_config={"zero_one_enabled": args.use_zero_1, "grad_clipping": True, "max_grad_norm": 1.0},
        sequence_parallel=False,
        model_init_config=None,
        mixed_precision_config=get_mixed_precision_config(args),
    )
    model = nxd.initialize_parallel_model(nxd_config, get_model)
    world_size = ps.get_data_parallel_size()
    if xm.is_master_ordinal():
        print('NEURON_CC_FLAGS: ', os.environ.get('NEURON_CC_FLAGS', None))
        print('XLA_IR_DEBUG: ', os.environ.get('XLA_IR_DEBUG', None))
        print('XLA_HLO_DEBUG: ', os.environ.get('XLA_HLO_DEBUG', None))
        print('TP groups:', ps.get_tensor_model_parallel_group(as_list=True))
        print('DP groups:', ps.get_data_parallel_group(as_list=True))
        print('Config: ', config)
        param_size, dtype = 0, None
        for param in set(model.parameters()):
            param_size += param.nelement()
            dtype = param.dtype
        print(f"Model size: {param_size / 10 ** 6:.1f}M parameters/core")
        print(f"Param dtype: {dtype}")

    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm", "norm", "A", "D"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if args.no_optimizer_fp32:
        optimizer_cls = AdamW
    else:
        optimizer_cls = AdamW_FP32OptimParams

    # Creating NxD Optimizer
    optimizer = nxd.initialize_parallel_optimizer(
        nxd_config,
        optimizer_cls,
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    optimizer.zero_grad()
    if args.use_zero_1:
        optimizer.optimizer.init_zero()

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        last_epoch=-1,
    )

    train_dataloader, _ = create_llama_pretraining_dataset(
        args.data_dir,
        args.batch,
        ps.get_data_parallel_size(),
        ps.get_data_parallel_rank(),
        args.seed,
    )
    # We wrap the dataloader with MpDeviceLoader. This dataloader should take
    # care of copying the tensors to device and also inserting the mark_step at
    # iteration end.
    train_device_loader = pl.MpDeviceLoader(train_dataloader, device)

    running_loss = torch.zeros(1).to(device)
    training_step, global_step = 0, 0
    t0 = time.time()
    xm.mark_step()
    for i, data in enumerate(train_device_loader):
        training_step += 1
        input_ids = data["input_ids"]
        labels = data["labels"]

        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss / args.grad_accum_usteps
        loss.backward()
        xm.mark_step()

        running_loss += loss.detach()

        if training_step % args.grad_accum_usteps == 0:
            xm.mark_step()
            if xm.is_master_ordinal():
                print(f"Global Step {global_step}")
                print(f"Loss: {loss.item()}")
                print(f"running_loss: {running_loss.item()}")

            running_loss.zero_()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            xm.mark_step()
            if xm.is_master_ordinal():
                print(f'process time of this batch is: {time.time() - t0}')
            t0 = time.time()

            if (args.checkpoint_freq > 0) and (global_step % args.checkpoint_freq == 0):
                xm.add_step_closure(
                    nxd.save_checkpoint, (
                        args.checkpoint_dir,  # checkpoint directory
                        f"{args.tag}_step_{global_step}",  # tag
                        model,  # model
                        optimizer,  # optimizer
                        scheduler,  # scheduler
                        {"global_step": global_step, "cli_args": args.__dict__},  # user content
                    )
                )

        if global_step >= args.max_steps:
            xm.mark_step()
            break
        # xm.mark_step()  # final mark_step not needed when using MpDeviceLoader


def init_processes(args, backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    run(args=args, backend=backend)
    xm.rendezvous("_mp_fn finished")


def tp_loader(state_dict, tp_rank, tp_size, config):
    """Load the correct slice of weights from a checkpoint for the current core give the tensor parallel degree."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.endswith('out_proj.weight'):  # row parallel
            dim_1_shape = v.shape[1]
            cv = torch.split(v, dim_1_shape // tp_size, dim=1)
            new_state_dict[k] = cv[tp_rank]
        elif k.endswith('in_proj_xBC.weight') or 'conv1d' in k:  # xBC and Conv Col para
            wx, wB, wC = torch.split(v, [config.hidden_size * 2, config.n_groups * config.state_size,
                                         config.n_groups * config.state_size],
                                     dim=0)
            wx_tp = torch.split(wx, wx.shape[0] // tp_size, dim=0)[tp_rank]
            wB_tp = torch.split(wB, wB.shape[0] // tp_size, dim=0)[tp_rank]
            wC_tp = torch.split(wC, wC.shape[0] // tp_size, dim=0)[tp_rank]
            xBC_tp = torch.cat((wx_tp, wB_tp, wC_tp), dim=0)
            new_state_dict[k] = xBC_tp
        elif 'norm' in k and 'mixer' not in k:
            new_state_dict[k] = v
        else:  # norm weight and z and dt
            dim_0_shape = v.shape[0]
            rv = torch.split(v, dim_0_shape // tp_size, dim=0)
            new_state_dict[k] = rv[tp_rank]

    return new_state_dict


if __name__ == '__main__':
    import torch_xla.core.xla_model as xm
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Mamba2Block Configuration")
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"], help="Data type")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--vocab_size", type=int, default=50272, help="Vocab size")
    parser.add_argument("--model", default="Mamba130M", help="Hugging face model to profile")
    parser.add_argument("--backend", type=str, default="xla", choices=['xla', 'nccl', 'gloo'])
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel Size")

    parser.add_argument("--no_optimizer_fp32", action="store_true", help="Do not use FP32 for the optimizer state.")
    parser.add_argument("--use_zero_1", action="store_true", help="Use ZeRO-1.")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta1 parameter for Adam optimizer")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta2 parameter for Adam optimizer")
    parser.add_argument("--data_dir", type=str, help="Pre-tokenized dataset directory.")

    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Number of warmup accumulation-steps for learning rate .")
    parser.add_argument("--max_steps", type=int, help="Maximum total accumulation-steps to run.")
    parser.add_argument("--grad_accum_usteps", type=int, default=1,
                        help="Gradient accumulation micro-steps (an accumulation-step has <value> micro-steps.")
    parser.add_argument("--rmsnorm_across_groups", action="store_true",
                        help="Uses (HF style) RMSNorm instead of the custom one that normalizes independently for each of the n_groups.")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable Neuron debugging flags to dump model graph and compiler logs.")
    parser.add_argument("--checkpoint_freq", type=int, help="ckpt save freq.")
    parser.add_argument("--checkpoint_dir", type=str, default="./", help="ckpt saving dir")
    parser.add_argument("--tag", type=str, default="exp", help="ckpt saving name")

    args = parser.parse_args()

    os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "0"
    args.dtype = getattr(torch, args.dtype)

    if args.dtype == torch.bfloat16:
        modeling_utils.get_parameter_dtype = lambda x: torch.bfloat16

    if args.debug:
        # Debug flags to dump the annotated HLO graph, useful for profiling
        os.environ["XLA_IR_DEBUG"] = "1"
        os.environ["XLA_HLO_DEBUG"] = "1"
        os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

    os.environ["NEURON_CC_FLAGS"] = " --model-type=transformer -O1"

    init_processes(args, backend=args.backend)
