import argparse
import math

import torch.distributed
from commons import init_distributed_env, ToyMLPsTP, generate_inputs, loss_check
from transformers import LlamaConfig
from neuronx_distributed.utils import cpu_mode

import torch
from neuronx_distributed.parallel_layers.random import model_parallel_xla_manual_seed
from neuronx_distributed.parallel_layers.parallel_state import get_data_parallel_size
from test_training_pp import training_loop
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.pipeline.comm import rmsg

from torch_xla.core import xla_model as xm
from neuronx_distributed.pipeline.model import NxDPPModel


def wrapped_cross_entropy_loss(logits, labels):
    if len(logits.shape) > 2:
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
    return torch.nn.functional.cross_entropy(logits, labels)

def _uneven_partition_layers(layers, num_stages):
    rst = {}
    cur_stage = 0
    while cur_stage < num_stages:
        # first stage get 1 layer
        if cur_stage == 0:
            rst[layers[0]] = cur_stage
            cur_stage += 1
        elif cur_stage < num_stages - 1:
            # middle stages get 2 layers
            rst[layers[cur_stage * 2 - 1]] = cur_stage
            rst[layers[cur_stage * 2]] = cur_stage
            cur_stage += 1
        else:
            # last stage get the rest layers 
            for i in range(cur_stage * 2 - 1, len(layers)):
                rst[layers[i]] = cur_stage
            cur_stage += 1
    return rst

def test_tp_pp_train(tp_size, pp_size, uneven_partition=False):
    logger = get_logger("test_tp_tp_train", rank0_only=False)

    model_config = LlamaConfig(
        hidden_size=8,
        intermediate_size=8,
        num_hidden_layers=8,
        initializer_range=0.5,
        hidden_act="relu",
    )
    n_iter = 10
    mbs = 4
    n_mb = 1
    seq_len = 2
    dp_size = get_data_parallel_size()
    optim_cls = torch.optim.SGD
    lr = 1e-3

    model_parallel_xla_manual_seed(1234)
    torch.manual_seed(1234)

    # create the parallelized model
    model = ToyMLPsTP(model_config, tp_size)
    rand_data = generate_inputs(
        n_iter=n_iter,
        dp_size=dp_size,
        bs=4,
        hidden_size=model_config.hidden_size,
        seq_len=seq_len,
    )

    # copy the layer master weights from the single device model
    no_tp_model = ToyMLPsTP(model_config, tp_size=1)
    with torch.no_grad():
        for idx in range(len(model.layers)):
            no_tp_model.layers[idx].gate_proj.weight.copy_(
                model.layers[idx].gate_proj.master_weight
            )
            no_tp_model.layers[idx].down_proj.weight.copy_(
                model.layers[idx].down_proj.master_weight
            )

    cpu_single_device_losses = _single_device_runs(
        no_tp_model.clone(),
        "cpu",
        optim_cls,
        wrapped_cross_entropy_loss,
        lr,
        rand_data,
    )
    if not cpu_mode():
        xla_single_device_losses = _single_device_runs(
            no_tp_model.clone(),
            xm.xla_device(),
            optim_cls,
            wrapped_cross_entropy_loss,
            lr,
            rand_data,
        )

    # pp run losses
    partition_fn = _uneven_partition_layers if uneven_partition else None
    multi_device_pp_losses = _pp_runs(
        model.clone(), optim_cls, wrapped_cross_entropy_loss, lr, rand_data, mbs, n_mb, partition_fn
    )

    if torch.distributed.get_rank() == 0:
        rtol = 1e-5
        xla_single_device_losses = None if cpu_mode() else xla_single_device_losses
        loss_check(logger, rtol, cpu_single_device_losses, xla_single_device_losses, multi_device_pp_losses)


def _single_device_runs(model, device, optim_cls, loss_fn, lr, rand_dataset):
    model = model.to(device)
    model.train()
    optimizer = optim_cls(model.parameters(), lr=lr)
    losses = training_loop(model, optimizer, rand_dataset, loss_fn, device=device)
    return losses


def _pp_runs(model, optim_cls, loss_fn, lr, rand_dataset, bs, n_mb, partition_fn=None):
    pp_model = NxDPPModel(
        module=model.get_layers(),
        num_microbatches=n_mb,
        manual_pp_partition=True,
        manual_pp_stage_partition_fn=partition_fn,
        manual_pp_loss_fn=loss_fn,
        _use_gloo_for_metadata_comm=False,
        _all_reduce_send_recv=False,
        return_loss_on_cpu=True,
        broadcast_and_average_loss=True,
    )
    optim = optim_cls(pp_model.local_parameters(), lr=lr)

    losses = training_loop(pp_model, optim, rand_dataset, loss_fn, nxd_pp=True, bs=bs)

    return losses


def _get_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--pp-size", default=4, type=int)

    arg_parser.add_argument("--tp-size", default=2, type=int)

    arg_parser.add_argument("--uneven-partition", default=False, action="store_true")

    return arg_parser.parse_args()


if __name__ == "__main__":
    args = _get_args()

    init_distributed_env(tp_size=args.tp_size, pp_size=args.pp_size)

    test_tp_pp_train(
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        uneven_partition=args.uneven_partition,
    )
