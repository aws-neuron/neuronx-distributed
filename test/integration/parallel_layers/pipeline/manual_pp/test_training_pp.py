import torch.distributed
from neuronx_distributed.pipeline.model import NxDPPModel
import torch

import math

from neuronx_distributed import initialize_parallel_optimizer
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_size,
    get_data_parallel_rank,
)
from neuronx_distributed.pipeline.comm import rmsg
from neuronx_distributed.utils.logger import get_logger
from commons import init_distributed_env, ToyModel, copy_model, generate_inputs, loss_check
from neuronx_distributed.utils import cpu_mode

import torch_xla.core.xla_model as xm

from argparse import ArgumentParser


logger = get_logger("Manual_PP_train", rank0_only=True)


def create_optimizer(parameters, use_nxd_wrapper=False, lr=1e-3, cls=torch.optim.AdamW):
    if use_nxd_wrapper:
        nxd_config = {}
        optimizer = initialize_parallel_optimizer(
            nxd_config,
            cls,
            parameters=parameters,
        )
        return optimizer
    else:
        return cls(parameters, lr=lr)


def _check_device_mark_step(device):
    if device == "xla" or device == xm.xla_device():
        xm.mark_step()


def training_loop(
    model,
    optimizer,
    dataset,
    loss_fn,
    nxd_pp=False,
    bs=None,
    device=torch.device("cpu"),
):
    dp_rank = get_data_parallel_rank()
    loss_vals = []
    for idx, (inputs, labels) in enumerate(dataset):
        if nxd_pp:
            assert bs is not None, "bs must be provided for nxd_pp"
            offset = dp_rank * bs
            sliced_inputs = inputs[offset : offset + bs, :]
            sliced_labels = labels[offset : offset + bs]
            # logger.debug(rmsg(f'pp test input sum {torch.sum(sliced_inputs)} labels sum {torch.sum(sliced_labels)}'))
            loss = model.run_train(inputs=sliced_inputs, labels=sliced_labels)
        else:
            _check_device_mark_step(device)

            inputs = inputs.to(device)
            labels = labels.to(device)
            _check_device_mark_step(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            _check_device_mark_step(device)
            loss.backward()
            _check_device_mark_step(device)

        logger.info(
            rmsg(f"nxd_pp {nxd_pp}, dev {device} iter {idx} loss: {loss.detach().cpu().item()}")
        )
        loss_vals.append(loss.detach().cpu().item())
        optimizer.step()
        _check_device_mark_step(device)
        optimizer.zero_grad()
        _check_device_mark_step(device)

    _check_device_mark_step(device)
    return loss_vals


def _cpu_single_device_run(model, optim_cls, loss_fn, rand_dataset):
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = create_optimizer(
        model.parameters(), use_nxd_wrapper=False, cls=optim_cls
    )
    model.train()
    single_device_losses = training_loop(
        model, optimizer, rand_dataset, loss_fn, device=device
    )
    return single_device_losses


def _xla_single_device_run(model, optim_cls, loss_fn, rand_dataset):
    device = xm.xla_device()
    model = model.to(device)
    optimizer = create_optimizer(
        model.parameters(), use_nxd_wrapper=False, cls=optim_cls
    )
    model.train()
    losses = training_loop(model, optimizer, rand_dataset, loss_fn, device=device)
    return losses


def _nxd_pp_run(model, optim_cls, loss_fn, rand_dataset, bs, n_mb, weight_sharing):
    """This func runs in xla or cpu mode.
    assuming model is a copied.
    """
    layers = model.get_layers()
    if weight_sharing:
        from neuronx_distributed.pipeline.manual_pipe_stage import PipelineStageModule
        PipelineStageModule.mark_weight_sharing(
            [(layers[0], "weight"), (layers[-1], "weight")],
            "shared_first_last_layer_weight",
        )

    nxdpp_model = NxDPPModel(
        module=layers,
        num_microbatches=n_mb,
        manual_pp_partition=True,
        manual_pp_stage_partition_fn=None,
        manual_pp_loss_fn=torch.nn.CrossEntropyLoss(),
        _use_gloo_for_metadata_comm=False,
        _all_reduce_send_recv=False,
        return_loss_on_cpu=True,
        broadcast_and_average_loss=True,
    )
    # nxd pp run
    nxd_optimizer = create_optimizer(
        nxdpp_model.local_parameters(), use_nxd_wrapper=False, cls=optim_cls
    )
    # for nxd pp keep the inputs on cpu first
    losses = training_loop(
        nxdpp_model, nxd_optimizer, rand_dataset, loss_fn, nxd_pp=True, bs=bs
    )
    return losses


def test_training(weight_sharing=False):
    """"""

    hidden_size = 8
    nlayers = 8
    mbs = 4
    n_mb = 2  # number of microbatches per pipeline group
    n_iter = 100  # number of iterations
    opt_cls = torch.optim.SGD

    dp_size = get_data_parallel_size()

    # fixing the weight init and data inputs
    torch.manual_seed(1234)
    single_device_model = ToyModel(
        nlayers=nlayers, hidden_size=hidden_size, weight_sharing=weight_sharing
    )

    rand_dataset = generate_inputs(
        n_iter=n_iter, dp_size=dp_size, bs=mbs, hidden_size=hidden_size
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # assuming evenly divide the layers to 4 stages

    cpu_single_device_losses = _cpu_single_device_run(
        copy_model(single_device_model, weight_sharing), opt_cls, loss_fn, rand_dataset
    )

    if not cpu_mode():
        xla_single_device_losses = _xla_single_device_run(
            copy_model(single_device_model, weight_sharing), opt_cls, loss_fn, rand_dataset
        )
    else:
        xla_single_device_losses = None

    nxd_pp_losses = _nxd_pp_run(
        copy_model(single_device_model, weight_sharing), opt_cls, loss_fn, rand_dataset, bs=mbs, n_mb=n_mb,
        weight_sharing=weight_sharing
    )

    rtol = 1e-5
    rank = torch.distributed.get_rank()
    if rank == 0:
        loss_check(logger, rtol, cpu_single_device_losses, xla_single_device_losses, nxd_pp_losses)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--pp-size", default=4, type=int)
    arg_parser.add_argument("--test-weight-sharing", default=False, action="store_true")
    args = arg_parser.parse_args()

    init_distributed_env(tp_size=1, pp_size=args.pp_size)
    test_training(weight_sharing=args.test_weight_sharing)
