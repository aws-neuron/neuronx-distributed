import argparse
import atexit
import json
import os
import traceback
from datetime import datetime
from typing import Tuple

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.debug.metrics as met
from commons import IdentityLayer, print_separator, set_random_seed

from neuronx_distributed.parallel_layers import (
    parallel_state,
    scatter_to_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy, from_parallel_logits_to_logprobs
from neuronx_distributed.parallel_layers.utils import requires_init_pg_override

datetime_str = str(datetime.now())


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument(
        "--s3_bucket",
        default="s3://ktf-test-runs/neuronx_distributed_parallel_layers/loss_functions",
    )
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    return S3_BUCKET_NAME, args

S3_BUCKET_NAME, args = parse_args()
results = {"inference_success": 1}


def test_parallel_cross_entropy(tensor_model_parallel_size):
    def torch_cross_entropy(
        batch_size: int,
        seq_length: int,
        vocab_size: int,
        logits_scale: float,
        seed: int,
        device,
        label_smoothing: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        set_random_seed(seed)
        identity = IdentityLayer((batch_size, seq_length, vocab_size), scale=logits_scale).to(device)
        logits = identity()
        target = torch.LongTensor(size=(batch_size, seq_length)).random_(0, vocab_size).to(device)
        loss = (
            F.cross_entropy(
                logits.view(-1, logits.size()[-1]),
                target.view(-1),
                reduction="none",
                label_smoothing=label_smoothing,
            )
            .view_as(target)
            .mean()
        )
        loss.backward()
        return loss, identity.weight.grad

    def tensor_sharded_cross_entropy(
        batch_size, seq_length, vocab_size, logits_scale, seed, device, label_smoothing=0.0
    ):
        set_random_seed(seed)
        identity = IdentityLayer((batch_size, seq_length, vocab_size), scale=logits_scale).to(device)
        logits = identity()
        logits_parallel = scatter_to_tensor_model_parallel_region(logits)
        target = torch.LongTensor(size=(batch_size, seq_length)).random_(0, vocab_size).to(device)
        loss = parallel_cross_entropy(logits_parallel, target, label_smoothing=label_smoothing).mean()
        loss.backward()
        return loss, identity.weight.grad

    def _test_parallel_cross_entropy():
        device = xm.xla_device()
        tensor_model_parallel_size_ = tensor_model_parallel_size
        parallel_state.initialize_model_parallel(tensor_model_parallel_size_)
        tensor_model_parallel_size_ = parallel_state.get_tensor_model_parallel_size()

        batch_size, sequence_length, vocab_size_per_partition = 13, 17, 11
        logits_scale = 1000.0
        seed = 1234

        vocab_size = vocab_size_per_partition * tensor_model_parallel_size_
        loss_torch, grad_torch = torch_cross_entropy(
            batch_size, sequence_length, vocab_size, logits_scale, seed, device
        )
        (
            loss_tensor_parallel,
            grad_tensor_parallel,
        ) = tensor_sharded_cross_entropy(batch_size, sequence_length, vocab_size, logits_scale, seed, device)
        error = loss_tensor_parallel.sub(loss_torch).abs()
        print("   error in loss (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
        assert error < 1.0e-3, "error: {}".format(error)

        error = grad_tensor_parallel.sub(grad_torch).abs().max()
        print("   error in grad (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
        assert error < 1.0e-3, "error: {}".format(error)

        # Reset groups
        parallel_state.destroy_model_parallel()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test passed")

        del device

    global results
    try:
        _test_parallel_cross_entropy()
    except Exception:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise

def test_parallel_logits_to_logprobs(tensor_model_parallel_size):
    def torch_logits_to_logprobs(
        logits,
        target,
        seed,
    ) -> torch.Tensor:
        target = target.roll(shifts=-1, dims=-1)
        set_random_seed(seed)
        log_probs = F.log_softmax(logits, dim=-1)
        gathered_log_probs = torch.gather(log_probs, -1, target.unsqueeze(-1)).squeeze(-1)
        return gathered_log_probs[:, :-1]

    def tensor_sharded_logits_to_logprobs(
        logits,
        target,
        seed,
    ):
        set_random_seed(seed)
        logits_parallel = scatter_to_tensor_model_parallel_region(logits)
        log_probs = from_parallel_logits_to_logprobs(logits_parallel, target, inference=True)
        return log_probs

    def _test_parallel_logits_to_logprobs():
        device = xm.xla_device()
        tensor_model_parallel_size_ = tensor_model_parallel_size
        parallel_state.initialize_model_parallel(tensor_model_parallel_size_)
        tensor_model_parallel_size_ = parallel_state.get_tensor_model_parallel_size()

        batch_size, sequence_length, vocab_size_per_partition = 2, 8, 16
        vocab_size = vocab_size_per_partition * tensor_model_parallel_size_
        logits_scale = 1.0
        seed = 1234

        # Generate full logits and target
        set_random_seed(seed)
        logits = torch.randn((batch_size, sequence_length, vocab_size), device=device) * logits_scale
        target = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)

        # Compute torch log probs
        log_probs_torch = torch_logits_to_logprobs(logits, target, seed)

        # Compute parallel log probs
        log_probs_parallel = tensor_sharded_logits_to_logprobs(logits, target, seed)

        error = log_probs_parallel.sub(log_probs_torch).abs().max()
        print(f"Max error in log_probs on rank {torch.distributed.get_rank()}: {error}")
        assert error < 1.0e-5, f"Error too large: {error}"

        # Reset groups
        parallel_state.destroy_model_parallel()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("Test passed")

    global results
    try:
        _test_parallel_logits_to_logprobs()
    except Exception:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise

def on_exit():
    print(met.metrics_report())


if __name__ == "__main__":
    if requires_init_pg_override():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    world_size = xr.world_size()
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator("test parallel cross entropy")
        test_parallel_cross_entropy(tensor_model_parallel_size)

        print_separator("test parallel logits to logprobs")
        test_parallel_logits_to_logprobs(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
    atexit.register(on_exit)
