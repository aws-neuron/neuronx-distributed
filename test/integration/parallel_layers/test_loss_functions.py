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
import torch_xla.debug.metrics as met
from commons import IdentityLayer, print_separator, set_random_seed

from neuronx_distributed.parallel_layers import (
    parallel_state,
    scatter_to_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy
from neuronx_distributed.parallel_layers.utils import is_pjrt_device

datetime_str = str(datetime.now())


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--test_json",
        required=False,
        help="input json listing the test spec for network to compile",
    )
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument(
        "--s3_bucket",
        default="s3://ktf-test-runs/neuronx_distributed_parallel_layers/loss_functions",
    )
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    with open(args.test_json, "r") as f:
        test_dict = json.load(f)
    return test_dict, S3_BUCKET_NAME, args


test_config, S3_BUCKET_NAME, args = parse_args()
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
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise


def upload_to_s3():
    os.system(f'aws s3 cp --no-progress "{datetime_str}" {S3_BUCKET_NAME}')
    print(met.metrics_report())


def on_exit():
    upload_to_s3()
    for k in test_config:
        os.system(f"rm {args.test_json}")
        with open(args.test_json, "w") as f:
            json.dump({k: results}, f)


if __name__ == "__main__":
    if is_pjrt_device():
        import torch_xla.experimental.pjrt_backend
        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    world_size = xm.xrt_world_size()
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator("test parallel cross entropy")
        test_parallel_cross_entropy(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
    atexit.register(on_exit)
