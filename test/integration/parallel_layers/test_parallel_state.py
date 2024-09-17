import argparse
import atexit
import json
import os
import traceback
from datetime import datetime

import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from commons import print_separator

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.utils import requires_init_pg_override

datetime_str = str(datetime.now())


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument(
        "--s3_bucket",
        default="s3://ktf-test-runs/neuronx_distributed_parallel_layers/parallel_state",
    )
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    return S3_BUCKET_NAME, args


S3_BUCKET_NAME, args = parse_args()
results = {"inference_success": 1}


def test_initialize_model_parallel(tensor_model_parallel_size):
    def _test_initialize_model_parallel():
        if torch.distributed.get_rank() == 0:
            print("testing initialize_model_parallel with size {}".format(tensor_model_parallel_size))
        tensor_model_parallel_size_ = min(tensor_model_parallel_size, torch.distributed.get_world_size())
        assert not parallel_state.model_parallel_is_initialized()
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tensor_model_parallel_size_)
        assert parallel_state.model_parallel_is_initialized()

        # Checks.
        def check(group, world_size, rank):
            assert world_size == torch.distributed.get_world_size(group=group)
            assert rank == torch.distributed.get_rank(group=group)

        # Model parallel.
        world_size = tensor_model_parallel_size_
        rank = torch.distributed.get_rank() % tensor_model_parallel_size_
        assert world_size == parallel_state.get_tensor_model_parallel_size()
        assert rank == parallel_state.get_tensor_model_parallel_rank()
        check(parallel_state.get_tensor_model_parallel_group(), world_size, rank)

        # Data parallel.
        world_size = torch.distributed.get_world_size() // tensor_model_parallel_size_
        rank = torch.distributed.get_rank() // tensor_model_parallel_size
        assert world_size == parallel_state.get_data_parallel_size()
        assert rank == parallel_state.get_data_parallel_rank()
        check(parallel_state.get_data_parallel_group(), world_size, rank)

        # Reset groups
        parallel_state.destroy_model_parallel()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test passed")

    global results
    try:
        _test_initialize_model_parallel()
    except Exception:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise


def test_get_tensor_model_parallel_src_rank(tensor_model_parallel_size_):
    def _test_get_tensor_model_parallel_src_rank():
        if torch.distributed.get_rank() == 0:
            print("testing get_tensor_model_parallel_src_rank with size {}".format(tensor_model_parallel_size_))
        tensor_model_parallel_size = min(tensor_model_parallel_size_, torch.distributed.get_world_size())
        assert not parallel_state.model_parallel_is_initialized()
        parallel_state.initialize_model_parallel(tensor_model_parallel_size)
        assert parallel_state.model_parallel_is_initialized()

        # Checks
        src_rank = torch.distributed.get_rank() - parallel_state.get_tensor_model_parallel_rank()
        assert parallel_state.get_tensor_model_parallel_src_rank() == src_rank

        # Reset groups
        parallel_state.destroy_model_parallel()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test passed")

    global results
    try:
        _test_get_tensor_model_parallel_src_rank()
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
    world_size = xm.xrt_world_size()
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator("test initialize model parallel")
        test_initialize_model_parallel(tensor_model_parallel_size)
        print_separator("test model parallel source rank")
        test_get_tensor_model_parallel_src_rank(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
    atexit.register(on_exit)
