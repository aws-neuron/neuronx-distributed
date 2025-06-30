import os
import argparse
import atexit
import traceback
from datetime import datetime

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
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


def run_test(test_fn, *args, **kwargs) -> None:
    try:
        print_separator(f"test {test_fn.__name__}")
        rank = torch.distributed.get_rank()
        if rank == 0:
            print(f"testing {test_fn.__name__} with args={args} and kwargs={kwargs}")
        assert not parallel_state.model_parallel_is_initialized()
        test_fn(*args, **kwargs)
        assert parallel_state.model_parallel_is_initialized()

        torch.distributed.barrier()
        if rank == 0:
            print("test passed")
    except Exception:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise
    finally:
        parallel_state.destroy_model_parallel()


def check(group, world_size: int, rank: int) -> None:
    assert world_size == torch.distributed.get_world_size(group=group)
    assert rank == torch.distributed.get_rank(group=group)


def on_exit() -> None:
    print(met.metrics_report())


def test_initialize_model_parallel(tp_size: int, cp_size=1) -> None:
    """test initialize model parallel"""
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size, context_parallel_size=cp_size)

    # Model parallel.
    dp_size = torch.distributed.get_world_size() // (tp_size*cp_size)
    rank = torch.distributed.get_rank()
    if tp_size == 4:  # a special case
        dp_rank = rank % dp_size
        tp_rank = (rank // dp_size) % tp_size
        cp_rank = rank % cp_size
    else:
        tp_rank = rank % tp_size
        dp_rank = (rank // (tp_size*cp_size)) % dp_size
        cp_rank = rank % cp_size
    assert tp_size == parallel_state.get_tensor_model_parallel_size()
    assert tp_rank == parallel_state.get_tensor_model_parallel_rank()
    check(parallel_state.get_tensor_model_parallel_group(), tp_size, tp_rank)

    assert cp_size == parallel_state.get_context_model_parallel_size()
    assert cp_rank == parallel_state.get_context_model_parallel_rank()
    check(parallel_state.get_context_model_parallel_group(), cp_size, cp_rank)

    # Data parallel.
    assert dp_size == parallel_state.get_data_parallel_size()
    assert dp_rank == parallel_state.get_data_parallel_rank()
    check(parallel_state.get_data_parallel_group(), dp_size, dp_rank)


def test_get_tensor_model_parallel_src_rank(tp_size: int) -> None:
    """test model parallel source rank"""
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_size)

    # Checks
    rank = torch.distributed.get_rank()
    if tp_size == 4:
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        src_rank = local_world_size * (rank // local_world_size) + rank % tp_size
    else:
        src_rank = (rank // tp_size) * tp_size
    assert parallel_state.get_tensor_model_parallel_src_rank() == src_rank


if __name__ == "__main__":
    if requires_init_pg_override():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    world_size = xr.world_size()
    assert world_size <= torch.distributed.get_world_size()
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        run_test(test_initialize_model_parallel, tensor_model_parallel_size)
        run_test(test_get_tensor_model_parallel_src_rank, tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
    
    # Test with TP and CP
    tensor_model_parallel_size = 1
    context_model_parallel_size = 1
    while context_model_parallel_size*tensor_model_parallel_size <= world_size:
        print_separator("test initialize model parallel")
        test_initialize_model_parallel(tensor_model_parallel_size, context_model_parallel_size)
        context_model_parallel_size *= 2

    atexit.register(on_exit)
