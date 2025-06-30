import argparse
import atexit
import json
import os
import time
import traceback

import torch

# Imports from MoE unit tests (for this import to succeed, test/unit_test/modules/moe must be added to PYTHONPATH)
from device_correctness_test_configs import (
    get_device_correctness_parallel_test_configs,
    get_neuron_cc_flags,
)
from checkpoint_test_runner import run_checkpoint_test

from neuronx_distributed.parallel_layers.utils import requires_init_pg_override
import torch_xla.core.xla_model as xm  # TRN enablement
import torch_xla.runtime as xr

SEPARATOR = "-" * 70

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--test_json",
        required=False,
        help="input json listing the test spec for network to compile",
    )
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument("--s3_bucket", default="s3://ktf-test-runs/neuronx_distributed_modules/moe")
    parser.add_argument("--test_dtype", required=True, choices=["fp32", "bf16"], help="Either fp32 or bf16")
    parser.add_argument("--test_mode", required=True, type=str, help="Either training or inference")
    parser.add_argument(
        "--test_tp_degree", required=True, type=int, choices=[1, 2, 8, 16, 32], help="One of 1, 2, 8, 16 or 32"
    )
    parser.add_argument(
        "--test_ep_degree", required=True, type=int, choices=[1, 2, 4, 8, 16, 32], help="One of 1, 2, 4, 8, 16 or 32"
    )
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    with open(args.test_json, "r") as f:
        test_dict = json.load(f)
    test_dtype = torch.float32 if args.test_dtype == "fp32" else torch.bfloat16
    return test_dict, S3_BUCKET_NAME, args, test_dtype, args.test_mode, args.test_tp_degree, args.test_ep_degree


test_config, S3_BUCKET_NAME, args, TEST_DTYPE, TEST_MODE, TEST_TP_DEGREE, TEST_EP_DEGREE = parse_args()
results = {"inference_success": 1}

# Set compiler flags before TRN enablement
os.environ["NEURON_CC_FLAGS"] = get_neuron_cc_flags(test_dtype=TEST_DTYPE)

def print_rank0(s):
    if xr.global_ordinal() == 0:
        print(s)


def summarize_test(start_time, num_tests, failed):
    print_rank0(f"{SEPARATOR}\nRan {num_tests} tests in {round(time.time()-start_time, 1)}s\n\n")
    if failed == 0:
        print_rank0("OK\n\n")
    else:
        raise Exception(f"Failed {failed}/{num_tests} tests")


def test_moe_layer_checkpoint_parallel():
    def _test_moe_layer_checkpoint_parallel():
        test_configs = get_device_correctness_parallel_test_configs(
            dtype=TEST_DTYPE, tp_degree=TEST_TP_DEGREE, ep_degree=TEST_EP_DEGREE, test_mode=TEST_MODE
        )

        start_time = time.time()
        failed = 0
        print_rank0(f"Running {len(test_configs)} tests")
        for i, cfg in enumerate(test_configs):
            print_rank0(f"Running test {i+1}/{len(test_configs)}: {str(cfg)}")
            try:
                run_checkpoint_test(cfg)
                clean_dir()
                print_rank0("ok\n")

            except Exception:
                print_rank0("Failed test")
                print_rank0(traceback.format_exc())
                failed += 1

            # running test only once
            break
        summarize_test(start_time, len(test_configs), failed)

    global results
    try:
        _test_moe_layer_checkpoint_parallel()
    except:
        results["inference_success"] = 0
        print_rank0(traceback.format_exc())
        raise


def clean_dir():
    xm.rendezvous("Cleaning directory")
    if xr.global_ordinal() == 0:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(cur_dir, "model")
        os.system(f"rm -rf {path}")
    xm.rendezvous("Cleaned directory")

def on_exit():
    if xr.global_ordinal() == 0:
        for k in test_config:
            os.system(f"rm {args.test_json}")
            with open(args.test_json, "w") as f:
                json.dump({k: results}, f)


if __name__ == "__main__":
    if requires_init_pg_override():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")

    print_rank0(
        f"test device correctness parallel, test_dtype={str(TEST_DTYPE)}, test_mode={TEST_MODE}, test_tp_degree={TEST_TP_DEGREE}, test_ep_degree={TEST_EP_DEGREE}"
    )
    test_moe_layer_checkpoint_parallel()
    atexit.register(on_exit)
