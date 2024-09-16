import argparse
import os
import time
import traceback

import torch

# Imports from MoE unit tests (for this import to succeed, test/unit_test/modules/moe must be added to PYTHONPATH)
from device_correctness_test_configs import (
    get_device_correctness_parallel_test_configs,
    get_neuron_cc_flags,
)
from device_correctness_test_runner import run_device_correctness_test
from checkpoint_test_runner import run_checkpoint_test

from neuronx_distributed.parallel_layers.parallel_state import rmsg
from neuronx_distributed.parallel_layers.utils import is_pjrt_device

SEPARATOR = "-" * 70

# FP32 test tolerances
OUTPUT_TEST_TOLS_FP32 = {
    "atol": 5e-4,
    "rtol": 1e-2,
}
GRAD_TEST_TOLS_FP32 = {
    "atol": 5e-4,
    "rtol": 1e-2,
}
# BF16 test tolerances
OUTPUT_TEST_TOLS_BF16 = {
    "atol": 5e-2,
    "rtol": 1e-2,
}
GRAD_TEST_TOLS_BF16 = {
    "atol": 5e-3,
    "rtol": 1e-2,
}


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument("--s3_bucket", default="s3://ktf-test-runs/neuronx_distributed_modules/moe")
    parser.add_argument("--test_dtype", required=True, choices=["fp32", "bf16"], help="Either fp32 or bf16")
    parser.add_argument("--test_mode", required=True, type=str, help="Either training or inference")
    parser.add_argument(
        "--test_tp_degree", required=True, type=int, choices=[1, 2, 8, 16, 32], help="One of 1, 2, 8, 16 or 32"
    )
    parser.add_argument(
        "--test_ep_degree", required=False, default=1, type=int, choices=[1, 2, 4, 8, 16, 32], help="One of 1, 2, 4, 8, 16 or 32"
    )
    parser.add_argument(
        "--zero1", required=False, default=1, type=int, choices=[0, 1], help="Enable zero-1"
    )
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    test_dtype = torch.float32 if args.test_dtype == "fp32" else torch.bfloat16
    return S3_BUCKET_NAME, args, test_dtype, args.test_mode, args.test_tp_degree


S3_BUCKET_NAME, args, TEST_DTYPE, TEST_MODE, TEST_TP_DEGREE = parse_args()
results = {"inference_success": 1}

# Set compiler flags before TRN enablement
os.environ["NEURON_CC_FLAGS"] = get_neuron_cc_flags(test_dtype=TEST_DTYPE)

# TRN enablement
import torch_xla.core.xla_model as xm  # noqa: E402


def print_rank0(s):
    if xm.get_ordinal() == 0:
        print(s)


def summarize_test(start_time, num_tests, failed):
    print_rank0(f"{SEPARATOR}\nRan {num_tests} tests in {round(time.time()-start_time, 1)}s\n\n")
    if failed == 0:
        print_rank0("OK\n\n")
    else:
        raise Exception(f"Failed {failed}/{num_tests} tests")


def test_moe_layer_device_correctness_parallel():
    if TEST_DTYPE == torch.float32:
        output_test_tols, grad_test_tols = OUTPUT_TEST_TOLS_FP32, GRAD_TEST_TOLS_FP32
    elif TEST_DTYPE == torch.bfloat16:
        output_test_tols, grad_test_tols = OUTPUT_TEST_TOLS_BF16, GRAD_TEST_TOLS_BF16
    else:
        raise ValueError(f"Unknown TEST_DTYPE: {str(TEST_DTYPE)}")

    def _test_moe_layer_device_correctness_parallel():
        test_configs = get_device_correctness_parallel_test_configs(
            dtype=TEST_DTYPE, tp_degree=TEST_TP_DEGREE, ep_degree=TEST_EP_DEGREE, test_mode=TEST_MODE, zero1=ZERO1,
        )
        start_time = time.time()
        failed = 0
        print_rank0(f"Running {len(test_configs)} tests")
        for i, cfg in enumerate(test_configs):
            print_rank0(f"Running test {i+1}/{len(test_configs)}: {str(cfg)}")
            try:
                run_device_correctness_test(cfg, output_test_tols, grad_test_tols)
                print_rank0("ok\n")
            except Exception as e:
                print(rmsg(f"Test failed: {e}"))
                print(rmsg(traceback.format_exc()))
                failed += 1
        summarize_test(start_time, len(test_configs), failed)

    global results
    try:
        _test_moe_layer_device_correctness_parallel()
    except Exception:
        results["inference_success"] = 0
        print_rank0(traceback.format_exc())
        raise


if __name__ == "__main__":
    if is_pjrt_device():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")

    print_rank0(
        f"test device correctness parallel, test_dtype={str(TEST_DTYPE)}, test_mode={TEST_MODE}, test_tp_degree={TEST_TP_DEGREE}, test_ep_degree={TEST_EP_DEGREE}, zero1={ZERO1}"
    )
    test_moe_layer_device_correctness_parallel()
