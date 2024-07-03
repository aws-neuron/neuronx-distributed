import argparse
import atexit
import json
import os
import time
import torch
import traceback

# Imports from MoE unit tests (for this import to succeed, test/unit_test/modules/moe must be added to PYTHONPATH)
from device_correctness_test_configs import get_device_correctness_parallel_test_configs
from device_correctness_test_runner import run_device_correctness_test

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
    parser.add_argument(
        "--test_json",
        required=False,
        help="input json listing the test spec for network to compile",
    )
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument("--s3_bucket", default="s3://ktf-test-runs/neuronx_distributed_modules/moe")
    parser.add_argument("--test_dtype", required=True, choices=["fp32", "bf16"], help="Either fp32 or bf16")
    parser.add_argument(
        "--test_tp_degree", required=True, type=int, choices=[2, 8, 16, 32], help="One of 2, 8, 16 or 32"
    )
    parser.add_argument(
        "--test_sp_mode", required=True, type=str, help="One of MoESequenceParallelMode"
    )
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    with open(args.test_json, "r") as f:
        test_dict = json.load(f)
    test_dtype = torch.float32 if args.test_dtype == "fp32" else torch.bfloat16
    return test_dict, S3_BUCKET_NAME, args, test_dtype, args.test_tp_degree, args.test_sp_mode


test_config, S3_BUCKET_NAME, args, TEST_DTYPE, TEST_TP_DEGREE, TEST_SP_MODE = parse_args()
results = {"inference_success": 1}

if "--model-type" not in os.environ.get("NEURON_CC_FLAGS", ""):
    os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + " --model-type=transformer "
else:
    assert any(s in os.environ["NEURON_CC_FLAGS"] for s in ["--model-type transformer", "--model-type=transformer"])


if TEST_DTYPE == torch.float32:
    # Set compiler flag to disable auto-casting before TRN enablement
    assert "--auto-cast" not in os.environ.get("NEURON_CC_FLAGS", "")
    os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + " --auto-cast=none"

import torch_xla.core.xla_model as xm  # TRN enablement


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
        test_configs = get_device_correctness_parallel_test_configs(dtype=TEST_DTYPE, tp_degree=TEST_TP_DEGREE, sp_mode=TEST_SP_MODE)
        start_time = time.time()
        failed = 0
        print_rank0(f"Running {len(test_configs)} tests")
        for i, cfg in enumerate(test_configs):
            print_rank0(f"Running test {i+1}/{len(test_configs)}: {str(cfg)}")
            try:
                run_device_correctness_test(cfg, output_test_tols, grad_test_tols)
                print_rank0(f"ok\n")
            except Exception as e:
                print_rank0(f"Failed test")
                print_rank0(traceback.format_exc())
                failed += 1
        summarize_test(start_time, len(test_configs), failed)

    global results
    try:
        _test_moe_layer_device_correctness_parallel()
    except:
        results["inference_success"] = 0
        print_rank0(traceback.format_exc())
        raise


def on_exit():
    if xm.get_ordinal() == 0:
        for k in test_config:
            os.system(f"rm {args.test_json}")
            with open(args.test_json, "w") as f:
                json.dump({k: results}, f)


if __name__ == "__main__":
    if is_pjrt_device():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")

    print_rank0(f"test device correctness parallel, test_dtype={str(TEST_DTYPE)}, test_tp_degree={TEST_TP_DEGREE}, test_sp_mode={TEST_SP_MODE}")
    test_moe_layer_device_correctness_parallel()
    atexit.register(on_exit)
