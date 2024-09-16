import argparse
import os
import time
import traceback

import loss_fn_correctness_test_helper as lch
import torch

# Imports from MoE unit tests (for this import to succeed, test/unit_test/modules/moe must be added to PYTHONPATH)
import utils_testing as ut
from device_correctness_test_configs import (
    get_device_correctness_test_configs,
    get_neuron_cc_flags,
)
from device_correctness_test_runner import run_device_correctness_test

from neuronx_distributed.modules.moe import (
    load_balancing_loss_func as neuron_load_balancing_loss_func,
)
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
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    test_dtype = torch.float32 if args.test_dtype == "fp32" else torch.bfloat16
    return S3_BUCKET_NAME, args, test_dtype


S3_BUCKET_NAME, args, TEST_DTYPE = parse_args()
results = {"inference_success": 1}

# Set compiler flags before TRN enablement
os.environ["NEURON_CC_FLAGS"] = get_neuron_cc_flags(test_dtype=TEST_DTYPE)

# TRN enablement
import torch_xla.core.xla_model as xm  # noqa: E402


def summarize_test(start_time, num_tests, failed):
    print(f"{SEPARATOR}\nRan {num_tests} tests in {round(time.time()-start_time, 1)}s\n\n")
    if failed == 0:
        print("OK\n\n")
    else:
        raise Exception(f"Failed {failed}/{num_tests} tests")


def test_moe_layer_device_correctness():
    if TEST_DTYPE == torch.float32:
        output_test_tols, grad_test_tols = OUTPUT_TEST_TOLS_FP32, GRAD_TEST_TOLS_FP32
    elif TEST_DTYPE == torch.bfloat16:
        output_test_tols, grad_test_tols = OUTPUT_TEST_TOLS_BF16, GRAD_TEST_TOLS_BF16
    else:
        raise ValueError(f"Unknown TEST_DTYPE: {str(TEST_DTYPE)}")

    def _test_moe_layer_device_correctness():
        test_configs = get_device_correctness_test_configs(dtype=TEST_DTYPE)
        start_time = time.time()
        failed = 0
        for i, cfg in enumerate(test_configs):
            print(f"Running test {i+1}/{len(test_configs)}: {str(cfg)}")
            try:
                run_device_correctness_test(cfg, output_test_tols, grad_test_tols)
                print("ok\n")
            except Exception:
                print("Failed test")
                print(traceback.format_exc())
                failed += 1
        summarize_test(start_time, len(test_configs), failed)

    global results
    try:
        _test_moe_layer_device_correctness()
    except Exception:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise


def test_loss_fn_device_correctness():
    def _test_loss_fn_device_correctness():
        test_configs = lch.get_loss_fn_correctness_test_configs(dtypes=[TEST_DTYPE])
        start_time = time.time()
        failed = 0
        for i, cfg in enumerate(test_configs):
            print(f"Running test {i+1}/{len(test_configs)}: {str(cfg)}")
            try:
                # Set random seed for reproducibility
                torch.manual_seed(cfg.num_experts)
                with torch.no_grad():
                    for it in range(cfg.num_iters):
                        concatenated_test_gate_logits = torch.randn(
                            cfg.num_layers * cfg.batch_size * cfg.seq_len,
                            cfg.num_experts,
                            device=torch.device("cpu"),
                            dtype=cfg.dtype,
                        )
                        cpu_loss = neuron_load_balancing_loss_func(
                            concatenated_test_gate_logits, cfg.num_experts, cfg.top_k
                        )

                        concatenated_test_gate_logits_xla = concatenated_test_gate_logits.to(device="xla")
                        neuron_loss = neuron_load_balancing_loss_func(
                            concatenated_test_gate_logits_xla, cfg.num_experts, cfg.top_k
                        )
                        xm.mark_step()

                        # Test correctness
                        assert neuron_loss.dtype == cpu_loss.dtype
                        TEST_TOLS = lch.FP32_TEST_TOLS if cfg.dtype == torch.float32 else lch.BF16_TEST_TOLS
                        ut.check_tensors(neuron_loss.cpu(), cpu_loss, **TEST_TOLS, additional_msg=f"Iteration {it}")
                print("ok\n")
            except Exception:
                print("Failed test")
                print(traceback.format_exc())
                failed += 1
        summarize_test(start_time, len(test_configs), failed)

    global results
    try:
        _test_loss_fn_device_correctness()
    except Exception:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    if is_pjrt_device():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")

    print(f"Running MoE layer device correctness test, test_dtype={str(TEST_DTYPE)}")
    test_moe_layer_device_correctness()
    print(f"Running loss fn device correctness test, test_dtype={str(TEST_DTYPE)}")
    test_loss_fn_device_correctness()
