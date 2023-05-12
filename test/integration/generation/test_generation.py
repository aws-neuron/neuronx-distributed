import argparse
import os
import json
import subprocess

import numpy as np

RUN_SAMPLING_SCRIPT_CMD = "python run_sampling.py"

def run_test(test_dict):
    XLA_CPU_RUN_CMD=f"NEURON_INTERNAL_USE_VANILLA_TORCH_XLA=1 NEURON_NUM_DEVICES=0 {RUN_SAMPLING_SCRIPT_CMD} --device xla --output_path xla_cpu_out.npy"
    XLA_NEURON_RUN_FP32_CMD=f"{RUN_SAMPLING_SCRIPT_CMD} --device xla --output_path xla_neuron_fp32_out.npy"
    XLA_NEURON_RUN_BF16_CMD=f"XLA_USE_BF16=1 {RUN_SAMPLING_SCRIPT_CMD} --device xla --output_path xla_neuron_bf16_out.npy"
    CPU_RUN_CMD=f"{RUN_SAMPLING_SCRIPT_CMD} --device cpu --output_path cpu_out.npy"
    subprocess.call(XLA_CPU_RUN_CMD, shell=True)
    subprocess.call(XLA_NEURON_RUN_FP32_CMD, shell=True)
    subprocess.call(XLA_NEURON_RUN_BF16_CMD, shell=True)
    subprocess.call(CPU_RUN_CMD, shell=True)

    with open("cpu_out.npy", "rb") as f:
        out_cpu = np.load(f, allow_pickle=True)
    
    with open("xla_cpu_out.npy", "rb") as f:
        out_xla_cpu = np.load(f, allow_pickle=True)
    
    with open("xla_neuron_fp32_out.npy", "rb") as f:
        out_neuron_fp32 = np.load(f, allow_pickle=True)
    
    with open("xla_neuron_bf16_out.npy", "rb") as f:
        out_neuron_bf16 = np.load(f, allow_pickle=True)
    
    try:
        assert np.array_equal(out_cpu, out_xla_cpu), "XLA CPU output doesn't match CPU only output"
        assert np.array_equal(out_cpu, out_neuron_fp32), "XLA Neuron FP32 output doesn't match CPU only output"
        assert np.array_equal(out_cpu, out_neuron_bf16), "XLA Neuron bf16 output doesn't match CPU only output"
    except:
        test_dict["inference_success"] = 0
    test_dict["inference_success"] = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--test_json",
        required=False,
        help="input json listing the test spec for network to compile",
    )
    parser.add_argument(
        "--s3_dir", required=False, help="location to upload all test artifacts"
    )
    parser.add_argument("--s3_bucket", default="test_generate")
    args, leftovers = parser.parse_known_args()
    with open(args.test_json, "r") as f:
        test_dict = json.load(f)

    inference_success = 1
    test_result_json = {}
    for k, v in test_dict.items():
        inference_success = inference_success and run_test(v)
        test_result_json[k] = v
    os.system(f"rm {args.test_json}")
    with open(args.test_json, "w") as f:
        json.dump(test_result_json, f, indent=2)
