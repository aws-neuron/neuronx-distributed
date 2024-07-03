import argparse
import atexit
import json
import os
import re
import signal
import subprocess
import sys
import traceback

SUCCEEDED = "succeeded"
ERRORS = "errors"
MEMORY_DEGRADATION = "memory degradation"
PERFORMANCE_DEGADATION = "performance degradation"


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--test_json",
        required=False,
        help="input json listing the test spec for network to compile",
    )
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument("--s3_bucket", default="s3://ktf-test-runs/neuronx_distributed_parallel_layers/layers")
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    with open(args.test_json, "r") as f:
        test_dict = json.load(f)
    return test_dict, S3_BUCKET_NAME, args


test_config, S3_BUCKET_NAME, args = parse_args()
results = {"inference_success": 1}


def run_job(seq_len=32768, mem_threshold=0, throughputs_threshold=99999):
    p1 = subprocess.run(
        [f"{seq_len}", "neuron_parallel_compile", "./run_llama_7b_tp_ptl.sh", f"{seq_len}"],
        stderr=sys.stderr,
        stdout=sys.stdout,
    )
    print(f"return code is {p1.returncode}")
    if p1.returncode > 0:
        print(f"Got error while running compilation for seqlen {seq_len}")
        return ERRORS

    pro1 = subprocess.Popen(
        "/opt/aws/neuron/bin/neuron-monitor",
        stdout=subprocess.PIPE,
        shell=True,
        preexec_fn=os.setsid,
    )

    p2 = subprocess.run(
        [f"{seq_len}", "./run_llama_7b_tp_ptl.sh", f"{seq_len}"],
        stderr=sys.stderr,
        stdout=sys.stdout,
    )
    if p2.returncode > 0:
        print(f"Got error while running execution for seqlen {seq_len}")
        return ERRORS
    os.killpg(os.getpgid(pro1.pid), signal.SIGTERM)
    output = str(pro1.communicate()[0])
    print(output)

    memory_usage = extract_peak_mem_usage(output)
    print(f"peak mem usage is {memory_usage}")
    if memory_usage > mem_threshold:
        print(f"memory usage {memory_usage} exceeded mem_threshold {mem_threshold} for seqlen {seq_len}")
        return MEMORY_DEGRADATION

    throughputs = extract_throughput(seq_len)
    print(f"throughputs are {throughputs}")
    if float(throughputs[25]) < throughputs_threshold:
        print(
            f"throughputs {throughputs[25]} doesn't match throughputs threshold {throughputs_threshold} for seqlen {seq_len}"
        )
        return PERFORMANCE_DEGADATION
    return SUCCEEDED


def extract_peak_mem_usage(nm_string):
    regex_pattern = """(?<=memory_used_bytes\"\:).[0-9]+"""
    usages = re.findall(regex_pattern, nm_string)
    max_usage = 0
    print(f"len usages are {len(usages)}")
    for usage in usages:
        max_usage = max(max_usage, int(usage))
    return max_usage


def extract_throughput(seq_len=32768):
    regex_pattern = """(?<=throughput\s)[0-9]+\.[0-9]+(?=\sseq)"""
    throughputs_extracted = [re.findall(regex_pattern, line) for line in open(f"log_exe-{seq_len}.log")]
    throughputs = []
    for extracted in throughputs_extracted:
        throughputs.extend(extracted)
    return throughputs


def on_exit():
    for k in test_config:
        os.system(f"rm {args.test_json}")
        with open(args.test_json, "w") as f:
            json.dump({k: results}, f)


if __name__ == "__main__":
    succeeded = []
    failed = []
    for seq_len, mem_thershold, perf_threshold in [
        # Threshold with 5%-8% tolarance
        [8192, 88590512128, 9.15],
        [16384, 109604828160, 3.10],
        [32768, 124354230272, 1.20],
    ]:
        return_status = run_job(seq_len, mem_thershold, perf_threshold)
        if return_status == SUCCEEDED:
            succeeded.append(seq_len)
        else:
            failed.append(seq_len)
    print(f"Succeeded: seq len {succeeded}, Failed: seq len {failed}")
    try:
        assert not failed, "Job failed"
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise
    print(f"Tests finished successfully!")
    atexit.register(on_exit)
