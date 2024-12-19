"""Load and compare metrics from TensorBoard event files or PyTorch
checkpoint files for GPU and Trn1 (Trainium) hardware.

This script supports comparing multiple tags (metrics) across a specified range of steps, 
using configurable tolerance levels for numerical comparisons.

Usage:
    python compare_gpu_trn1_metrics.py <gpu_event_file> <trn1_event_file> <tags...>
                                       [--comparison_start_step STEP]
                                       [--comparison_stop_step STEP]
                                       [--atol ATOL] [--rtol RTOL]
                                       [--confidence_interval INTERVAL]

Arguments:
    gpu_event_file: Path to the GPU TensorBoard file or PyTorch checkpoint
    trn1_event_file: Path to the Trn1 TensorBoard file
    tags: List of metric tags to compare

Optional arguments:
    --comparison_start_step: First step for data comparison (default: 450)
    --comparison_stop_step: Last step for data comparison (default: None, use all steps)
    --atol: Absolute tolerance for numerical comparisons (default: 1e-08)
    --rtol: Relative tolerance for numerical comparisons (default: 0.05)
    --confidence_interval: Required similarity threshold (default: 0.95)
"""

import argparse
import os
import tensorboard.backend.event_processing.event_accumulator as event_acc
import torch


def load_events(event_file):
    accumulator = event_acc.EventAccumulator(event_file)
    accumulator.Reload()
    tags = accumulator.Tags()

    data = {}
    for tag in tags["scalars"]:
        data[tag] = accumulator.Scalars(tag)
    return data


def get_gpu_values(args):
    if args.gpu_event_file.endswith("pt"):
        if len(args.tags) > 1:
            raise ValueError(
                "Too many tags provided to use gpu pt benchmark. Only one tag is supported for gpu pt benchmark"
            )
        else:
            tag = args.tags[0]
        print("GPU benchmark is a pt file")
        gpu_benchmark = torch.load(args.gpu_event_file)
        gpu = {tag: [tensor.item() for tensor in gpu_benchmark]}
    else:
        gpu = load_events(args.gpu_event_file)
    return gpu


def main():
    parser = argparse.ArgumentParser(description="Compare GPU and Trn1 TensorBoard event files.")
    parser.add_argument("gpu_event_file", help="Path to the GPU TensorBoard file.")
    parser.add_argument("trn1_event_file", help="Path to the Trn1 TensorBoard file.")
    parser.add_argument("tags", nargs="+", help="List of tags to compare.")
    parser.add_argument(
        "--comparison_start_step", type=int, help="the first step for which to compare data", default=450
    )
    parser.add_argument(
        "--comparison_stop_step", type=int, help="the last step for which to compare data", default=None
    )
    parser.add_argument("--atol", type=float, help="Absolute tolerance to use", default=1e-08)
    parser.add_argument("--rtol", type=float, help="Relative tolerance to use", default=0.05)

    parser.add_argument("--confidence_interval", type=float, help="Relative tolerance to use", default=0.95)

    args = parser.parse_args()

    if not os.path.exists(args.gpu_event_file):
        raise FileNotFoundError(f"{args.gpu_event_file} not found")
    if not os.path.exists(args.trn1_event_file):
        raise FileNotFoundError(f"{args.trn1_event_file} not found")

    gpu_benchmark = get_gpu_values(args)
    trn1_events = load_events(args.trn1_event_file)

    for tag in args.tags:
        trn1_tag = "step loss" if (tag == "loss" and tag not in trn1_events) else tag
        if tag in gpu_benchmark and trn1_tag in trn1_events:
            # extract values for the relevant tag from GPU to handle different benchmark types
            gpu = [val.value if not isinstance(val, float) else val for val in gpu_benchmark[tag]]
            trn = [val.value if not isinstance(val, float) else val for val in trn1_events[trn1_tag]]

            # Apply the start and stop steps
            start_step = args.comparison_start_step
            stop_step = args.comparison_stop_step if args.comparison_stop_step is not None else len(gpu)

            gpu = gpu[start_step:stop_step]
            trn = trn[start_step:stop_step]

            are_close = torch.isclose(
                torch.Tensor(trn), torch.Tensor(gpu), rtol=args.rtol, atol=args.atol, equal_nan=False
            )

            if torch.mean(are_close.float()).item() < args.confidence_interval:
                raise ValueError(f"Tolerance exceeded for values for tag '{tag}', GPU value={gpu}, trn1 value={trn}")
        else:
            raise ValueError(f"Tag '{tag}' not found in one of the event files")


if __name__ == "__main__":
    main()
