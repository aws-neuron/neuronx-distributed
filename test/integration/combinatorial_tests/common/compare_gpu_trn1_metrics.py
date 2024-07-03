import argparse
import math
import os
from collections import defaultdict

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
    if args.gpu_event_file.endswith('pt'):
        if len(args.tags) > 1:
            raise ValueError(f"Too many tags provided to use gpu pt benchmark. Only one tag is supported for gpu pt "
                             f"benchmark")
        else:
            tag = args.tags[0]
        print('GPU benchmark is a pt file')
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
        "--atol", type=float, help="Absolute tolerance to use", default=1e-08
    )
    parser.add_argument(
        "--rtol", type=float, help="Relative tolerance to use", default=0.05
    )

    args = parser.parse_args()

    if not os.path.exists(args.gpu_event_file):
        raise FileNotFoundError(f"{args.gpu_event_file} not found")
    if not os.path.exists(args.trn1_event_file):
        raise FileNotFoundError(f"{args.trn1_event_file} not found")

    gpu_benchmark = get_gpu_values(args)
    trn1_events = load_events(args.trn1_event_file)

    for tag in args.tags:
        if tag in gpu_benchmark and tag in trn1_events:
            # extract values for the relevant tag from GPU to handle different benchmark types
            gpu = [val.value if type(val) is not float else val for val in gpu_benchmark[tag]]
            trn = [val.value if type(val) is not float else val for val in trn1_events[tag]]

            are_all_close = torch.allclose(torch.Tensor(trn), torch.Tensor(gpu), rtol=args.rtol, atol=args.atol, equal_nan=False)
            if not are_all_close:
                raise ValueError(f"Tolerance exceeded for values for tag '{tag}'")
        else:
            raise ValueError(f"Tag '{tag}' not found in one of the event files")



if __name__ == "__main__":
    main()
