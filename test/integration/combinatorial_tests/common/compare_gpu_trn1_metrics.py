import tensorboard.backend.event_processing.event_accumulator as event_acc
import argparse
from collections import defaultdict
import os
import math


def load_events(event_file):
    accumulator = event_acc.EventAccumulator(event_file)
    accumulator.Reload()
    tags = accumulator.Tags()

    data = {}
    for tag in tags['scalars']:
        data[tag] = accumulator.Scalars(tag)
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Compare GPU and Trn1 TensorBoard event files.")
    parser.add_argument("gpu_event_file",
                        help="Path to the GPU TensorBoard file.")
    parser.add_argument("trn1_event_file",
                        help="Path to the Trn1 TensorBoard file.")
    parser.add_argument("tags", nargs='+',
                        help="List of tags to compare.")
    parser.add_argument("--smoothed_weight", type=float, default=0,
                        help="Smoothing factor for the values.Value between 0 and 1")
    parser.add_argument("--delta_percentage", type=float, default=1.0,
                        help="The tolerated percentage difference.")
    parser.add_argument("--comparison_start_step", type=int, help="the first step for which to compare data", default=450)

    args = parser.parse_args()

    if not os.path.exists(args.gpu_event_file):
        raise FileNotFoundError(f"{args.gpu_event_file} not found")
    if not os.path.exists(args.trn1_event_file):
        raise FileNotFoundError(f"{args.trn1_event_file} not found")

    gpu_events = load_events(args.gpu_event_file)
    trn1_events = load_events(args.trn1_event_file)
    for tag in args.tags:
        if tag in gpu_events and tag in trn1_events:
            trn1_last_value = trn1_events[tag][0].value # First value in the trn plot (first timestep)
            gpu_last_value = gpu_events[tag][0].value # First value in the gpu plot (first timestep)

            # Create a lookup for step to smoothed gpu value for efficient comparisons
            gpu_events_lookup = defaultdict(lambda: None)
            for gpu in gpu_events[tag]:
                gpu_value = gpu.value
                smoothed_gpu_val = gpu_last_value * args.smoothed_weight + (1 - args.smoothed_weight) * gpu_value
                gpu_events_lookup[gpu.step] = smoothed_gpu_val
                gpu_last_value = smoothed_gpu_val
            missing_steps = 0

            for trn in trn1_events[tag]:
                trn1_value = trn.value
                assert not math.isnan(trn1_value), f"trn1 value is nan for {tag}"
                smoothed_trn_val = trn1_last_value * args.smoothed_weight + (1 - args.smoothed_weight) * trn1_value
                smoothed_gpu_val = gpu_events_lookup[trn.step]

                if(smoothed_gpu_val is not None):

                    smoothed_gpu_val = gpu_events_lookup[trn.step]

                    delta = abs(smoothed_gpu_val - smoothed_trn_val)
                    max_val = max(abs(smoothed_gpu_val), abs(smoothed_gpu_val))
                    trn1_last_value = smoothed_trn_val
                    if trn.step > args.comparison_start_step and max_val > 0 and (delta / max_val) * 100 > args.delta_percentage:
                        raise ValueError(f"Delta percentage exceeds tolerance value for tag '{tag}' at step {trn.step} {smoothed_trn_val} {smoothed_gpu_val}")
                elif trn.step > args.comparison_start_step:
                    missing_steps+=1
            print(f"Missing steps were {missing_steps}")
        else:
            raise ValueError(f"Tag '{tag}' not found in one of the event files")

if __name__ == "__main__":
    main()