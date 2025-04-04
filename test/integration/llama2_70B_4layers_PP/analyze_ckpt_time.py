import re
import sys
from statistics import mean

def parse_logs(log_path, expected_checkpoint_time):
    checkpointing_steps = set()
    step_times = []

    # Regular expressions for parsing
    step_line_regex = re.compile(r"step (\d+) step_time ([0-9.]+)s")
    checkpoint_start_regex = re.compile(r"async saving of checkpoint step_(\d+) began")

    # Read log file
    with open(log_path, "r") as log_file:
        log_lines = log_file.readlines()

    # Collect step times and identify checkpointing steps
    for line in log_lines:
        # Match checkpoint start logs
        checkpoint_match = checkpoint_start_regex.search(line)
        if checkpoint_match:
            checkpoint_step = int(checkpoint_match.group(1))
            checkpointing_steps.add(checkpoint_step)

        # Match step logs
        step_match = step_line_regex.search(line)
        if step_match:
            step_num = int(step_match.group(1))
            step_time = float(step_match.group(2))
            step_times.append((step_num, step_time))
    
    # Ignore the first 3 steps which are usually outliers
    step_times = step_times[3:]
    
    # Separate checkpointing and non-checkpointing step times
    non_checkpoint_times = [
        time for step, time in step_times if step not in checkpointing_steps
    ]
    checkpoint_times = [
        time for step, time in step_times if step in checkpointing_steps
    ]

    # Calculate means
    mean_non_checkpoint = mean(non_checkpoint_times) if non_checkpoint_times else 0
    mean_checkpoint = mean(checkpoint_times) if checkpoint_times else 0

    # Mean checkpointing time
    mean_checkpointing_time = mean_checkpoint - mean_non_checkpoint

    return {
        "mean_non_checkpoint_time": mean_non_checkpoint,
        "mean_checkpoint_time": mean_checkpoint,
        "mean_checkpointing_time": mean_checkpointing_time,
    }, mean_checkpointing_time >= expected_checkpoint_time


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_checkpoint_time.py <log_path> <expected_checkpoint_time>")
        sys.exit(1)

    log_path = sys.argv[1]
    expected_checkpoint_time = float(sys.argv[2])

    results, passed = parse_logs(log_path, expected_checkpoint_time)

    print("Results:")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")

    if not passed:
        raise ValueError(f"Mean checkpointing time {results['mean_checkpointing_time']:.6f} is below expected {expected_checkpoint_time}")

    print("Checkpointing time meets expectations.")
