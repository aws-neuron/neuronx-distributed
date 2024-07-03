import time
from functools import partial

import numpy as np

BENCHMARK_REPORT_FILENAME = "benchmark_report.json"


class Benchmark:
    def __init__(self, benchmark_func, input_param, config, num_runs=20, preprocess_func=None) -> None:
        if isinstance(input_param, (tuple, list)):
            self.benchmark_func = partial(benchmark_func, *input_param)
        elif isinstance(input_param, dict):
            self.benchmark_func = partial(benchmark_func, **input_param)
        else:
            self.benchmark_func = partial(benchmark_func, input_param)

        self.config = config
        self.num_runs = num_runs
        self.preprocess_func = preprocess_func

    def run(self):
        # Warmp up
        if self.preprocess_func:
            self.preprocess_func()
        self.benchmark_func()

        latency_list = []
        e2e_start = time.time()
        for _ in range(self.num_runs):
            start = time.time()
            if self.preprocess_func:
                self.preprocess_func()
            self.benchmark_func()
            latency_list.append(time.time() - start)
        e2e_time = time.time() - e2e_start

        return self.process_metrics(latency_list, e2e_time, self.config)

    def process_metrics(self, latency_list, e2e_time, config):
        latency_array = np.array(latency_list)

        max_length = config.max_length
        batch_size = config.max_batch_size
        n_runs = self.num_runs
        throughput = (max_length * n_runs * batch_size) / e2e_time

        metrics = {
            "latency_ms_p50": np.percentile(latency_array, 50) * 1000,
            "latency_ms_p90": np.percentile(latency_array, 90) * 1000,
            "latency_ms_p95": np.percentile(latency_array, 95) * 1000,
            "latency_ms_p99": np.percentile(latency_array, 99) * 1000,
            "latency_ms_p100": np.percentile(latency_array, 100) * 1000,
            "latency_ms_avg": np.average(latency_array) * 1000,
            "throughput": throughput,
        }
        return metrics
