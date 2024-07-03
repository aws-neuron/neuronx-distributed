import json
import math
import os
import queue
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import datasets
from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from transformers import default_data_collator, set_seed

try:
    from lr import CosineAnnealing
except ImportError:
    CosineAnnealing = None

from collections import namedtuple

Metric = namedtuple("Metric", ["name", "value", "units", "additional_data"])


def get_learning_rate_scheduler(optimizer, max_steps, min_lr, warmup_steps, constant_steps, last_epoch=-1):
    lr_scheduler = CosineAnnealing(
        optimizer,
        max_steps=max_steps,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        constant_steps=constant_steps,
        last_epoch=last_epoch,
    )
    return lr_scheduler


def create_mixtral_pretraining_dataset(data_dir, mini_batch_size, dp_size, dp_rank, seed):
    # Workaround because python functions are not picklable
    class WorkerInitObj(object):
        def __init__(self, seed):
            self.seed = seed

        def __call__(self, id):
            set_seed(self.seed)

    worker_init = WorkerInitObj(seed)
    train_data = datasets.load_from_disk(data_dir)
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=False,
        drop_last=True,
    )
    train_dataloader = DataLoader(
        train_data,
        collate_fn=default_data_collator,
        sampler=train_sampler,
        batch_size=mini_batch_size,
        num_workers=0,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=True,
    )
    return train_dataloader


class TrainingMetrics:
    """
    This class is used for logging metrics to a json file. One can provide a
    dictionary of metrics that needs to be stored, and it wpuld get
    written to the file.
    Arguments:
        json_file: File used for logging. If no file exists, new file would be created.
    """

    def __init__(self, json_file):
        self.json_file = json_file

    def read_modify_write_file(self, data, key: str = "metrics") -> None:
        """
        data (dict of training parameters or list of metrics): Data to update in the file.
        key (str): the dictionary key under which data is to be recorded
        """
        result_dict = {}
        print(f"Writing data to the provided results file: {self.json_file}")
        if os.path.exists(self.json_file):
            with open(self.json_file, "r") as json_file:
                content = json_file.read()
                if not content.strip():  # Check if content is empty or contains only whitespace
                    print("File is empty or contains only whitespace.")
                else:
                    result_dict = json.loads(content) or result_dict
        print(f"Updating with {key} data: {data}")
        if result_dict:
            try:
                # handle internal named entity if present
                results = result_dict[next(iter(result_dict))]
            except Exception:
                results = result_dict
            current = results.get(key)
            if not current:
                results[key] = data
            else:
                if isinstance(current, list):
                    current.extend(data)
                elif isinstance(current, dict):
                    current.update(data)
        else:
            result_dict["results"] = {key: data}
        with open(self.json_file, "w") as json_file:
            json.dump(result_dict, json_file)

    def store_metrics(self, metrics: List[Metric]) -> None:
        """
        Writes collected metrics to the file.
        """
        data = [
            {
                "MetricName": metric.name,
                "MeasuredValue": metric.value,
                "Units": metric.units,
                "Timestamp": datetime.now(timezone.utc).isoformat(),
                "AdditionalData": metric.additional_data,
            }
            for metric in metrics
        ]
        self.update(data=data, key="metrics")

    def store_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Writes specified model and configuration parameters to the file.
        """
        self.update(data=parameters, key="parameters")

    def update(self, **kwargs: Any) -> None:
        """
        Write specified data to the output file.
        """
        self.read_modify_write_file(**kwargs)


class Throughput:
    def __init__(self, batch_size, world_size, grad_accum_usteps, moving_avg_window_size=10, logging_interval=1):
        """
        Used to calculate the throughput over a moving window. It records the step time
        between two calls and uses that time to calculate the throughput.
        """
        self.seqs_per_iteration = batch_size * world_size * grad_accum_usteps * logging_interval
        self.moving_avg_window_size = math.ceil(moving_avg_window_size / logging_interval)
        self.moving_avg_window = queue.Queue()
        self.window_time = 0
        self.start_time = time.time()

    def get_throughput(self):
        step_time = time.time() - self.start_time
        self.start_time += step_time
        self.window_time += step_time
        self.moving_avg_window.put(step_time)
        window_size = self.moving_avg_window.qsize()
        if window_size > self.moving_avg_window_size:
            self.window_time -= self.moving_avg_window.get()
            window_size -= 1
        throughput = window_size * self.seqs_per_iteration / self.window_time
        return throughput


def get_mixed_precision_config(use_gpu_compatible_precision):
    return {
        "use_master_weights": bool(use_gpu_compatible_precision),
        "use_fp32_grad_acc": bool(use_gpu_compatible_precision),
        "use_master_weights_in_ckpt": False,
    }
