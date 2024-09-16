import os

import numpy as np
import torch
from module_llama_orig import NeuronLlamaLTModule as NeuronLlamaLTModuleOrigin
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_events(event_file):
    accumulator = EventAccumulator(event_file)
    accumulator.Reload()
    tags = accumulator.Tags()

    data = {}
    for tag in tags["scalars"]:
        data[tag] = accumulator.Scalars(tag)
    return data


class NeuronLlamaLTModule(NeuronLlamaLTModuleOrigin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # For pipeline testing, comparing with golden event
        self.golden_steploss = []
        event_file = os.getenv("GOLDEN_EVENT_FILE") if not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None) else None
        if event_file is not None:
            accumulator = EventAccumulator(event_file)
            accumulator.Reload()
            tags = accumulator.Tags()

            data = {}
            for tag in tags["scalars"]:
                data[tag] = accumulator.Scalars(tag)
            self.golden_steploss = []
            for step in data["step loss"]:
                self.golden_steploss.append(step.value)

    def on_train_batch_end(self, *args, **kwargs):
        # Compare to golden
        if self.should_print:
            if (
                self.trainer.strategy.data_parallel_rank == 0
                and self.trainer.strategy.tensor_parallel_rank == 0
                and self.trainer.strategy.pipeline_parallel_rank == self.trainer.strategy.pipeline_parallel_size - 1
            ):
                print(
                    f"step {self.global_step} loss is {self.loss.detach().cpu().item()}, lr is {self.lr}, throughput {self.tps} seq/s, input_ids {torch.sum(self.input_ids.detach().cpu()).item()}"
                )

                step_now = self.global_step - 1
                if step_now < len(self.golden_steploss) and step_now >= 0:
                    assert np.allclose(
                        self.loss.detach().cpu().item(), self.golden_steploss[step_now], rtol=2.3e-1
                    ), f"Loss mismatch with golden, PTL {self.loss.detach().cpu().item()}, Non-PTL {self.golden_steploss[step_now]}"
