import numbers
import os
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from torch import Tensor
import torch_xla.core.xla_model as xm

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import _METRIC
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torchmetrics import Metric, MetricCollection
from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import _FxValidator
from lightning_utilities.core.apply_func import apply_to_collection
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature

from neuronx_distributed.lightning import NeuronLTModule
from neuronx_distributed.trainer import initialize_parallel_model, initialize_parallel_optimizer
from neuronx_distributed.parallel_layers.grads import get_grad_norm
from neuronx_distributed.parallel_layers import parallel_state
from module_llama_orig import NeuronLlamaLTModule as NeuronLlamaLTModuleOrigin

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
            for step in data["loss"]:
                self.golden_steploss.append(step.value)
                print(f"self golden is {self.golden_steploss}")

    def on_train_batch_end(self, *args, **kwargs):
        # Compare to golden
        if self.should_print:
            if (
                self.trainer.strategy.data_parallel_rank == 0 
                and self.trainer.strategy.tensor_parallel_rank == 0
                and self.trainer.strategy.pipeline_parallel_rank == self.trainer.strategy.pipeline_parallel_size - 1
            ):
                print(f"step {self.global_step} loss is {self.loss.detach().cpu().item()}, lr is {self.lr}, input_ids {torch.sum(self.input_ids.detach().cpu()).item()}")
    
                step_now = self.global_step - 1
                if step_now < len(self.golden_steploss) and step_now >= 0:
                    assert np.allclose(
                        self.loss.detach().cpu().item(), self.golden_steploss[step_now], rtol=2.3e-1
                    ), f"Loss mismatch with golden, PTL {self.loss.detach().cpu().item()}, Non-PTL {self.golden_steploss[step_now]}"


            

            

