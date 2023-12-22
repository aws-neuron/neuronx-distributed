import numbers
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

class NeuronLlamaLTModule(NeuronLTModule):

    def training_step(self, batch, batch_idx):
        xm.mark_step()
        for logger in self.trainer.loggers:
            logger.print_step = -1

        self.should_print = False
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss / self.grad_accum_steps
        loss.backward()
        self.averaged_loss += loss.detach()
        xm.mark_step()
        # doing manual optimization
        if not self.automatic_optimization and (batch_idx +1) % self.grad_accum_steps == 0:
            self.should_print = True

            loss_div = self.averaged_loss / self.trainer.strategy.data_parallel_size
            loss_reduced = xm.all_reduce(
                xm.REDUCE_SUM,
                loss_div,
                groups=parallel_state.get_data_parallel_group(as_list=True),
            )
            loss_reduced_detached = loss_reduced.detach()
            self.averaged_loss.zero_()
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
            optimizer.step()

            optimizer.zero_grad()
            scheduler.step()
            xm.mark_step()

            
            # Setup items for logging
            self.loss = loss_reduced_detached
            self.lr = scheduler.get_lr()[0]
            self.input_ids = batch["input_ids"]
        return loss


    def configure_optimizers(self):
        param_groups = self.get_param_groups_by_weight_decay()
        optimizer = initialize_parallel_optimizer(
            self.nxd_config, self.opt_cls, param_groups, **self.opt_kwargs
        )
        optimizer.zero_grad()

        scheduler = self.scheduler_cls(optimizer, *self.scheduler_args, **self.scheduler_kwargs)
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                }
            ],
        )


    def on_train_batch_end(self, *args, **kwargs):
        if self.should_print:
            if (
                self.trainer.strategy.data_parallel_rank == 0 
                and self.trainer.strategy.tensor_parallel_rank == 0
                and self.trainer.strategy.pipeline_parallel_rank == self.trainer.strategy.pipeline_parallel_size - 1
            ):
                print(f"step {self.global_step} loss is {self.loss.detach().cpu().item()}, lr is {self.lr}, input_ids {torch.sum(self.input_ids.detach().cpu()).item()}")

            # # Logging, need to revisit when automatic_optimization enabled
            if not self.automatic_optimization:
                self.log(
                    "loss",
                    self.loss.detach().cpu().item() if self.loss is not None else torch.zeros(1, device="cpu", requires_grad=False),
                    prog_bar=True,
                )
                self.log(
                    "lr",
                    self.lr,
                    prog_bar=True,
                )
                self.log(
                    "input_ids",
                    torch.sum(self.input_ids.detach().cpu()).item(),
                    prog_bar=True,
                )
                self.log(
                    "global_step",
                    self.global_step,
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                )
                for logger in self.trainer.loggers:
                    logger.print_step = self.global_step
