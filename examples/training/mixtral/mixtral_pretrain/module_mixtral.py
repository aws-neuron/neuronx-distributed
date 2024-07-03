import math
import os
import time
from collections import namedtuple

import torch
import torch_xla.core.xla_model as xm
from training_utils import Throughput, TrainingMetrics

from neuronx_distributed.lightning import NeuronLTModule
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trainer import initialize_parallel_optimizer

Metric = namedtuple("Metric", ["name", "value", "units", "additional_data"])


class NeuronMixtralLTModule(NeuronLTModule):
    def setup(self, stage=None):
        super().setup()

        self.throughput = Throughput(
            self.train_batch_size,
            parallel_state.get_data_parallel_size(),
            self.grad_accum_steps,
            10,
            self.logging_interval,
        )
        if (
            self.trainer.strategy.data_parallel_rank == 0
            and self.trainer.strategy.tensor_parallel_rank == 0
            and self.trainer.strategy.pipeline_parallel_rank == self.print_pp_rank
        ):
            self.tps_history = []
            self.metric_writer = TrainingMetrics("results.json")
            self.metric_writer.store_parameters(
                {
                    "Model": self.model_args[0].model_type,
                    "Model configuration": str(self.model_args[0]),
                    "World size": xm.xrt_world_size(),
                    "Data parallel degree": self.trainer.strategy.data_parallel_size,
                    "Batch size": self.train_batch_size,
                    "Optimizer": str(self.opt_cls),
                    "Optimizer Parameters": str(self.opt_kwargs),
                    "Gradient accumulation microsteps": self.grad_accum_steps,
                    "Environment variables": {
                        variable: value
                        for variable, value in os.environ.items()
                        if variable.startswith("NEURON") or variable.startswith("XLA")
                    },
                }
            )

    def training_step(self, batch, batch_idx):
        xm.mark_step()
        for logger in self.trainer.loggers:
            logger.print_step = -1

        self.should_print = False
        outputs = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
        loss = outputs.loss / self.grad_accum_steps
        loss.backward()
        self.averaged_loss += loss.detach()
        xm.mark_step()
        # doing manual optimization
        if not self.automatic_optimization and (batch_idx + 1) % self.grad_accum_steps == 0:
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
            self.global_norm = optimizer.grad_norm
            optimizer.zero_grad()
            scheduler.step()
            xm.mark_step()

            # Setup items for logging
            self.loss = loss_reduced_detached
            self.lr = scheduler.get_lr()[0]
            self.input_ids = batch["input_ids"]
            self.tps = self.throughput.get_throughput()
            if (
                self.trainer.strategy.data_parallel_rank == 0
                and self.trainer.strategy.tensor_parallel_rank == 0
                and self.trainer.strategy.pipeline_parallel_rank == self.print_pp_rank
            ):
                self.tps_history.append(self.tps)
        return loss

    def configure_optimizers(self):
        param_groups = self.get_param_groups_by_weight_decay()
        optimizer = initialize_parallel_optimizer(self.nxd_config, self.opt_cls, param_groups, **self.opt_kwargs)
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
                and self.trainer.strategy.pipeline_parallel_rank == self.print_pp_rank
            ):
                print(
                    f"step {self.global_step} loss is {self.loss.detach().cpu().item()}, lr is {self.lr}, input_ids {torch.sum(self.input_ids.detach().cpu()).item()}, norm {self.global_norm}, global rank {xm.get_ordinal()}"
                )

            # Logging, need to revisit when automatic_optimization enabled
            if not self.automatic_optimization:
                self.log(
                    "loss",
                    self.loss.detach().cpu().item()
                    if self.loss is not None
                    else torch.zeros(1, device="cpu", requires_grad=False),
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
                self.log("throughput", self.tps, prog_bar=True)
                self.log(
                    "global_step",
                    self.global_step,
                    prog_bar=True,
                    on_step=True,
                    on_epoch=True,
                )
                for logger in self.trainer.loggers:
                    logger.print_step = self.global_step

    def on_train_start(self, *args, **kwargs):
        if (
            self.trainer.strategy.data_parallel_rank == 0
            and self.trainer.strategy.tensor_parallel_rank == 0
            and self.trainer.strategy.pipeline_parallel_rank == self.print_pp_rank
        ):
            print(f"Training started!")
            # record training start time
            self.start_time = time.time()

    def on_train_end(self, *args, **kwargs):
        if (
            self.trainer.strategy.data_parallel_rank == 0
            and self.trainer.strategy.tensor_parallel_rank == 0
            and self.trainer.strategy.pipeline_parallel_rank == self.print_pp_rank
        ):
            print(f"Training finished!")
            extract_graphs_only = os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None)
            if not extract_graphs_only:
                # record aggregate & final statistics in the metrics file
                final_time = time.time()
                time_diff = final_time - self.start_time
                additional_data = {"Epoch": self.current_epoch, "Global step": self.global_step}
                # Skip the first a few steps for collecting accurate average throughput.
                min_throughput_index = math.ceil(10 / self.logging_interval)
                if len(self.tps_history) > min_throughput_index:
                    throughputs_to_average = self.tps_history[min_throughput_index:]
                else:
                    throughputs_to_average = self.tps_history
                average_throughput = (
                    round(sum(throughputs_to_average) / len(throughputs_to_average), 4)
                    if len(throughputs_to_average) > 0
                    else None
                )
                metric_data = [
                    Metric(
                        "Final loss", self.loss.detach().item() if self.loss is not None else None, "", additional_data
                    ),
                    Metric(
                        "Time to train",
                        round(time_diff / 60, 4),
                        "minutes",
                        additional_data,
                    ),
                    Metric(
                        "Average throughput",
                        average_throughput,
                        "seq/s",
                        additional_data,
                    ),
                    Metric(
                        "Peak throughput",
                        round(max(self.tps_history), 4) if len(self.tps_history) > 0 else None,
                        "seq/s",
                        additional_data,
                    ),
                ]
                self.metric_writer.store_metrics(metric_data)


class NeuronMixtralPPLTModule(NeuronMixtralLTModule):
    def training_step(self, batch, batch_idx):
        xm.mark_step()

        for logger in self.trainer.loggers:
            logger.print_step = -1

        self.should_print = True

        loss = self.model.run_train(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )

        loss_detached = loss.detach() if self.trainer.strategy.pipeline_parallel_rank == self.print_pp_rank else None

        if not self.automatic_optimization:
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
            optimizer.step()
            self.global_norm = optimizer.grad_norm
            optimizer.zero_grad()
            scheduler.step()
            xm.mark_step()

            # Setup items for logging
            self.loss = loss_detached
            self.lr = self.lr_schedulers().get_lr()[0]
            self.input_ids = batch["input_ids"]
            self.tps = self.throughput.get_throughput()
            if (
                self.trainer.strategy.data_parallel_rank == 0
                and self.trainer.strategy.tensor_parallel_rank == 0
                and self.trainer.strategy.pipeline_parallel_rank == self.print_pp_rank
            ):
                self.tps_history.append(self.tps)

        return loss
