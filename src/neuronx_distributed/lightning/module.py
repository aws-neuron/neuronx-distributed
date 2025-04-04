import numbers
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch_xla.core.xla_model as xm
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.rank_zero import rank_zero_warn
from pytorch_lightning import LightningModule
from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import (
    _FxValidator,
)
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import _METRIC
from torch import Tensor
from torchmetrics import Metric

from neuronx_distributed.trainer import (
    initialize_parallel_model,
    initialize_parallel_optimizer,
)


class NeuronLTModule(LightningModule):
    def __init__(
        self,
        nxd_config: Dict,
        opt_cls: Callable,
        scheduler_cls: Callable,
        model_args: Tuple = (),
        model_kwargs: Dict = {},
        opt_args: Tuple = (),
        opt_kwargs: Dict = {},
        scheduler_args: Tuple = (),
        scheduler_kwargs: Dict = {},
        model_fn: Optional[Callable[..., Any]] = None,
        grad_accum_steps: int = 1,
        train_batch_size: int = 16,
        logging_interval: int = 1,
        log_rank0: bool = False,
        manual_opt: bool = True,
    ):
        super().__init__()
        self.model_fn = model_fn
        self.nxd_config = nxd_config
        self.opt_cls = opt_cls
        self.scheduler_cls = scheduler_cls
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.opt_args = opt_args
        self.opt_kwargs = opt_kwargs
        self.scheduler_args = scheduler_args
        self.scheduler_kwargs = scheduler_kwargs
        self.grad_accum_steps = grad_accum_steps
        self.train_batch_size = train_batch_size
        self.logging_interval = logging_interval
        self.log_rank0 = log_rank0

        self.automatic_optimization = not manual_opt

        # metrics to log
        self.loss = None
        self.lr = None
        self.input_ids = None
        self.global_norm = None

        self.should_print = False
        self._metric_attributes: Dict[int, str]

    def forward(self, batch):
        return self.model.forward()

    def configure_optimizers(self):
        param_groups = self.get_param_groups_by_weight_decay()
        optimizer = initialize_parallel_optimizer(
            self.nxd_config, self.opt_cls, param_groups, *self.opt_args, **self.opt_kwargs
        )

        scheduler = self.scheduler_cls(optimizer, *self.scheduler_args, **self.scheduler_kwargs)
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                }
            ],
        )

    def configure_gradient_clipping(self, *args, **kwargs):
        # Since we handle the gradient clipping inside the optimizer
        # wrapper or within Zero1, we pass here
        pass

    def clip_gradients(self, *args, **kwargs):
        pass

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    def on_train_batch_end(self, *args, **kwargs):
        pass  # Customer defined

    def setup(self, stage=None, include_buffers=False):
        self.model = initialize_parallel_model(
            self.nxd_config,
            self.model_fn,
            include_buffers,
            *self.model_args,
            **self.model_kwargs,
        )
        self.averaged_loss = torch.zeros(1, dtype=torch.double).to(xm.xla_device())
        self.print_pp_rank = 0 if self.log_rank0 else self.trainer.strategy.pipeline_parallel_size - 1

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_param_groups_by_weight_decay(self):
        """Get param groups. Customers can override this to have their own way of weight_decay"""
        if hasattr(self.model, "local_named_parameters") and hasattr(self.model, "partitioned") and self.model.partitioned:
            # Zero1 use the first param in opt to decide the device
            param_optimizer = list(self.model.local_named_parameters())
        else:
            param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "norm"]  # gamma/beta are in LayerNorm.weight

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters

    def __to_tensor(self, value: Union[Tensor, numbers.Number], name: str) -> Tensor:
        value = value.clone().detach() if isinstance(value, Tensor) else torch.tensor(value, device="cpu")
        if not torch.numel(value) == 1:
            raise ValueError(
                f"`self.log({name}, {value})` was called, but the tensor must have a single element."
                f" You can try doing `self.log({name}, {value}.mean())`"
            )
        value = value.squeeze()
        return value

    def log(
        self,
        name: str,
        value: _METRIC,
        prog_bar: bool = False,
        logger: Optional[bool] = None,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: Union[str, Callable] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = False,
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,
    ) -> None:
        """Log a key, value pair.

        Example::

            self.log('train_loss', loss)

        The default behavior per hook is documented here: :ref:`extensions/logging:Automatic Logging`.

        Args:
            name: key to log.
            value: value to log. Can be a ``float``, ``Tensor``, or a ``Metric``.
            prog_bar: if ``True`` logs to the progress bar.
            logger: if ``True`` logs to the logger.
            on_step: if ``True`` logs at this step. The default value is determined by the hook.
                See :ref:`extensions/logging:Automatic Logging` for details.
            on_epoch: if ``True`` logs epoch accumulated metrics. The default value is determined by the hook.
                See :ref:`extensions/logging:Automatic Logging` for details.
            reduce_fx: reduction function over step values for end of epoch. :meth:`torch.mean` by default.
            enable_graph: if ``True``, will not auto detach the graph.
            sync_dist: if ``True``, reduces the metric across devices. Use with care as this may lead to a significant
                communication overhead.
            sync_dist_group: the DDP group to sync across.
            add_dataloader_idx: if ``True``, appends the index of the current dataloader to
                the name (when using multiple dataloaders). If False, user needs to give unique names for
                each dataloader to not mix the values.
            batch_size: Current batch_size. This will be directly inferred from the loaded batch,
                but for some data structures you might need to explicitly provide it.
            metric_attribute: To restore the metric state, Lightning requires the reference of the
                :class:`torchmetrics.Metric` in your model. This is found automatically if it is a model attribute.
            rank_zero_only: Whether the value will be logged only on rank 0. This will prevent synchronization which
                would produce a deadlock as not all processes would perform this log call.

        """
        if self._fabric is not None:
            self._log_dict_through_fabric(dictionary={name: value}, logger=logger)
            return

        # check for invalid values
        apply_to_collection(value, dict, self.__check_not_nested, name)
        apply_to_collection(
            value, object, self.__check_allowed, name, value, wrong_dtype=(numbers.Number, Metric, Tensor)
        )

        trainer = self._trainer
        if trainer is None:
            # not an error to support testing the `*_step` methods without a `Trainer` reference
            rank_zero_warn(
                "You are trying to `self.log()` but the `self.trainer` reference is not registered on the model yet."
                " This is most likely because the model hasn't been passed to the `Trainer`"
            )
            return
        if trainer.barebones:
            rank_zero_warn(
                "You are trying to `self.log()` but `Trainer(barebones=True)` is configured."
                " Logging can impact raw speed so it is disabled under this setting."
            )
            return
        results = trainer._results
        if results is None:
            raise MisconfigurationException(
                "You are trying to `self.log()` but the loop's result collection is not registered"
                " yet. This is most likely because you are trying to log in a `predict` hook,"
                " but it doesn't support logging"
            )
        if self._current_fx_name is None:
            raise MisconfigurationException(
                "You are trying to `self.log()` but it is not managed by the `Trainer` control flow"
            )

        on_step, on_epoch = _FxValidator.check_logging_and_get_default_levels(
            self._current_fx_name, on_step=on_step, on_epoch=on_epoch
        )

        # make sure user doesn't introduce logic for multi-dataloaders
        if "/dataloader_idx_" in name:
            raise MisconfigurationException(
                f"You called `self.log` with the key `{name}`"
                " but it should not contain information about `dataloader_idx`"
            )
        value = apply_to_collection(value, (Tensor, numbers.Number), self.__to_tensor, name)
        value = apply_to_collection(value, (Tensor, numbers.Number), self.__to_tensor, name)
        if trainer._logger_connector.should_reset_tensors(self._current_fx_name):
            # if we started a new epoch (running its first batch) the hook name has changed
            # reset any tensors for the new hook name
            results.reset(metrics=False, fx=self._current_fx_name)

        if metric_attribute is None and isinstance(value, Metric):
            if self._metric_attributes is None:
                # compute once
                self._metric_attributes = {
                    id(module): name for name, module in self.named_modules() if isinstance(module, Metric)
                }
                if not self._metric_attributes:
                    raise MisconfigurationException(
                        "Could not find the `LightningModule` attribute for the `torchmetrics.Metric` logged."
                        " You can fix this by setting an attribute for the metric in your `LightningModule`."
                    )
            # try to find the passed metric in the LightningModule
            metric_attribute = self._metric_attributes.get(id(value), None)
            if metric_attribute is None:
                raise MisconfigurationException(
                    "Could not find the `LightningModule` attribute for the `torchmetrics.Metric` logged."
                    f" You can fix this by calling `self.log({name}, ..., metric_attribute=name)` where `name` is one"
                    f" of {list(self._metric_attributes.values())}"
                )

        if (
            trainer.training
            and is_param_in_hook_signature(self.training_step, "dataloader_iter", explicit=True)
            and batch_size is None
        ):
            raise MisconfigurationException(
                "With `def training_step(self, dataloader_iter)`, `self.log(..., batch_size=...)` should be provided."
            )

        if logger and trainer.logger is None:
            rank_zero_warn(
                f"You called `self.log({name!r}, ..., logger=True)` but have no logger configured. You can enable one"
                " by doing `Trainer(logger=ALogger(...))`"
            )
        if logger is None:
            # we could set false here if there's no configured logger, however, we still need to compute the "logged"
            # metrics anyway because that's what the evaluation loops use as return value
            logger = True
        val: Union[Metric, Tensor] = apply_to_collection(value, (Tensor, numbers.Number), self.__to_tensor, name)
        results.log(
            self._current_fx_name,
            name,
            val,
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,  # type: ignore[arg-type]
            enable_graph=enable_graph,
            add_dataloader_idx=add_dataloader_idx,
            batch_size=batch_size,
            sync_dist=sync_dist and trainer._accelerator_connector.is_distributed,
            sync_dist_fn=trainer.strategy.reduce,
            sync_dist_group=sync_dist_group,
            metric_attribute=metric_attribute,
            rank_zero_only=rank_zero_only,
        )

        trainer._logger_connector._current_fx = self._current_fx_name

    @staticmethod
    def __check_not_nested(value: dict, name: str) -> None:
        # self-imposed restriction. for simplicity
        if any(isinstance(v, dict) for v in value.values()):
            raise ValueError(f"`self.log({name}, {value})` was called, but nested dictionaries cannot be logged")

    @staticmethod
    def __check_allowed(v: Any, name: str, value: Any) -> None:
        raise ValueError(f"`self.log({name}, {value})` was called, but `{type(v).__name__}` values cannot be logged")
