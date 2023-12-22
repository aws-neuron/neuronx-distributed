import os
from typing import Any, Callable, Mapping, Optional

from lightning_fabric.utilities.cloud_io import _is_dir
from lightning_fabric.utilities.logger import _add_prefix
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch import Tensor

from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_data_parallel_size,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_rank,
    model_parallel_is_initialized,
)


class NeuronTensorBoardLogger(TensorBoardLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._print_step = -1

    @property
    def print_step(self):
        return self._print_step

    @print_step.setter
    def print_step(self, value: int = -1):
        self._print_step = value

    @property
    def experiment(self) -> "SummaryWriter":
        """Actual tensorboard object. To use TensorBoard features anywhere in your code, do the following.

        Example::

            logger.experiment.some_tensorboard_function()

        """
        """Neuron change, log on the last PP rank"""
        if not self.should_print():
            return _DummyExperiment()
        """End of Neuron change"""

        if self._experiment is not None:
            return self._experiment

        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)

        from torch.utils.tensorboard import SummaryWriter

        self._experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._experiment

    def save(self) -> None:
        """Neuron change, log on the last PP rank"""
        if not self.should_print():
            return
        """End of Neuron change"""

        super(TensorBoardLogger, self).save()
        dir_path = self.log_dir

        # prepare the file path
        hparams_file = os.path.join(dir_path, self.NAME_HPARAMS_FILE)

        # save the metatags file if it doesn't exist and the log directory exists
        if _is_dir(self._fs, dir_path) and not self._fs.isfile(hparams_file):
            save_hparams_to_yaml(hparams_file, self.hparams)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        """Neuron change, log on the last PP rank"""
        if not self.should_print():
            return
        """End of Neuron change"""

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        step = self.print_step
        for k, v in metrics.items():
            if isinstance(v, Tensor):
                v = v.item()

            if isinstance(v, dict):
                self.experiment.add_scalars(k, v, step)
            else:
                try:
                    self.experiment.add_scalar(k, v, step)
                # TODO(fabric): specify the possible exception
                except Exception as ex:
                    raise ValueError(
                        f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
                    ) from ex

    def log_graph(  # type: ignore[override]
        self, model: "pl.LightningModule", input_array: Optional[Tensor] = None
    ) -> None:
        """Neuron change, log on the last PP rank"""
        if not self.should_print():
            return
        """End of Neuron change"""

        if not self._log_graph:
            return

        input_array = model.example_input_array if input_array is None else input_array

        if input_array is None:
            rank_zero_warn(
                "Could not log computational graph to TensorBoard: The `model.example_input_array` attribute"
                " is not set or `input_array` was not given."
            )
        elif not isinstance(input_array, (Tensor, tuple)):
            rank_zero_warn(
                "Could not log computational graph to TensorBoard: The `input_array` or `model.example_input_array`"
                f" has type {type(input_array)} which can't be traced by TensorBoard. Make the input array a tuple"
                f" representing the positional arguments to the model's `forward()` implementation."
            )
        else:
            input_array = model._on_before_batch_transfer(input_array)
            input_array = model._apply_batch_transfer_handler(input_array)
            with pl.core.module._jit_is_scripting():
                self.experiment.add_graph(model, input_array)

    def should_print(self):
        # For NxD we should log on the last PP
        assert model_parallel_is_initialized(), f"NxD model parallel not initialized"
        return (
            get_data_parallel_rank() == 0
            and get_tensor_model_parallel_rank() == 0
            and get_pipeline_model_parallel_rank() == get_pipeline_model_parallel_size() - 1
            and self.print_step >= 0
        )


class _DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args: Any, **kw: Any) -> None:
        pass

    def __getattr__(self, _: Any) -> Callable:
        return self.nop

    def __getitem__(self, idx: int) -> "_DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args: Any, **kwargs: Any) -> None:
        pass
