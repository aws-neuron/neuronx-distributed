import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from neuronx_distributed.parallel_layers.parallel_state import (
    get_pipeline_model_parallel_size,
)


class NeuronTQDMProgressBar(TQDMProgressBar):
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self._trainer = trainer
        print_pp_rank = 0 if pl_module.log_rank0 else get_pipeline_model_parallel_size() - 1
        # For NxD we should log on the last PP rank
        if not (
            trainer.strategy.data_parallel_rank == 0
            and trainer.strategy.tensor_parallel_rank == 0
            and trainer.strategy.pipeline_parallel_rank == print_pp_rank
        ):
            self.disable()
