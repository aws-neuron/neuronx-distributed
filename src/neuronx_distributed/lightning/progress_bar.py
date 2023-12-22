import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar


class NeuronTQDMProgressBar(TQDMProgressBar):
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self._trainer = trainer

        # For NxD we should log on the last PP rank
        if not (
            trainer.strategy.data_parallel_rank == 0
            and trainer.strategy.tensor_parallel_rank == 0
            and trainer.strategy.pipeline_parallel_rank == trainer.strategy.pipeline_parallel_size - 1
        ):
            self.disable()
