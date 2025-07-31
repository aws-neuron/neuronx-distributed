from typing import cast
import lightning.pytorch as pl

from neuronx_distributed.parallel_layers.parallel_state import (
    get_pipeline_model_parallel_size,
)
from neuronx_distributed.lightning.strategy import NeuronXLAStrategy


class NeuronTQDMProgressBar(pl.callbacks.TQDMProgressBar):
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self._trainer = trainer
        assert isinstance(trainer.strategy, NeuronXLAStrategy)
        strategy: NeuronXLAStrategy = cast(NeuronXLAStrategy, trainer.strategy)
        print_pp_rank = 0 if pl_module.log_rank0 else get_pipeline_model_parallel_size() - 1
        # For NxD we should log on the last PP rank
        if not (
            strategy.data_parallel_rank == 0
            and strategy.tensor_parallel_rank == 0
            and strategy.pipeline_parallel_rank == print_pp_rank
        ):
            self.disable()
