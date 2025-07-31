import os

import torch
from module_llama_orig import NeuronLlamaLTModule as NeuronLlamaLTModuleOrigin
from lightning.pytorch.trainer.connectors.logger_connector.fx_validator import (
    _FxValidator,
)

from neuronx_distributed.trainer import (
    initialize_parallel_model,
    initialize_parallel_optimizer,
)


class NeuronLlamaLTModule(NeuronLlamaLTModuleOrigin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # For pipeline testing, comparing with golden event
        self.gpu_losses_to_compare = []
        event_file = os.getenv("GOLDEN_EVENT_FILE") if not os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None) else None
        if event_file is not None:
            self.gpu_losses_to_compare = torch.load(os.getenv("GOLDEN_EVENT_FILE"))

    def on_train_batch_end(self, *args, **kwargs):
        # Compare to golden
        if (
            self.trainer.strategy.data_parallel_rank == 0
            and self.trainer.strategy.tensor_parallel_rank == 0
            and self.trainer.strategy.pipeline_parallel_rank == self.trainer.strategy.pipeline_parallel_size - 1
        ):
            print(
                f"step {self.global_step} loss is {self.loss.detach().cpu().item()}, lr is {self.lr}, throughput {self.tps} seq/s, input_ids {torch.sum(self.input_ids.detach().cpu()).item()}"
            )

            step_now = self.global_step - 1
            if step_now < len(self.gpu_losses_to_compare) and step_now >= 0:
                if not torch.allclose(
                    self.loss.cpu().float(), self.gpu_losses_to_compare[step_now].float(), rtol=1.5e-1
                ):
                    raise RuntimeError(
                        f"Loss mismtach with golden, Trn {self.loss.item()} GPU {self.gpu_losses_to_compare[step_now].item()}"
                    )
