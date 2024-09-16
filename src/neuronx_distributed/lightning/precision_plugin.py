from typing import Any, Callable, TYPE_CHECKING

from lightning_fabric.accelerators.xla import _XLA_AVAILABLE
from lightning_fabric.utilities.types import Optimizable
from pytorch_lightning.plugins.precision import XLAPrecisionPlugin

if TYPE_CHECKING:
    import pytorch_lightning as pl


class NeuronXLAPrecisionPlugin(XLAPrecisionPlugin):
    def __init__(self, mixed_precision_enabled: bool = False) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))

        self.mixed_precision_enabled = mixed_precision_enabled

    def optimizer_step(  # type: ignore[override]
        self,
        optimizer: Optimizable,
        model: "pl.LightningModule",
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        # TODO: currently using manual optimization, need further modification here for auto optimization
        optimizer.step()
