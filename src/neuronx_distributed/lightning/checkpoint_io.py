import os
from typing import Any, Dict, Optional

from lightning.fabric.plugins.io import XLACheckpointIO 
from lightning.fabric.utilities.cloud_io import get_filesystem 
from lightning.fabric.utilities.types import _PATH 
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.imports import RequirementCache 

from neuronx_distributed.parallel_layers.checkpointing import load, save


class NeuronCheckpointIO(XLACheckpointIO):
    def __init__(self, save_load_xser: bool = True, weights_only: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_load_xser = save_load_xser
        self.weights_only = weights_only

    def load_checkpoint(
        self,
        checkpoint_path: _PATH,
        master_dp_only: bool = True,
    ) -> Dict[str, Any]:
        return load(
            chkpt_path=checkpoint_path,
            load_xser=self.save_load_xser,
            master_dp_only=master_dp_only,
            weights_only=self.weights_only,
        )

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
        master_dp_only: bool = True,
    ) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``XLACheckpointIO.save_checkpoint``

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in

        """
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO`"
                " to define how you'd like to use `storage_options`."
            )
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        if RequirementCache("omegaconf"):
            # workaround for https://github.com/pytorch/xla/issues/2773
            from omegaconf import DictConfig, ListConfig, OmegaConf

            checkpoint = apply_to_collection(checkpoint, (DictConfig, ListConfig), OmegaConf.to_container)

        save(checkpoint=checkpoint, output_dir=path, save_xser=self.save_load_xser, master_dp_only=master_dp_only)
