import os
from typing import Any, Dict, Optional, Union

import torch
from lightning_fabric.plugins.environments import (
    TorchElasticEnvironment,
    XLAEnvironment,
)
from lightning_fabric.utilities.types import _PATH, ReduceOp
from pytorch_lightning.strategies import XLAStrategy
from torch import Tensor

from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_data_parallel_size,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from .accelerator import NeuronXLAAccelerator
from .checkpoint_io import NeuronCheckpointIO
from .launcher import _NeuronXLALauncher


class NeuronXLAStrategy(XLAStrategy):
    def __init__(
        self,
        nxd_config: Dict = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        debug: bool = False,
        sync_module_states: bool = False,
        checkpoint_io: bool = None,
        save_load_xser: bool = True,
    ):
        if os.environ.get("TORCHELASTIC_RUN_ID") is not None:
            cluster_environment = TorchElasticEnvironment()
        else:
            cluster_environment = XLAEnvironment()

        super(XLAStrategy, self).__init__(
            accelerator=NeuronXLAAccelerator(),
            cluster_environment=cluster_environment,
            debug=debug,
        )

        if not checkpoint_io:
            self.checkpoint_io = NeuronCheckpointIO(save_load_xser=save_load_xser)
        elif isinstance(checkpoint_io, NeuronCheckpointIO):
            self.checkpoint_io = checkpoint_io
        else:
            raise NotImplementedError(f"NeuronXLAStrategy only supports NeuronCheckpointIO")

        self.debug = debug
        self._launched = False
        self._sync_module_states = sync_module_states

        self.nxd_config = nxd_config

        if self.nxd_config is not None:
            self.tensor_parallel_size = self.nxd_config["tensor_parallel_size"]
            self.pipeline_parallel_size = self.nxd_config["pipeline_parallel_size"]
        else:
            self.tensor_parallel_size = tensor_parallel_size
            self.pipeline_parallel_size = pipeline_parallel_size

    def _configure_launcher(self) -> None:
        self._launcher = _NeuronXLALauncher(self)

    def broadcast(self, obj, src: int = 0):
        return obj

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, int]:
        return {"num_replicas": get_data_parallel_size(), "rank": get_data_parallel_rank()}

    def setup_distributed(self) -> None:
        import torch.distributed as dist

        if self.cluster_environment.creates_processes_externally:
            global_rank = int(os.environ.get("RANK"))
        else:
            import torch_xla.core.xla_model as xm

            global_rank = xm.get_ordinal()
        if torch.__version__.startswith("2.0"):
            import torch_xla.experimental.pjrt_backend  # noqa

            dist.init_process_group("xla", init_method="pjrt://", rank=global_rank)
        else:
            dist.init_process_group("xla", rank=global_rank)
        super().setup_distributed()

        # init model parallel if needed
        if not model_parallel_is_initialized():
            initialize_model_parallel(
                tensor_model_parallel_size=self.tensor_parallel_size,
                pipeline_model_parallel_size=self.pipeline_parallel_size,
            )

        self.data_parallel_rank = get_data_parallel_rank()
        self.data_parallel_size = get_data_parallel_size()
        self.tensor_parallel_rank = get_tensor_model_parallel_rank()
        self.pipeline_parallel_rank = get_pipeline_model_parallel_rank()

    def teardown(self):
        assert self.cluster_environment is not None
        self.cluster_environment.teardown()
        self.precision_plugin.teardown()
        assert self.accelerator is not None
        self.accelerator.teardown()
        self.checkpoint_io.teardown()

    def model_to_device(self) -> None:
        # PTL will call model_to_device during setup
        pass

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0) -> Any:
        """Moves the batch to the correct device.

        The returned batch is of the same type as the input batch, just
        having all tensors on the correct device.

        Args:
            batch: The batch of samples to move to the correct device
            device: The target device
            dataloader_idx: The index of the dataloader to which the batch belongs.

        """

        model = self.lightning_module
        device = device or self.root_device

        # For PP we're not moving to cpu
        if self.pipeline_parallel_size > 1:
            device = torch.device("cpu")

        if model is not None:
            return model._apply_batch_transfer_handler(batch, device=device, dataloader_idx=dataloader_idx)
        return move_data_to_device(batch, device)

    def reduce(
        self, output: Union[Tensor, Any], group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None
    ) -> Tensor:
        if not isinstance(output, Tensor):
            output = torch.tensor(output, device=self.root_device)

        invalid_reduce_op = isinstance(reduce_op, ReduceOp) and reduce_op != ReduceOp.SUM
        invalid_reduce_op_str = isinstance(reduce_op, str) and reduce_op.lower() not in ("sum", "mean", "avg")
        if invalid_reduce_op or invalid_reduce_op_str:
            raise ValueError(
                "Currently, the XLAStrategy only supports `sum`, `mean`, `avg` for the reduce operation, got:"
                f" {reduce_op}"
            )

        # Replace xm.mesh_reduce since it will cause error in PT2.x
        import torch_xla.core.xla_model as xm

        xm.mark_step()

        torch.distributed.all_reduce(output, op=torch.distributed.ReduceOp.SUM)
        if isinstance(reduce_op, str) and reduce_op.lower() in ("avg", "mean"):
            output = output / self.world_size

        xm.mark_step()
        return output.cpu()

    def process_dataloader(self, dataloader: object):
        from torch_xla.distributed.parallel_loader import MpDeviceLoader

        # PP requires input data on CPU
        if self.pipeline_parallel_size > 1:
            if isinstance(dataloader, MpDeviceLoader):
                print(f"convertine dataloader {dataloader} to {dataloader._loader}")
                return dataloader._loader
            return dataloader

        if isinstance(dataloader, MpDeviceLoader):
            # dataloader is already wrapped by MpDeviceLoader
            return dataloader

        dataloader = MpDeviceLoader(dataloader, self.root_device)
        # Mimic interface to torch.utils.data.DataLoader
        dataloader.dataset = dataloader._loader.dataset
        dataloader.batch_sampler = getattr(dataloader._loader, "batch_sampler", None)
        return dataloader

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None
    ) -> None:
        # When zero1 enabled, save to all dp rank. TODO: Optimize by saving model to dp rank 0 and opt to all ranks
        self.checkpoint_io.save_checkpoint(
            checkpoint,
            filepath,
            storage_options=storage_options,
            master_dp_only=not self.nxd_config["optimizer_config"]["zero_one_enabled"],
        )

    def load_checkpoint(self, checkpoint_path: _PATH) -> Dict[str, Any]:
        # torch.cuda.empty_cache()
        return self.checkpoint_io.load_checkpoint(
            checkpoint_path,
            master_dp_only=not self.nxd_config["optimizer_config"]["zero_one_enabled"],
        )
