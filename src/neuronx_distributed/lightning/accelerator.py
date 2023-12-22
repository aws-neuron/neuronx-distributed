import functools
from typing import Any, Dict, List, Union

import torch
from pytorch_lightning.accelerators import XLAAccelerator
from lightning_fabric.accelerators.xla import _XLA_AVAILABLE, _using_pjrt, _parse_tpu_devices_str
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
from lightning_fabric.utilities.device_parser import _check_data_type

class NeuronXLAAccelerator(XLAAccelerator):
    """
    Neuron overrided XLAAccelerator
    parse_devices(), get_parallel_devices() are directly copied from XLAAccelerator
    since they have call to _check_tpu_devices_valid() method which we're overriding
    """
    
    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Union[int, List[int]]:
        """Accelerator device parsing logic."""
        return _parse_tpu_devices(devices)
    
    @staticmethod
    def get_parallel_devices(devices: Union[int, List[int]]) -> List[torch.device]:
        """Gets parallel devices for the Accelerator."""
        devices = _parse_tpu_devices(devices)
        # In XLA XRT index 0 maps to CPU, in fact, a `xla_device()` with no arguments has index 1
        # since the user passes a 0-based index, we need to adjust the indices
        device_offset = 0 if _using_pjrt() else 1

        if isinstance(devices, int):
            return [torch.device("xla", i) for i in range(device_offset, devices + device_offset)]
        # list of devices is not supported, just a specific index, fine to access [0]
        return [torch.device("xla", devices[0] + device_offset)]
        # we cannot create `xla_device` here because processes have not been spawned yet (this is called in the
        # accelerator connector init). However, there doesn't seem to be a problem with instantiating `torch.device`.
        # it will be replaced with `xla_device` (also a torch.device`, but with extra logic) in the strategy


    

    @staticmethod
    # XLA's multiprocessing will pop the TPU_NUM_DEVICES key, so we need to cache it
    # https://github.com/pytorch/xla/blob/v2.0.0/torch_xla/distributed/xla_multiprocessing.py#L280
    @functools.lru_cache(maxsize=1)
    def auto_device_count() -> int:
        """
        Overriding since we don't have tpu.version() in place https://github.com/Lightning-AI/pytorch-lightning/blob/2.1.1/src/lightning/fabric/accelerators/xla.py#L83
        """
        if not _XLA_AVAILABLE:
            return 0
        
        import torch_xla.core.xla_env_vars as xenv
        from torch_xla.utils.utils import getenv_as
        
        return getenv_as(xenv.TPU_NUM_DEVICES, int, 8)

    
    @staticmethod
    @functools.lru_cache(maxsize=1)
    def is_available() -> bool:
        try:
            return NeuronXLAAccelerator.auto_device_count() > 0
        except (ValueError, AssertionError, OSError):
            # XLA may raise these exceptions if it's not properly configured. This needs to be avoided for the cases
            # when `torch_xla` is imported but not used
            return False
    


    @classmethod
    def register_accelerators(cls, accelerator_registry: _AcceleratorRegistry) -> None:
        accelerator_registry.register("neuron", cls, description=cls.__name__)


def _parse_tpu_devices(devices: Union[int, str, List[int]]) -> Union[int, List[int]]:
    """
    Directly copied from PTL code, to let it call _check_tpu_devices_valid() method which we're overriding
    """
    _check_data_type(devices)
    if isinstance(devices, str):
        devices = _parse_tpu_devices_str(devices)
    _check_tpu_devices_valid(devices)
    return devices

def _check_tpu_devices_valid(devices: object) -> None:
    # Changing XLAAccelerator to NeuronXLAAccelerator
    device_count = NeuronXLAAccelerator.auto_device_count()
    if (
        # support number of devices
        isinstance(devices, int)
        and devices in {1, device_count}
        # support picking a specific device
        or isinstance(devices, (list, tuple))
        and len(devices) == 1
        and 0 <= devices[0] <= device_count - 1
    ):
        return
    raise ValueError(
        f"`devices` can only be 'auto', 1, {device_count} or [<0-{device_count - 1}>] for TPUs. Got {devices!r}"
    )