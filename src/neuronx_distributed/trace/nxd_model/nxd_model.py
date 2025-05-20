from typing import Any, Optional, List, Dict, Tuple, Union

import torch
import torch_neuronx
from torch_neuronx.xla_impl import structure
from torch_neuronx.xla_impl.trace import get_torch_dtype
from torch_neuronx.proto import metaneff_pb2
from torch_neuronx.pyhlo import hlo_pb2

from .base_nxd_model import BaseNxDModel, StateInitializer


class NxDModel(torch.nn.Module, BaseNxDModel):
    def __init__(
        self,
        world_size: int,
        start_rank: int = 0,
        local_ranks_size: Optional[int] = None,
        state_initializer: Optional[StateInitializer] = None,
        layout_transformer: Optional[torch.classes.neuron.LayoutTransformation] = None
    ):
        torch.nn.Module.__init__(self)
        self.world_size = world_size
        self.start_rank = start_rank
        self.local_ranks_size = world_size if local_ranks_size is None else local_ranks_size
        self.layout_transformer = layout_transformer

        self.ordered_kwargs: List[str] = []

        self.input_shape_map: Dict[str, str] = {}
        self.hlo_map: Dict[str, hlo_pb2.HloModule] = {}
        self.spmd_models: Dict[str, torch.classes.neuron.SPMDModel] = {}
        self.flattener_map: Dict[str, torch.jit.ScriptModule] = {}
        self.packer_map: Dict[str, torch.jit.ScriptModule] = {}

        self.weights: List[Dict[str, torch.Tensor]] = []
        self.states: List[Dict[str, torch.Tensor]] = []
        self.state_initializer = state_initializer

        self.loaded_on_neuron = False

        self.torchscript_model = None

    def add(self, key: str, hlo, neff, metaneff, example_inputs,
            flattener: structure.Flattener, packer: structure.Packer) -> "NxDModel":

        raise NotImplementedError

    def get_available_keys(self) -> List[str]:
        raise NotImplementedError

    def get_hlo(self, key: str) -> hlo_pb2.HloModuleProto:
        raise NotImplementedError

    def get_neff(self, key: str) -> bytes:
        raise NotImplementedError

    def get_metaneff(self, key: str) -> metaneff_pb2.MetaNeff:
        raise NotImplementedError

    @torch.jit.export
    def set_weights(self, sharded_checkpoint: List[Dict[str, torch.Tensor]]):
        raise NotImplementedError

    @torch.jit.export
    def to_neuron(self):
        raise NotImplementedError

    @torch.jit.export
    def replace_weights(self, sharded_checkpoint: List[Dict[str, torch.Tensor]]):
        raise NotImplementedError

    @torch.jit.export
    def read_from_neuron_buffer(self, state_buffer_key: str) -> torch.Tensor:
        raise NotImplementedError

    @torch.jit.export
    def write_to_neuron_buffer(self, tensor: torch.Tensor, state_buffer_key: str):
        raise NotImplementedError

    def convert_dict_to_ordered_list(self, inputs: Dict[str, torch.Tensor], num_pos_args: int = -1) -> List[torch.Tensor]:
        """
        NOTE: This must be torchscriptable
        """
        raise NotImplementedError

    @torch.jit.export
    def router(self, inputs: List[torch.Tensor]) -> str:
        raise NotImplementedError

    def _torchscript_forward(  # type: ignore
        self,
        args: Any,
        kwargs: Any
    ) -> Any:
        """
        The forward method ran if serialized, this must be compatible
        with torch.jit.script
        """

        raise NotImplementedError

    def forward(
        self,
        *args,
        model_name=None,
        **kwargs
    ):
        """
        The forward method of the NxDModel class, which will take in inputs and run the respective neff.

        Parameters:
        The tensors whose properties match what was used for tracing.

        Returns:
        Expected format of tensor outputs based on what was originally traced
        """
        raise NotImplementedError

    def save(self, path_to_save: str, save_weights: bool = False):
        raise NotImplementedError

    @classmethod
    def load(cls, path_to_model: str, start_rank: int = 0) -> torch.jit.ScriptModule:
        raise NotImplementedError

class TorchScriptNxDModel(torch.nn.Module):
    """
    Wrapper for torchscripted NxDModel
    """

    def __init__(self, nxd_model: NxDModel):
        super().__init__()
        self.nxd_model = nxd_model

    def forward(  # type: ignore
            self,
            args: Any,
            kwargs: Any,
        ):
        raise NotImplementedError

def convert_nxd_model_to_torchscript_model(nxd_model: "NxDModel", save_weights: bool = False) -> torch.jit.ScriptModule:
    """
    Converts an NxDModel to a TorchScriptNxDModel object which can
    be Torchscripted with torch.jit.script.
    """

    raise NotImplementedError
