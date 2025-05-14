from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple

import torch
from torch_neuronx.proto import metaneff_pb2
from torch_neuronx.pyhlo import hlo_pb2
from torch_neuronx.xla_impl import structure

from neuronx_distributed.parallel_layers.utils import is_torch_version_greater_than_2

class StateInitializer(torch.nn.Module):
    """
    A module to initialize state buffers onto Neuron
    """
    # torchscript cannot script dict of with values of different types
    # so we store shapes and dtypes in separate dicts
    def __init__(self, shapes, dtypes, local_ranks_size):
        super().__init__()
        self.shapes = shapes
        self.dtypes = dtypes
        self.local_ranks_size = local_ranks_size

    def forward(self):
        results: List[Dict[str, torch.Tensor]] = []
        for rank in range(0, self.local_ranks_size):
            states = {}
            for key in self.shapes.keys():
                if is_torch_version_greater_than_2() and self.dtypes[key] == torch.float8_e4m3fn:
                    states[key] = torch.zeros(self.shapes[key], dtype=torch.bfloat16).to(dtype=self.dtypes[key]).to(device=f"privateuseone:{rank}")
                else:
                    states[key] = torch.zeros(self.shapes[key], dtype=self.dtypes[key], device=f"privateuseone:{rank}")
            results.append(states)

        return results

class BaseNxDModel(ABC):
    """
    The Python Wrapper for neffs compiled with the NxD ModelBuilder & tracing API
    """
    @abstractmethod
    def __init__(
        self,
        world_size: int,
        start_rank: int = 0,
        state_initializer: Optional[StateInitializer] = None,
        layout_transformer: Optional[torch.classes.neuron.LayoutTransformation] = None
    ):
        """
        Instantiates the NxDModel object which is used to run neffs compiled with the ModelBuilder.

        Parameters:
        1) world_size: int - This specifies the world size for distributed model.
        2) start_rank_tensor: Optional[int] - This specifies the start rank tensor for a multi node distributed model. The default is 0.
        3) state_initializer: StateInitializer - This is an optional parameter which is used to specify a module that initializes state buffers on Neuron.
        4) layout_transformer: LayoutTransformationModel - This is an optional parameter which is used for transforming weights from their original layout to the optimized layout. The default is None which means that the weights will be using their original layout.
        """
        pass

    @abstractmethod
    def add(self, key: str, hlo, neff, metaneff, example_inputs,
            flattener: structure.Flattener, packer: structure.Packer):
        """
        Adds a neff to the NxDModel. If a layout_transformer is available in the NxDModel, add can fail if it is identified that the neff was compiled with a different weight layout than what layout_transformer was made for.

        Parameters:
        1) key: str - This is used for identifying the neff.
        2) neff: Union[NeffArtifacts, str] - This is either the NeffArtifacts object returned by the ModelBuilder.compile_to_neff function, or a path to a .neff file
        3) metaneff: Union[metaneff_pb2.MetaNeff, str] - This is either the metaneff proto object found in the HloArtifacts object, or a path to the metaneff.pb file associated with the respective neff.
        4) flattener: Optional[torch.jit.ScriptModule] - This is a parameter to specify a flattener function. This converts the inputs to a 1d flattened list, and is also responsible for removing unused inputs in the neff. This is returned by the spmd_trace() function as part of the TraceArtifacts object.
        5) packer: Optional[torch.jit.ScriptModule] - This is a parameter to specify a packer function. This converts the outputs from a flattened list to the original representation from the traced model, and is also responsible for removing unused inputs in the neff. This is returned by the spmd_trace() function as part of the TraceArtifacts object.
        """
        pass

    @abstractmethod
    def get_available_keys(self) -> List[str]:
        """
        Gets all the available keys which are associated with a hlo/neff
        """
        pass

    @abstractmethod
    def get_hlo(self, key: str) -> hlo_pb2.HloModuleProto:
        """
        Returns the hlo proto object
        """
        pass

    @abstractmethod
    def get_neff(self, key: str) -> bytes:
        """
        Returns the neff bytes
        """
        pass

    @abstractmethod
    def get_metaneff(self, key: str) -> metaneff_pb2.MetaNeff:
        """
        Returns the metaneff proto object
        """
        pass

    @abstractmethod
    def set_weights(self, sharded_checkpoint: List[Dict[str, torch.Tensor]]):
        """
        Loads the provided weights in the Neuron Model. This must be called if the model was compiled with  initialize_model_weights=False . If weights were already associated with the model, and new ones were supplied, this function will replace the existing weights in the NxDModel.

        Parameters:
        1) sharded_checkpoint: List[Dict[str, torch.Tensor]]- This is a loaded sharded checkpoint. Each dictionary in the list represents the weights for the respective rank.
        """
        pass

    @abstractmethod
    def to_neuron(self):
        """
        Initializes the model with collectives and performs a one time transformation of the weight’s layout if using the layout transformer. This only needs to be called once before performing inferences with NxDModel
        """
        pass

    @abstractmethod
    def replace_weights(self, sharded_checkpoint: List[Dict[str, torch.Tensor]]):
        """
        Replaces the already loaded weights inn a Neuron Model.

        Parameters:
        1) sharded_checkpoint: List[Dict[str, torch.Tensor]] - This is a loaded sharded checkpoint. Each dictionary in the list represents the weights for the respective rank.
        """
        pass

    @abstractmethod
    def read_from_neuron_buffer(self, state_buffer_key: str) -> torch.Tensor:
        """
        Reads the data from the specified state buffer key allocated on Neuron HBM to a CPU torch tensor

        Parameters:
        1) state_buffer_key: str - the key corresponding to the original torch parameter/buffer
        """
        pass

    @abstractmethod
    def write_to_neuron_buffer(self, tensor: torch.Tensor, state_buffer_key: str):
        """
        Reads the data from the specified state buffer key allocated on Neuron HBM to a CPU torch tensor

        Parameters:
        1) tensor: torch.Tensor - the tensor containing the data to write to the buffer.
            Its dtype and shape must match the allocated Neuron buffer,
        2) state_buffer_key: str - the key corresponding to the original torch parameter/buffer
        """
        pass

    @abstractmethod
    def router(self, inputs: List[torch.Tensor]) -> str:
        """
        Returns a model key based on properties of the inputs
        """
        pass

    @abstractmethod
    def save(self, path_to_save: str, save_weights: bool = False):
        """
        Serializes the model to an .pt file which can be loaded with `NxDModel.load` or `torch.jit.load` (`NxDModel.load` is preferred).
        This is accomplished by converting the NxDModel class to a TorchScriptNxDModel class, which is then converted to TorchScript.

        WARNING: Serialization will deactivate certain features and make the loaded object opaque to Python, due to Torchscript conversion.

        Parameters:
        1) path_to_save: str - Path to save NxDModel.
        2) save_weights: bool - Optional boolean to indicate if the sharded weights should be saved with the model. The default is False, which means that both NxDModel.load_from_file and set_weights must both be called to run the NxDModel.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path_to_model: str, start_rank: int = 0):
        """
        A classmethod which initializes an NxDModel object from a .pt file.

        Parameters:
        1) path_to_model: str - Path to save NxDModel.
        2) start_rank_tensor: Optional[int] - This specifies the start rank tensor for a multi node distributed model. The default is 0.

        Returns:
        NxDModel
        """
        pass
