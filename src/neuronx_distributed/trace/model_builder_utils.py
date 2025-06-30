from typing import Any, Dict, Optional, List, Tuple, Set
import os
import hashlib
import torch
from dataclasses import dataclass

from torch_neuronx.pyhlo import hlo_pb2

# Constants
class ModelBuilderConstants:
    """
    A class containing constants used in the model builder.
    """
    NEFF_FILE = "graph.neff"
    WRAPPED_NEFF_FILE = "wrapped_neff.hlo"
    GRAPH_HLO_FILE = "graph.hlo"
    LOG_FILE_DEFAULT_NAME = "log-neuron-cc.txt"
    METANEFF_FILE = "metaneff.pb"
    DEFAULT_COMPILER_WORKDIR = "/tmp/nxd_models"
    LAYOUT_TRANSFORMER_KEY = "layout_transformer"
    DEFAULT_WORLD_SIZE = 1


@dataclass
class ModelParamInfo:
    """
    Information about a parameter in the model's function signature.

    Attributes:
        param_name: Name of the parameter in the function signature.
        is_positional: Whether this parameter is positional (required) or keyword (optional).
    """
    param_name: str
    is_positional: bool


@dataclass
class ProvidedArgInfo:
    """
    Information about an argument provided to the model during tracing.

    Attributes:
        param_name: Name of the parameter this argument corresponds to.
        is_positional: Whether this argument is positional (required) or keyword (optional).
        tensor: The tensor value provided for this argument.
    """
    param_name: str
    is_positional: bool
    tensor: torch.Tensor


# Artifacts
class TraceArtifacts:
    """
    A class containing trace artifacts.

    Attributes:
        hlo: HLO representation.
        metaneff: The meta information for the Neuron Executable File Format (NEFF).
        flattener: The function to flatten inputs.
        packer: The function to pack outputs.
        weight_name_to_idx: A dictionary mapping weight names to their indices.
        weight_names_to_skip: A set of weight names to exclude during weight layout optimization.
        provided_args: List of ProvidedArgInfo for all provided args.
        model_params: List of ModelParamInfo for all parameters in the model's function signature.
    """
    def __init__(
        self,
        hlo: Any,
        metaneff: Any,
        flattener: Any,
        packer: Any,
        weight_name_to_idx: Dict[str, int],
        weight_names_to_skip: Set,
        provided_args: List[ProvidedArgInfo],
        model_params: List[ModelParamInfo],
    ):
        self.hlo = hlo
        self.metaneff = metaneff
        self.flattener = flattener
        self.packer = packer
        self.weight_name_to_idx = weight_name_to_idx
        self.weight_names_to_skip = weight_names_to_skip
        self.provided_args = provided_args
        self.model_params = model_params


class CompilationArtifacts:
    """
    A class containing compilation artifacts.

    Attributes:
        neff_filepath (str): The filepath of the compiled Neuron Executable File Format (NEFF).
    """
    def __init__(
        self,
        neff_filepath: str
    ):
        self.neff_filepath = neff_filepath


    def get_neff_bytes(self) -> bytes:
        """
        Read the neff bytes from the provided neff_filename
        """
        with open(self.neff_filepath, 'rb') as f:
            return f.read()


class WLOArtifacts(CompilationArtifacts):
    """
    A class containing Weight Layout Optimized (WLO) compilation artifacts.

    Attributes:
        neff_filepath (str): The filepath of the compiled Neuron Executable File Format (NEFF).
        wrapped_neff_hlo_filepath (Optional[str]): The filepath of the wrapped NEFF HLO. Can be None.
    """
    def __init__(
        self,
        neff_filepath: str,
        wrapped_neff_hlo_filepath: Optional[str] = None
    ):
        super().__init__(neff_filepath)
        self.wrapped_neff_hlo_filepath = wrapped_neff_hlo_filepath


class LayoutTransformerArtifacts:
    """
    A class containing layout transformation artifacts.

    Attributes:
        hlo_filepath (str): The filepath of the HLO (High-Level Optimization) file.
        neff_filepath (str): The filepath of the compiled Neuron Executable File Format (NEFF).
        metaneff_filepath (str): The filepath of metaneff file.
    """
    def __init__(
        self,
        hlo_filepath: str,
        neff_filepath: str,
        metaneff_filepath: str,
    ):
        self.hlo_filepath = hlo_filepath
        self.neff_filepath = neff_filepath
        self.metaneff_filepath = metaneff_filepath

    def construct_layout_transformer_object(self, local_ranks_size):
        with open(self.neff_filepath, 'rb') as f:
            neff = f.read()
        with open(self.metaneff_filepath, 'rb') as f:
            metaneff = f.read()

        return torch.classes.neuron.LayoutTransformation(
            neff,
            metaneff,
            local_ranks_size
        )


def generate_key(
    hlo_module: hlo_pb2.HloModuleProto,
    key: Optional[str] = None
) -> str:
    """
    Generate a unique key for the HLO module.

    Args:
        hlo_module: The HLO module to generate a key for
        key: Optional pre-defined key. If None, a key will be generated based on the HLO hash

    Returns:
        str: The key to use for the HLO module
    """
    # Use the provided key if available
    if key is not None:
        return key

    serialized_hlo = hlo_module.SerializeToString()
    hlo_hash = hashlib.sha256(serialized_hlo).hexdigest()[:8]
    return f"model_{hlo_hash}"
