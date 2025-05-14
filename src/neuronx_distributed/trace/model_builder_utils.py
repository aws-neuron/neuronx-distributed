from typing import Any, Dict, Optional
import os
import hashlib

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
    """
    def __init__(
        self,
        hlo: Any,
        metaneff: Any,
        flattener: Any,
        packer: Any,
        weight_name_to_idx: Dict[str, int]
    ):
        self.hlo = hlo
        self.metaneff = metaneff
        self.flattener = flattener
        self.packer = packer
        self.weight_name_to_idx = weight_name_to_idx


class CompilationArtifacts:
    """
    A class containing compilation artifacts.
    
    Attributes:
        neff_filename (str): The filename of the compiled Neuron Executable File Format (NEFF).
    """
    def __init__(
        self,
        neff_filename: str
    ):
        self.neff_filename = neff_filename


class WLOArtifacts:
    """
    A class containing Weight Layout Optimized (WLO) compilation artifacts.
    
    Attributes:
        neff_filename (str): The filename of the compiled Neuron Executable File Format (NEFF).
        wrapped_neff_hlo_filename (str): The filename of the wrapped NEFF HLO.
    """
    def __init__(
        self,
        neff_filename: str,
        wrapped_neff_hlo_filename: str
    ):
        self.neff_filename = neff_filename
        self.wrapped_neff_hlo_filename = wrapped_neff_hlo_filename


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
