import tempfile
from typing import Dict, List, Tuple, Union
import torch
import torch_neuronx

from neuronx_distributed.trace.model_builder_utils import ProvidedArgInfo

# All available torch dtypes
TORCH_DTYPES = [getattr(torch, attr) for attr in dir(torch) if isinstance(getattr(torch, attr), torch.dtype)]

@torch.jit.script
def get_dtype_enum(dtype: torch.dtype) -> int:
    """
    Get the TorchScript enum value for a torch.dtype.

    Args:
        dtype: A torch.dtype object

    Returns:
        The corresponding integer enum value
    """
    return dtype  # TorchScript automatically converts this to the enum value

def get_dtype_from_enum(dtype_enum: int) -> torch.dtype:
    """
    Get the torch.dtype corresponding to an enum value.

    Args:
        dtype_enum: An integer representing a torch.dtype enum value

    Returns:
        The corresponding torch.dtype object
    """
    for dtype in TORCH_DTYPES:
        if get_dtype_enum(dtype) == dtype_enum:
            return dtype

    raise ValueError(f"Could not find torch.dtype corresponding to enum value: {dtype_enum}")


def retrieve_artifact_from_model(
    model: torch.classes.neuron.SPMDModel,
    artifact: str) -> Union[None, bytes]:
    """
    Retrieves the specified artifact (neff/metaneff)

    This function performs the following steps:
    1. Saves artifact bytes to a tempfile from provided model
    2. Reads the contents of the file
    3. Removes the tempfile
    4. Returns the read contents

    Args:
        model (torch.classes.neuron.SPMDModel): The SPMD model object
        artifact (str): Artifact to retrieve. Must be one of ["neff", "metaneff"]

    Returns:
        Union[None, bytes]: The artifact as bytes if successful, None if the method call fails
    """
    ARTIFACT_TO_METHOD = {
        "neff": "save_neff",
        "metaneff": "save_metaneff"
    }
    assert artifact in ARTIFACT_TO_METHOD, f"{artifact=} is not valid. Choose from {ARTIFACT_TO_METHOD.keys()}"
    method = ARTIFACT_TO_METHOD[artifact]

    # we save the neff/metaneff and read it back because
    # they can't be encoded as utf8 and python strings are utf8
    # while c++ strings have no specific encoding.
    with tempfile.NamedTemporaryFile(suffix=f'.{artifact}') as neff_file:
        successful = getattr(model, method)(neff_file.name)
        if not successful:
            return None
        with open(neff_file.name, "rb") as f:
            content: bytes = f.read()

    return content

def generate_route_key_from_provided_args(
    provided_args: List[ProvidedArgInfo]
) -> str:
    names = tuple(provided_arg.param_name for provided_arg in provided_args)
    shapes = [list(provided_arg.tensor.shape) for provided_arg in provided_args]

    signature = ", ".join(f"{name}: {shape}" for name, shape in zip(names, shapes))
    signature = f"({signature})"
    return signature

######################## TorchScript Functions ########################
# This section contains torchscripted functions
#
# Here are some notes for developers
# 1. The torch.jit.script_if_tracing is needed for all functions defined below
# 2. There are duplicate implementations for functions for different types
#    a. Typically these are for handling tensors and List[torch.Tensor] types
#    b. Unions are poorly supported/behave unintuitively in Torchscript, which is why we duplicate implementations
#    c. Functions that end in type_<type>, will usually have an additional implementation
#######################################################################

# This one is used as the base generic implementation for non torchscript mode
@torch.jit.script_if_tracing
def ts_convert_dict_to_ordered_list_type_tensor(model_params: List[Tuple[str, bool, bool]], inputs: Dict[str, torch.Tensor], num_pos_args: int) -> Tuple[List[torch.Tensor], List[str]]:
    ordered_list: List[torch.Tensor] = []
    ordered_kwargs: List[Tuple[str, bool, bool]]  = []
    if torch.jit.is_scripting():
        ordered_kwargs = model_params
    else:
        ordered_kwargs = [(param.param_name, param.is_positional, False) for param in model_params] #  type: ignore[attr-defined]
    names: List[str] = []
    num_true_pos_args = 0
    kwargs_as_pos = 0
    for name, is_positional, _ in ordered_kwargs:
        if is_positional:
            num_true_pos_args += 1
            names.append(name)
            if name in inputs:
                ordered_list.append(inputs[name])
        elif name not in inputs:
            if num_pos_args > num_true_pos_args:
                kwargs_as_pos += 1
                names.append(name)
        else:
            names.append(name)
            ordered_list.append(inputs[name])

    assert len(ordered_list) == len(inputs), "Not all kwargs were parsed" +" (got " + str(len(ordered_list)) + " but expected " + str(len(inputs)) +"). This means that there is a kwarg that doesn't belong to the traced function signature."
    assert  num_true_pos_args >= num_pos_args - kwargs_as_pos, "Found insufficient number of positional args. Expected " + str(num_true_pos_args) + " but found " + str(num_pos_args - kwargs_as_pos)
    return ordered_list, names

@torch.jit.script_if_tracing
def ts_convert_dict_to_ordered_list_type_list_tensor(model_params: List[Tuple[str, bool, bool]], inputs: Dict[str, List[torch.Tensor]], num_pos_args: int) -> Tuple[List[List[torch.Tensor]], List[str]]:
    ordered_list: List[List[torch.Tensor]] = []
    ordered_kwargs: List[Tuple[str, bool, bool]]  = []
    if torch.jit.is_scripting():
        ordered_kwargs = model_params
    else:
        ordered_kwargs = [(param.param_name, param.is_positional, False) for param in model_params]  # type: ignore[attr-defined]
    names: List[str] = []
    num_true_pos_args = 0
    kwargs_as_pos = 0
    for name, is_positional, _ in ordered_kwargs:
        if is_positional:
            num_true_pos_args += 1
            names.append(name)
            if name in inputs:
                ordered_list.append(inputs[name])
        elif name not in inputs:
            if num_pos_args > num_true_pos_args:
                kwargs_as_pos += 1
                names.append(name)
        else:
            names.append(name)
            ordered_list.append(inputs[name])

    assert len(ordered_list) == len(inputs), "Not all kwargs were parsed" +" (got " + str(len(ordered_list)) + " but expected " + str(len(inputs)) +"). This means that there is a kwarg that doesn't belong to the traced function signature."
    assert  num_true_pos_args >= num_pos_args - kwargs_as_pos, "Found insufficient number of positional args. Expected " + str(num_true_pos_args) + " but found " + str(num_pos_args - kwargs_as_pos)
    return ordered_list, names
