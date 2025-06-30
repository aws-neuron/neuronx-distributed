import ast
import copy
import logging
import traceback
from typing import Any, Optional, List, Dict, Set, Tuple, Union
import warnings

import torch
import torch_neuronx
from torch_neuronx.xla_impl import structure
from torch_neuronx.xla_impl.trace import get_torch_dtype
from torch_neuronx.proto import metaneff_pb2
from torch_neuronx.pyhlo import hlo_pb2

from neuronx_distributed.trace.model_builder_utils import TraceArtifacts, CompilationArtifacts, WLOArtifacts, LayoutTransformerArtifacts, ModelParamInfo
from .base_nxd_model import BaseNxDModel, StateInitializer
from .utils import (
    retrieve_artifact_from_model,
    generate_route_key_from_provided_args,
    ts_convert_dict_to_ordered_list_type_tensor,
    ts_convert_dict_to_ordered_list_type_list_tensor
)

logger = logging.getLogger("Neuron")

InputTensorType = Union[torch.Tensor, Tuple[torch.Tensor], Dict[str, torch.Tensor]]

class JITWrapper(torch.nn.Module):
    """
    Makes a python object like Flattener and Packer JIT traceable.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, inputs):
        return self.func(inputs)

class NxDModel(torch.nn.Module, BaseNxDModel):
    def __init__(
        self,
        world_size: int,
        start_rank: Optional[int] = None,
        local_ranks_size: Optional[int] = None,
        state_initializer: Optional[StateInitializer] = None,
        layout_transformer: Optional[LayoutTransformerArtifacts] = None
    ):
        torch.nn.Module.__init__(self)
        self.world_size = world_size
        if start_rank is None:
            assert local_ranks_size is None or local_ranks_size == world_size, f"{local_ranks_size=} but start_rank is not defined. If local_ranks_size is set, the start rank must also be set."
            self.start_rank = 0
            self.local_ranks_size = self.world_size
        else:
            assert local_ranks_size is not None, f"{start_rank=} but found local_ranks_size to be unset. If setting start_rank, local_ranks_size must also be set."
            self.start_rank = start_rank
            self.local_ranks_size = local_ranks_size

        self.layout_transformer: Optional[torch.classes.neuron.LayoutTransformation] = None
        if layout_transformer is not None:
            self.layout_transformer = layout_transformer.construct_layout_transformer_object(
                self.local_ranks_size
            )

        # type annoted as tuple for torchscript purposes, but is usually
        # of type List[ModelParamInfo]
        self.model_params: List[Tuple[str, bool, bool]] = []  # type: ignore

        self.reserved_example_inputs: Dict[str, List[torch.Tensor]] = {}
        self.reserved_example_outputs: Dict[str, List[torch.Tensor]] = {}
        self.input_shape_map: Dict[str, List[str]] = {}
        self.spmd_models: Dict[str, torch.classes.neuron.SPMDModel] = {}
        # type annoted as ModuleDict for torchscript purposes, but is usually
        # of type Dict[str, Callable]
        self.flattener_map: torch.nn.ModuleDict = torch.nn.ModuleDict()  # type: ignore
        self.packer_map: torch.nn.ModuleDict = torch.nn.ModuleDict() # type: ignore

        self.weights: List[Dict[str, torch.Tensor]] = [{}]
        self.states: List[Dict[str, torch.Tensor]] = [{}]
        self.state_initializer = state_initializer

        self.loaded_on_neuron = False

    @torch.jit.unused
    def add(
        self,
        key: str,
        trace_artifacts: TraceArtifacts,
        compilation_artifacts: Union[CompilationArtifacts, WLOArtifacts],
    ) -> "NxDModel":
        """
        Add a compiled submodel to this NxDModel instance.

        Parameters
        ----------
        key : str
            Unique identifier for this submodel within the NxDModel.
        trace_artifacts : TraceArtifacts
            Artifacts produced from the `trace()` function.
        compilation_artifacts : CompilationArtifacts
            Artifacts produced from the `compile()` or `compile_wlo()` functions.

        Returns
        -------
        NxDModel
            Self reference, enabling builder-style method chaining.

        Notes
        -----
        - Creates a state initializer if none exists and state tensors are present in metaneff
        - Sets up SPMDModel instances and input/output processing components
        """
        metaneff = trace_artifacts.metaneff
        flattener = trace_artifacts.flattener
        packer = trace_artifacts.packer
        provided_args = trace_artifacts.provided_args
        model_params = trace_artifacts.model_params
        neff = compilation_artifacts.get_neff_bytes()

        # initialize a default state initializer if not initialized
        if self.state_initializer is None:
            shapes = {}
            dtypes = {}

            for tensor in metaneff.input_tensors:
                if tensor.type is metaneff_pb2.MetaTensor.Type.INPUT_STATE:
                    # proto keys are bytes not strings, and casting as a string causes it to be "b'key'"
                    checkpoint_key = str(tensor.checkpoint_key).replace("b'", "").replace("'", "")
                    shapes[checkpoint_key] = list(tensor.shape)
                    dtypes[checkpoint_key] = get_torch_dtype(tensor.data_type)

            if len(shapes) != 0:
                self.state_initializer = StateInitializer(
                    shapes,
                    dtypes,
                    self.local_ranks_size
                )

        # create SPMDModel class
        metaneff_bytes = metaneff.SerializeToString()
        self.spmd_models[key] = torch.classes.neuron.SPMDModel(
            neff,
            metaneff_bytes,
            self.local_ranks_size,
            self.world_size,
        )

        # make sure we're only adding models that use the same original forward signature
        if len(self.model_params) == 0:
            self.model_params = model_params  # type: ignore[assignment]
        else:
            assert all(
                [
                    expected_param == actual_param
                    for expected_param, actual_param
                    in zip(
                        self.model_params,
                        model_params
                    )
                ]
            ), "The model provided appears to have a different signature than all other models added prior."

        # change layout of Flattener to accept List[torch.Tensor] form of example_inputs
        assert isinstance(flattener.layout, tuple), f"Expected flattener layout type to be tuple but found {type(flattener.layout)}"
        flattener.layout = list(flattener.layout)
        self.flattener_map[key] = JITWrapper(flattener)

        # register input shape to input shape map
        self.reserved_example_inputs[key] = [provided_arg.tensor for provided_arg in provided_args]
        # key will have the following structure: (arg1_name,arg2_name,...)[arg1_shape, arg2_shape, ...]
        input_shape_map_key: str = generate_route_key_from_provided_args(provided_args)
        # support one shape to multiple keys
        if input_shape_map_key in self.input_shape_map:
            self.input_shape_map[input_shape_map_key].append(key)
        else:
            self.input_shape_map[input_shape_map_key] = [key]

        # create reserved example outputs, which is necessary for packer tracing
        example_outputs = []
        for out_tensor in metaneff.output_tensors:
            torch_dtype = get_torch_dtype(out_tensor.data_type)
            shape = out_tensor.shape
            tensor = torch.zeros(list(shape), dtype=torch_dtype)
            example_outputs.append(tensor)
        self.reserved_example_outputs[key] = example_outputs
        self.packer_map[key] = JITWrapper(packer)

        return self # to allow for builder like syntax

    @torch.jit.unused
    def get_available_keys(self) -> Set[str]:
        return set(self.spmd_models.keys())

    @torch.jit.unused
    def get_neff(self, key: str) -> bytes:
        """
        Retrieves the NEFF (Neuron Executable File Format) from the specified model.

        Args:
            key (str): The identifier for the model whose NEFF should be retrieved.

        Returns:
            bytes: The NEFF for the specified model.

        Raises:
            KeyError: If the specified key is not found in the available keys.
            RuntimeError: If there is an error retrieving the NEFF.
        """
        if key not in self.get_available_keys():
            raise KeyError(f"{key=} not found.")

        model = self.spmd_models[key]
        neff = retrieve_artifact_from_model(model, "neff")
        if neff is None:
            raise RuntimeError("Something went wrong while retrieving the neff. Set NEURON_CPP_LOG_LEVEL=1 to get more debug logging.")

        return neff

    @torch.jit.unused
    def get_metaneff(self, key: str) -> metaneff_pb2.MetaNeff:
        """
        Retrieves the metaneff from the specified model.

        Args:
            key (str): The identifier for the model whose metaneff should be retrieved.

        Returns:
            metaneff_pb2.MetaNeff: The metaneff proto object for the specified model.

        Raises:
            KeyError: If the specified key is not found in the available keys.
            RuntimeError: If there is an error retrieving the metaneff.
        """
        if key not in self.get_available_keys():
            raise KeyError(f"{key=} not found.")

        model = self.spmd_models[key]
        metaneff_bytes = retrieve_artifact_from_model(model, "metaneff")
        if metaneff_bytes is None:
            raise RuntimeError("Something went wrong while retrieving the metaneff. Set NEURON_CPP_LOG_LEVEL=1 to get more debug logging.")

        metaneff = metaneff_pb2.MetaNeff()
        metaneff.ParseFromString(metaneff_bytes)
        return metaneff

    @torch.jit.unused
    def get_hlo(self, key: str) -> hlo_pb2.HloModuleProto:
        """
        Retrieves the HLO from the specified model.

        Args:
            key (str): The identifier for the model whose HLO should be retrieved.

        Returns:
            hlo_pb2.HloModuleProto: The HLO module proto object for the specified model.

        Raises:
            KeyError: If the specified key is not found in the available keys.
            RuntimeError: If there is an error retrieving the HLO.
        """
        if key not in self.get_available_keys():
            raise KeyError(f"{key=} not found.")

        hlo_module = hlo_pb2.HloModuleProto()
        metaneff = self.get_metaneff(key)
        hlo_bytes = metaneff.serialized_graph_def
        hlo_module.ParseFromString(hlo_bytes)

        return hlo_module

    @torch.jit.export
    def set_weights(self, sharded_checkpoint: List[Dict[str, torch.Tensor]]):
        """
        Set the model's weights from a sharded checkpoint.

        This function initializes the model's weights using a sharded checkpoint. The checkpoint is processed
        and loaded using either a layout transformer (if provided) or a direct parallel loading mechanism.

        This function should only be called before the model is loaded onto a neuron device. Once the model
        is loaded, use `replace_weights()` to update weights.

        Parameters:
            sharded_checkpoint (List[Dict[str, torch.Tensor]]): A list of state dicts mapping parameter names
                to their corresponding tensor values for each rank.

        Raises:
            ValueError: If the model is already loaded on a Neuron device.

        Notes:
            - This function is TorchScript compatible.
        """
        # MUST BE TORCHSCRIPT COMPATIBLE
        if self.loaded_on_neuron:
            raise ValueError("set_weights() was called when the model was already loaded on neuron. If you need to replace your weights, call `replace_weights()` instead.")

        if self.layout_transformer is not None:
            self.weights = self.layout_transformer.forward(sharded_checkpoint, False)
        else:
            self.weights = torch.ops.neuron._parallel_load(sharded_checkpoint)

        if (self.state_initializer is not None):
            self.states = self.state_initializer()

    @torch.jit.export
    def to_neuron(self):
        """
        Load the model onto AWS Neuron devices.

        This function initializes the model on Neuron hardware.

        Returns:
            None

        Notes:
            - This function is TorchScript compatible.
        """
        # MUST BE TORCHSCRIPT COMPATIBLE
        if self.loaded_on_neuron:
            return
        for model in self.spmd_models.values():
            model.initialize(self.states, self.weights, self.start_rank)
        self.loaded_on_neuron = True

    @torch.jit.export
    def replace_weights(self, sharded_checkpoint: List[Dict[str, torch.Tensor]]):
        """
        Replace the model's weights and reload onto Neuron devices.

        This method should be used instead of `set_weights()` when the model is already loaded on
        Neuron devices and weights need to be updated.

        Parameters:
            sharded_checkpoint (List[Dict[str, torch.Tensor]]): A list of state dicts mapping parameter names
                to their corresponding tensor values for each rank.

        Returns:
            None

        Notes:
            - This function is TorchScript compatible.
        """
        # MUST BE TORCHSCRIPT COMPATIBLE
        self.loaded_on_neuron = False
        self.set_weights(sharded_checkpoint)
        self.to_neuron()

    @torch.jit.export
    def read_from_neuron_buffer(self, buffer_key: str, rank: int) -> torch.Tensor:
        """
        Reads a tensor value from a Neuron device buffer to CPU, based on given key and rank.

        Parameters:
            buffer_key (str): The key identifying the specific buffer to retrieve.
            rank (int): The rank from which to retrieve the buffer.

        Returns:
            torch.Tensor: The requested tensor buffer copied to CPU memory.

        Raises:
            AssertionError: If this method is called before to_neuron()
            KeyError: If the specified state_buffer_key does not exist in the states for the given rank.

        Notes:
            - This function is TorchScript compatible.
        """
        # MUST BE TORCHSCRIPT COMPATIBLE
        assert self.loaded_on_neuron, "Model must be loaded on neuron before calling these methods"

        if len(self.states) > rank and buffer_key in self.states[rank]:
            return self.states[rank][buffer_key].cpu()
        elif len(self.weights) > rank and buffer_key in self.weights[rank]:
            return self.weights[rank][buffer_key].cpu()

        raise KeyError("Could not find neuron buffer: " + buffer_key + " in rank " + str(rank))

    @torch.jit.export
    def write_to_neuron_buffer(self, tensor: torch.Tensor, buffer_key: str, rank: int):
        """
        Write a tensor to a specific Neuron device buffer.

        This function updates a state buffer on a Neuron device by copying values from the provided tensor.
        The destination buffer must already exist and have the same shape as the input tensor.

        Parameters:
            tensor (torch.Tensor): The tensor containing the data to be written to the buffer.
            buffer_key (str): The key identifying the specific buffer to update.
            rank (int): The rank where the buffer is located.

        Raises:
            AssertionError: If this method is called before to_neuron()
            KeyError: If the specified state_buffer_key does not exist in the states for the given rank,
                    or if the shapes of the input tensor and target buffer don't match.

        Notes:
            - This function is TorchScript compatible.
        """
        # MUST BE TORCHSCRIPT COMPATIBLE
        assert self.loaded_on_neuron, "Model must be loaded on neuron before calling these methods"

        # checks both rank bounds and existence of key
        if (
            (len(self.states) > rank and buffer_key not in self.states[rank]) and
            (len(self.weights) > rank and buffer_key not in self.weights[rank])
        ):
            raise KeyError("Could not find neuron buffer: " + buffer_key + " in rank " + str(rank))

        buffer_type = self.states if buffer_key in self.states[rank] else self.weights
        if tensor.shape != buffer_type[rank][buffer_key].shape:
            raise KeyError("Cannot write tensor of shape " + str(tensor.shape) + " to neuron buffer with shape " + str(buffer_type[rank][buffer_key].shape))

        # inplace copy, and self.weights/states will get update since buffer_type is a reference
        buffer_type[rank][buffer_key].copy_(tensor)

    def convert_dict_to_ordered_list(self, inputs: Dict[str, Union[torch.Tensor, List[torch.Tensor]]], num_pos_args: int) -> Tuple[Union[List[torch.Tensor], List[List[torch.Tensor]]], List[str]]:
        # MUST BE TORCHSCRIPT COMPATIBLE
        ordered_list: Union[List[torch.Tensor], List[List[torch.Tensor]]] = []
        ordered_kwargs: List[Tuple[str, bool, bool]]  = []
        if torch.jit.is_scripting():
            ordered_kwargs = self.model_params
        else:
            ordered_kwargs = [(param.param_name, param.is_positional, False) for param in self.model_params]  # type: ignore[attr-defined]
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

        assert len(ordered_list) == len(inputs), "Not all kwargs were parsed" +"(got " + str(len(ordered_list)) + " but expected " + str(len(inputs)) +"). This means that there is a kwarg that doesn't belong to the traced function signature."
        # subtract the kwargs_as_pos from num_pos_args to handle case where user passes in kwargs as positional
        assert  num_true_pos_args >= num_pos_args - kwargs_as_pos, "Found insufficient number of required positional args. Expected " + str(num_true_pos_args) + " but found " + str(num_pos_args - kwargs_as_pos)
        return ordered_list, names

    @torch.jit.export
    def router(self, inputs: List[torch.Tensor], arg_names: List[str]) -> List[str]:
        signature: str = ", ".join(name + ": " + str(list(tensor.shape)) for name,tensor in zip(arg_names, inputs))
        key: str = "(" + signature + ")"
        return self.input_shape_map[key]

    def _ts_fake_forward(self, x):
        # Used exclusively for monkeypatching a fake forward for torchscript conversion
        return x

    def forward(
        self,
        *args,
        model_name: Optional[str] = None,
        forward_mode='default',
        **kwargs
    ):
        """
        The forward method of the NxDModel class, which will take in inputs and run the respective neff.

        Args:
            *args (Union[torch.Tensor, List[torch.Tensor]]): Positional tensor inputs to model. List form must be used if `forward_mode != 'default'`.

            model_name (Optional[str]): Parameter to pass in a specific key to execute.
                This must be used in cases of ambiguous routing.

            forward_mode (str): There are 3 supported modes: default, ranked, async.

                *default*: This takes in inputs, replicates them across ranks,
                    executes the model, and returns the outputs from rank 0

                *ranked*: This takes in inputs in ranked form, meaning each
                    individual tensor input must be a list of tensors whose
                    length is equal to the world size of the compiled model.
                    The model will execute, and return a ranked output, which
                    is a List of all outputs by rank (ie a List[List[torch.Tensor]])

                *async*: Like ranked, this takes in inputs and returns outputs in ranked form
                    except the major difference is that the outputs will be returned
                    instantly, and will be references to buffers where the model will
                    write the output once the neff is done executing. To block on
                    the neff call, you must call `.cpu()` for each tensor in the
                    output.

            **kwargs (torch.Tensor, List[torch.Tensor]): Key word arguments corresponding to specific input tensors to the model. List form must be used if `forward_mode != 'default'`.

        Returns:
            It depends on the forward_mode setting:

            *default*: Expected format of tensor outputs based on what was originally traced.
            *ranked & async*: List[List[torch.Tensor]] of shape (num_out_tensors, world_size)

        Raises:
            AssertionError
            RuntimeError
            KeyError
        """
        if not self.loaded_on_neuron:
            raise RuntimeError("Model not initialized. Call set_weights() followed by to_neuron()")
        SUPPORTED_FORWARD_MODES = {'default', 'ranked', 'async'}
        assert forward_mode in SUPPORTED_FORWARD_MODES, f"{forward_mode=} is not supported. It must be one of {SUPPORTED_FORWARD_MODES}"

        kwargs, arg_names = self.convert_dict_to_ordered_list(  # type: ignore[assignment]
            kwargs,
            len(args)
        )
        # kwargs is either an empty list or an ordered list of tensors
        if forward_mode == 'default':
            inputs: List[torch.Tensor] = list(args) + kwargs  # type: ignore[operator]
            model_names: List[str] = self.router(inputs, arg_names)
        else:
            # convert pos args with ranks to ranked form
            _args: List[List[torch.Tensor]] = [[] for _ in range(self.local_ranks_size)]
            for arg_all_ranks in args:
                for rank, arg in enumerate(arg_all_ranks):
                    _args[rank].append(arg)

            # convert ordered kwargs to ordered ranked positional args
            for ordered_kwarg_all_ranks in kwargs:
                for rank, ordered_kwarg in enumerate(ordered_kwarg_all_ranks):
                    _args[rank].append(ordered_kwarg)

            inputs = _args
            model_names = self.router(inputs[0], arg_names)

        if len(model_names) > 1:
            assert model_name is not None, f"Got {len(model_names)} possible routes but model_name wasn't provided. The Model Name must be provided if input routing is ambiguous."
        else:
            assert model_name is None or model_name == model_names[0], f"Provided model_name does not match model name found by the shape router. Found {model_names[0]} but got {model_name}"
            model_name = model_names[0]

        if isinstance(inputs[0], list):
            flattened_inputs: List[List[torch.Tensor]] = [self.flattener_map[model_name](inp) for inp in inputs]  # type: ignore[no-redef]
        else:
            flattened_inputs: List[torch.Tensor] = self.flattener_map[model_name](inputs)  # type: ignore[no-redef]

        if forward_mode == 'default':
            outputs: List[torch.Tensor] = self.spmd_models[model_name].forward(flattened_inputs)  # type: ignore[no-redef]
            return self.packer_map[model_name](outputs)
        elif forward_mode == 'ranked':
            outputs: List[List[torch.Tensor]] = self.spmd_models[model_name].forward_ranked(flattened_inputs)  # type: ignore[no-redef]
        else: # async
            outputs: List[List[torch.Tensor]] = self.spmd_models[model_name].forward_async(flattened_inputs)  # type: ignore[no-redef]

        # only runs for 'ranked' and 'async' modes
        transposed_outputs: List[List[torch.Tensor]] = [[] for _ in range(len(outputs[0]))]
        for rank,output in enumerate(outputs):
            for output_tensor in output:
                transposed_outputs[rank].append(output_tensor)

        return transposed_outputs

    def save(self, path_to_save: str, save_weights: bool = False):
        """
        Saves the model as a TorchScript module to the specified path.
        This can be loaded with `NxDModel.load` or `torch.jit.load`

        Args:
            path_to_save (str): The file path where the TorchScript model should be saved.
            save_weights (bool, optional): If True, preserves the weights within the torchscript
            model. It is False by default.

        Returns:
            None
        """
        temp_weights = self.weights
        temp_states = self.states
        torchscript_nxd_model: TorchScriptNxDModel = convert_nxd_model_to_torchscript_model(
            self, save_weights
        )
        # save some metadata indicating it's not forwards/backwards compatible wrt un-torchscripting the NxDModel.
        torch.jit.save(torchscript_nxd_model, path_to_save, _extra_files={'nxd_torchscript_metadata.json':'{"forwards_compatibile": false, "backwards_compatible": false}'})
        del torchscript_nxd_model
        if not save_weights:
            self.weights = temp_weights
            self.states = temp_states

    @classmethod
    def load(
        cls,
        path_to_model: str,
        start_rank: Optional[int] = None,
        local_ranks_size: Optional[int] = None
    ) -> Union["NxDModel", torch.jit.ScriptModule]:
        """
        Attempts to load and restore an NxDModel from a saved TorchScript model.

        This classmethod tries to reconstruct an NxDModel instance from a previously saved
        TorchScript model. If the restoration process fails, it returns the loaded TorchScript
        model instead, as backwards compatibility is not guaranteed across different versions
        of NxD.

        Args:
            path_to_model (str): Path to the saved TorchScript model file.
            start_rank (Optional[int], optional): Starting rank for distributed processing.
                If None and local_ranks_size is set, an error will be raised. Defaults to None.
            local_ranks_size (Optional[int], optional): Size of local ranks for distributed processing.
                Must be set if start_rank is provided. Defaults to None.

        Returns:
            Union[NxDModel, torch.jit.ScriptModule]: Either the restored NxDModel instance
            or the loaded TorchScript model if restoration fails.

        Raises:
            ValueError: If the provided model was not originally saved using NxDModel.save().
            AssertionError: If start_rank/local_ranks_size parameters are inconsistently set.

        Note:
            - If weights were saved with the model, they will be loaded onto Neuron hardware.
        """
        # check if torchscript model was originally an NxDModel saved with NxDModel.save
        torchscript_model = torch.jit.load(path_to_model) # if weights were saved, it'll be loaded on Neuron here
        if not (
            torchscript_model.original_name == TorchScriptNxDModel.__name__
            and hasattr(torchscript_model, "nxd_model")
            and torchscript_model.nxd_model.original_name == NxDModel.__name__
        ):
            raise ValueError("The Torchscript Model Supplied was not produced with the NxDModel.save method.")

        try:
            # restore distributed attributed
            world_size = torchscript_model.nxd_model.world_size
            if start_rank is None:
                assert local_ranks_size is None or local_ranks_size == world_size, f"{local_ranks_size=} but start_rank is not defined. If local_ranks_size is set, the start rank must also be set."
                torchscript_model.nxd_model.start_rank = 0
                torchscript_model.nxd_model.local_ranks_size = world_size
            else:
                assert local_ranks_size is not None, f"{start_rank=} but found local_ranks_size to be unset. If setting start_rank, local_ranks_size must also be set."
                torchscript_model.nxd_model.start_rank = start_rank
                torchscript_model.nxd_model.local_ranks_size = local_ranks_size

            nxd_model = NxDModel(
                world_size,
                start_rank,
                local_ranks_size,
                StateInitializer(
                    torchscript_model.nxd_model.state_initializer.shapes,
                    torchscript_model.nxd_model.state_initializer.dtypes,
                    local_ranks_size
                ) if torchscript_model.nxd_model.state_initializer is not None else None,
                torchscript_model.nxd_model.layout_transformer # these can remain opaque under torchscript, since it's opaque to python anyways
            )

            # restore model_params
            nxd_model.model_params = [ModelParamInfo(param_info[0], param_info[1]) for param_info in torchscript_model.nxd_model.model_params]  # type: ignore[misc]

            # all of these, except spmd_models, are loaded with the original python dtypes
            nxd_model.reserved_example_inputs = torchscript_model.nxd_model.reserved_example_inputs
            nxd_model.reserved_example_outputs = torchscript_model.nxd_model.reserved_example_outputs
            nxd_model.input_shape_map = torchscript_model.nxd_model.input_shape_map
            nxd_model.spmd_models = torchscript_model.nxd_model.spmd_models # SPMDModel is opaque to python anyways

            # reconstruct flattener map from saved flattener metadata
            # the flattener metadata are all strings that should be parseable by ast.literal_eval
            restored_flattener_map: torch.nn.ModuleDict = torch.nn.ModuleDict()
            for key, flattener_metadata in torchscript_model.flattener_metadata.items():
                restored_flattener_map[key] = JITWrapper(structure.Flattener(
                    ast.literal_eval(flattener_metadata['layout']),
                    ast.literal_eval(flattener_metadata['exclude'])
                ))
            nxd_model.flattener_map = restored_flattener_map

            # reconstruct packer map from saved packer metadata
            # the packer metadata are all strings that should be parseable by ast.literal_eval
            restored_packer_map: torch.nn.ModuleDict = torch.nn.ModuleDict()
            for key, packer_metadata in torchscript_model.packer_metadata.items():
                restored_packer_map[key] = JITWrapper(structure.Packer(
                    ast.literal_eval(packer_metadata['layout']),
                    ast.literal_eval(packer_metadata['identifiers']),
                    ast.literal_eval(packer_metadata['constants'])
                ))
            nxd_model.packer_map = restored_packer_map

            # check if weights in torchscript model were original weights, if so initialize the NxDModel
            if '__neuronprivatetensor__' not in torchscript_model.nxd_model.weights[0]:
                nxd_model.weights = torchscript_model.nxd_model.weights
                nxd_model.states = torchscript_model.nxd_model.states

                # in this scenario the weights are on neuron already in the correct layout,
                # so we just assign the weights to the model
                nxd_model.to_neuron()

        except Exception as e:
            warnings.warn("Unable to reconstruct NxDModel from torchscript model, will return loaded torchscript model.")
            traceback_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
            logger.debug(traceback_str)
            return torchscript_model

        return nxd_model

class TorchScriptNxDModel(torch.nn.Module):
    """
    Wrapper for torchscripted NxDModel
    """

    def __init__(self, nxd_model: NxDModel, orig_flatteners: torch.nn.ModuleDict, orig_packers: torch.nn.ModuleDict):
        super().__init__()
        self.nxd_model = nxd_model
        self.model_name: Optional[str] = None
        self.flattener_metadata: Dict[str, Dict[str, str]] = {}
        self.packer_metadata: Dict[str, Dict[str, str]] = {}

        # stringify flattener attributes for simpler type attributing
        # and easy restoration of Flattener object in load
        for key, flattener in orig_flatteners.items():
            self.flattener_metadata[key] = {
                'layout': str(flattener.func.layout),
                'exclude': str(flattener.func.exclude)
            }

        # stringify packer attributes for simpler type attributing
        # and easy restoration of Packer object in load
        for key, packer in orig_packers.items():
            self.packer_metadata[key] = {
                'layout': str(packer.func.layout),
                'identifiers': str(packer.func.identifiers),
                'constants': str(packer.func.constants)
            }

    def _torchscript_forward_default(  # type: ignore
        self,
        args: List[torch.Tensor],
        kwargs: Dict[str, torch.Tensor],
        model_name: Optional[str],
    ) -> Any:
        """
        The forward method ran if serialized, this must be compatible
        with torch.jit.script
        """

        assert torch.jit.isinstance(kwargs, Dict[str, torch.Tensor])
        ordered_results: Tuple[List[torch.Tensor], List[str]] = ts_convert_dict_to_ordered_list_type_tensor(
            self.nxd_model.model_params,
            kwargs,
            len(args)
        )
        ordered_kwargs: List[torch.Tensor] = ordered_results[0]
        arg_names: List[str] = ordered_results[1]
        assert torch.jit.isinstance(args, List[torch.Tensor])
        inputs: List[torch.Tensor] = args + ordered_kwargs
        model_names: List[str] = self.nxd_model.router(inputs, arg_names)
        if len(model_names) > 1:
            assert model_name is not None, f"Got {len(model_names)} possible routes but model_name wasn't provided. The Model Name must be provided if input routing is ambiguous."
        else:
            assert model_name is None or model_name == model_names[0], f"Provided model_name does not match model name found by the shape router. Found {model_names[0]} but got {model_name}"
            model_name = model_names[0]

        assert torch.jit.isinstance(model_name, str)

        # can't use dictionary indexing syntax here because torchscript doesn't support it
        flattened_inputs: List[torch.Tensor] = []
        for name, flattener in self.nxd_model.flattener_map.items():
            if name == model_name:
                flattened_inputs = flattener(inputs)

        results: Optional[Any] = None
        outputs: List[torch.Tensor] = self.nxd_model.spmd_models[model_name].forward(flattened_inputs)
        # can't use dictionary indexing syntax here because torchscript doesn't support it
        for name, _packer in self.nxd_model.packer_map.items():
            if name == model_name:
                results = _packer(outputs)

        assert results is not None, "Could not find associated packer"
        return results

    def _torchscript_forward_ranked(  # type: ignore
        self,
        args: List[List[torch.Tensor]],
        kwargs: Dict[str, List[torch.Tensor]],
        model_name: Optional[str],
        async_mode: bool
    ) -> Any:
        ordered_results: Tuple[List[List[torch.Tensor]], List[str]] = ts_convert_dict_to_ordered_list_type_list_tensor(
            self.nxd_model.model_params,
            kwargs,
            len(args)
        )
        ordered_kwargs: List[List[torch.Tensor]] = ordered_results[0]
        arg_names: List[str] = ordered_results[1]

        # convert ranked args from rank in form to rank out form
        # since that's what SPMDModel.forward_ranked/async expect.
        # ie List[List[torch.Tensor]] of shape (num_args, local_ranks_size) -> (local_ranks_size, num_args)
        _args: List[List[torch.Tensor]] = [[] for _ in range(self.nxd_model.local_ranks_size)]
        for arg_all_ranks in args:
            for rank, arg in enumerate(arg_all_ranks):
                _args[rank].append(arg)

        # convert ordered kwargs to ordered rank out positional args
        for ordered_kwarg_all_ranks in ordered_kwargs:
            for rank, ordered_kwarg in enumerate(ordered_kwarg_all_ranks):
                _args[rank].append(ordered_kwarg)

        inputs: List[List[torch.Tensor]] = _args
        model_names: List[str] = self.nxd_model.router(inputs[0], arg_names)
        if len(model_names) > 1:
            assert model_name is not None, f"Got {len(model_names)} possible routes but model_name wasn't provided. The Model Name must be provided if input routing is ambiguous."
        else:
            assert model_name is None or model_name == model_names[0], f"Provided model_name does not match model name found by the shape router. Found {model_names[0]} but got {model_name}"
            model_name = model_names[0]

        assert torch.jit.isinstance(model_name, str)

        # can't use dictionary indexing syntax here because torchscript doesn't support it
        flattened_inputs: List[List[torch.Tensor]] = [[torch.tensor(0)]]
        for name, flattener in self.nxd_model.flattener_map.items():
            if name == model_name:
                flattened_inputs = [flattener(inp) for inp in inputs]

        if async_mode:
            return self.nxd_model.spmd_models[model_name].forward_async(flattened_inputs)
        else:
            return self.nxd_model.spmd_models[model_name].forward_ranked(flattened_inputs)

    def forward(  # type: ignore
        self,
        args: Optional[List[torch.Tensor]],
        kwargs: Optional[Dict[str, torch.Tensor]],
    ) -> Any:
        if not self.nxd_model.loaded_on_neuron:
            raise RuntimeError("Model not initialized. Call set_weights() followed by to_neuron()")

        if args is None:
            assert kwargs is not None
            return self._torchscript_forward_default(
                [],
                kwargs,
                self.model_name,
            )
        elif kwargs is None:
            assert args is not None
            return self._torchscript_forward_default(
                args,
                {},
                self.model_name,
            )

        return self._torchscript_forward_default(
            args,
            kwargs,
            self.model_name,
        )

    @torch.jit.export
    def forward_ranked(
        self,
        args: Optional[List[List[torch.Tensor]]],
        kwargs: Optional[Dict[str, List[torch.Tensor]]],
        async_mode: bool
    ):
        if not self.nxd_model.loaded_on_neuron:
            raise RuntimeError("Model not initialized. Call set_weights() followed by to_neuron()")

        if args is None:
            assert kwargs is not None
            return self._torchscript_forward_ranked(
                [],
                kwargs,
                self.model_name,
                async_mode
            )
        elif kwargs is None:
            assert args is not None
            _kwargs: Dict[str, List[torch.Tensor]] = {}
            return self._torchscript_forward_ranked(
                args,
                _kwargs,
                self.model_name,
                async_mode
            )

        return self._torchscript_forward_ranked(
            args,
            kwargs,
            self.model_name,
            async_mode
        )

    @torch.jit.export
    def set_weights(self, sharded_checkpoint: List[Dict[str, torch.Tensor]]):
        self.nxd_model.set_weights(sharded_checkpoint)

    @torch.jit.export
    def to_neuron(self):
        self.nxd_model.to_neuron()

    @torch.jit.export
    def replace_weights(self, sharded_checkpoint: List[Dict[str, torch.Tensor]]):
        self.nxd_model.replace_weights(sharded_checkpoint)

    @torch.jit.export
    def read_from_neuron_buffer(self, buffer_key: str, rank: int) -> torch.Tensor:
        return self.nxd_model.read_from_neuron_buffer(buffer_key, rank)

    @torch.jit.export
    def write_to_neuron_buffer(self, tensor: torch.Tensor, buffer_key: str, rank: int):
        self.nxd_model.write_to_neuron_buffer(tensor, buffer_key, rank)

def convert_nxd_model_to_torchscript_model(nxd_model: "NxDModel", save_weights: bool = False) -> torch.jit.ScriptModule:
    """
    Converts an NxDModel to a TorchScriptNxDModel object which can
    be Torchscripted with torch.jit.script.
    """
    # unable to save empty containers in torchscript
    if not save_weights or (len(nxd_model.weights) == 1 and len(nxd_model.weights[0]) == 0):
        nxd_model.weights = [{'__neuronprivatetensor__':torch.tensor(0)}]
    if not save_weights or (len(nxd_model.states) == 1 and len(nxd_model.states[0]) == 0):
        nxd_model.states = [{'__neuronprivatetensor__':torch.tensor(0)}]

    new_flattener_map: List[Tuple[str, torch.jit.ScriptModule]] = []
    for key,flattener in nxd_model.flattener_map.items():
        new_flattener_map.append((key, torch.jit.trace(flattener, (nxd_model.reserved_example_inputs[key],), strict=False)))
    orig_flatteners = nxd_model.flattener_map
    nxd_model.flattener_map = torch.nn.ModuleDict(new_flattener_map)

    new_packer_map: List[Tuple[str, torch.jit.ScriptModule]] = []
    for key, packer in nxd_model.packer_map.items():
        new_packer_map.append((key, torch.jit.trace(packer, (nxd_model.reserved_example_outputs[key],), strict=False)))
    orig_packers = nxd_model.packer_map
    nxd_model.packer_map = torch.nn.ModuleDict(new_packer_map)

    # needed for torchscript serialization due to possibility of it being an empty container
    orig_model_params = copy.deepcopy(nxd_model.model_params)
    nxd_model.model_params = [(param_info.param_name, param_info.is_positional, False) for param_info in nxd_model.model_params]  # type: ignore[attr-defined]

    # monkeypatch forward with _torchscript_forward so jit.script doesn't complain about forward
    orig_forward = nxd_model.forward
    nxd_model.forward = nxd_model._ts_fake_forward  # type: ignore[method-assign, assignment]

    # all loaded torchscript models are considered uninitialized
    orig_loaded_status = nxd_model.loaded_on_neuron
    nxd_model.loaded_on_neuron = False

    with torch_neuronx.contexts.disable_nrt_load():
        torchscript_model = torch.jit.script(TorchScriptNxDModel(nxd_model, orig_flatteners, orig_packers))

    # restore original nxd model
    nxd_model.forward = orig_forward  # type: ignore[method-assign]
    nxd_model.model_params = orig_model_params
    nxd_model.flattener_map = orig_flatteners
    nxd_model.packer_map = orig_packers
    nxd_model.loaded_on_neuron = orig_loaded_status

    return torchscript_model
