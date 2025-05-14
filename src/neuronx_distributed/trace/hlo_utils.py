import os
import torch
import shutil
import pathlib
import importlib.util
import subprocess
import tempfile
from safetensors.torch import load_file, save_file
from functools import partial
import logging
from typing import Dict, List, Optional, Set, Any
from copy import deepcopy

from neuronx_distributed.trace.model_builder_utils import (
    WLOArtifacts,
    generate_key
)

from torch_neuronx.proto import metaneff_pb2
from torch_neuronx.pyhlo import hlo_pb2, xla_data_pb2
from torch_neuronx.xla_impl.trace import (
    hlo_entry_computation,
    get_hlo_computation_by_id,
    get_hlo_root_instruction,
    HloArtifacts,
    XLA_DTYPE_TO_METANEFF_DTYPE,
)

TRANSPOSABLE_WEIGHT_IDX = "transposable_weight_idx"
REQUIRE_TRANSPOSE_WEIGHT_IDX = "require_transpose_weight_idx"
REQUIRE_TRANSPOSE_CUSTOM_CALL = "require_transpose_custom_call"
FRONTEND_ATTRIBUTES_DELIMITER = ","

# Constants for an experimental feature on weight layout transformation
NXD_LAYOUT_TRANSFORMATION_OPTIONS = "NXD_LAYOUT_TRANSFORMATION_OPTIONS"
NXD_LAYOUT_ON_CPU_AND_SERIALIZE = "NXD_LAYOUT_ON_CPU_AND_SERIALIZE"
NXD_LAYOUT_ON_DEVICE_AND_SERIALIZE = "NXD_LAYOUT_ON_DEVICE_AND_SERIALIZE"

# Constants for NKI Kernel
AWS_NEURON_CUSTOM_NATIVE_KERNEL = "AwsNeuronCustomNativeKernel"

logger = logging.getLogger("Neuron")


def read_hlo(hlo_path: str):
    """Read a HLOModuleProto from given path"""
    hlo = hlo_pb2.HloModuleProto()
    with open(hlo_path, "rb") as f:
        hlo.ParseFromString(f.read())
    return hlo


def write_hlo(hlo_path: str, hlo_module: hlo_pb2.HloModuleProto):
    """ Serialize and write the hlo object into the specified location """
    with open(hlo_path, 'wb') as f:
        f.write(hlo_module.SerializeToString())


def add_weight_idx_attr_to_hlo(hlo: hlo_pb2.HloModuleProto, weight_name_to_idx: Dict[str, int], weight_names_to_skip=None):
    """
    Add frontend attributes on weight indices for weights
    """
    # Avoid updating the original dict, otherwise it will impact the creation
    # of weight layout transformation HLO
    weight_name_to_idx = deepcopy(weight_name_to_idx)
    # Remove the weights which should not be transposed
    if weight_names_to_skip:
        logger.info(f"Removing {len(weight_names_to_skip)} kernel weights from the frontend attributes")
        for weight_name in weight_names_to_skip:
            weight_name_to_idx.pop(weight_name, None)

    weight_idx = sorted(weight_name_to_idx.values())
    weight_idx_list_str = ",".join([str(idx) for idx in weight_idx])
    hlo.frontend_attributes.map[TRANSPOSABLE_WEIGHT_IDX] = weight_idx_list_str
    return hlo


def mark_weights_for_wlo(
    priority_model_trace_hlo: hlo_pb2.HloModuleProto,
    priority_model_weight_name_to_idx: Dict[str, int],
    weights_to_skip_layout_optimization: Optional[Set] = None
) -> None:
    """
    Mark weights in the priority model for Weight Layout Optimization (WLO).
    
    Args:
        priority_model_trace_hlo: Priority model trace HLO
        priority_model_weight_name_to_idx: A dictionary mapping weight names to their indices for the priority model
        weights_to_skip_layout_optimization: Set of weight names to exclude from optimization
        
    Returns:
        None
    """
    if weights_to_skip_layout_optimization is None:
        weights_to_skip_layout_optimization = set()

    try:
        # Input validation
        if not priority_model_trace_hlo:
            raise ValueError("Priority model trace HLO is None")
            
        # Validate that all skipped weights exist in the weight_name_to_idx
        invalid_weights = weights_to_skip_layout_optimization - set(priority_model_weight_name_to_idx.keys())
        if invalid_weights:
            raise ValueError(f"Invalid weights in skip set: {invalid_weights}")

        logger.info("Marking weights in the priority model for weight layout optimization.")

        add_weight_idx_attr_to_hlo(
            hlo=priority_model_trace_hlo,
            weight_name_to_idx=priority_model_weight_name_to_idx,
            weight_names_to_skip=weights_to_skip_layout_optimization
        )

    except Exception as e:
        logger.error(f"Failed to mark weights for WLO: {str(e)}")
        raise RuntimeError(f"Weight layout optimization marking failed: {str(e)}") from e


def apply_layout_transformation(
    hlo_module: hlo_pb2.HloModuleProto,
    flattener: Any,
    packer: Any,
    metaneff: Any,
    weight_name_to_idx: Dict[str, int],
    wlo_artifacts: WLOArtifacts,
    key: Optional[str] = None
) -> None:
    """
    Apply the layout transformation suggestion from the priority HLO to the given non-priority model trace artifacts.

    Args:
        hlo_module: The HLO module to apply the layout optimization to.
        flattener: Function to flatten inputs.
        packer: Function to pack outputs.
        metaneff: The meta information for the Neuron Executable File Format (NEFF).
        weight_name_to_idx: Dictionary mapping weight names to their indices.
        wlo_artifacts: Artifacts containing the weight layout optimization information.
        key: Optional key used to tag the bucket with a meaningful name.

    Returns:
        None
    """
    try:
        # Input validation
        if not hlo_module:
            raise ValueError("HLO module is None")
        if not weight_name_to_idx:
            raise ValueError("Weight name to index mapping is empty")
        if not wlo_artifacts:
            raise ValueError("WLO artifacts is missing")

        # Generate key for logging and tracking
        key = generate_key(hlo_module, key)
        logger.info(f"Applying layout transformation to {key}")

        # Read the WLO stub
        wlo_stub = read_hlo(wlo_artifacts.wrapped_neff_hlo_filename)
        if not wlo_stub:
            raise ValueError("Failed to read WLO stub")

        # Get the layout transformation map
        weight_name_to_transform_cpt = get_layout_transform_map(
            hlo_stub=wlo_stub, 
            weight_name_to_idx=weight_name_to_idx
        )

        # Create HLO artifacts
        hlo_artifacts = HloArtifacts(
            hlo_module=hlo_module,
            flattener=flattener,
            packer=packer,
            metaneff=metaneff,
            weights=None,
            constant_parameter_tensors=None,
            weight_name_to_idx=weight_name_to_idx
        )

        # Apply layout transformation
        append_layout_computation_to_hlo(hlo_artifacts, weight_name_to_transform_cpt)

        # Use temporary files
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.hlo') as original_hlo_file, \
             tempfile.NamedTemporaryFile(mode='w+', suffix='.hlo') as optimized_hlo_file:
            
            # Write original HLO to temporary file
            write_hlo(original_hlo_file.name, hlo_artifacts.hlo_module)

            # Convert the inputs in HLO to optimal shape
            convert_inputs_to_optimal_shape(original_hlo_file.name, optimized_hlo_file.name)

            # Read the optimized HLO and update the trace artifacts
            optimized_hlo = read_hlo(optimized_hlo_file.name)
            hlo_artifacts.hlo_module = cleanup_after_layout_transformation(
                optimized_hlo, 
                hlo_artifacts.hlo_module.frontend_attributes.map
            )

            logger.info(f"Successfully applied layout optimization to {key}")

    except Exception as e:
        raise RuntimeError(f"Layout transformation failed for {key}: {str(e)}") from e

    logger.info(f"Completed layout optimization process for {key}")


def get_layout_transform_map(hlo_stub: hlo_pb2.HloModuleProto, weight_name_to_idx: Dict[str, int]):
    """
    Return a map of weight layout transformation from the HLO stub, if the weight
    is transformed in the HLO stub
        {"weight_name": hlo_computation_proto}

    This map might not contain all the weights from weight_name_to_idx, because
    some of them could already be in the optimal layout, so there won't be
    a transformation for them in the hlo_stub.
    """
    weight_idx_to_name = {}
    for weight_name, idx in weight_name_to_idx.items():
        weight_idx_to_name[idx] = weight_name

    weight_name_to_transform_cpt = {}
    entry_cpt = hlo_entry_computation(hlo_stub)
    for instr in entry_cpt.instructions:
        if TRANSPOSABLE_WEIGHT_IDX in instr.frontend_attributes.map:
            priority_weight_idx = int(instr.frontend_attributes.map[TRANSPOSABLE_WEIGHT_IDX])
            # Compiler will always wrap the layout transformation into just one
            # custom-call, so getting the first called computation id is enough
            cpt = get_hlo_computation_by_id(hlo_stub, instr.called_computation_ids[0])
            weight_name = weight_idx_to_name[priority_weight_idx]
            weight_name_to_transform_cpt[weight_name] = cpt
    return weight_name_to_transform_cpt


def update_computation_id_and_name(src_cpt: hlo_pb2.HloComputationProto, start_id: int, name_prefix: str):
    """
    Update the id and name inside a computation.

    It will increase all ids inside the computation by `start_id`, and add a
    prefix of `name_prefix` for all var names inside the computation.
    """
    # Create a new one to avoid polluting the existing one
    cpt = hlo_pb2.HloComputationProto()
    cpt.CopyFrom(src_cpt)

    # update the id
    cpt.id += start_id
    cpt.root_id += start_id

    for instr in cpt.instructions:
        instr.id += start_id
        if len(instr.operand_ids) == 0:
            continue
        for idx in range(len(instr.operand_ids)):
            instr.operand_ids[idx] += start_id

    # update the name
    cpt.name = name_prefix + cpt.name
    for idx in range(len(cpt.program_shape.parameter_names)):
        cpt.program_shape.parameter_names[idx] = name_prefix + cpt.program_shape.parameter_names[idx]

    for instr in cpt.instructions:
        instr.name = name_prefix + instr.name

    return cpt


def append_layout_computation_to_hlo(
        hlo_artifact: HloArtifacts,
        weight_name_to_transform_cpt: Dict[str, hlo_pb2.HloComputationProto],
    ):
    """
    For each weight mentioned in the hlo_artifact.ho_module, if there is a
    computation corresponds to that weight in the map of
    `weight_name_to_transform_cpt`, append that computation to the end of
    `hlo_artifact.ho_module`.
    """
    hlo = hlo_artifact.hlo_module

    weight_idx_to_append = []
    layout_transform_cpt_to_append = []
    for weight_name, weight_idx in hlo_artifact.weight_name_to_idx.items():
        # We skip the weights if it is not in `weight_name_to_transform_cpt`
        # because they are already in optimal layout
        if weight_name in weight_name_to_transform_cpt:
            weight_idx_to_append.append(weight_idx)
            cpt = weight_name_to_transform_cpt[weight_name]
            # Need to update the id and name for the computation, to avoid
            # duplicate name or duplicate id in the whole hlo
            cpt = update_computation_id_and_name(cpt, start_id=hlo.id+1, name_prefix="wlt_")
            layout_transform_cpt_to_append.append(cpt)
    weight_idx_str = ",".join([str(idx) for idx in weight_idx_to_append])
    layout_transform_cpt_str = ",".join([cpt.name for cpt in layout_transform_cpt_to_append])

    hlo.frontend_attributes.map[REQUIRE_TRANSPOSE_WEIGHT_IDX] = weight_idx_str
    hlo.frontend_attributes.map[REQUIRE_TRANSPOSE_CUSTOM_CALL] = layout_transform_cpt_str

    # Compiler will be responsible to insert the layout transformation custom-calls into the HLO compute graph.
    hlo.computations.extend(layout_transform_cpt_to_append)  # extend() will copy the value
    return hlo_artifact


def extract_weight_layout_transform_hlo(
        hlo_stub: hlo_pb2.HloModuleProto,
        weight_name_to_idx: Dict[str, int],
    ):
    """
    Build a new HLO for weight layout transformation following the suggestion
    from the `hlo_stub`.

    After the transformation, the layout of some weights will change, but
    the other could stay the same, because they are already in the optimal
    layout.

    The resulting HLO will take all the weights in original layout as input,
    and output them in optimal layout. The number and order of the inputs and
    outputs are the same. The order of inputs (weights) is decided by the its
    index from the `weight_name_to_idx`, and it is in ascending order.

    This is because during the transformation, we will provide all the weights
    as input and expect to get all the weights in the same order from the output.
    """
    wlt_hlo = hlo_pb2.HloModuleProto()  # weight layout transformation HLO
    wlt_hlo.CopyFrom(hlo_stub)

    entry_cpt = hlo_entry_computation(wlt_hlo)
    weight_idx_to_info = {}

    # Step 1. Update the output of the root instruction in the entry computation
    # Find the weights whose layout will change, as part of the output
    for instr in entry_cpt.instructions:
        is_changed_weight = TRANSPOSABLE_WEIGHT_IDX in instr.frontend_attributes.map
        if is_changed_weight:
            idx = int(instr.frontend_attributes.map[TRANSPOSABLE_WEIGHT_IDX])
            weight_idx_to_info[idx] = (instr.id, instr.shape)

    # Find the weights whose layout won't change, as part of the output
    all_weight_idx = set(weight_name_to_idx.values())
    changed_weight_idx = set(weight_idx_to_info.keys())
    unchanged_weight_idx = list(all_weight_idx - changed_weight_idx)
    for instr in entry_cpt.instructions:
        is_unchanged_weight = instr.opcode == "parameter" and instr.parameter_number in unchanged_weight_idx
        if is_unchanged_weight:
            idx = instr.parameter_number
            weight_idx_to_info[idx] = (instr.id, instr.shape)

    weight_idx_list = sorted(weight_idx_to_info.keys())
    var_id_list = [weight_idx_to_info[w_idx][0] for w_idx in weight_idx_list]
    var_shape_list = [weight_idx_to_info[w_idx][1] for w_idx in weight_idx_list]

    # Update the root instrution to return all the weights
    root_instr = get_hlo_root_instruction(entry_cpt)
    root_instr.name = "last"
    root_instr.opcode = "tuple"
    root_instr.ClearField("shape")
    root_instr.shape.element_type = xla_data_pb2.PrimitiveType.TUPLE
    root_instr.shape.tuple_shapes.extend(var_shape_list)
    root_instr.ClearField("operand_ids")
    root_instr.operand_ids.extend(var_id_list)

    output_shape = root_instr.shape

    # Clear irrelevant fields
    # TODO: create a new instr instead of updating the old one
    root_instr.ClearField("custom_call_target")
    root_instr.ClearField("backend_config")
    root_instr.ClearField("constrain_layout")
    root_instr.ClearField("operand_shapes_with_layout")
    root_instr.ClearField("frontend_attributes")
    root_instr.ClearField("custom_call_api_version")
    root_instr.ClearField("precision_config")
    root_instr.ClearField("feature_group_count")
    root_instr.ClearField("batch_group_count")
    root_instr.ClearField("statistics_viz")

    # Step 2: Update output shape of the entry computation
    program_shape = entry_cpt.program_shape
    program_shape.result.CopyFrom(output_shape)

    # Step 3: Clean up instructions inside the entry computation that are not
    # for weight layout transformation
    reduced_instrs = []
    for instr in entry_cpt.instructions:
        is_not_weight = instr.opcode == "parameter" and instr.parameter_number not in all_weight_idx
        if is_not_weight:
            continue
        reduced_instrs.append(instr)
    entry_cpt.ClearField("instructions")
    entry_cpt.instructions.extend(reduced_instrs)

    # Step 4: Clean up inputs that are not used in the entry computation
    program_shape = entry_cpt.program_shape
    reduced_param_names = []
    reduced_param_shapes = []
    input_id_mapping = {}
    input_id_in_updated_hlo = 0
    for id, (param_name, param_shape) in enumerate(zip(program_shape.parameter_names, program_shape.parameters)):
        if id not in weight_idx_list:
            continue
        reduced_param_names.append(param_name)
        reduced_param_shapes.append(param_shape)
        input_id_mapping[id] = input_id_in_updated_hlo
        input_id_in_updated_hlo += 1

    program_shape.ClearField("parameter_names")
    program_shape.parameter_names.extend(reduced_param_names)
    program_shape.ClearField("parameters")
    program_shape.parameters.extend(reduced_param_shapes)

    for instr in entry_cpt.instructions:
        if instr.opcode == "parameter":
            instr.parameter_number = input_id_mapping[instr.parameter_number]

    # Step 5: Update the input and output of the HLO module
    host_program_shape = wlt_hlo.host_program_shape
    host_program_shape.result.CopyFrom(output_shape)
    host_program_shape.ClearField("parameters")
    host_program_shape.parameters.extend(reduced_param_shapes)
    host_program_shape.ClearField("parameter_names")
    host_program_shape.parameter_names.extend(reduced_param_names)

    # Step 6: add alias attributes for input and output
    # This is needed to avoid overwrite on the buffers when reusing the same
    # buffers for inputs and outputs in NEFF execution.
    # This assumes that the number of inputs and the number of outputs are
    # the same and assumes that they are in the same order.
    io_alias_proto = hlo_pb2.HloInputOutputAliasProto()
    for input_id in range(len(reduced_param_names)):
        alias_entry_proto = io_alias_proto.AliasEntryProto()

        alias_entry_proto.output_shape_index.append(input_id)
        alias_entry_proto.parameter_number = input_id
        alias_entry_proto.kind = hlo_pb2.Kind.MUST_ALIAS

        io_alias_proto.entries.append(alias_entry_proto)
    wlt_hlo.input_output_alias.CopyFrom(io_alias_proto)

    return wlt_hlo


def prepare_metaneff_for_wlt_hlo(
        wlt_hlo: hlo_pb2.HloModuleProto,
        weight_name_to_idx: Dict[str, int],
    ):
    """
    Generate a metaneff for weight layout transformation HLO.
    """
    metaneff = metaneff_pb2.MetaNeff()
    weight_name_sorted_by_idx: List[str] = [name.replace("->", ".") for _, name in sorted([(idx, name) for name, idx in weight_name_to_idx.items()])]

    entry_cpt = hlo_entry_computation(wlt_hlo)
    # Prepare meta for input_tensors
    for index, param_meta in enumerate(entry_cpt.program_shape.parameters):
        input_tensor = metaneff.input_tensors.add()
        # Needs to be `input#` to avoid a `ddrs_create_lookup_key` error
        input_tensor.name = f"input{index}".encode("utf8")
        input_tensor.shape[:] = list(param_meta.dimensions)
        input_tensor.data_type = XLA_DTYPE_TO_METANEFF_DTYPE[param_meta.element_type]
        input_tensor.type = metaneff_pb2.MetaTensor.Type.INPUT_WEIGHT
        input_tensor.checkpoint_key = weight_name_sorted_by_idx[index].encode("utf8")

    # Prepare meta for output_tensors
    for index, output_meta in enumerate(entry_cpt.program_shape.result.tuple_shapes):
        output_tensor = metaneff.output_tensors.add()
        output_tensor.name = f"output{index}".encode("utf8")
        output_tensor.shape[:] = list(output_meta.dimensions)
        output_tensor.data_type = XLA_DTYPE_TO_METANEFF_DTYPE[output_meta.element_type]
        output_tensor.checkpoint_key = weight_name_sorted_by_idx[index].encode("utf8")

    return metaneff


################################################################################
# Experimental feature for weight layout transformation
################################################################################

def update_weight(original_shape, second_shape, permute_order, x):
    """Get weight layout optimization function in torch"""
    assert x.shape == original_shape, f"actual shape is {x.shape}, but expected shape is {original_shape}"
    x = x.reshape(second_shape)
    x = x.permute(permute_order)
    x = x.contiguous()
    return x


def get_wlt(cpt):
    """Get weight layout optimization function in torch from HLO computation"""
    assert len(cpt.instructions) == 3

    read_instr = cpt.instructions[0]
    assert read_instr.opcode == "parameter"
    original_shape = tuple(read_instr.shape.dimensions[:])

    reshape_instr = cpt.instructions[1]
    assert reshape_instr.opcode == "reshape"
    second_shape = tuple(reshape_instr.shape.dimensions[:])

    permute_instr = cpt.instructions[2]
    assert permute_instr.opcode == "transpose"
    permute_order = permute_instr.dimensions[:]
    return partial(update_weight, original_shape, second_shape, permute_order)


def get_wlt_map(hlo):
    """Get a map from a weight to its corresponding layout optimization"""
    wlt_map = {}

    entry_cpt = hlo_entry_computation(hlo)
    param_id = 0
    for instr in entry_cpt.instructions:
        if instr.opcode == "parameter":
            logger.debug(f"- param_id = {param_id}, weight name: {instr.name}, weight shape: {instr.shape.dimensions[:]}")
            param_id += 1
        elif instr.opcode == "call":
            logger.debug(f"-- param_id = {param_id-1}, transformed weight shape: {instr.shape.dimensions[:]}")
            cpt_for_weight = get_hlo_computation_by_id(hlo, instr.called_computation_ids[0])
            wlt_map[param_id-1] = get_wlt(cpt_for_weight)
    logger.debug("Done, get_wlt_map")
    return wlt_map


def read_metaneff(metaneff_path):
    metaneff = metaneff_pb2.MetaNeff()
    with open(metaneff_path, "rb") as f:
        metaneff.ParseFromString(f.read())
    return metaneff


def get_input_order(metaneff):
    """Get inputs in an ordered list based on metaneff"""
    ckpt_names = []
    ckpt_shapes = []
    for input in metaneff.input_tensors:
        ckpt_names.append(input.checkpoint_key.decode())
        ckpt_shapes.append(input.shape)
    return ckpt_names, ckpt_shapes


def transform_weight_layout_on_cpu(
        hlo_filename,
        metaneff_filename,
        start_rank_id,
        local_ranks_size,
        sharded_checkpoint_dir,
    ):
    """
    Transform the weights on CPU using torch.

    This code path will be slower, but it is the most reliable way.
    """
    hlo = read_hlo(hlo_filename)
    wlt_map = get_wlt_map(hlo)

    metaneff = read_metaneff(metaneff_filename)
    ckpt_names, ckpt_shapes = get_input_order(metaneff)

    for rank in range(start_rank_id, start_rank_id + local_ranks_size):
        logger.info(f"- transforming weight for tp rank {rank}")
        ckpt_file = os.path.join(sharded_checkpoint_dir, f"tp{rank}_sharded_checkpoint.safetensors")
        ckpt = load_file(ckpt_file)

        for id, ckpt_name in enumerate(ckpt_names):
            logger.debug(f"-- processing weight, id: {id}, name: {ckpt_name}, shape: {ckpt_shapes[id]}")
            weight = ckpt[ckpt_name]
            if id in wlt_map:
                wlt = wlt_map[id]
                weight = wlt(weight)
                ckpt[ckpt_name] = weight
                logger.debug(f"--- updated layout for weight {ckpt_name} to be {weight.shape}")

        os.remove(ckpt_file) # Delete the orignal sharded checkpoint
        save_file(ckpt, ckpt_file)
        logger.info(f"Done, ckpt_file = {ckpt_file}")

    logger.info("Done layout transformation on CPU!")


def transform_weight_layout_on_device_and_save_to_disk(
        metaneff_filename,
        start_rank_id,
        local_ranks_size,
        wlt_neff_path,
        sharded_checkpoint_dir,
    ):
    """
    Transform the weights on neuron device, and then serialize them.

    This option needs the disk space to store the transformed weights,
    but it avoids the overhead to transform it everytime during load.
    """
    with open(wlt_neff_path, "rb") as f:
        wlt_neff =  f.read()

    with open(metaneff_filename, "rb") as f:
        metaneff = f.read()

    wlt_model = torch.classes.neuron.LayoutTransformation(wlt_neff, metaneff, local_ranks_size)

    checkpoint = []
    for rank in range(start_rank_id, start_rank_id + local_ranks_size):
        checkpoint_file = os.path.join(sharded_checkpoint_dir, f"tp{rank}_sharded_checkpoint.safetensors")
        checkpoint.append(load_file(checkpoint_file))

    transposed_ckpt = wlt_model.forward(checkpoint, True)
    for rank, transposed_ckpt_per_rank in zip(range(start_rank_id, start_rank_id + local_ranks_size), transposed_ckpt):
        for name, weight in transposed_ckpt_per_rank.items():
            logger.debug(f"tp_rank: {rank}, name: {name}, original shape: {checkpoint[0][name].shape}, new shape: {weight.shape}")

        os.remove(checkpoint_file)
        save_file(transposed_ckpt_per_rank, checkpoint_file)
        logger.debug(f"Done, ckpt_file = {checkpoint_file}")

    logger.info("Done layout transformation on device and serialization!")

def get_compiler_package_dir():
    return importlib.util.find_spec("neuronxcc").loader.get_filename() if importlib.util.find_spec("neuronxcc") else ''


def get_executable_full_qualified_path(executable: str) -> str:
    """ Get executable file's full qualified path from the `neuronxcc` folder provided in compiler .whl """
    package_dir = os.path.dirname(get_compiler_package_dir())
    paths = os.get_exec_path()

    for relative_path in [pathlib.Path("starfish/bin")]:
        paths.append(os.path.join(package_dir, str(relative_path)))

    path_directories = os.pathsep.join(paths)
    executable_location = shutil.which(executable, path=path_directories)
    if not executable_location:
        raise RuntimeError(f"Could not find {executable} in {paths}")

    return executable_location


def convert_inputs_to_optimal_shape(input_hlo_path: str, output_hlo_path: str) -> None:
    """ Convert the weights in input HLO to optimal shape using the `hlo-opt` executable """
    hlo_opt_executable_path = get_executable_full_qualified_path("hlo-opt")

    process_status = subprocess.run([
        hlo_opt_executable_path, "--input", input_hlo_path, "--output", output_hlo_path,
        "--passes", "convert-inputs-to-optimal-shape", "--input-type", "proto", "--output-type", "proto"
    ])
    if process_status.returncode != 0:
        raise RuntimeError(f"Error while converting hlo to optimized layout: {str(process_status.stderr)}")
    return


def cleanup_after_layout_transformation(
        optimized_hlo_module: hlo_pb2.HloModuleProto, old_frontend_attrs_map: dict
    ) -> hlo_pb2.HloModuleProto:
    """ Cleanup the transformation functions from HLO module once it has been transformed into optimal layout. """
    lambda_func_names = old_frontend_attrs_map[REQUIRE_TRANSPOSE_CUSTOM_CALL].split(FRONTEND_ATTRIBUTES_DELIMITER)

    if optimized_hlo_module.frontend_attributes.map:
        # Cleanup frontend attributes
        optimized_hlo_module.frontend_attributes.map.pop(REQUIRE_TRANSPOSE_CUSTOM_CALL)
        optimized_hlo_module.frontend_attributes.map.pop(REQUIRE_TRANSPOSE_WEIGHT_IDX)

    if lambda_func_names is None:
        # There's nothing to cleanup
        return optimized_hlo_module

    # Need to iterate over a copy to avoid modifying the list being iterated over
    computations_copy = deepcopy(optimized_hlo_module.computations)
    for cpt in computations_copy:
        if cpt.name in lambda_func_names:
            optimized_hlo_module.computations.remove(cpt)

    return optimized_hlo_module


def is_nki_kernel_called(id_to_computation: Dict[int, hlo_pb2.HloComputationProto], cpt_id: int) -> bool:
    """
    Identify if the given computation is making call to NKI Kernel or not. It recursively traces back the
    computations and looks for the call to AWS_NEURON_CUSTOM_NATIVE_KERNEL.
    """
    cpt = id_to_computation[cpt_id]
    for inst in cpt.instructions:
        # Iterate over all the instructions which are making calls to another computation or custom calls
        if inst.opcode == "custom-call":
            if inst.custom_call_target and inst.custom_call_target == AWS_NEURON_CUSTOM_NATIVE_KERNEL:
                return True
        elif inst.opcode == "call" and inst.called_computation_ids:
            return is_nki_kernel_called(id_to_computation, inst.called_computation_ids[0])

    return False


def prepare_parameter_usage_map(
    id_to_instruction_dict: Dict[int, hlo_pb2.HloInstructionProto], parameters_list: list
) -> dict:
    """
    Identify the instructions using parameters of entry computation. This function prepares a mapping of
    parameter id to the list of instruction ids consuming that paramter.
    """
    parameter_usage_map: Dict[int, list] = {}
    param_ids = [param.id for param in parameters_list]
    for inst in id_to_instruction_dict.values():
        if inst.opcode == "parameter":
            continue
        for operand in inst.operand_ids:
            if operand in param_ids:
                consumers = parameter_usage_map.get(operand, [])
                consumers.append(inst.id)
                parameter_usage_map[operand] = consumers
    return parameter_usage_map


def traceback_instruction_to_parameter(
        id_to_instruction_dict: Dict[int, hlo_pb2.HloInstructionProto], inst_id: int, traced_inst_ids: set
    ) -> Optional[hlo_pb2.HloInstructionProto]:
    """ Trace the given instruction id to the entry computation parameter it used recursively """
    inst = id_to_instruction_dict[inst_id]

    # If an instruction has already been traced
    if inst_id in traced_inst_ids:
        return None

    # If instruction is a parameter, return it and terminate recursion.
    if inst.opcode == "parameter":
        return inst

    # Proceed with recursion only if operation is reshape or transpose
    if inst.opcode == "reshape" or inst.opcode == "transpose":
        assert len(inst.operand_ids) == 1, f"reshape/transpose instruction '{inst.id}' should only take one operand"
        traced_inst_ids.add(inst_id)
        return traceback_instruction_to_parameter(id_to_instruction_dict, inst.operand_ids[0], traced_inst_ids)

    return None


def get_nki_kernel_weight_names(
        entry_cpt: hlo_pb2.HloComputationProto,
        idx_to_weight_name: Dict[int, str],
        nki_kernel_weight_names: set,
        id_to_computation: Dict[int, hlo_pb2.HloInstructionProto]
    ) -> set:
    """
    Utility function to identify the kernel weight names from HLO. It performs following operations:
    1. For each "call" instruction in Entry Cpt, identify if it makes kernel call.
    2. If a call to kernel is being made, identify the weight if:
        a. It is a parameter to entry computation call
        b. There are only reshape and transpose operations
        c. This weight is not being used for any other instruction in the entry cpt
    """
    kernel_call_inst_ids = set()
    id_to_instruction_dict = {i.id: i for i in entry_cpt.instructions}
    parameters_list = [i for i in entry_cpt.instructions if i.opcode == "parameter"]
    parameter_usage_map = prepare_parameter_usage_map(id_to_instruction_dict, parameters_list)
    for inst in entry_cpt.instructions:
        if inst.opcode == "call":
            assert len(inst.called_computation_ids) == 1, f"Multiple computation Ids found for {inst.id}"
            if is_nki_kernel_called(id_to_computation, inst.called_computation_ids[0]):
                kernel_call_inst_ids.update(inst.operand_ids)

    for inst_id in kernel_call_inst_ids:
        # Check if the instruction is a parameter and with only single consumer
        if inst_id in parameters_list and len(parameter_usage_map[inst_id]) == 1:
            param_number = id_to_instruction_dict[inst_id].parameter_number
            if param_number in idx_to_weight_name:
                nki_kernel_weight_names.add(idx_to_weight_name[param_number])
        else:
            # Traceback the instuction to a parameter
            traced_parameter = traceback_instruction_to_parameter(id_to_instruction_dict, inst_id, set())
            # Check if it only has single consumer
            if traced_parameter and len(parameter_usage_map[traced_parameter.id]) == 1:
                if traced_parameter.parameter_number in idx_to_weight_name:
                    nki_kernel_weight_names.add(idx_to_weight_name[traced_parameter.parameter_number])

    return nki_kernel_weight_names
