from typing import Dict

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


def read_hlo(hlo_path: str):
    """Read a HLOModuleProto from given path"""
    hlo = hlo_pb2.HloModuleProto()
    with open(hlo_path, "rb") as f:
        hlo.ParseFromString(f.read())
    return hlo


def add_weight_idx_attr_to_hlo(hlo: hlo_pb2.HloModuleProto, weight_name_to_idx: Dict[str, int]):
    """
    Add frontend attributes on weight indices for weights
    """
    weight_idx = sorted(weight_name_to_idx.values())
    weight_idx_list_str = ",".join([str(idx) for idx in weight_idx])
    hlo.frontend_attributes.map[TRANSPOSABLE_WEIGHT_IDX] = weight_idx_list_str
    return hlo


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
    
    return wlt_hlo


def prepare_metaneff_for_wlt_hlo(
        wlt_hlo: hlo_pb2.HloModuleProto,
        weight_name_to_idx: Dict[str, int],
    ):
    """
    Generate a metaneff for weight layout transformation HLO.
    """
    metaneff = metaneff_pb2.MetaNeff()
    weight_name_sorted_by_idx = sorted([(idx, name) for name, idx in weight_name_to_idx.items()])
    weight_name_sorted_by_idx = [name.replace("->", ".") for (idx, name) in weight_name_sorted_by_idx]

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
