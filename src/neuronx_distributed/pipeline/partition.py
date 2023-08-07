from collections import OrderedDict
from typing import Any, List, Optional

import torch
from torch.fx.passes.split_module import split_module

from neuronx_distributed.parallel_layers.parallel_state import rmsg
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.utils.serialization import TensorMeta

logger = get_logger()


def partition_traced_model(traced_model):
    """
    Partition a traced model based on annotations
    """
    curr_stage_id = 0
    # "partition" will mark the cut stage
    # node.meta["partition"] will mark the stage for current op
    for node in traced_model.graph.nodes:
        if "partition" not in node.meta:
            node.meta["partition"] = curr_stage_id
        else:
            node.meta["partition"] = curr_stage_id
            curr_stage_id += 1
    mod_after_split = split_module(
        traced_model,
        None,
        lambda node: node.meta["partition"],
        keep_original_order=True,
    )
    return mod_after_split


class PipelineIO:
    def __init__(
        self,
        name: str,
        input_idx: Optional[int] = None,
        output_idx: Optional[int] = None,
        metadata: Optional[List[TensorMeta]] = None,
        obj: Any = None,
    ):
        """
        IO object that is used as annotation of communications between the pipeline stages
        Inputs:
            name: FX graph node name for this IO object
            input_idx: The index of this object if it is used as stage input,
                       None means it is not used as stage input (either output or pass along)
            output_idx: The index of this object if it is used as stage output,
                        None means it is not used as stage output (either input or pass along)
            metadata: If this IO contains tensors, this will be the metadata of the tensors
            obj: The python objection that carry this IO, None means it is a tensor.
        """
        self.name = name
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.metadata = metadata
        self.obj = obj

    def __repr__(self):
        return f"PipelineIO_{self.name}_input_idx_{self.input_idx}_output_idx_{self.output_idx}_metadata_{self.metadata}"  # noqa: E501


def adding_live_obj_for_previous_stages(
    stage_id_to_IO_input_names, stage_id_2_IO_output_names, obj_name, current_stage
):
    """
    If the object is not from the model input, it must from one of the outputs of previous stages.
    We need to pass this object through all stages.
    This function will track from current_stage to the first stage and pass the object along the way.
    """
    if current_stage < 0:
        raise RuntimeError(
            f"{obj_name} is missing from all previous stages, stage_id_2_IO_output_names {stage_id_2_IO_output_names}"  # noqa: E501
        )
    if (
        obj_name not in stage_id_to_IO_input_names[current_stage]
        and obj_name not in stage_id_2_IO_output_names[current_stage]
    ):
        # This object is from previous stage
        adding_live_obj_for_previous_stages(
            stage_id_to_IO_input_names, stage_id_2_IO_output_names, obj_name, current_stage - 1
        )
        # Add the obj as both input and output for current stage
        stage_id_to_IO_input_names[current_stage][obj_name] = PipelineIO(obj_name)
        stage_id_2_IO_output_names[current_stage][obj_name] = PipelineIO(obj_name)
    else:
        if obj_name in stage_id_2_IO_output_names[current_stage]:
            # Do nothing, this tensor will be passed to next stage
            return
        else:
            # Object is consumed in this stage, pass it to next stage as well
            stage_id_2_IO_output_names[current_stage][obj_name] = PipelineIO(obj_name)


def iterate_graph_model_outputs(output_node_args):
    """
    Input:
        output_node_args: The args of the output node in FX GraphModule
    Assume the args of output node always has the following rules:
        - args of the output node will always be a tuple with length 1, i.e. output_node_args = (`output_`, )
        - if there is only a single output, `output_` will be a graph node
        - if there are multiple outputs, `output_` will be a tuple as well
    """

    # [TODO] Verify it with multiple models
    assert (
        isinstance(output_node_args, tuple) and len(output_node_args) == 1
    ), f"Unsupported output args found {output_node_args}"
    # Current stage contains multiple outputs
    if isinstance(output_node_args[0], tuple):
        outputs = output_node_args[0]
    else:
        # Current stage only has a single output
        outputs = output_node_args
    for out in outputs:
        yield out


def analyze_pipeline_module(top_mod):
    """
    Analyze the stage inputs/outputs. Anything with IO in the name requires communication between stages
    Ouputs:
        stage_id_to_IO_input_names(dict): Stage id to a dict mapping IO input names to PipelineIOs.
                                         Inputs sequence is guarantee across PP ranks, so that send/recv will be in same order # noqa: E501
        stage_id_to_model_input_names(dict): Stage id to a dict mapping model input name to the index of current stage input
        stage_id_to_input_count(dict): Stage id to current stage input counts
        stage_id_to_output_count(dict): Stage id to current stage output counts
    """

    def get_name(node_or_str):
        if isinstance(node_or_str, torch.fx.node.Node):
            return node_or_str.name
        if isinstance(node_or_str, str):
            return node_or_str
        raise RuntimeError(f"Unsupported type {node_or_str} {type(node_or_str)}")

    curr_stage_id = 0
    stage_id_to_IO_input_names = {}
    stage_id_2_IO_output_names = {}
    stage_id_to_model_input_names = {}
    stage_id_to_input_count = {}
    stage_id_to_output_count = {}
    model_inputs = set()
    for node in top_mod.graph.nodes:
        if node.op == "placeholder":
            model_inputs.add(get_name(node))

    # Each named child is a partition
    for n, mod in top_mod.named_children():
        assert n.startswith("submod_"), f"The partition model needs to start with submod, but getting {n}"
        # Use ordered dict to enforce order when iterating through model inputs/outputs
        # This is required to make the send/recvs between stages
        stage_id_to_IO_input_names[curr_stage_id] = OrderedDict()
        stage_id_2_IO_output_names[curr_stage_id] = OrderedDict()
        stage_id_to_model_input_names[curr_stage_id] = {}
        stage_id_to_input_count[curr_stage_id] = 0
        stage_id_to_output_count[curr_stage_id] = 0
        for node in mod.graph.nodes:
            # Stage input
            if node.op == "placeholder":
                if get_name(node) not in model_inputs:
                    # From previous stage
                    adding_live_obj_for_previous_stages(
                        stage_id_to_IO_input_names,
                        stage_id_2_IO_output_names,
                        get_name(node),
                        curr_stage_id - 1,
                    )
                    stage_id_to_IO_input_names[curr_stage_id][get_name(node)] = PipelineIO(
                        get_name(node), input_idx=stage_id_to_input_count[curr_stage_id]
                    )
                else:
                    # Model input
                    stage_id_to_model_input_names[curr_stage_id][get_name(node)] = stage_id_to_input_count[
                        curr_stage_id
                    ]
                stage_id_to_input_count[curr_stage_id] += 1
            # Stage output
            elif node.op == "output":
                for idx, arg in enumerate(iterate_graph_model_outputs(node.args)):
                    stage_id_2_IO_output_names[curr_stage_id][get_name(arg)] = PipelineIO(get_name(arg), output_idx=idx)
                    stage_id_to_output_count[curr_stage_id] += 1
        curr_stage_id += 1

    logger.debug(rmsg(f"stage_id_to_IO_input_names {stage_id_to_IO_input_names}"))
    logger.debug(rmsg(f"stage_id_2_IO_output_names {stage_id_2_IO_output_names}"))

    # Current stage output should align with next stage input
    # We just need to keep one copy, since we do not need last stage output info
    # we keep the stage_id_to_IO_input_names
    for i in range(1, curr_stage_id):
        if set(stage_id_to_IO_input_names[i].keys()) != set(stage_id_2_IO_output_names[i - 1].keys()):
            raise RuntimeError(
                f"Stage {i}'s IO inputs {set(stage_id_to_IO_input_names[i].keys())} does not match stage {i-1}'s output {set(stage_id_2_IO_output_names[i-1].keys())}"  # noqa: E501
            )
        # Update the output index as the source of the current input io
        for name, io in stage_id_to_IO_input_names[i].items():
            io.output_idx = stage_id_2_IO_output_names[i - 1][name].output_idx

    logger.debug(rmsg(f"stage_id_to_model_input_names {stage_id_to_model_input_names}"))
    logger.debug(rmsg(f"stage_id_to_input_count {stage_id_to_input_count}"))
    logger.debug(rmsg(f"stage_id_to_output_count {stage_id_to_input_count}"))
    return (
        stage_id_to_IO_input_names,
        stage_id_to_model_input_names,
        stage_id_to_input_count,
        stage_id_to_output_count,
    )


def analyze_shared_weights_across_stages(top_module, partitions):
    """
    Find the shared weight between stages.
    Output:
        shared_weights(list): A list that each entry is a list of shared parameter info tuple
                              with (name, stage). Note name is from local module.
    """
    param_to_partition = {p: [] for p in top_module.parameters()}
    for stage, partition in enumerate(partitions):
        for name, p in partition.named_parameters():
            if p not in param_to_partition:
                # not shared parameter or shared parameter on first occurance stage
                # this attr will be use to calculate global grad norm
                setattr(p, "shared", False)
            else:
                # shared parameter on rest stages
                setattr(p, "shared", True)
            param_to_partition[p].append((name, stage))
    shared_weights = []
    for weights in param_to_partition.values():
        if len(weights) > 1:
            shared_weights.append(weights)
    return shared_weights
