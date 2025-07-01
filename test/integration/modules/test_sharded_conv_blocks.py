import argparse
import atexit
import json
import os
import traceback
from datetime import datetime
import sys
import copy

from typing import Tuple

import torch
import torch.nn.init as init  # noqa: F401
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.debug.metrics as met

from neuronx_distributed.parallel_layers import layers, parallel_state
from neuronx_distributed.parallel_layers.pad import pad_model  # noqa: F401
from neuronx_distributed.parallel_layers.random import model_parallel_xla_manual_seed  # noqa: F401
from neuronx_distributed.parallel_layers.utils import requires_init_pg_override

import diffusers
from diffusers.models.unets.unet_2d_blocks import CrossAttnUpBlock2D, UpBlock2D, CrossAttnDownBlock2D, UNetMidBlock2DCrossAttn

import gc

datetime_str = str(datetime.now())

# Get the parent directory of the current directory
parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Add the parent directory to the sys.path
sys.path.append(parentdir)

# Import the module from the parent directory
from common.integration_test_utils import test_init, test_cleanup, test_modules, print_separator, set_random_seed  # noqa: E402, F401

UP_BLOCKS = (CrossAttnUpBlock2D, UpBlock2D)
CROSS_ATTN_BLOCKS = (CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn)


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--test_json",
        required=False,
        help="input json listing the test spec for network to compile",
    )
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument("--s3_bucket", default="s3://ktf-test-runs/neuronx_distributed_parallel_layers/layers")
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    with open(args.test_json, "r") as f:
        test_dict = json.load(f)
    return test_dict, S3_BUCKET_NAME, args


test_config, S3_BUCKET_NAME, args = parse_args()
results = {"inference_success": 1}

DOWN_BLOCK = "down_blocks"
MID_BLOCK = "mid_block"
UP_BLOCK = "up_blocks"


def get_sharded_data(data: torch.Tensor, dim: int) -> torch.Tensor:
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    per_partition_size = data.shape[dim] // parallel_state.get_tensor_model_parallel_size()
    if dim == 0:
        return data[
            per_partition_size * tp_rank: per_partition_size * (tp_rank + 1)
        ].clone()
    elif dim == 1:
        return data[
            :, per_partition_size * tp_rank: per_partition_size * (tp_rank + 1)
        ].clone()
    else:
        raise Exception(
            f"Partiton value of 0,1 are supported, found {dim}."
        )


# Shard a given Conv2d to be an OutputChannelParallelConv2d or InputChannelParallelConv2d,
# including copying the weight/bias data to the sharded conv from the original
def shard_conv2d(conv: torch.nn.Module, layer_type: type, gather_output: bool = None, input_is_parallel: bool = None) -> torch.nn.Module:
    allowed_layer_types = [layers.InputChannelParallelConv2d, layers.OutputChannelParallelConv2d]
    assert layer_type in allowed_layer_types, f"Requested layer must be one of {allowed_layer_types} but got {layer_type}"
    assert (layer_type == layers.InputChannelParallelConv2d and input_is_parallel is not None) or (layer_type == layers.OutputChannelParallelConv2d and gather_output is not None), "Must specify gather_output for OutputChannelParallelConv2d or input_is_parallel for InputChannelParallelConv2d"

    orig_conv = conv
    partition_dim = 0 if layer_type == layers.OutputChannelParallelConv2d else 1
    kw = {'gather_output': gather_output} if layer_type == layers.OutputChannelParallelConv2d else {'input_is_parallel': input_is_parallel}
    conv = layer_type(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None, **kw)
    conv.weight.data = get_sharded_data(orig_conv.weight.data, partition_dim)
    if orig_conv.bias is not None:
        if layer_type == layers.OutputChannelParallelConv2d:
            conv.bias.data = get_sharded_data(orig_conv.bias.data, 0)
        else:
            # InputChannelParallel bias not sharded
            conv.bias.data.copy_(orig_conv.bias.data)

    del orig_conv

    return conv


def shard_groupnorm(norm: torch.nn.Module) -> torch.nn.Module:
    tp_degree = parallel_state.get_tensor_model_parallel_size()
    if norm.num_channels % tp_degree != 0 or (norm.num_channels // tp_degree) % norm.num_groups != 0:
        raise NotImplementedError(f"Have not implemented padding for norms yet. Cannot shard {norm} to TP degree {tp_degree}")

    orig_norm = norm
    norm = torch.nn.GroupNorm(orig_norm.num_groups, orig_norm.num_channels // tp_degree, orig_norm.eps, orig_norm.affine)
    norm.weight.data = get_sharded_data(orig_norm.weight.data, 0)
    norm.bias.data = get_sharded_data(orig_norm.bias.data, 0)

    return norm


def shard_sd_resnet_block(block: torch.nn.Module) -> torch.nn.Module:
    assert hasattr(block, 'conv1') and hasattr(block, 'conv2'), f"Expected the module being tested has a conv1 and conv2 to shard but found it doesn't! Selected module: {block}"

    # We can shard all the operators between conv1 and conv2 iff the sharded conv will still have an integer number of
    # groups in norm1
    # Otherwise, we'd have to pad the GroupNorm, which we don't support yet
    # TODO: short-circuiting to turn off this sharding routine because gradient checks fail if we shard this way
    #       V1319168276
    # Also, to do this we'll need to update the gradient gathering functions in integration_test_utils.py to account for the
    # sharded GroupNorm operator
    # Leaving this scaffolding in because this is the intended design for SD resnet block sharding
    if False and (block.conv1.out_channels // parallel_state.get_tensor_model_parallel_size()) % block.norm1.num_groups == 0:
        block.conv1 = shard_conv2d(block.conv1, layers.OutputChannelParallelConv2d, gather_output=False)
        block.conv2 = shard_conv2d(block.conv2, layers.InputChannelParallelConv2d, input_is_parallel=True)
        block.norm2 = shard_groupnorm(block.norm2)
        orig_time_emb_proj = block.time_emb_proj
        block.time_emb_proj = layers.ColumnParallelLinear(orig_time_emb_proj.in_features, orig_time_emb_proj.out_features, orig_time_emb_proj.bias is not None, gather_output=False)
        block.time_emb_proj.weight.data = get_sharded_data(orig_time_emb_proj.weight.data, 0)
        if orig_time_emb_proj.bias is not None:
            block.time_emb_proj.bias.data = get_sharded_data(orig_time_emb_proj.bias.data, 0)
    else:
        block.conv1 = shard_conv2d(block.conv1, layers.OutputChannelParallelConv2d, gather_output=True)
        block.conv2 = shard_conv2d(block.conv2, layers.OutputChannelParallelConv2d, gather_output=True)

    if hasattr(block, 'conv_shortcut') and block.conv_shortcut is not None:
        block.conv_shortcut = shard_conv2d(block.conv_shortcut, layers.OutputChannelParallelConv2d, gather_output=True)

    return block


# model: HuggingFace model ID, e.g. stabilityai/stable-diffusion-2-1-base
# block_type: down block, mid block, or up block
# block_idx: index of the block in the list of selected block type, e.g. which downblock
# resnet_idx: which resnet of the block to test
# input_spatial_dim: H/W of the input tensor
# batch_size: batch size for input
def test_stable_diffusion_resnet_block(tensor_model_parallel_size: int, model_id: str, block_type: str, block_idx: int, resnet_idx: int, input_spatial_dim: int, batch_size: int) -> Tuple[bool, bool, bool]:
    def _test_stable_diffusion_resnet_block():
        test_init(tensor_model_parallel_size, 1234)

        # All ranks other than master wait here so only one rank is downloading the model
        if not xm.is_master_ordinal():
            xm.rendezvous("download model")
        # Download model from HuggingFace and extract UNet
        model = diffusers.UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        if xm.is_master_ordinal():
            xm.rendezvous("download model")

        assert hasattr(model, block_type), f"Selected model doesn't have attribute {block_type}!"

        blocks = getattr(model, block_type)
        # Mid block isn't a list so wrap it in one so the rest of our flow is common
        if block_type == MID_BLOCK:
            blocks = [blocks]

        assert hasattr(blocks[block_idx], 'resnets'), f"{blocks[block_idx]}\nExpected the module being tested (see above) has an attribute 'resnets' (a list of resnet blocks) but found it doesn't!"

        control_module = copy.deepcopy(blocks[block_idx].resnets[resnet_idx])
        test_module = copy.deepcopy(blocks[block_idx].resnets[resnet_idx])
        del model

        # Build the input tuple
        input_channels = test_module.conv1.in_channels
        temb_in_features = test_module.time_emb_proj.in_features
        hidden_states_shape = (batch_size, input_channels, input_spatial_dim, input_spatial_dim)
        temb_shape = (batch_size, temb_in_features)

        # Create input tensor, copy weights from test layer to control layer
        input_hidden_states = torch.randn(hidden_states_shape, requires_grad=True)
        input_temb = torch.randn(temb_shape, requires_grad=True)
        input_tuple = (input_hidden_states, input_temb)

        # Shard the test module:
        test_module = shard_sd_resnet_block(test_module)

        # Don't check the output tensor, some configurations fail but still pass the gradient check
        # See V1310769999
        pass_fail = test_modules(test_module, control_module, input_tuple, check_output_tensor=False)
        test_cleanup()

        return pass_fail

    global results
    try:
        ret = None
        ret = _test_stable_diffusion_resnet_block()
        assert all(ret), "Test failed!"
        # If we reach this point, test has passed
        xm.master_print("test passed")
    except Exception:
        results["inference_success"] = 0
        print(traceback.format_exc())
        print(f"test_stable_diffusion_resnet_block FAILED for {model_id}.{block_type}[{block_idx}].resnets[{resnet_idx}], input size {input_size}")
        test_cleanup()
        # raise
        if ret is None:
            # Compilation failed
            ret = (False, False, False)

    gc.collect()

    return ret


# model: HuggingFace model ID, e.g. stabilityai/stable-diffusion-2-1-base
# block_type: down block, mid block, or up block
# block_idx: index of the block in the list of selected block type, e.g. which downblock
# input_spatial_dim: H/W of the input tensor
# batch_size: batch size for input
def test_stable_diffusion_unet_block(tensor_model_parallel_size: int, model_id: str, block_type: str, block_idx: int, input_spatial_dim: int, batch_size: int) -> Tuple[bool, bool, bool]:
    def _test_stable_diffusion_unet_block():
        test_init(tensor_model_parallel_size, 1234)

        # All ranks other than master wait here so only one rank is downloading the model
        if not xm.is_master_ordinal():
            xm.rendezvous("download model")
        # Download model from HuggingFace and extract UNet
        model = diffusers.UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        if xm.is_master_ordinal():
            xm.rendezvous("download model")

        assert hasattr(model, block_type), f"Selected model doesn't have attribute {block_type}!"

        blocks = getattr(model, block_type)
        # Mid block isn't a list so wrap it in one so the rest of our flow is common
        if block_type == MID_BLOCK:
            blocks = [blocks]

        control_module = copy.deepcopy(blocks[block_idx])
        test_module = copy.deepcopy(blocks[block_idx])
        del model

        assert hasattr(control_module, 'resnets'), f"{control_module}\nExpected the module being tested (see above) has an attribute 'resnets' (a list of resnet blocks) but found it doesn't!"

        input_channels = test_module.resnets[0].conv1.in_channels
        temb_in_features = test_module.resnets[0].time_emb_proj.in_features

        # Build the input tuple
        input_list = []

        if isinstance(test_module, UP_BLOCKS):
            input_channels = input_channels // 2

        hidden_states_shape = (batch_size, input_channels, input_spatial_dim, input_spatial_dim)
        input_hidden_states = torch.randn(hidden_states_shape, requires_grad=True)

        input_list.append(input_hidden_states)

        if isinstance(test_module, UP_BLOCKS):
            # Up blocks take an extra argument res_hidden_states_tuple of the Resnet hidden states from
            # the downblocks
            res_hidden_states = []
            for i, resnet in enumerate(test_module.resnets):
                # Cross attn upblocks vary their resnet sizes, so need to choose the right number of channels
                if isinstance(test_module, CROSS_ATTN_BLOCKS) and i != 0:
                    input_channels = resnet.conv1.in_channels - test_module.attentions[i - 1].proj_out.out_features
                else:
                    input_channels = resnet.conv1.in_channels // 2

                shape = (batch_size, input_channels, input_spatial_dim, input_spatial_dim)
                xm.master_print(f"computed shape of {shape} for resnet {i}")
                res_hidden_states.append(torch.randn(shape, requires_grad=True))
            # UNet blocks iterate the res_hidden_states tuple from the back, see
            # https://github.com/huggingface/diffusers/blob/5266ab7935dd9e9aec596cdc2464badf1eacd99a/src/diffusers/models/unets/unet_2d_blocks.py#L2363-L2366
            res_hidden_states.reverse()
            res_hidden_states = tuple(res_hidden_states)
            input_list.append(res_hidden_states)

        temb_shape = (batch_size, temb_in_features)
        input_temb = torch.randn(temb_shape, requires_grad=True)

        input_list.append(input_temb)

        if isinstance(test_module, CROSS_ATTN_BLOCKS):
            # Cross attn blocks need an extra arguments encoder_hidden_states
            encoder_hidden_states_shape = (batch_size, input_spatial_dim**2, input_spatial_dim**2)
            input_encoder_hidden_states = torch.randn(encoder_hidden_states_shape, requires_grad=True)
            input_list.append(input_encoder_hidden_states)

        input_tuple = tuple(input_list)

        # Shard the test module
        for i, resnet_block in enumerate(test_module.resnets):
            test_module.resnets[i] = shard_sd_resnet_block(resnet_block)

        # TODO: the mark_step_between_fwd_bwd is a hack to make fewer tests fail
        #       V1318912667
        #       V1319115668
        pass_fail = test_modules(test_module, control_module, input_tuple, mark_step_between_fwd_bwd=True, check_output_tensor=False)
        # If we reach this point, test has passed
        test_cleanup()

        return pass_fail

    global results
    try:
        ret = None
        ret = _test_stable_diffusion_unet_block()
        assert all(ret), "Test failed!"
        xm.master_print("test passed")
    except Exception:
        results["inference_success"] = 0
        print(traceback.format_exc())
        print(f"test_stable_diffusion_unet_block FAILED for {model_id}.{block_type}[{block_idx}], input size {input_size}")
        test_cleanup()
        # raise
        if ret is None:
            # Compilation failed
            ret = (False, False, False)

    gc.collect()

    return ret


def upload_to_s3():
    os.system(f'aws s3 cp --no-progress "{datetime_str}" {S3_BUCKET_NAME}')
    print(met.metrics_report())


def on_exit():
    upload_to_s3()
    for k in test_config:
        os.system(f"rm {args.test_json}")
        with open(args.test_json, "w") as f:
            json.dump({k: results}, f)


if __name__ == "__main__":
    # We want cnn-training to enable attention kernel matching
    # Otherwise compilation takes forever and gets really poor QoR
    compiler_flags = """ --model-type=cnn-training """
    os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + compiler_flags

    if requires_init_pg_override():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    world_size = xr.world_size()
    # TODO: Data parallel disabled for now because of V1320578643
    # TODO: Iterate over other tensor parallel sizes like the other tests do. Ran into compiler issues when trying it
    #       while building the test.
    tensor_model_parallel_size = 2

    if xm.is_master_ordinal():
        test_results_csv_file = open("sharded_conv_functional_block_test_results.csv", "w+")
        test_results_csv_file.write("module_id,batch_size,input_shape,compile_pass,output_pass,grads_pass,test_pass\n")
        test_results_csv_file.flush()

    # TODO: test more models - at least SD 1.5 and SD 2.1 non-base
    model_id = "stabilityai/stable-diffusion-2-1-base"

    unet = diffusers.UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    # Dict of BLOCK_TYPE : list[int], where the list is the number of resnets that each block of that type has
    # This allows us to test every block in the UNet
    num_resnets_per_block = {DOWN_BLOCK: [], MID_BLOCK: [], UP_BLOCK: []}
    for block in unet.down_blocks:
        num_resnets_per_block[DOWN_BLOCK].append(len(block.resnets))
    num_resnets_per_block[MID_BLOCK].append(len(unet.mid_block.resnets))
    for block in unet.up_blocks:
        num_resnets_per_block[UP_BLOCK].append(len(block.resnets))
    del unet
    for block_type in [DOWN_BLOCK, MID_BLOCK, UP_BLOCK]:
        for block_idx, num_resnets in enumerate(num_resnets_per_block[block_type]):
            for resnet_idx in range(0, num_resnets):
                # TODO: Enable more resolutions. Fails today, see V1310769999
                for input_size in [32, 64]:
                    block_identifier = f"{model_id}.{block_type}[{block_idx}].resnets[{resnet_idx}]"
                    print_separator(f"test {block_identifier}, input size {input_size}")
                    pass_fail = test_stable_diffusion_resnet_block(tensor_model_parallel_size, model_id, block_type, block_idx, resnet_idx, input_size, 1)
                    if xm.is_master_ordinal():
                        test_results_csv_file.write(f"{block_identifier},{1},{input_size},{pass_fail[0]},{pass_fail[1]},{pass_fail[2]},{all(pass_fail)}\n")
                        test_results_csv_file.flush()
                    xm.mark_step()

    # TODO: mid and up blocks fail for various issues, tracked in the following tickets
    #       V1317582055
    #       V1317572506
    #       V1317582055
    for block_type in [DOWN_BLOCK]:
        for block_idx, _ in enumerate(num_resnets_per_block[block_type]):
            # TODO: can we test more resolutions?
            for input_size in [32]:
                block_identifier = f"{model_id}.{block_type}[{block_idx}]"
                print_separator(f"test {block_identifier}, input size {input_size}")
                pass_fail = test_stable_diffusion_unet_block(tensor_model_parallel_size, model_id, block_type, block_idx, input_size, 1)
                if xm.is_master_ordinal():
                    test_results_csv_file.write(f"{block_identifier},{1},{input_size},{pass_fail[0]},{pass_fail[1]},{pass_fail[2]},{all(pass_fail)}\n")
                    test_results_csv_file.flush()
                xm.mark_step()

    if xm.is_master_ordinal():
        test_results_csv_file.close()
    atexit.register(on_exit)
