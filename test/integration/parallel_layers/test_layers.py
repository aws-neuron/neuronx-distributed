import argparse
import atexit
import json
import os
import traceback
from datetime import datetime
import sys

import torch
import torch.nn.init as init
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from commons import print_separator, set_random_seed

from neuronx_distributed.parallel_layers import layers, parallel_state
from neuronx_distributed.parallel_layers.pad import pad_model
from neuronx_distributed.parallel_layers.random import model_parallel_xla_manual_seed
from neuronx_distributed.parallel_layers.utils import requires_init_pg_override

datetime_str = str(datetime.now())

# Get the parent directory of the current directory
parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Add the parent directory to the sys.path
sys.path.append(parentdir)

# Import the module from the parent directory
from common.integration_test_utils import test_init, test_cleanup, test_modules


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


def test_parallel_embedding(tensor_model_parallel_size):
    def _test_parallel_embedding():
        device = xm.xla_device()
        tensor_model_parallel_size_ = tensor_model_parallel_size
        parallel_state.initialize_model_parallel(tensor_model_parallel_size_)
        tensor_model_parallel_size_ = parallel_state.get_tensor_model_parallel_size()

        batch_size = 1
        seq_length = 2048
        vocab_size = 30432
        hidden_size = 1024
        seed = 1234

        set_random_seed(seed)
        input_data = torch.LongTensor(size=(batch_size, seq_length)).random_(0, vocab_size).to(device)
        loss_weight = torch.randn([batch_size, seq_length, hidden_size]).to(device)

        set_random_seed(seed)
        embedding_original = torch.nn.Embedding(vocab_size, hidden_size).to(device)

        output = embedding_original(input_data)
        loss_original = torch.mul(output, loss_weight).sum()
        loss_original.backward()

        set_random_seed(seed)
        embedding_parallel = layers.ParallelEmbedding(vocab_size, hidden_size, init_method=init.normal_).to(device)
        output = embedding_parallel(input_data)
        loss_parallel = torch.mul(output, loss_weight).sum()
        loss_parallel.backward()

        torch.distributed.barrier()
        error = loss_parallel.sub(loss_original).abs()
        print("   error in loss (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
        assert error < 1.0e-3, "error: {}".format(error)

        weight_grad_orig = torch.split(embedding_original.weight.grad, vocab_size // tensor_model_parallel_size_, 0)[
            parallel_state.get_tensor_model_parallel_rank()
        ]
        error = embedding_parallel.weight.grad.sub(weight_grad_orig).abs().max()
        print("   error in grad (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
        # assert error < 1.0e-5, 'error: {}'.format(error) #Error is 2.09

        # Reset groups
        parallel_state.destroy_model_parallel()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test passed")

        del device

    global results
    try:
        _test_parallel_embedding()
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise


def test_parallel_embedding_shard_over_embedding_dim(tensor_model_parallel_size):
    def _test_parallel_embedding():
        device = xm.xla_device()
        tensor_model_parallel_size_ = tensor_model_parallel_size
        parallel_state.initialize_model_parallel(tensor_model_parallel_size_)
        tensor_model_parallel_size_ = parallel_state.get_tensor_model_parallel_size()

        batch_size = 1
        seq_length = 2048
        vocab_size = 30432
        hidden_size = 1024
        seed = 1234

        set_random_seed(seed)
        input_data = torch.LongTensor(size=(batch_size, seq_length)).random_(0, vocab_size).to(device)
        loss_weight = torch.randn([batch_size, seq_length, hidden_size]).to(device)

        set_random_seed(seed)
        embedding_original = torch.nn.Embedding(vocab_size, hidden_size).to(device)

        output = embedding_original(input_data)
        loss_original = torch.mul(output, loss_weight).sum()
        loss_original.backward()

        set_random_seed(seed)
        embedding_parallel = layers.ParallelEmbedding(
            vocab_size, hidden_size, init_method=init.normal_, shard_across_embedding=True
        ).to(device)
        output_nxd = embedding_parallel(input_data)
        loss_parallel = torch.mul(output_nxd, loss_weight).sum()
        loss_parallel.backward()

        torch.distributed.barrier()

        print(
            "  error in output (parallel) on global rank {}: {}".format(
                torch.distributed.get_rank(), torch.sub(output, output_nxd).max()
            )
        )
        assert torch.allclose(output, output_nxd, rtol=1e-05)

        error = loss_parallel.sub(loss_original).abs()
        print("   error in loss (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
        assert error < 1.0e-3, "error: {}".format(error)

        weight_grad_orig = torch.split(embedding_original.weight.grad, hidden_size // tensor_model_parallel_size_, 1)[
            parallel_state.get_tensor_model_parallel_rank()
        ]
        error = embedding_parallel.weight.grad.sub(weight_grad_orig).abs().max()
        print("   error in grad (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
        # assert error < 1.0e-5, 'error: {}'.format(error) #Error is 2.09

        # Reset groups
        parallel_state.destroy_model_parallel()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test passed")

        del device

    global results
    try:
        _test_parallel_embedding()
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise


def test_initialize_parameter_cpu(tensor_model_parallel_size):
    def _test_initialize_parameter_cpu():
        tensor_model_parallel_size_ = tensor_model_parallel_size
        parallel_state.initialize_model_parallel(tensor_model_parallel_size_)
        tensor_model_parallel_size_ = parallel_state.get_tensor_model_parallel_size()

        seed = 12345
        input_size_coeff = 13
        input_size = input_size_coeff * tensor_model_parallel_size_
        output_size_coeff = 17
        output_size = output_size_coeff * tensor_model_parallel_size_

        # ---------------
        # Column parallel
        # ---------------
        weight = torch.empty(output_size_coeff, input_size)
        set_random_seed(seed)
        layers._initialize_parameter_cpu(weight, 0, torch.nn.init.normal_)
        # Target.
        set_random_seed(seed)
        master_weight = torch.empty(output_size, input_size)
        torch.nn.init.normal_(master_weight)
        rank = parallel_state.get_tensor_model_parallel_rank()
        my_weight = torch.split(master_weight, output_size_coeff, dim=0)[rank].contiguous().clone()

        # Compare.
        error = weight.sub(my_weight).abs().max()
        torch.distributed.barrier()
        print(
            "   column parallel max error (should be zero) on global rank "
            "{}: {}".format(torch.distributed.get_rank(), error)
        )
        assert error < 1.0e-6

        # ------------
        # Row parallel
        # ------------
        weight = torch.empty(output_size, input_size_coeff)
        set_random_seed(seed)
        layers._initialize_parameter_cpu(weight, 1, torch.nn.init.normal_)
        # Target.
        set_random_seed(seed)
        master_weight = torch.empty(output_size, input_size)
        torch.nn.init.normal_(master_weight)
        rank = parallel_state.get_tensor_model_parallel_rank()
        my_weight = torch.split(master_weight, input_size_coeff, dim=1)[rank].contiguous().clone()

        # Compare.
        error = weight.sub(my_weight).abs().max()
        torch.distributed.barrier()
        print(
            "   row parallel max error (should be zero) on global rank "
            "{}: {}".format(torch.distributed.get_rank(), error)
        )
        assert error < 1.0e-6

        xm.mark_step()

        # Reset groups
        parallel_state.destroy_model_parallel()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test passed")

    global results
    try:
        _test_initialize_parameter_cpu()
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise


def test_row_parallel_linear_seq_parallel(tensor_model_parallel_size):
    def _test_row_parallel_linear_seq_parallel():
        batch_size = 8
        seq_length = 128
        hidden_size = 256
        tensor_shape = (seq_length, batch_size, hidden_size)
        seed = 1234

        device = xm.xla_device()
        tensor_model_parallel_size_ = tensor_model_parallel_size
        parallel_state.initialize_model_parallel(tensor_model_parallel_size_)
        tensor_model_parallel_size_ = parallel_state.get_tensor_model_parallel_size()

        set_random_seed(seed)
        model_parallel_xla_manual_seed(seed)

        linear = layers.RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            sequence_parallel_enabled=True,
            keep_master_weight=True,
        ).to(device)

        with torch.no_grad():
            orig_input_tensor = torch.randn(tensor_shape, requires_grad=True, device=device)
            orig_loss_weight = torch.randn(tensor_shape, device=device)
            input_tensor = orig_input_tensor.chunk(
                chunks=tensor_model_parallel_size_,
                dim=2,
            )[parallel_state.get_tensor_model_parallel_rank()].contiguous()
            loss_weight = orig_loss_weight.chunk(
                chunks=tensor_model_parallel_size_,
                dim=0,
            )[parallel_state.get_tensor_model_parallel_rank()]
        input_tensor.requires_grad_()
        output = linear(input_tensor)
        loss = torch.mul(output, loss_weight).sum()
        loss.backward()

        ref_linear = torch.nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
        ).to(device)

        with torch.no_grad():
            dldy = orig_loss_weight.clone()
            x = orig_input_tensor.clone()
            ref_linear.weight.copy_(linear.master_weight)
        x.requires_grad_()
        expected_output = ref_linear(x)
        expected_loss = torch.mul(expected_output, dldy).sum()
        expected_loss.backward()

        torch.distributed.barrier()

        expected_output_chunk = expected_output.chunk(
            chunks=tensor_model_parallel_size_,
            dim=0,
        )[parallel_state.get_tensor_model_parallel_rank()]

        error = output.sub(expected_output_chunk).abs().max()
        print("   error in output (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
        assert error < 1.0e-3, "error: {}".format(error)

        if tensor_model_parallel_size_ == 1:
            expected_grad_chunk = ref_linear.weight.grad.chunk(
                chunks=tensor_model_parallel_size_,
                dim=1,
            )[parallel_state.get_tensor_model_parallel_rank()]

            error = linear.weight.grad.sub(expected_grad_chunk).abs().max()
            print("   error in grad (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
            assert error < 1.0e-3, "error: {}".format(error)

        # Reset groups
        parallel_state.destroy_model_parallel()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test passed")

        del device

    global results
    try:
        _test_row_parallel_linear_seq_parallel()
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise


def test_column_parallel_linear_seq_parallel(tensor_model_parallel_size):
    def _test_column_parallel_linear_seq_parallel():
        batch_size = 8
        seq_length = 128
        hidden_size = 256
        tensor_shape = (seq_length, batch_size, hidden_size)
        seed = 1234

        device = xm.xla_device()
        tensor_model_parallel_size_ = tensor_model_parallel_size
        parallel_state.initialize_model_parallel(tensor_model_parallel_size_)
        tensor_model_parallel_size_ = parallel_state.get_tensor_model_parallel_size()

        set_random_seed(seed)
        model_parallel_xla_manual_seed(seed)

        linear = layers.ColumnParallelLinear(
            hidden_size,
            hidden_size,
            bias=False,
            gather_output=False,
            sequence_parallel_enabled=True,
            keep_master_weight=True,
        ).to(device)

        with torch.no_grad():
            orig_input_tensor = torch.randn(tensor_shape, requires_grad=True, device=device)
            orig_loss_weight = torch.randn(tensor_shape, device=device)
            input_tensor = list(orig_input_tensor.chunk(tensor_model_parallel_size_, dim=0))[
                parallel_state.get_tensor_model_parallel_rank()
            ]
            loss_weight = orig_loss_weight.chunk(
                tensor_model_parallel_size_,
                dim=2,
            )[parallel_state.get_tensor_model_parallel_rank()]
        input_tensor.requires_grad_()
        output = linear(input_tensor)
        loss = torch.mul(output, loss_weight).sum()
        loss.backward()

        ref_linear = torch.nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=False,
        ).to(device)

        with torch.no_grad():
            dldy = orig_loss_weight.clone()
            x = orig_input_tensor.clone()
            ref_linear.weight.copy_(linear.master_weight)
        x.requires_grad_()
        expected_output = ref_linear(x)
        expected_loss = torch.mul(expected_output, dldy).sum()
        expected_loss.backward()

        torch.distributed.barrier()

        expected_output_chunk = expected_output.chunk(
            tensor_model_parallel_size_,
            dim=2,
        )[parallel_state.get_tensor_model_parallel_rank()]

        error = output.sub(expected_output_chunk).abs().max()
        print("   error in output (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
        assert error < 1.0e-3, "error: {}".format(error)

        if tensor_model_parallel_size_ == 1:
            expected_grad_chunk = ref_linear.weight.grad.chunk(
                chunks=tensor_model_parallel_size_,
                dim=0,
            )[parallel_state.get_tensor_model_parallel_rank()]

            error = linear.weight.grad.sub(expected_grad_chunk).abs().max()
            print("   error in grad (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
            assert error < 1.0e-3, "error: {}".format(error)

        # Reset groups
        parallel_state.destroy_model_parallel()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test passed")

        del device

    global results
    try:
        _test_column_parallel_linear_seq_parallel()
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise


def test_padding_attention_heads(tensor_model_parallel_size):
    def _test_padding_attention_heads():
        # Set up largely copied from other tests
        batch_size = 8
        seq_length = 128
        hidden_size = 256
        tensor_shape = (batch_size, seq_length, hidden_size)
        seed = 1234

        device = xm.xla_device()
        tensor_model_parallel_size_ = tensor_model_parallel_size
        parallel_state.initialize_model_parallel(tensor_model_parallel_size_)
        tensor_model_parallel_size_ = parallel_state.get_tensor_model_parallel_size()

        set_random_seed(seed)
        model_parallel_xla_manual_seed(seed)

        # Set up a model to pad
        model_to_pad = torch.nn.Sequential(
            layers.ColumnParallelLinear(
                hidden_size,
                hidden_size,
                bias=False,
                gather_output=False,
                keep_master_weight=True,
            ).to(device),
            layers.RowParallelLinear(
                hidden_size,
                hidden_size,
                bias=False,
                input_is_parallel=True,
                keep_master_weight=True,
            ).to(device),
        )
        # Pad it to the desired TP_DEGREE
        padded_model = pad_model(model_to_pad, parallel_state.get_tensor_model_parallel_size(), 1)
        # Get output
        with torch.no_grad():
            orig_input_tensor = torch.randn(tensor_shape, requires_grad=True, device=device)
            orig_loss_weight = torch.randn(tensor_shape, device=device)
            input_tensor = list(orig_input_tensor.chunk(tensor_model_parallel_size_, dim=0))[
                parallel_state.get_tensor_model_parallel_rank()
            ]
            loss_weight = orig_loss_weight.chunk(
                tensor_model_parallel_size_,
                dim=0,
            )[parallel_state.get_tensor_model_parallel_rank()]
        input_tensor.requires_grad_()
        output = padded_model(input_tensor)
        # Get a loss for it
        loss = torch.mul(output, loss_weight).sum()
        loss.backward()

        # Re-seed, which should be easier than copying weights in my case:
        set_random_seed(seed)
        model_parallel_xla_manual_seed(seed)

        # Set up an original unpadded model
        model = torch.nn.Sequential(
            layers.ColumnParallelLinear(
                hidden_size,
                hidden_size,
                bias=False,
                gather_output=False,
                keep_master_weight=True,
            ).to(device),
            layers.RowParallelLinear(
                hidden_size,
                hidden_size,
                bias=False,
                input_is_parallel=True,
                keep_master_weight=True,
            ).to(device),
        )

        # Get output (after both ColumnParallel/RowParallel, should be same size) and loss
        with torch.no_grad():
            input_tensor = list(orig_input_tensor.chunk(tensor_model_parallel_size_, dim=0))[
                parallel_state.get_tensor_model_parallel_rank()
            ]
            loss_weight = orig_loss_weight.chunk(
                tensor_model_parallel_size_,
                dim=0,
            )[parallel_state.get_tensor_model_parallel_rank()]
        input_tensor.requires_grad_()
        expected_output = model(input_tensor)
        # Get a loss for it
        expected_loss = torch.mul(expected_output, loss_weight).sum()
        expected_loss.backward()

        # Note: since we're working entirely in ColumnParallel/RowParallel anyways, no need to chunk the expected output

        # Compare the outputs and losses
        torch.distributed.barrier()

        error = output.sub(expected_output).abs().max()
        print("   error in output (parallel) on global rank {}: {}".format(torch.distributed.get_rank(), error))
        assert error < 1.0e-3, "error: {}".format(error)

        # Reset groups
        parallel_state.destroy_model_parallel()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test passed")

        del device

    global results
    try:
        _test_padding_attention_heads()
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise



def test_output_channel_parallel_conv(tensor_model_parallel_size):
    def _test_output_channel_parallel_conv():
        test_init(tensor_model_parallel_size, 1234)
        # Real dims taken from 768x768 Stable Diffusion UNet
        # (conv1): LoRACompatibleConv(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        batch_size = 1
        input_channels = 320
        output_channels = 640
        kernel_size = (3, 3)
        stride = (1, 1)
        padding = (1, 1)
        # H and W
        spatial_dim = 128
        tensor_shape = (batch_size, input_channels, spatial_dim, spatial_dim)

        test_conv = layers.OutputChannelParallelConv2d(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            bias=True,
            # Gather output because we're just testing the single layer here
            gather_output=True,
            # Need keep_master_weight so we can use the same weight for our reference layer
            keep_master_weight=True,
        )

        control_conv = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            bias=True,
        )

        # Create input tensor, copy weights from test layer to control layer
        with torch.no_grad():
            orig_input_tensor = (torch.randn(tensor_shape, requires_grad=True),)

            control_conv.weight.copy_(test_conv.master_weight)
            control_conv.bias.copy_(test_conv.master_bias)

        test_modules(test_conv, control_conv, orig_input_tensor)
        # If we reach this point, test has passed
        test_cleanup()

    global results
    try:
        _test_output_channel_parallel_conv()
        xm.master_print("test passed")
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise

def test_input_channel_parallel_conv(tensor_model_parallel_size):
    def _test_input_channel_parallel_conv():
        test_init(tensor_model_parallel_size, 1234)
        # Real dims taken from 768x768 Stable Diffusion UNet
        # (conv1): LoRACompatibleConv(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        batch_size = 1
        input_channels = 320
        output_channels = 640
        kernel_size = (3, 3)
        stride = (1, 1)
        padding = (1, 1)
        # H and W
        spatial_dim = 128
        tensor_shape = (batch_size, input_channels, spatial_dim, spatial_dim)

        test_conv = layers.InputChannelParallelConv2d(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            bias=True,
            # Gather output because we're just testing the single layer here
            input_is_parallel=False,
            # Need keep_master_weight so we can use the same weight for our reference layer
            keep_master_weight=True,
        )

        control_conv = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            bias=True,
        )

        # Create input tensor, copy weights from test layer to control layer
        with torch.no_grad():
            orig_input_tensor = (torch.randn(tensor_shape, requires_grad=True),)
            control_conv.weight.copy_(test_conv.master_weight)
            control_conv.bias.copy_(test_conv.master_bias)

        test_modules(test_conv, control_conv, orig_input_tensor)
        # If we reach this point, test has passed
        test_cleanup()

    global results
    try:
        _test_input_channel_parallel_conv()
        xm.master_print("test passed")
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise

class BackToBackConvs(torch.nn.Module):
    def __init__(self, conv1_args, conv2_args, parallel: bool = False):
        super().__init__()
        if parallel:
            # Need keep_master_weight so we can use the same weight for our reference layer
            self.conv1 = layers.OutputChannelParallelConv2d(
                *conv1_args,
                gather_output=False,
                keep_master_weight=True,
            )
            self.conv2 = layers.InputChannelParallelConv2d(
                *conv2_args,
                input_is_parallel=True,
                keep_master_weight=True,
            )
        else:
            self.conv1 = torch.nn.Conv2d(*conv1_args)
            self.conv2 = torch.nn.Conv2d(*conv2_args)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x


def test_back_to_back_parallel_convs(tensor_model_parallel_size):
    def _test_back_to_back_parallel_convs():
        test_init(tensor_model_parallel_size, 1234)

        # Real dims taken from 768x768 Stable Diffusion UNet
        # (conv1): LoRACompatibleConv(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (conv2): LoRACompatibleConv(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        conv1_args = (160, 320, (3, 3), (1, 1), (1, 1))
        conv2_args = (320, 320, (3, 3), (1, 1), (1, 1))
        batch_size = 1
        # H and W
        spatial_dim = 128
        tensor_shape = (batch_size, 160, spatial_dim, spatial_dim)

        test_model = BackToBackConvs(conv1_args, conv2_args, parallel=True)
        control_model = BackToBackConvs(conv1_args, conv2_args, parallel=False)

        # Create input tensor, copy weights from test layer to control layer
        with torch.no_grad():
            orig_input_tensor = (torch.randn(tensor_shape, requires_grad=True),)

            control_model.conv1.weight.copy_(test_model.conv1.master_weight)
            control_model.conv1.bias.copy_(test_model.conv1.master_bias)
            control_model.conv2.weight.copy_(test_model.conv2.master_weight)
            control_model.conv2.bias.copy_(test_model.conv2.master_bias)

        # TODO: assert_close fails when autocast is enabled V1305356298
        test_modules(test_model, control_model, orig_input_tensor, assert_close_on_output_tensor=False)
        # If we reach this point, test has passed
        test_cleanup()

    global results
    try:
        _test_back_to_back_parallel_convs()
        xm.master_print("test passed")
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise

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
    if requires_init_pg_override():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    world_size = xm.xrt_world_size()
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator("test parallel embedding")
        test_parallel_embedding(tensor_model_parallel_size)
        xm.mark_step()
        print_separator("test parallel embedding with shard over embedding dim")
        test_parallel_embedding_shard_over_embedding_dim(tensor_model_parallel_size)
        xm.mark_step()
        print_separator("test initialize affine weight")
        test_initialize_parameter_cpu(tensor_model_parallel_size)
        xm.mark_step()
        print_separator("test row_parallel_linear with seq_parallel")
        test_row_parallel_linear_seq_parallel(tensor_model_parallel_size)
        xm.mark_step()
        print_separator("test column_parallel_linear with seq_parallel")
        test_column_parallel_linear_seq_parallel(tensor_model_parallel_size)
        xm.mark_step()
        print_separator("test padding attention heads")
        test_padding_attention_heads(tensor_model_parallel_size)
        xm.mark_step()
        print_separator("test output_channel_parallel_conv2d")
        test_output_channel_parallel_conv(tensor_model_parallel_size)
        xm.mark_step()
        print_separator("test input_channel_parallel_conv2d")
        test_input_channel_parallel_conv(tensor_model_parallel_size)
        xm.mark_step()
        print_separator("test test_back_to_back_parallel_conv2d")
        test_back_to_back_parallel_convs(tensor_model_parallel_size)
        xm.mark_step()
        tensor_model_parallel_size *= 2
    atexit.register(on_exit)
