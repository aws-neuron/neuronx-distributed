import argparse
import os
import random
import traceback
from datetime import datetime
import regex as re

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm

from neuronx_distributed.modules import qkv_linear
from neuronx_distributed.parallel_layers import layers, parallel_state
from neuronx_distributed.parallel_layers.random import model_parallel_xla_manual_seed
from neuronx_distributed.parallel_layers.utils import requires_init_pg_override

datetime_str = str(datetime.now())


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument("--s3_bucket", default="s3://ktf-test-runs/neuronx_distributed_parallel_layers/layers")
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    return S3_BUCKET_NAME, args

S3_BUCKET_NAME, args = parse_args()
results = {"inference_success": 1}


def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_qkv_linear_with_kv_multipler_1(tensor_model_parallel_size, fuse_qkv=False):
    def _test_qkv_linear_with_kv_multipler_1():
        batch_size = 8
        seq_length = 128
        hidden_size = 256
        tensor_shape = (seq_length, batch_size, hidden_size)
        seed = 1234

        device = xm.xla_device()
        tensor_model_parallel_size_ = tensor_model_parallel_size
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tensor_model_parallel_size_)
        tensor_model_parallel_size_ = parallel_state.get_tensor_model_parallel_size()

        set_random_seed(seed)
        model_parallel_xla_manual_seed(seed)

        col_linear = qkv_linear.GQAQKVColumnParallelLinear(
            hidden_size,
            [tensor_model_parallel_size * hidden_size, tensor_model_parallel_size * hidden_size],
            bias=False,
            gather_output=False,
            sequence_parallel_enabled=True,
            keep_master_weight=True,
            kv_size_multiplier=1,
            fuse_qkv=fuse_qkv
        ).to(device)

        row_linear = layers.RowParallelLinear(
            tensor_model_parallel_size * hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            sequence_parallel_enabled=False,
            keep_master_weight=True,
        ).to(device)

        with torch.no_grad():
            orig_input_tensor = torch.randn(tensor_shape, requires_grad=True).to(device)
            orig_loss_weight = torch.randn(tensor_shape).transpose(0, 1).to(device)
            input_tensor = list(orig_input_tensor.chunk(tensor_model_parallel_size_, dim=0))[
                parallel_state.get_tensor_model_parallel_rank()
            ]
        input_tensor.requires_grad_()
        output_q, output_k, output_v = col_linear(input_tensor)
        q, k, v = (
            output_q.view(128, 8, 1, 256).permute(1, 2, 0, 3),
            output_k.view(128, 8, 1, 256).permute(1, 2, 0, 3),
            output_v.view(128, 8, 1, 256).permute(1, 2, 0, 3),
        )
        intermediate_tensor = torch.matmul(q, k.transpose(2, 3))
        intermediate_tensor = torch.matmul(intermediate_tensor, v)
        intermediate_tensor = intermediate_tensor.transpose(1, 2)
        intermediate_tensor = intermediate_tensor.reshape(8, 128, -1)
        output = row_linear(intermediate_tensor)

        loss = torch.mul(output, orig_loss_weight).sum()
        loss.backward()
        xm.mark_step()

        ref_q_linear = torch.nn.Linear(
            in_features=hidden_size,
            out_features=tensor_model_parallel_size * hidden_size,
            bias=False,
        ).to(device)
        ref_k_linear = torch.nn.Linear(
            in_features=hidden_size,
            out_features=tensor_model_parallel_size * hidden_size,
            bias=False,
        ).to(device)
        ref_v_linear = torch.nn.Linear(
            in_features=hidden_size,
            out_features=tensor_model_parallel_size * hidden_size,
            bias=False,
        ).to(device)
        ref_mlp_linear = torch.nn.Linear(
            in_features=tensor_model_parallel_size * hidden_size,
            out_features=hidden_size,
            bias=False,
        ).to(device)

        with torch.no_grad():
            dldy = orig_loss_weight.clone()
            x = orig_input_tensor.clone()
            if fuse_qkv:
                sizes = [tensor_model_parallel_size * hidden_size, tensor_model_parallel_size * hidden_size, tensor_model_parallel_size * hidden_size]
                master_weight_q, master_weight_k, master_weight_v = torch.split(col_linear.master_weight_qkv, sizes, dim=0)
                ref_q_linear.weight.copy_(master_weight_q)
                ref_k_linear.weight.copy_(master_weight_k)
                ref_v_linear.weight.copy_(master_weight_v)
                ref_mlp_linear.weight.copy_(row_linear.master_weight)
            else:
                ref_q_linear.weight.copy_(col_linear.master_weight_q)
                ref_k_linear.weight.copy_(col_linear.master_weight_k)
                ref_v_linear.weight.copy_(col_linear.master_weight_v)
                ref_mlp_linear.weight.copy_(row_linear.master_weight)
        x.requires_grad_()
        expected_q, expected_k, expected_v = ref_q_linear(x), ref_k_linear(x), ref_v_linear(x)
        e_q, e_k, e_v = (
            expected_q.view(128, 8, tensor_model_parallel_size, 256).permute(1, 2, 0, 3),
            expected_k.view(128, 8, tensor_model_parallel_size, 256).permute(1, 2, 0, 3),
            expected_v.view(128, 8, tensor_model_parallel_size, 256).permute(1, 2, 0, 3),
        )
        expected_intermediate_tensor = torch.matmul(e_q, e_k.transpose(2, 3))
        expected_intermediate_tensor = torch.matmul(expected_intermediate_tensor, e_v)
        expected_intermediate_tensor = expected_intermediate_tensor.transpose(1, 2)
        expected_intermediate_tensor = expected_intermediate_tensor.reshape(8, 128, -1)
        expected_output = ref_mlp_linear(expected_intermediate_tensor)
        expected_loss = torch.mul(expected_output, dldy).sum()
        expected_loss.backward()

        torch.distributed.barrier()

        xm.mark_step()

        assert np.allclose(
            output.detach().cpu().numpy(), expected_output.detach().cpu().numpy(), rtol=1e-2, atol=1e-2
        ), "final output doesn't match rank{}".format(xm.get_ordinal())
        assert np.allclose(
            output_q.detach().cpu().numpy(),
            expected_q.chunk(tensor_model_parallel_size_, dim=2)[parallel_state.get_tensor_model_parallel_rank()]
            .detach()
            .cpu()
            .numpy(),
            rtol=1e-2,
            atol=1e-2,
        ), "output_q doesn't match rank{}".format(xm.get_ordinal())
        assert np.allclose(
            output_k.detach().cpu().numpy(),
            expected_k.chunk(tensor_model_parallel_size_, dim=2)[parallel_state.get_tensor_model_parallel_rank()]
            .detach()
            .cpu()
            .numpy(),
            rtol=1e-2,
            atol=1e-2,
        ), "output_k doesn't match rank{}".format(xm.get_ordinal())
        assert np.allclose(
            output_v.detach().cpu().numpy(),
            expected_v.chunk(tensor_model_parallel_size_, dim=2)[parallel_state.get_tensor_model_parallel_rank()]
            .detach()
            .cpu()
            .numpy(),
            rtol=1e-2,
            atol=1e-2,
        ), "output_v doesn't match rank{}".format(xm.get_ordinal())

        expected_q_grad_chunk = ref_q_linear.weight.grad.chunk(
            chunks=tensor_model_parallel_size_,
            dim=0,
        )[parallel_state.get_tensor_model_parallel_rank()]
        expected_k_grad_chunk = ref_k_linear.weight.grad.chunk(
            chunks=tensor_model_parallel_size_,
            dim=0,
        )[parallel_state.get_tensor_model_parallel_rank()]
        expected_v_grad_chunk = ref_v_linear.weight.grad.chunk(
            chunks=tensor_model_parallel_size_,
            dim=0,
        )[parallel_state.get_tensor_model_parallel_rank()]

        if fuse_qkv:
            expected_qkv_grad_chunk = torch.cat([expected_q_grad_chunk, expected_k_grad_chunk, expected_v_grad_chunk], dim=0)
            assert np.allclose(
                col_linear.weight_qkv.grad.detach().cpu().numpy(),
                expected_qkv_grad_chunk.detach().cpu().numpy(),
                rtol=5e-2,
                atol=1e-2,
            ), "grad_qkv doesn't match rank{}".format(xm.get_ordinal())
        else:
            assert np.allclose(
                col_linear.weight_q.grad.detach().cpu().numpy(),
                expected_q_grad_chunk.detach().cpu().numpy(),
                rtol=1e-2,
                atol=1e-2,
            ), "grad_q doesn't match rank{}".format(xm.get_ordinal())

            assert np.allclose(
                col_linear.weight_k.grad.detach().cpu().numpy(),
                expected_k_grad_chunk.detach().cpu().numpy(),
                rtol=1e-2,
                atol=1e-2,
            ), "grad_k doesn't match rank{}".format(xm.get_ordinal())

            assert np.allclose(
                col_linear.weight_v.grad.detach().cpu().numpy(),
                expected_v_grad_chunk.detach().cpu().numpy(),
                rtol=1e-2,
                atol=1e-2,
            ), "grad_v doesn't match rank{}".format(xm.get_ordinal())

        # Reset groups
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_kv_group()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test passed")

        del device

    global results
    try:
        _test_qkv_linear_with_kv_multipler_1()
    except Exception:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise


def test_qkv_linear_with_kv_multipler_4(tensor_model_parallel_size, fuse_qkv=False):
    def _test_qkv_linear_with_kv_multipler_4():
        batch_size = 8
        seq_length = 128
        hidden_size = 256
        tensor_shape = (seq_length, batch_size, hidden_size)
        seed = 1234
        kv_shared_group_size = 4

        device = xm.xla_device()
        tensor_model_parallel_size_ = tensor_model_parallel_size
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tensor_model_parallel_size_)
        tensor_model_parallel_size_ = parallel_state.get_tensor_model_parallel_size()

        set_random_seed(seed)
        xm.set_rng_state(seed)
        model_parallel_xla_manual_seed(seed)

        col_linear = qkv_linear.GQAQKVColumnParallelLinear(
            hidden_size,
            [
                tensor_model_parallel_size * hidden_size,
                tensor_model_parallel_size * hidden_size // kv_shared_group_size,
            ],
            bias=False,
            gather_output=False,
            sequence_parallel_enabled=True,
            keep_master_weight=True,
            kv_size_multiplier=kv_shared_group_size,
            fuse_qkv=fuse_qkv
        ).to(device)

        row_linear = layers.RowParallelLinear(
            tensor_model_parallel_size * hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            sequence_parallel_enabled=False,
            keep_master_weight=True,
        ).to(device)

        with torch.no_grad():
            orig_input_tensor = torch.randn(tensor_shape, requires_grad=True).to(device)
            orig_loss_weight = torch.randn(tensor_shape).transpose(0, 1).to(device)
            input_tensor = list(orig_input_tensor.chunk(tensor_model_parallel_size_, dim=0))[
                parallel_state.get_tensor_model_parallel_rank()
            ]
        input_tensor.requires_grad_()
        output_q, output_k, output_v = col_linear(input_tensor)
        q, k, v = (
            output_q.view(seq_length, batch_size, 1, hidden_size).permute(1, 2, 0, 3),
            output_k.view(seq_length, batch_size, 1, hidden_size).permute(1, 2, 0, 3),
            output_v.view(seq_length, batch_size, 1, hidden_size).permute(1, 2, 0, 3),
        )
        intermediate_tensor = torch.matmul(q, k.transpose(2, 3))
        intermediate_tensor = torch.matmul(intermediate_tensor, v)
        intermediate_tensor = intermediate_tensor.transpose(1, 2)
        intermediate_tensor = intermediate_tensor.reshape(batch_size, seq_length, -1)
        output = row_linear(intermediate_tensor)

        loss = torch.mul(output, orig_loss_weight).sum()
        loss.backward()
        xm.mark_step()

        ref_q_linear = torch.nn.Linear(
            in_features=hidden_size,
            out_features=tensor_model_parallel_size * hidden_size,
            bias=False,
        ).to(device)
        ref_k_linear = torch.nn.Linear(
            in_features=hidden_size,
            out_features=(tensor_model_parallel_size * hidden_size // kv_shared_group_size),
            bias=False,
        ).to(device)
        ref_v_linear = torch.nn.Linear(
            in_features=hidden_size,
            out_features=(tensor_model_parallel_size * hidden_size // kv_shared_group_size),
            bias=False,
        ).to(device)
        ref_mlp_linear = torch.nn.Linear(
            in_features=tensor_model_parallel_size * hidden_size,
            out_features=hidden_size,
            bias=False,
        ).to(device)

        with torch.no_grad():
            dldy = orig_loss_weight.clone()
            x = orig_input_tensor.clone()
            if fuse_qkv:
                sizes = [tensor_model_parallel_size * hidden_size, tensor_model_parallel_size * hidden_size // kv_shared_group_size , tensor_model_parallel_size * hidden_size // kv_shared_group_size]
                master_weight_q, master_weight_k, master_weight_v = torch.split(col_linear.master_weight_qkv, sizes, dim=0)
                ref_q_linear.weight.copy_(master_weight_q)
                ref_k_linear.weight.copy_(master_weight_k)
                ref_v_linear.weight.copy_(master_weight_v)
                ref_mlp_linear.weight.copy_(row_linear.master_weight)
            else:
                ref_q_linear.weight.copy_(col_linear.master_weight_q)
                ref_k_linear.weight.copy_(col_linear.master_weight_k)
                ref_v_linear.weight.copy_(col_linear.master_weight_v)
                ref_mlp_linear.weight.copy_(row_linear.master_weight)
        x.requires_grad_()
        expected_q, expected_k, expected_v = ref_q_linear(x), ref_k_linear(x), ref_v_linear(x)
        e_q, e_k, e_v = (
            expected_q.view(seq_length, batch_size, -1, hidden_size).permute(1, 2, 0, 3),
            expected_k.view(seq_length, batch_size, -1, hidden_size).permute(1, 2, 0, 3),
            expected_v.view(seq_length, batch_size, -1, hidden_size).permute(1, 2, 0, 3),
        )
        e_k = e_k.repeat(1, kv_shared_group_size, 1, 1)
        e_v = e_v.repeat(1, kv_shared_group_size, 1, 1)
        expected_intermediate_tensor = torch.matmul(e_q, e_k.transpose(2, 3))
        expected_intermediate_tensor = torch.matmul(expected_intermediate_tensor, e_v)
        expected_intermediate_tensor = expected_intermediate_tensor.transpose(1, 2)
        expected_intermediate_tensor = expected_intermediate_tensor.reshape(batch_size, seq_length, -1)
        expected_output = ref_mlp_linear(expected_intermediate_tensor)
        expected_loss = torch.mul(expected_output, dldy).sum()
        expected_loss.backward()

        torch.distributed.barrier()
        xm.mark_step()
        assert np.allclose(
            output.detach().cpu().numpy(), expected_output.detach().cpu().numpy(), rtol=1e-2, atol=1e-2
        ), "final output doesn't match rank{}".format(xm.get_ordinal())

        expected_q_grad_chunk = ref_q_linear.weight.grad.chunk(
            chunks=tensor_model_parallel_size_,
            dim=0,
        )[parallel_state.get_tensor_model_parallel_rank()]
        expected_k_grad_chunk = ref_k_linear.weight.grad.chunk(
            chunks=tensor_model_parallel_size_ // kv_shared_group_size,
            dim=0,
        )[parallel_state.get_tensor_model_parallel_rank() % 8]

        expected_v_grad_chunk = ref_v_linear.weight.grad.chunk(
            chunks=tensor_model_parallel_size_ // kv_shared_group_size,
            dim=0,
        )[parallel_state.get_tensor_model_parallel_rank() % 8]
        if fuse_qkv:
            grad_q_chunk, grad_k_chunk, grad_v_chunk = torch.split(col_linear.weight_qkv.grad.detach().cpu(), [hidden_size,hidden_size,hidden_size], dim=0)
            assert np.allclose(
                grad_q_chunk.numpy(),
                expected_q_grad_chunk.detach().cpu().numpy(),
                rtol=5e-2,
                atol=1e-2,
            ), "grad_q doesn't match rank{}".format(xm.get_ordinal())
            assert np.allclose(
                grad_k_chunk.numpy(),
                expected_k_grad_chunk.detach().cpu().numpy(),
                rtol=1e-2,
                atol=1,
            ), "grad_k doesn't match rank{}".format(xm.get_ordinal())
            assert np.allclose(
                grad_v_chunk.numpy(),
                expected_v_grad_chunk.detach().cpu().numpy(),
                rtol=5e-2,
                atol=1e-1,
            ), "grad_v doesn't match rank{}".format(xm.get_ordinal())
        else:
            assert np.allclose(
                col_linear.weight_q.grad.detach().cpu().numpy(),
                expected_q_grad_chunk.detach().cpu().numpy(),
                rtol=1e-2,
                atol=1e-2,
            ), "grad_q doesn't match rank{}".format(xm.get_ordinal())

            assert np.allclose(
                col_linear.weight_k.grad.detach().cpu().numpy(), expected_k_grad_chunk.cpu().numpy(), rtol=1e-2, atol=1
            ), "grad_k doesn't match rank{}".format(xm.get_ordinal())

            assert np.allclose(
                col_linear.weight_v.grad.detach().cpu().numpy(),
                expected_v_grad_chunk.detach().cpu().numpy(),
                rtol=5e-2,
                atol=1e-1,
            ), "grad_v doesn't match rank{}".format(xm.get_ordinal())

        # Reset groups
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_kv_group()

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test passed")

        del device

    global results
    try:
        _test_qkv_linear_with_kv_multipler_4()
    except Exception:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise

def test_qkv_linear_with_kv_multipler_4_HLO_test(tensor_model_parallel_size, fuse_qkv=False):
    os.environ["XLA_DOWNCAST_BF16"] = "1"
    def _test_qkv_linear_with_kv_multipler_4_hlo_test():
        batch_size = 8
        seq_length = 128
        hidden_size = 256
        tensor_shape = (seq_length, batch_size, hidden_size)
        seed = 1234
        kv_shared_group_size = 4

        device = xm.xla_device()
        tensor_model_parallel_size_ = tensor_model_parallel_size
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tensor_model_parallel_size_)
        tensor_model_parallel_size_ = parallel_state.get_tensor_model_parallel_size()

        set_random_seed(seed)
        xm.set_rng_state(seed)
        model_parallel_xla_manual_seed(seed)

        col_linear = qkv_linear.GQAQKVColumnParallelLinear(
            hidden_size,
            [
                tensor_model_parallel_size * hidden_size,
                tensor_model_parallel_size * hidden_size // kv_shared_group_size,
            ],
            bias=False,
            gather_output=False,
            sequence_parallel_enabled=True,
            keep_master_weight=True,
            kv_size_multiplier=kv_shared_group_size,
            fuse_qkv=fuse_qkv
        ).to(device)

        row_linear = layers.RowParallelLinear(
            tensor_model_parallel_size * hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            sequence_parallel_enabled=False,
            keep_master_weight=True,
        ).to(device)

        with torch.no_grad():
            orig_input_tensor = torch.randn(tensor_shape, requires_grad=True).to(device)
            orig_loss_weight = torch.randn(tensor_shape).transpose(0, 1).to(device)
            input_tensor = list(orig_input_tensor.chunk(tensor_model_parallel_size_, dim=0))[
                parallel_state.get_tensor_model_parallel_rank()
            ]
        input_tensor.requires_grad_()
        output_q, output_k, output_v = col_linear(input_tensor)
        q, k, v = (
            output_q.view(seq_length, batch_size, 1, hidden_size).permute(1, 2, 0, 3),
            output_k.view(seq_length, batch_size, 1, hidden_size).permute(1, 2, 0, 3),
            output_v.view(seq_length, batch_size, 1, hidden_size).permute(1, 2, 0, 3),
        )
        intermediate_tensor = torch.matmul(q, k.transpose(2, 3))
        intermediate_tensor = torch.matmul(intermediate_tensor, v)
        intermediate_tensor = intermediate_tensor.transpose(1, 2)
        intermediate_tensor = intermediate_tensor.reshape(batch_size, seq_length, -1)
        output = row_linear(intermediate_tensor)

        loss = torch.mul(output, orig_loss_weight).sum()
        loss.backward()
        hlo_text = torch_xla._XLAC._get_xla_tensors_text([col_linear.weight_k.grad])
        if tensor_model_parallel_size_ > 1:
            assert re.search(r".*f32.*f32.*xla::cross_replica_sum.*f32.*f32.*", hlo_text) is not None, "dtype mismatch for reduce scatter, not all F32 detected"

    global results
    try:
        _test_qkv_linear_with_kv_multipler_4_hlo_test()
    except Exception:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    if requires_init_pg_override():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    world_size = xm.xrt_world_size()
    tensor_model_parallel_size = 32
    # Set the XLA_DISABLE_FUNCTIONALIZATION flag to avoid accuracy issues with PT2.1 and fused_qkv
    os.environ['XLA_DISABLE_FUNCTIONALIZATION'] = '0'
    test_qkv_linear_with_kv_multipler_1(tensor_model_parallel_size)
    test_qkv_linear_with_kv_multipler_1(tensor_model_parallel_size, fuse_qkv=True)
    test_qkv_linear_with_kv_multipler_4(tensor_model_parallel_size)
    test_qkv_linear_with_kv_multipler_4(tensor_model_parallel_size, fuse_qkv=True)
    test_qkv_linear_with_kv_multipler_4_HLO_test(tensor_model_parallel_size)
