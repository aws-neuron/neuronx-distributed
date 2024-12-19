import argparse
import atexit
import json
import os
import random
import traceback
from datetime import datetime
import time

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from neuronx_distributed.modules import qkv_linear
from neuronx_distributed.parallel_layers import layers, parallel_state
from neuronx_distributed.parallel_layers.random import model_parallel_xla_manual_seed
from neuronx_distributed.parallel_layers.utils import requires_init_pg_override, is_torch_version_greater_than_2

import sys
sys.path.append('/home/ubuntu/ktest/NeuronxDistributed/examples/training/llama/modeling_llama_nxd')

from modeling_llama_nxd import LlamaAttention
from transformers.models.llama.configuration_llama import LlamaConfig

datetime_str = str(datetime.now())


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument("--s3_bucket", default="s3://ktf-test-runs/neuronx_distributed_parallel_layers/layers")
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    return S3_BUCKET_NAME, args

def set_random_seed(seed):
    """Set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

S3_BUCKET_NAME, args = parse_args()
results = {"inference_success": 1}


def test_llama_attention(tensor_model_parallel_size=8, seq_len=8192):
    def run_fwd_and_measure_throughput(self_attn, attn_input, config):
        self_attn.eval()

        warmup_iters = 5
        for _ in range(warmup_iters):
            with torch.no_grad():
                attn_output = self_attn.forward(
                    hidden_states=attn_input,
                    output_attentions=False,
                    use_cache=False)

        num_iterations = 5
        start_time = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                attn_output = self_attn.forward(
                    hidden_states=attn_input,
                    output_attentions=False,
                    use_cache=False)

        end_time = time.time()
        total_time = end_time - start_time
        samples = config.batch_size * num_iterations
        throughput = samples / total_time

        return attn_output, throughput


    def _test_attention(tensor_model_parallel_size=8, seq_len=8192):
        seed = 1234

        device = xm.xla_device()
        tensor_model_parallel_size_ = tensor_model_parallel_size
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tensor_model_parallel_size_)
        tensor_model_parallel_size_ = parallel_state.get_tensor_model_parallel_size()

        set_random_seed(seed)
        xm.set_rng_state(seed)
        model_parallel_xla_manual_seed(seed)

        training_config_file = '/home/ubuntu/ktest/NeuronxDistributed/examples/training/llama/tp_zero1_llama_hf_pretrain/8B_config_llama3/'
        config = LlamaConfig.from_pretrained(training_config_file)
        config.use_cache = False
        config.return_dict = False
        config.qkv_linear = True
        config.kv_shared_group_size = 4
        config.fuse_qkv = True
        config.num_hidden_layers = 4
        config.batch_size = 8
        config.seq_length = seq_len

        tensor_shape = (config.seq_length, config.batch_size, config.hidden_size)
        with torch.no_grad():
            attn_input = torch.randn(tensor_shape, device=device, requires_grad=False)

        # init attns
        self_attn_fused = LlamaAttention(config=config).to(device)
        config.fuse_qkv = False
        self_attn = LlamaAttention(config=config).to(device)

        # keep weights the same
        with torch.no_grad():
            for param1, param2 in zip(self_attn_fused.parameters(), self_attn.parameters()):
                param2.copy_(param1)
                assert torch.equal(param1, param2), "Attention parameters are unexpectedly different. Differences in parameter values will produce different outputs"

        # fused qkv
        attn_output_fused, throughput_fused = run_fwd_and_measure_throughput(self_attn_fused, attn_input, config)

        # without fused qkv
        attn_output, throughput = run_fwd_and_measure_throughput(self_attn, attn_input, config)

        if xm.get_ordinal()==0:
            # assert outputs are equal
            assert torch.allclose(
                attn_output[0].detach().cpu().numpy(), attn_output_fused[0].detach().cpu().numpy(), rtol=1e-1, atol=1e-1
            ), "output doesn't match rank{}".format(xm.get_ordinal())

            # assert that the throughput with fused qkv is equal to or better than the throughput without fused qkv
            print(f"Throughput fused: {throughput_fused:.2f} and throughput not fused: {throughput:.2f} tokens/second")
            assert throughput_fused > throughput, "Throughput is unexpectedly lower for fused qkv than non fused qkv."


    global results
    try:
        result = _test_attention(tensor_model_parallel_size=tensor_model_parallel_size, seq_len=seq_len)
    except Exception:
        results["inference_success"] = 0
        print(results)
        print(traceback.format_exc())
        raise
    return result


if __name__ == "__main__":
    if requires_init_pg_override():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    world_size = xm.xrt_world_size()
    if is_torch_version_greater_than_2():
        # Set the XLA_DISABLE_FUNCTIONALIZATION flag to avoid accuracy issues with PT2.1 and fused_qkv
        os.environ['XLA_DISABLE_FUNCTIONALIZATION'] = '0'

    seq_lengths = [8192, 16384]
    tp_values = [8, 16, 32]
    for tp in tp_values:
        for seq_len in seq_lengths:
            test_llama_attention(seq_len=seq_len)
