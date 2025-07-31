import os
import re
import copy
import json
import subprocess
import fire
from functools import partial
from typing import List

import torch
import torch_neuronx
from safetensors.torch import load_file

from neuronx_distributed import ModelBuilder, shard_checkpoint, NxDModel, NxDParallelState
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.mock_torchdist import mock_distributed
from neuronx_distributed.utils.model_utils import init_on_device

# TODO load config from path
from config import Llama3_2_1B
from model import Attention, Transformer, load_llama_checkpoint, precompute_rope
from tokenizer import Tokenizer


def generate(model: torch.nn.Module, max_len: int, prompt_tokens: List[List[int]], stop_tokens: List[int]):
    prompt_tokens = copy.deepcopy(prompt_tokens)

    # Track max pos per batch
    last_pos = torch.tensor([len(prompt) - 1 for prompt in prompt_tokens], dtype=torch.int32)

    # Pad all batch lines to the same sequence length
    pad_token = Llama3_2_1B.pad_token
    padded_tokens = [prompt + [pad_token] * (max_len - len(prompt)) for prompt in prompt_tokens]
    tokens = torch.tensor(padded_tokens, dtype=torch.int32)

    input_tokens = tokens
    input_bs, input_len = input_tokens.shape

    attention_mask = torch.where(tokens != pad_token, 1, 0).to(torch.int32)

    # A tensor to keep track of generation completion per batch line
    is_gen_complete = torch.full((input_bs, 1), False)

    while True:

        logits = model.forward(input_tokens, last_pos, attention_mask)
        last_pos = last_pos + 1

        # assuming we are doing greedy sampling
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        input_tokens = next_token.to(torch.int32)

        # Add the new token to prompt
        for idx, prompt in enumerate(prompt_tokens):
            if not is_gen_complete[idx][0].item():
                prompt.append(next_token[idx].item())

        for stop_token in stop_tokens:
            is_gen_complete = is_gen_complete.logical_or(next_token == stop_token)

        # Stop generation when all batch lines are complete
        if is_gen_complete.all():
            break

        if torch.max(last_pos).item() >= max_len:
            break

        # Update mask
        attention_mask[torch.arange(last_pos.shape[0]), last_pos] = 1

    return prompt_tokens


def _load_model_config(compiled_model_path, prompts):
    if not compiled_model_path.endswith("/"):
        compiled_model_path += "/"

    with open(compiled_model_path + "config.json", "r") as file:
        cfg = json.load(file)

    bs, seq_len, tp_degree = cfg["batch_size"], cfg["seq_len"], cfg["tp_degree"]
    shard_on_load = cfg.get("shard_on_load", True)

    if len(prompts) != bs:
        raise ValueError(f"Prompts size does not match batch size {cfg['batch_size']}")

    return bs, seq_len, tp_degree, shard_on_load


def _save_model_config(output_path, batch_size, seq_len, tp_degree, shard_on_load):
    config = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tp_degree": tp_degree,
        "shard_on_load": shard_on_load
    }

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


def _load_sharded_weights(model_path, compiled_model_path, shard_on_load, tp_degree, bs, seq_len):
    if shard_on_load:
        # Shard weights during loading
        checkpoint = load_llama_checkpoint(Llama3_2_1B, model_path, tp_degree)

        with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree), \
                init_on_device(torch.device("meta")):
            sharded_weights = shard_checkpoint(
                checkpoint=checkpoint,
                model=Transformer(Llama3_2_1B, bs, seq_len),
                start_rank=0,
                end_rank=tp_degree-1,
            )
    else:
        # Load pre-sharded weights
        sharded_weights = []
        for rank in range(tp_degree):
            ckpt = load_file(os.path.join(compiled_model_path, f"weights/tp{rank}_sharded_checkpoint.safetensors"))
            sharded_weights.append(ckpt)

    return sharded_weights


@torch.inference_mode()
def generate_cpu(
    batch_size=2,
    seq_len=128,
    model_path="~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth",
    tokenizer_path="~/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model",
    prompts=["How tall is the Space Needle?", "What is the capital of France?"],
):

    checkpoint = load_llama_checkpoint(Llama3_2_1B, model_path)

    model: Transformer = Transformer(Llama3_2_1B, batch_size, seq_len)
    model.load_state_dict(checkpoint, strict=False)

    tokenizer = Tokenizer(model_path=tokenizer_path)
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    output_tokens = generate(model, seq_len, prompt_tokens, stop_tokens=tokenizer.stop_tokens)

    return [tokenizer.decode(tokens) for tokens in output_tokens]


@torch.inference_mode()
def generate_nxd(
    compiled_model_path="~/neuron_models/Llama3.2-1B-Instruct",
    model_path="~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth",
    tokenizer_path="~/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model",
    prompts=["How tall is the Space Needle?", "What is the capital of France?"],
):
    compiled_model_path = os.path.expanduser(compiled_model_path)
    model_path = os.path.expanduser(model_path)
    tokenizer_path = os.path.expanduser(tokenizer_path)
    
    # Load model config
    bs, seq_len, tp_degree, shard_on_load = _load_model_config(compiled_model_path, prompts)

    # Load NxD model
    nxd_model = NxDModel.load(os.path.join(compiled_model_path, "nxd_model.pt"))
    sharded_weights = _load_sharded_weights(model_path, compiled_model_path, shard_on_load, tp_degree, bs, seq_len)
    nxd_model.set_weights(sharded_weights)
    nxd_model.to_neuron()

    # Generate text
    tokenizer = Tokenizer(model_path=os.path.expanduser(tokenizer_path))
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    output_tokens = generate(nxd_model, seq_len, prompt_tokens, stop_tokens=tokenizer.stop_tokens)

    return [tokenizer.decode(tokens) for tokens in output_tokens]


def _compile_no_mock(
    batch_size=2,
    seq_len=128,
    tp_degree=32,
    model_path="~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth",
    output_path="~/neuron_models/Llama3.2-1B-Instruct",
    shard_on_load=True,
):
    rank = int(os.environ["RANK"])
    torch.distributed.init_process_group(backend="xla", rank=0, world_size=tp_degree)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_degree)
    torch.multiprocessing.set_sharing_strategy("file_system")
    parallel_state.set_aot_mode(True)

    if rank == 0:
        model = Transformer(Llama3_2_1B, batch_size, seq_len)

        # Initialize ModelBuilder
        builder = ModelBuilder(
            model=model,
        )

        # Add prefill trace
        builder.trace(
            kwargs={
                "tokens": torch.ones((batch_size, seq_len), dtype=torch.int32),
                "last_pos": torch.tensor([0] * batch_size, dtype=torch.int32),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.int32),
            },
            tag="prefill",
        )

        # Add decode trace
        builder.trace(
            kwargs={
                "tokens": torch.ones((batch_size, 1), dtype=torch.int32),
                "last_pos": torch.tensor([0] * batch_size, dtype=torch.int32),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.int32),
            },
            tag="decode",
        )

        # Compile
        traced_model = builder.compile()

        # Save model and config
        os.makedirs(output_path, exist_ok=True)
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))
        _save_model_config(output_path, batch_size, seq_len, tp_degree, shard_on_load)
        print(f"Saved compiled model to {output_path}")

    parallel_state.set_aot_mode(False)
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()

    if not shard_on_load and rank == 0:
        serialize_path = os.path.join(output_path, "weights/")

        with init_on_device(torch.device("meta")):
            torch.distributed.init_process_group(backend="xla", rank=0, world_size=tp_degree)
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_degree)

            shard_checkpoint(
                checkpoint=load_llama_checkpoint(Llama3_2_1B, model_path, tp_degree),
                model=Transformer(Llama3_2_1B, batch_size, seq_len),
                start_rank=0,
                end_rank=tp_degree-1,
                serialize_path=serialize_path
            )

            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()

        print(f"Saved sharded checkpoints to {serialize_path}")


def compile_no_mock(
    batch_size=2,
    seq_len=128,
    tp_degree=32,
    model_path="~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth",
    output_path="~/neuron_models/Llama3.2-1B-Instruct",
    shard_on_load=True,
):
    script_path = os.path.abspath(__file__)
    cmd = [
        "torchrun",
        f"--nproc_per_node={tp_degree}",
        script_path,
        "_compile_no_mock",
        f"--batch_size={batch_size}",
        f"--seq_len={seq_len}",
        f"--tp_degree={tp_degree}",
        f"--model_path={model_path}",
        f"--output_path={output_path}",
        f"--shard_on_load={shard_on_load}"
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    print(stdout.decode())
    if process.returncode != 0:
        print("Error:", stderr.decode())
        raise RuntimeError("Compilation failed")


def compile_with_mock(
    batch_size=2,
    seq_len=128,
    tp_degree=32,
    model_path="~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth",
    output_path="~/neuron_models/Llama3.2-1B-Instruct",
    shard_on_load=True,
):
    with mock_distributed(world_size=tp_degree):
        torch.distributed.init_process_group(backend="xla", rank=0, world_size=tp_degree)
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_degree, skip_collective_init=True)
        parallel_state.set_aot_mode(True)

        model = Transformer(Llama3_2_1B, batch_size, seq_len)

        # Initialize ModelBuilder
        builder = ModelBuilder(model=model)

        # Add prefill trace
        builder.trace(
            kwargs={
                "tokens": torch.ones((batch_size, seq_len), dtype=torch.int32),
                "last_pos": torch.tensor([0] * batch_size, dtype=torch.int32),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.int32),
            },
            tag="prefill",
        )

        # Add decode trace
        builder.trace(
            kwargs={
                "tokens": torch.ones((batch_size, 1), dtype=torch.int32),
                "last_pos": torch.tensor([0] * batch_size, dtype=torch.int32),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.int32),
            },
            tag="decode",
        )

        # Compile
        traced_model = builder.compile()

        # Save model and config
        os.makedirs(output_path, exist_ok=True)
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))
        _save_model_config(output_path, batch_size, seq_len, tp_degree, shard_on_load)
        print(f"Saved compiled model to {output_path}")

        parallel_state.set_aot_mode(False)
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

    if not shard_on_load:
        # Pre-shard weights and save them
        serialize_path = os.path.join(output_path, "weights/")

        with mock_distributed(world_size=tp_degree), init_on_device(torch.device("meta")):
            torch.distributed.init_process_group(backend="xla", rank=0, world_size=tp_degree)
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp_degree, skip_collective_init=True)

            shard_checkpoint(
                checkpoint=load_llama_checkpoint(Llama3_2_1B, model_path, tp_degree),
                model=Transformer(Llama3_2_1B, batch_size, seq_len),
                start_rank=0,
                end_rank=tp_degree-1,
                serialize_path=serialize_path
            )

            parallel_state.destroy_model_parallel()
            torch.distributed.destroy_process_group()

        print(f"Saved sharded checkpoints to {serialize_path}")


def compile(
    batch_size=2,
    seq_len=128,
    tp_degree=32,
    model_path="~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth",
    output_path="~/neuron_models/Llama3.2-1B-Instruct",
    shard_on_load=True,
):
    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        model = Transformer(Llama3_2_1B, batch_size, seq_len)

        # Initialize ModelBuilder
        builder = ModelBuilder(model=model)

        # Add prefill trace
        builder.trace(
            kwargs={
                "tokens": torch.ones((batch_size, seq_len), dtype=torch.int32),
                "last_pos": torch.tensor([0] * batch_size, dtype=torch.int32),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.int32),
            },
            tag="prefill",
        )

        # Add decode trace
        builder.trace(
            kwargs={
                "tokens": torch.ones((batch_size, 1), dtype=torch.int32),
                "last_pos": torch.tensor([0] * batch_size, dtype=torch.int32),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.int32),
            },
            tag="decode",
        )

        # Compile
        traced_model = builder.compile()

        # Save model and config
        os.makedirs(output_path, exist_ok=True)
        traced_model.save(os.path.join(output_path, "nxd_model.pt"))
        _save_model_config(output_path, batch_size, seq_len, tp_degree, shard_on_load)
        print(f"Saved compiled model to {output_path}")

    if not shard_on_load:
        # Pre-shard weights and save them
        serialize_path = os.path.join(output_path, "weights/")

        with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree), \
                init_on_device(torch.device("meta")):
            shard_checkpoint(
                checkpoint=load_llama_checkpoint(Llama3_2_1B, model_path, tp_degree),
                model=Transformer(Llama3_2_1B, batch_size, seq_len),
                start_rank=0,
                end_rank=tp_degree-1,
                serialize_path=serialize_path
            )

        print(f"Saved sharded checkpoints to {serialize_path}")


def test_attention(
    model_path="~/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth",
    output_path="~/neuron_models/Llama3.2-1B-Instruct_test_attention",
    tp_degree=32,
    batch_size=2,
    seq_len=128,
):
    torch.manual_seed(0)

    # Let's test with float32
    cfg = copy.deepcopy(Llama3_2_1B)
    cfg.dtype = torch.float32

    hidden_size = 2048
    head_dim = 64
    hidden = torch.randn((batch_size, seq_len, hidden_size), dtype=cfg.dtype)
    start_pos = torch.tensor([0] * batch_size)
    mask = torch.full((seq_len, seq_len), True).tril(diagonal=0)
    rope_cache = precompute_rope("cpu", 500000.0, head_dim, seq_len)

    # Save weights for testing
    neuron_attn_state_dict = _load_attn_state_dict(cfg, model_path, tp_degree)
    cpu_attn_state_dict = _load_attn_state_dict(cfg, model_path, 1)

    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree):
        attn = Attention(cfg, batch_size, seq_len)
        neuron_attn = ModelBuilder(model=attn) \
                        .trace(args=(hidden, start_pos, mask, rope_cache), tag="prefill") \
                        .compile()

        # Save the traced model
        os.makedirs(output_path, exist_ok=True)
        neuron_attn.save(os.path.join(output_path, "nxd_model.pt"))

    # Test CPU version
    attn = Attention(cfg, batch_size, seq_len)
    attn.load_state_dict(cpu_attn_state_dict, strict=False)
    cpu_o = attn(hidden, start_pos, mask, rope_cache)
    # Capture CPU cache_k and cache_v for comparison
    cpu_cache_k = attn.cache_k
    cpu_cache_v = attn.cache_v

    # Test Neuron version
    with NxDParallelState(world_size=tp_degree, tensor_model_parallel_size=tp_degree), init_on_device(torch.device("meta")):
        sharded_ckpts = shard_checkpoint(
            checkpoint=neuron_attn_state_dict,
            model=Attention(cfg, batch_size, seq_len),
            start_rank=0,
            end_rank=tp_degree-1,
        )
    
    neuron_attn = NxDModel.load(os.path.join(output_path, "nxd_model.pt"))
    neuron_attn.set_weights(sharded_ckpts)
    neuron_attn.to_neuron()
    
    # Note: We cannot return a single tensor is because
    # each core has its own copy of the tensor. As we are SPMD the output
    # is assumed to be the same, so return the output of the first core.
    neuron_o = neuron_attn(hidden, start_pos, mask, rope_cache)
    torch_neuronx.testing.assert_close(cpu_o, neuron_o)

    # But we can access them this way per rank. This is how you do it.
    if tp_degree == 1:
        rank = 0
        neuron_cache_k_n = neuron_attn.states[rank]["cache_k"].to("cpu")
        neuron_cache_v_n = neuron_attn.states[rank]["cache_v"].to("cpu")
        neuron_cache_k = neuron_cache_k_n.expand(batch_size, seq_len, cfg.n_kv_heads, head_dim)
        neuron_cache_v = neuron_cache_v_n.expand(batch_size, seq_len, cfg.n_kv_heads, head_dim)
        torch_neuronx.testing.assert_close(cpu_cache_k, neuron_cache_k, rtol=1e-5, atol=1e-5)
        torch_neuronx.testing.assert_close(cpu_cache_v, neuron_cache_v, rtol=1e-5, atol=1e-5)

    print("Attention test passed!")


def _load_attn_state_dict(cfg, model_path, tp_degree):
    model_path = os.path.expanduser(model_path)

    state_dict = load_llama_checkpoint(
        cfg=cfg,
        model_path=model_path,
        tp_degree=tp_degree
    )

    # Use the weights from the first layer.
    attn_dict = {
        re.sub(r"layers\.0\.attention\.", "", k): v.to(cfg.dtype)
        for (k, v) in state_dict.items()
        if re.search(r"layers\.0\.attention\.", k)
    }

    return attn_dict


def _print_string_list_output(func, *args, **kwargs):
    # Print output with newlines. Fire removes newlines when printing a list of strings.
    string_list = func(*args, **kwargs)
    for string in string_list:
        print(string)


if __name__ == "__main__":
    fire.Fire(
        {
            "generate_cpu": partial(_print_string_list_output, generate_cpu),
            "generate_nxd": partial(_print_string_list_output, generate_nxd),
            "compile": compile,
            "compile_with_mock": compile_with_mock,
            "compile_no_mock": compile_no_mock,
            "_compile_no_mock": _compile_no_mock,
            "test_attention": test_attention,
        }
    )
