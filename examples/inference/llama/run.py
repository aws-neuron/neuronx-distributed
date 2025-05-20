import copy
import json
import os
import re
from functools import partial
from typing import List

import fire
import torch
import torch_neuronx
from neuronx_distributed.trace import ModelBuilder
from neuronx_distributed.trace.model_builder import BaseModelInstance
from safetensors.torch import load_file

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


@torch.inference_mode()
def generate_cpu(
    batch_size=2,
    seq_len=128,
    model_path="/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth",
    tokenizer_path="/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model",
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
    compiled_model_path="/home/ubuntu/neuron_models/Llama3.2-1B-Instruct",
    tokenizer_path="/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/tokenizer.model",
    prompts=["How tall is the Space Needle?", "What is the capital of France?"],
):
    if not compiled_model_path.endswith("/"):
        compiled_model_path += "/"
    with open(compiled_model_path + "config.json", "r") as file:
        cfg = json.load(file)

    bs, seq_len, tp_degree = cfg["batch_size"], cfg["seq_len"], cfg["tp_degree"]

    if len(prompts) != bs:
        raise ValueError(f"Prompts size does not match batch size {cfg['batch_size']}")

    weights = []
    for rank in range(tp_degree):
        ckpt = load_file(os.path.join(compiled_model_path, f"weights/tp{rank}_sharded_checkpoint.safetensors"))
        weights.append(ckpt)

    model = torch.jit.load(compiled_model_path + "nxd_model.pt")
    start_rank_tensor = torch.tensor([0], dtype=torch.int32, device="cpu")
    model.nxd_model.initialize(weights, start_rank_tensor)

    tokenizer = Tokenizer(model_path=tokenizer_path)
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    output_tokens = generate(model, seq_len, prompt_tokens, stop_tokens=tokenizer.stop_tokens)

    return [tokenizer.decode(tokens) for tokens in output_tokens]


def compile(
    batch_size=2,
    seq_len=128,
    tp_degree=32,
    model_path="/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct/consolidated.00.pth",
    output_path="/home/ubuntu/neuron_models/Llama3.2-1B-Instruct",
):

    # ModelBuilder takes in a object of type BaseModelInstance. This object
    # should have two functions `load_module` and `get`. The object declares
    # how the object can be initialized so the tracing function can create its own
    # instance.
    #
    # What is an alias? and why do I need it?
    # If you have any state that you want to use across model invocations
    # you would want to use an alias. Aliasing tells the compiler that
    # the output can be written to the same input buffer. This avoids
    # the creation of duplicate memory allocations for the output.
    #
    # On NxD, all output tensors are copied back from device to CPU
    # after a model invocation. But the aliased tensors are not returned
    # and are retained on device.
    #
    # So if you have a buffer that is expensive to repeatedly copy
    # to and from the device, you should use an alias. KV Cache is a good
    # candidate for aliasing.
    #
    # How do I define an alias?
    # Alias is a map. It maps buffer -> output index.
    #
    # Say we have Module defined as,
    #
    #  Module(torch.nn.Module):
    #
    #    def __init__(self):
    #      self.register_buffer("cache", torch.zeros(...))
    #
    #    def forward(input_A, input_B):
    #       ...
    #       return output, output_A
    #
    # And we want to alias input_A and output_A. The alias would say,
    #
    #  module = Module()
    #  alias = { module.cache : 1 }
    #
    #  This means `cache` is aliased to the output number 1 which is `output_A`
    class Instance(BaseModelInstance):

        def __init__(self):
            self.module = None

        def load_module(self):
            self.module = Transformer(Llama3_2_1B, batch_size, seq_len)

        def get(self, bucket_rank, **kwargs):

            # The Transformer model return logits as index 0. We want to start
            # aliasing from output index 1
            #
            # Transformer() -> (logits,
            #                   k_cache_lay1, .., k_cache_layN,
            #                   v_cache_lay1, ... v_cache_layN)

            aliases = {}
            output_index = 1
            for i, layer in enumerate(self.module.layers):
                aliases[layer.attention.cache_k] = output_index
                output_index = output_index + 1
            for i, layer in enumerate(self.module.layers):
                aliases[layer.attention.cache_v] = output_index
                output_index = output_index + 1

            return self.module, aliases

    builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=partial(load_llama_checkpoint, Llama3_2_1B, model_path, tp_degree),
        debug=True,
    )
    builder.add(
        key="prefill",
        model_instance=Instance(),
        example_inputs=[
            (
                torch.ones((batch_size, seq_len), dtype=torch.int32),  # input tokens
                torch.tensor([0] * batch_size, dtype=torch.int32),
                torch.ones((batch_size, seq_len), dtype=torch.int32),  # attention mask
            )
        ],  # last_pos
        compiler_args="--auto-cast=none",
    )
    builder.add(
        key="decode",
        model_instance=Instance(),
        example_inputs=[
            (
                torch.ones((batch_size, 1), dtype=torch.int32),  # input tokens
                torch.tensor([0] * batch_size, dtype=torch.int32),
                torch.ones((batch_size, seq_len), dtype=torch.int32),  # attention mask
            )
        ],  # last_pos
        compiler_args="--auto-cast=none",
    )

    traced_model = builder.trace(initialize_model_weights=False)

    if not output_path.endswith("/"):
        output_path += "/"
    builder.shard_checkpoint(serialize_path=output_path + "weights/")
    torch.jit.save(traced_model, output_path + "nxd_model.pt")

    # Lets store the config along with the saved model
    data = {"batch_size": batch_size, "seq_len": seq_len, "tp_degree": tp_degree}
    with open(output_path + "config.json", "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved compiled model to {output_path}")


def test_attention(
    model_path="/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct",
    tp_degree=32,
    batch_size=2,
    seq_len=128,
):
    torch.manual_seed(0)

    # Lets test with float32
    cfg = copy.deepcopy(Llama3_2_1B)
    cfg.dtype = torch.float32

    hidden_size = 2048
    head_dim = 64
    hidden = torch.randn((batch_size, seq_len, hidden_size), dtype=cfg.dtype)
    start_pos = torch.tensor([0] * batch_size)
    mask = torch.full((seq_len, seq_len), True).tril(diagonal=0)
    rope_cache = precompute_rope("cpu", 500000.0, head_dim, seq_len)

    neuron_attn_state_dict = _load_attn_state_dict(cfg, model_path, tp_degree)
    cpu_attn_state_dict = _load_attn_state_dict(cfg, model_path, 1)

    class Instance(BaseModelInstance):

        def __init__(self):
            self.module = None

        def load_module(self):
            self.module = Attention(cfg, batch_size, seq_len)

        def get(self, bucket_rank, **kwargs):
            return self.module, {self.module.cache_k: 1, self.module.cache_v: 2}

    builder = ModelBuilder(
        router=None, tp_degree=tp_degree, checkpoint_loader=lambda: neuron_attn_state_dict, debug=True
    )
    builder.add(
        key="prefill",
        model_instance=Instance(),
        example_inputs=[(hidden, start_pos, mask, rope_cache)],
        compiler_args="--auto-cast=none",
    )
    neuron_attn = builder.trace()

    start_rank_tensor = torch.tensor([0], dtype=torch.int32, device="cpu")
    neuron_attn.nxd_model.initialize_with_saved_weights(start_rank_tensor)

    attn = Attention(cfg, batch_size, seq_len)
    attn.load_state_dict(cpu_attn_state_dict, strict=False)

    cpu_o, cpu_cache_k, cpu_cache_v = attn(hidden, start_pos, mask, rope_cache)

    # Note: If you alias the tensors, the compiled model will not return it.
    # They are kept on device. We cannot return a single tensor is because
    # each core has its own copy of the tensor. As we are SPMD the output
    # is assumed to be the same, so return the output of the first core.
    neuron_o = neuron_attn(hidden, start_pos, mask, rope_cache)
    torch_neuronx.testing.assert_close(cpu_o, neuron_o)

    # But we can access them this way per rank. This is how you do it.
    if tp_degree == 1:
        rank = 0
        neuron_cache_k_n = neuron_attn.nxd_model.state[rank]["cache_k"].to("cpu")
        neuron_cache_v_n = neuron_attn.nxd_model.state[rank]["cache_v"].to("cpu")
        neuron_cache_k = neuron_cache_k_n.expand(batch_size, seq_len, cfg.n_kv_heads, head_dim)
        neuron_cache_v = neuron_cache_v_n.expand(batch_size, seq_len, cfg.n_kv_heads, head_dim)
        torch_neuronx.testing.assert_close(cpu_cache_k, neuron_cache_k, rtol=1e-5, atol=1e-5)
        torch_neuronx.testing.assert_close(cpu_cache_v, neuron_cache_v, rtol=1e-5, atol=1e-5)

    print("Attention test passed!")


def _load_attn_state_dict(cfg, model_path, tp_degree):
    if not model_path.endswith("/"):
        model_path += "/"
    state_dict = load_llama_checkpoint(model_path=model_path + "consolidated.00.pth", cfg=cfg, tp_degree=tp_degree)
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
            "test_attention": test_attention,
        }
    )
