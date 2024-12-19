import os

import torch
from llama2.llama2_runner import LlamaRunner
from transformers import GenerationConfig

model_path = "/home/ubuntu/LLama7b/"
traced_model_path = "/home/ubuntu/traced_model/LLama7b_quantized/"

def prune_state_dict(state_dict):
    """
    A helper function that deletes None values in the state_dict before saving 
    as torch.save does not like None values in the state dict.
    """
    keys_to_delete = []
    for key in state_dict:
        if state_dict[key] is None:
            keys_to_delete.append(key)

    print(f"Will be deleting following keys as its Value is None: {keys_to_delete}")

    pruned_state_dict = {k:v for k,v in state_dict.items() if v is not None}
    return pruned_state_dict

def llama_get_quantized_checkpoint(path_to_save):
    """
    This example generates the quantized checkpoints and returns a state dict
    """
    # Compile the model for a specific configuration
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.top_k = 1
    generation_config.do_sample = True

    runner = LlamaRunner(model_path=model_path, tokenizer_path=model_path, generation_config=generation_config)
    batch_size = 2
    max_prompt_length = 128
    sequence_length = 512

    quantized_state_dict = runner.generate_quantized_hf_checkpoints_on_cpu(
        max_prompt_length=max_prompt_length,
        sequence_length=sequence_length,
        batch_size=batch_size,
        quantized=True,
        quantized_checkpoints_path="",
        quantization_type="per_channel_symmetric",
    )

    # delete None values in the quantized_state_dict. torch.save crashes if None values exist.
    quantized_state_dict = prune_state_dict(quantized_state_dict)
    torch.save(quantized_state_dict, path_to_save)

    return quantized_state_dict


def llama_cpu_sample():
    """
    This example generates the LLama model on CPU and quantize it using the Pytorch dynamic quantization api.
    """
    # Compile the model for a specific configuration
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.top_k = 1
    generation_config.do_sample = True

    runner = LlamaRunner(model_path=model_path, tokenizer_path=model_path, generation_config=generation_config)

    batch_size = 2
    max_prompt_length = 128
    sequence_length = 512
    prompt = ["I believe the meaning of life is", "The color of the sky is"]

    _, outputs = runner.generate_on_cpu(
        prompt=prompt,
        batch_size=batch_size,
        max_prompt_length=max_prompt_length,
        sequence_length=sequence_length,
        quantized=True,
        quantized_checkpoints_path="",
        quantization_type="per_channel_symmetric",
    )
    print("\nGenerating ..")
    for idx, output in enumerate(outputs):
        print(f"output {idx}: {output}")


def llama_sample(generate_checkpoint=False):
    # Compile the model for a specific configuration
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.top_k = 1
    generation_config.do_sample = True

    runner = LlamaRunner(model_path=model_path, tokenizer_path=model_path, generation_config=generation_config)

    batch_size = 2
    max_prompt_length = 128
    sequence_length = 512

    if generate_checkpoint:
        quantized_checkpoints_path = os.path.join(model_path, "model_quant.pt")
        quantized_state_dict = runner.generate_quantized_hf_checkpoints_on_cpu(
            max_prompt_length=max_prompt_length, sequence_length=sequence_length, batch_size=batch_size
        )
        # delete None values in the quantized_state_dict. torch.save crashes if None values exist.
        quantized_state_dict = prune_state_dict(quantized_state_dict)
        torch.save(quantized_state_dict, quantized_checkpoints_path)
        

    runner.trace(
        traced_model_path=traced_model_path,
        tp_degree=32,
        batch_size=batch_size,
        max_prompt_length=max_prompt_length,
        sequence_length=sequence_length,
        on_device_sampling=True,
        quantized=True,
        quantized_checkpoints_path=os.path.join(model_path, "model_quant.pt"),
        quantization_type="per_channel_symmetric",
    )

    # Perform inference
    print("\nLoading model to Neuron device ..")
    neuron_model = runner.load_neuron_model(traced_model_path)

    prompt = ["I believe the meaning of life is", "The color of the sky is"]
    print("\nGenerating ..")
    _, outputs = runner.generate_on_neuron(prompt, neuron_model)
    print("Generated outputs:")
    for idx, output in enumerate(outputs):
        print(f"output {idx}: {output}")

    runner.benchmark_sampling(neuron_model)


if __name__ == "__main__":
    # llama_cpu_sample()
    llama_sample()
    # llama_get_quantized_checkpoint(os.path.join(model_path, "model_quant.pt"))
