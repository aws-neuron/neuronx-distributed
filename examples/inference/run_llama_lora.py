import os

import torch
from llama2.llama2_runner import LlamaRunner
from transformers import GenerationConfig
from modules.lora_serving import LoraServingConfig

model_path = "/home/ubuntu/LLama7b/"
traced_model_path = "/home/ubuntu/traced_model/LLama7b_lora/"

def llama_cpu_sample():
    # Compile the model for a specific configuration
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.top_k = 1
    generation_config.do_sample = True

    runner = LlamaRunner(model_path=model_path, tokenizer_path=model_path, generation_config=generation_config)

    batch_size = 1
    max_prompt_length = 32
    sequence_length = 32

    # # Load model weights into Neuron device
    # # We will use the returned model to run accuracy and perf tests
    # print("\nLoading model to Neuron device ..")
    # neuron_model = runner.load_neuron_model(traced_model_path)

    prompt = ["I believe the meaning of life is"]

    max_loras = 1
    lora_config = LoraServingConfig(
        max_loras = max_loras,
        max_lora_rank = 16,
        target_modules = ["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )

    output_token_ids, outputs = runner.generate_on_cpu(
        prompt,
        batch_size,
        max_prompt_length,
        sequence_length,
        lora_config = lora_config,
    )

    print("Generated CPU outputs:")
    for idx, output in enumerate(outputs):
        print(f"output {idx}: {output}")


if __name__ == "__main__":
    llama_cpu_sample()
