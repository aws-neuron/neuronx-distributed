from llama2.llama2_runner import LlamaRunner
from transformers import GenerationConfig

model_path = "/home/ubuntu/LLama7b/"
traced_model_path = "/home/ubuntu/traced_model/quantized_Llama-2-7b/"


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
    max_context_length = 128
    max_new_tokens = 384
    prompt = ["I believe the meaning of life is", "The color of the sky is"]

    _, outputs = runner.generate_on_cpu(
        prompt=prompt,
        batch_size=batch_size,
        max_context_length=max_context_length,
        max_new_tokens=max_new_tokens,
        quantized=True,
    )
    print("\nGenerating ..")
    for idx, output in enumerate(outputs):
        print(f"output {idx}: {output}")


def llama_sample():
    # Compile the model for a specific configuration
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.top_k = 1
    generation_config.do_sample = True

    runner = LlamaRunner(model_path=model_path, tokenizer_path=model_path, generation_config=generation_config)

    batch_size = 2
    max_context_length = 128
    max_new_tokens = 384

    runner.trace(
        traced_model_path=traced_model_path,
        tp_degree=32,
        batch_size=batch_size,
        context_lengths=max_context_length,
        new_token_counts=max_new_tokens,
        on_device_sampling=True,
        quantized=True,
        quantized_checkpoints_path="/home/ubuntu/LLama7b/model_quant.pt",
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


if __name__ == "__main__":
    # llama_cpu_sample()
    llama_sample()
