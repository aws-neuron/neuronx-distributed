import torch
from dbrx.dbrx_runner import DbrxRunner
from transformers import GenerationConfig

model_path = "/data/model_hf/dbrx-base/"
traced_model_path = "/data/traced_model/dbrx-base/"

torch.manual_seed(0)

def dbrx_sample():
    # Compile the model for a specific configuration
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config.top_k = 1
    generation_config.do_sample = True

    runner = DbrxRunner(model_path=model_path, tokenizer_path=model_path, generation_config=generation_config)

    batch_size = 1
    max_prompt_length = 1024
    sequence_length = 1024 + 128

    runner.trace(
        traced_model_path=traced_model_path,
        tp_degree=32,
        batch_size=batch_size,
        max_prompt_length=max_prompt_length,
        sequence_length=sequence_length,
    )
    # Load model weights into Neuron devise
    # We will use the returned model to run accuracy and perf tests
    print("\nLoading model to Neuron device ..")
    neuron_model = runner.load_neuron_model(traced_model_path)

    # Confirm the traced model matches the huggingface model run on cpu
    print("\nChecking accuracy ..")
    runner.check_accuracy_logits(neuron_model, batch_size, sequence_length)

    # Perform inference
    prompt = ["I believe the meaning of life is", "The color of the sky is"]
    print("\nGenerating ..")
    _, outputs = runner.generate_on_neuron(prompt, neuron_model)
    print("Generated outputs:")
    for idx, output in enumerate(outputs):
        print(f"output {idx}: {output}")

    print("\nBenchmarking ..")
    # Now lets benchmark
    runner.benchmark_sampling(neuron_model)


if __name__ == "__main__":
    dbrx_sample()
