from llama2.llama2_runner import LlamaRunner

target_model_path = "/home/ubuntu/open_llama_7b/"
draft_model_path = "/home/ubuntu/open_llama_3b/"
traced_target_model_path = "/home/ubuntu/traced_model/open_llama_7b/"
traced_draft_model_path = "/home/ubuntu/traced_model/open_llama_3b/"


def llama_sample():
    # Compile the model for a specific configuration
    target_runner = LlamaRunner(model_path=target_model_path, tokenizer_path=target_model_path)
    draft_runner = LlamaRunner(model_path=draft_model_path, tokenizer_path=draft_model_path)

    # Batch size must be 1 for speculative decoding
    batch_size = 1
    max_prompt_length = 256
    sequence_length = 512

    # Need to trace both target and draft models
    # We don't need to trace token generation model for target
    # Here we specify the speculation length with the target model
    target_runner.trace(
        traced_model_path=traced_target_model_path,
        tp_degree=32,
        batch_size=batch_size,
        max_prompt_length=max_prompt_length,
        sequence_length=sequence_length,
        speculation_length=5,
        trace_tokengen_model=False,
    )
    draft_runner.trace(
        traced_model_path=traced_draft_model_path,
        tp_degree=32,
        batch_size=batch_size,
        max_prompt_length=max_prompt_length,
        sequence_length=sequence_length,
    )

    target_model = target_runner.load_neuron_model(traced_target_model_path)
    draft_model = draft_runner.load_neuron_model(traced_draft_model_path)

    # Confirm the traced model matches the huggingface model run on cpu
    print("\nChecking accuracy ..")
    target_runner.check_accuracy(
        target_model,
        batch_size,
        sequence_length,
        traced_draft_model=draft_model,
        speculation_length=5,
    )

    # Perform inference
    prompt = ["I believe the meaning of life is"]
    print("\nGenerating ..")
    _, outputs = target_runner.generate_on_neuron(prompt, target_model, draft_model=draft_model)
    print("Generated outputs:")
    for idx, output in enumerate(outputs):
        print(f"output {idx}: {output}")

    print("\nBenchmarking ..")
    # Now lets benchmark
    target_runner.benchmark_sampling(target_model, draft_model=draft_model)


if __name__ == "__main__":
    llama_sample()
