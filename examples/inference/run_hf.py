from transformers import AutoTokenizer, AutoConfig

from llama2.neuron_modeling_llama import NeuronLlamaForCausalLM, NeuronConfig
import torch


if __name__ == "__main__":
    model_path = "/home/ubuntu/model_hf/Llama-2-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    hf_config = AutoConfig.from_pretrained(model_path)
    hf_config.torch_dtype = torch.bfloat16
    config = NeuronConfig(
        hf_config,
        tp_degree=32,
        seq_len=128,
        max_new_tokens=32,
        batch_size=1,
    )

    model = NeuronLlamaForCausalLM.from_pretrained(model_path, config)

    model.to_neuron()


    prompt = ["I believe the meaning of life is", "The color of the sky is"]

    inputs = tokenizer(prompt, padding=True, return_tensors="pt")
    
    output_sequences = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=50,
        max_length=150,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    generated_texts = [tokenizer.decode(output_sequence, skip_special_tokens=True) for output_sequence in output_sequences]
    for i, generated_text in enumerate(generated_texts):
        print(f"Generated Text {i + 1}:\n{generated_text}\n")
