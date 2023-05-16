import os
import neuronx_distributed
from neuronx_distributed.parallel_layers import layers, parallel_state
import torch
import torch_neuronx
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput


def encode(tokenizer, *inputs, max_length=128, batch_size=1):
    tokens = tokenizer.encode_plus(
        *inputs,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    return (
        torch.repeat_interleave(tokens['input_ids'], batch_size, 0),
        torch.repeat_interleave(tokens['attention_mask'], batch_size, 0),
        torch.repeat_interleave(tokens['token_type_ids'], batch_size, 0),
    )


# Create the tokenizer and model
name = "bert-base-cased-finetuned-mrpc"
tokenizer = AutoTokenizer.from_pretrained(name)


# Set up some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

paraphrase = encode(tokenizer, sequence_1, sequence_2)
not_paraphrase = encode(tokenizer, sequence_1, sequence_1)

def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
    class ParallelSelfAttention(BertSelfAttention):
        def __init__(self, config, position_embedding_type=None):
            super().__init__(config, position_embedding_type)
            self.query = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
            self.key = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
            self.value = layers.ColumnParallelLinear(config.hidden_size, self.all_head_size, gather_output=False)
            self.num_attention_heads = self.num_attention_heads // parallel_state.get_tensor_model_parallel_size()
            self.all_head_size = self.all_head_size // parallel_state.get_tensor_model_parallel_size()

    class ParallelSelfOutput(BertSelfOutput):
        def __init__(self, config):
            super().__init__(config)
            self.dense = layers.RowParallelLinear(config.hidden_size,
                                       config.hidden_size,
                                       input_is_parallel=True)

    for layer in model.bert.encoder.layer:
        layer.attention.self = ParallelSelfAttention(model.config)
        layer.attention.output = ParallelSelfOutput(model.config)
    
    neuronx_distributed.parallel_layers.load("bert/bert.pt", model, sharded=False)
    
    return model

model = neuronx_distributed.trace.parallel_model_trace(get_model, paraphrase, tp_degree=2)

neuronx_distributed.trace.parallel_model_save(model, "tp_models")
model = neuronx_distributed.trace.parallel_model_load("tp_models")

model_cpu = AutoModelForSequenceClassification.from_pretrained(name, torchscript=True)
model_neuron = torch_neuronx.trace(model_cpu, paraphrase)
print(model(*paraphrase), model_cpu(*paraphrase), model_neuron(*paraphrase))