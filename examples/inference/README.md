## Running llama2
1. If you are using meta-llama, get access to it. Follow instructions on huggingface.co to get access.
1. Use the runner.py to trace the model
1. Use the runner.py to perform inference

With runner.py, there are a few actions you can perform,

* trace - Traces the neuron wrapper
* infer - Runs the traced model on Neuron
* infer-on-cpu - Runs the neuron wrapper on CPU
* infer-with-hf - Runs inference with huggingface model on CPU


### Inference example

Trace the model
```
python runner.py trace \
                --model LlamaForCausalLM \
                --model_path "/home/ubuntu/model_hf/llama-2-7b/" \
                --tokenizer_path "/home/ubuntu/model_hf/llama-2-7b/" \
                --tp_degree 16 \
                --batch_size 2 \
                --max_context_length 128 \
                --max_new_tokens 384 \
                --traced_model_path "/home/ubuntu/traced_model/llama2_7b/"
```

Run greedy sampling on neuron
```
python runner.py infer \
                --model LlamaForCausalLM \
                --traced_model_path "/home/ubuntu/traced_model/llama2_7b/" \
                -p "I believe the meaning of life is" \
                -p "The color of the sky is"
```

Run greedy sampling using the neuron wrapper on cpu

```
  python runner.py infer-on-cpu \
                        --model LlamaForCausalLM \
                        --model_path "/home/ubuntu/model_hf/llama-2-7b/" \
                        --tokenizer_path "/home/ubuntu/model_hf/llama-2-7b/" \
                        -p "I believe the meaning of life is" \
                        -p "The color of the sky is" \
                        --batch_size 2 \
                        --max_context_length 128 \
                        --max_new_tokens 384
```

Run greedy sampling using huggingface llama implementation. This can be used to validate the Neuron results.

```
  python runner.py  infer-with-hf \
                            --model LlamaForCausalLM \
                            --model_path "/home/ubuntu/model_hf/llama-2-7b/" \
                            --tokenizer_path "/home/ubuntu/model_hf/llama-2-7b/" \
                            -p "I believe the meaning of life is"  \
                            -p "The color of the sky is" \
                            --batch_size 2 \
                            --max_context_length 128 \
                            --max_new_tokens 384
```

Benchmark the greedy sampling

```
  python runner.py  benchmark-sampling \
                            --model LlamaForCausalLM \
                            --traced_model_path "/home/ubuntu/traced_model/llama2_7b/"
```

Check the accuracy of the neuron model
```
python runner.py check-accuracy \
                --model LlamaForCausalLM \
                --model_path "/home/ubuntu/model_hf/llama-2-7b/" \
                --tokenizer_path "/home/ubuntu/model_hf/llama-2-7b/" \
                --batch_size 2 \
                --max_context_length 128 \
                --max_new_tokens 384 \
                --traced_model_path "/home/ubuntu/traced_model/llama2_7b/"
```
