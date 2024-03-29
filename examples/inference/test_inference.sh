#!/bin/bash

PROMPT1="I believe the meaning of life is"
PROMPT2="The color of the sky is"
MAX_CONTEXT_LENGTH=128
MAX_NEW_TOKENS=384
BATCH_SIZE=2
TP_DEGREE=32

# The directory separator (/) at the end is needed
STORE_PATH="$HOME/model_hf/llama-2-7b/"
TRACED_PATH="$HOME/traced_model/llama2_7b/"

if [ "$1" = "trace" ]
then
  python runner.py trace \
                      --model LlamaForCausalLM \
                      --model_path $STORE_PATH \
                      --tokenizer_path $STORE_PATH \
                      --tp_degree $TP_DEGREE \
                      --batch_size $BATCH_SIZE \
                      --max_context_length $MAX_CONTEXT_LENGTH \
                      --max_new_tokens $MAX_NEW_TOKENS \
                      --traced_model_path $TRACED_PATH \

elif [ "$1" = "infer" ]
then
  python runner.py  infer \
                        --model LlamaForCausalLM \
                        --traced_model_path $TRACED_PATH \
                        -p "$PROMPT1" \
                        -p "$PROMPT2" \
                        2>&1 | tee stdout.log

elif [ "$1" = "infer-on-cpu" ]
then
  python runner.py  infer-on-cpu \
                            --model LlamaForCausalLM \
                            --model_path $STORE_PATH \
                            --tokenizer_path $STORE_PATH \
                            -p "$PROMPT1" \
                            -p "$PROMPT2" \
                            --batch_size $BATCH_SIZE \
                            --max_context_length $MAX_CONTEXT_LENGTH \
                            --max_new_tokens $MAX_NEW_TOKENS \
                            2>&1 | tee stdout.log

elif [ "$1" = "infer-with-hf" ]
then
  python runner.py  infer-with-hf \
                            --model LlamaForCausalLM \
                            --model_path $STORE_PATH \
                            --tokenizer_path $STORE_PATH \
                            -p "$PROMPT1" \
                            -p "$PROMPT2" \
                            --batch_size $BATCH_SIZE \
                            --max_context_length $MAX_CONTEXT_LENGTH \
                            --max_new_tokens $MAX_NEW_TOKENS \
                            2>&1 | tee stdout.log

elif [ "$1" = "benchmark" ]
then
  python runner.py  benchmark-sampling \
                            --model LlamaForCausalLM \
                            --model_path $STORE_PATH \
                            --tokenizer_path $STORE_PATH \
                            --traced_model_path $TRACED_PATH \
                            2>&1 | tee stdout.log

elif [ "$1" = "check-accuracy" ]
then
  python runner.py check-accuracy \
                      --model LlamaForCausalLM \
                      --model_path $STORE_PATH \
                      --tokenizer_path $STORE_PATH \
                      --batch_size $BATCH_SIZE \
                      --max_context_length $MAX_CONTEXT_LENGTH \
                      --max_new_tokens $MAX_NEW_TOKENS \
                      --traced_model_path $TRACED_PATH \

else
  echo "Run as ./test_inference.sh <trace|infer|infer-on-cpu|infer-with-hf|benchmark>"
fi
