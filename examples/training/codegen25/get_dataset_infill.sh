#!/usr/bin/bash

TOKENIZER="~/examples/codegen25-7b-mono/"

python get_dataset_infill.py \
    --dataset_name="ammarnasr/the-stack-java-clean" \
    --json-keys=content \
    --tokenizer-model=$TOKENIZER \
    --output_path="~/example_datasets/bigcode-stack-java_tokenized_infill" \
    --block_size=2048