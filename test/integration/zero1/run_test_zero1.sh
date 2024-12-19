#! /bin/bash
set -x

export NEURON_CC_FLAGS="--model-type=transformer --distribution-strategy=llm-training"
export XLA_DISABLE_FUNCTIONALIZATION=0
if [[ "$1" == "use_bf16" ]]; then
  export XLA_USE_BF16=1
fi
torchrun --nproc_per_node=32 test_zero1.py
