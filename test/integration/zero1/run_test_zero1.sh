#! /bin/bash
set -x

export NEURON_CC_FLAGS="--model-type=transformer "
export XLA_DISABLE_FUNCTIONALIZATION=0
use_bf16=false
model_dtype="fp32"
optimizer_dtype="fp32"

for arg in "$@"
do
    case $arg in
        use_bf16)
        use_bf16=true
        shift
        ;;
        model_dtype=*)
        model_dtype="${arg#*=}"
        shift
        ;;
        optimizer_dtype=*)
        optimizer_dtype="${arg#*=}"
        shift
        ;;
    esac
done

if [ "$use_bf16" = true ] ; then
  export XLA_USE_BF16=1
fi

torchrun --nproc_per_node=32 test_zero1.py "$model_dtype" "$optimizer_dtype"