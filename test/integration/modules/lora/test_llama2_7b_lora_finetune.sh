#!/bin/bash

#############################################
# Override transformers and Optimum-Neuron packages, can be removed once ON released changes in https://github.com/huggingface/optimum-neuron/pull/370
pip install git+https://github.com/huggingface/optimum-neuron.git
pip install --no-warn-conflicts transformers==4.32.1 nltk

#############################################
# User defined parameters and env vars

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export NEURON_CC_FLAGS="--model-type=transformer --distribution-strategy=llm-training "
export NEURON_FUSE_SOFTMAX=1

# Async Runtime
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3

# HOST OOM
export MALLOC_ARENA_MAX=64

# TP degree
TP_DEGREE=8
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=1
# 0: use pure DP; 1: use ZeRO-1
USE_ZERO_1=0
# global batch size
GBS=8
# micro batch size
MBS=1
# number of steps to run
TOTAL_STEPS=1000
# number of epochs to run
TOTAL_EPOCHS=2
# warmup steps
WARMUP_STEPS=5
# learning rate
LR=5.0e-4
# model path
MODEL_PATH=$SCRIPT_DIR/finetune_config
# pretrained weight path
PRETRAINED_PATH="$HOME/llama-2-7b-sharded"
# base model name
BASE_MODEL="NousResearch/Llama-2-7b-hf"
# sequence length
SEQ_LEN=4096
# golden rouge score path
GOLDEN_ROUGE_SCORE_PATH="llama2_7b_rouge_score_goldens_lora.json"

#############################################

export NUM_NEURONCORES=${TP_DEGREE}
NODE_ID=0
WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"
if [ ! -z "$SLURM_NTASKS" ]; then
    WORLD_SIZE=$SLURM_NTASKS
    NODE_ID=$SLURM_NODEID
    MASTER_ADDRESS=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
    DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $WORLD_SIZE --node_rank $NODE_ID --master_addr $MASTER_ADDRESS --master_port 44000"
    if [ $NODE_ID -eq 0 ]; then
        echo "WORLD_SIZE=$WORLD_SIZE"
        echo "NODE_ID=$NODE_ID"
        echo "MASTER_ADDRESS=$MASTER_ADDRESS"
        echo "DISTRIBUTED_ARGS=$DISTRIBUTED_ARGS"
    fi
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
fi

echo "WORLD_SIZE=$WORLD_SIZE"
echo "NODE_ID=$NODE_ID"
echo "MASTER_ADDRESS=$MASTER_ADDRESS"

sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620

#############################################

EXTRA_ARGS=" "
if [ $USE_MIX_PRECISION -gt 0 ]; then
    EXTRA_ARGS+=" --use_mix_precision"
fi
if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero_1"
fi

ACC_STEPS=$(($GBS / $MBS))


if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    STEPS_THIS_RUN=10
    OUTPUT_LOG=log_compile-$NODE_ID.log
else
    STEPS_THIS_RUN=1000
    OUTPUT_LOG=log_exe-$NODE_ID.log
fi

echo TP_DEGREE=$TP_DEGREE
echo USE_MIX_PRECISION=$USE_MIX_PRECISION
echo USE_ZERO_1=$USE_ZERO_1
echo GBS=$GBS
echo MBS=$MBS
echo TOTAL_STEPS=$TOTAL_STEPS
echo TOTAL_EPOCHS=$TOTAL_EPOCHS
echo WARMUP_STEPS=$WARMUP_STEPS
echo LR=$LR
echo MODEL_PATH=$MODEL_PATH
echo SEQ_LEN=$SEQ_LEN

echo EXTRA_ARGS=$EXTRA_ARGS
echo DP=$DP
echo ACC_STEPS=$ACC_STEPS
echo STEPS_THIS_RUN=$STEPS_THIS_RUN
echo OUTPUT_LOG=$OUTPUT_LOG

torchrun $DISTRIBUTED_ARGS \
    tp_llama_hf_finetune_ptl.py \
    --model_path $MODEL_PATH \
    --model_name $BASE_MODEL \
    --data_dir "databricks/databricks-dolly-15k" \
    --tensor_parallel_size $TP_DEGREE \
    --batch_size $MBS \
    --steps_this_run $STEPS_THIS_RUN \
    --max_steps $TOTAL_STEPS \
    --num_train_epochs $TOTAL_EPOCHS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --grad_accum_usteps $ACC_STEPS \
    --seq_len $SEQ_LEN \
    --selective_checkpoint_enabled \
    --golden_rouge_score_path $GOLDEN_ROUGE_SCORE_PATH  \
    --pretrained_ckpt $PRETRAINED_PATH \
    --task "open_qa" \
    --fuse_qkv 0 \
    $EXTRA_ARGS \
    --enable_lora \
    --use_gpu_compatible_precision 0 \


ret_val=${PIPESTATUS[0]}
echo ret_val=$ret_val

if [ -v PERF_TEST ];
then
    echo "Performance test complete"
else
  if [ $ret_val -eq 0 ]; then
      success=1
  else
      success=0
  fi

  if [ -z "$NEURON_EXTRACT_GRAPHS_ONLY" ]; then
      echo "success=$success"
      echo "update json with $HOME/ktest/dump_to_s3_update_test_json.sh"
      dump_to_s3_update_json_scr=$HOME/ktest/dump_to_s3_update_test_json.sh
      if [ -e $dump_to_s3_update_json_scr ]; then
          $dump_to_s3_update_json_scr $@ --key=inference_success --value=$success || echo "Unable to update test result JSON."
      else
          echo "WARNING: Script $dump_to_s3_update_json_scr not found. Not updating test result JSON."
      fi
  fi
fi
