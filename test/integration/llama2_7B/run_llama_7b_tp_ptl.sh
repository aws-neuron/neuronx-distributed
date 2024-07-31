#!/bin/bash

#############################################
# User defined parameters and env vars

if [ -z "$SEQ_LEN" ];
then
    DATA_PATH="$HOME/wikicorpus_datasets/wikicorpus_llama_v2_tokenized_4k"
    SEQ_LEN=4096
    n_layers=-1
    GBS=256
    total_steps=150
    TP_DEGREE=8
    use_flash_attention=0
elif [[ $SEQ_LEN = 8192 ]];
then
    DATA_PATH="$HOME/wikicorpus_datasets/wikicorpus_llama_v2_tokenized_8k"
    SEQ_LEN=8192
    n_layers=8
    GBS=16
    total_steps=30
    TP_DEGREE=32
    use_flash_attention=1
elif [[ $SEQ_LEN = 16384 ]];
then
    DATA_PATH="$HOME/wikicorpus_datasets/wikicorpus_llama_v2_tokenized_16k"
    SEQ_LEN=16384
    n_layers=8
    GBS=16
    total_steps=30
    TP_DEGREE=32
    use_flash_attention=1
elif [[ $SEQ_LEN = 32768 ]];
then
    DATA_PATH="$HOME/wikicorpus_datasets/wikicorpus_llama_v2_tokenized_32k"
    SEQ_LEN=32768
    n_layers=8
    GBS=16
    total_steps=30
    TP_DEGREE=32
    use_flash_attention=1
else
    echo "got unexpected seq len $SEQ_LEN"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export NEURON_CC_FLAGS="--model-type transformer --distribution-strategy=llm-training --cache_dir=$HOME/neuron_compile_cache$SEQ_LEN/ --retry_failed_compilation"
export NEURON_FUSE_SOFTMAX=1

# Async Runtime
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3

# HOST OOM
export MALLOC_ARENA_MAX=64

# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=1
# 0: use pure DP; 1: use ZeRO-1
USE_ZERO_1=1
# micro batch size
MBS=1
# number of steps to run
TOTAL_STEPS=10000
# warmup steps
WARMUP_STEPS=100
# learning rate
LR=3.0e-4
# model path
MODEL_PATH=$SCRIPT_DIR
# Output dir
: ${OUTPUT_DIR="./output"}

#############################################

export NEURON_NUM_DEVICES=32
NODE_ID=0
WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $NEURON_NUM_DEVICES"
if [ ! -z "$SLURM_NTASKS" ]; then
    WORLD_SIZE=$SLURM_NTASKS
    NODE_ID=$SLURM_NODEID
    MASTER_ADDRESS=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
    DISTRIBUTED_ARGS="--nproc_per_node $NEURON_NUM_DEVICES --nnodes $WORLD_SIZE --node_rank $NODE_ID --master_addr $MASTER_ADDRESS --master_port 44000"
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

export NEURON_RT_NUM_CORES=32
export NEURON_NUM_DEVICES=$NEURON_RT_NUM_CORES
export TPU_NUM_DEVICES=$NEURON_RT_NUM_CORES
export TPU_CHIPS_PER_HOST_BOUNDS=$NEURON_RT_NUM_CORES

#############################################

EXTRA_ARGS=" "
if [ $USE_MIX_PRECISION -gt 0 ]; then
    EXTRA_ARGS+=" --use_mix_precision"
fi
if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero_1"
fi

DP=$(($NEURON_RT_NUM_CORES * $WORLD_SIZE / $TP_DEGREE))
ACC_STEPS=$(($GBS / $MBS / $DP))


if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    STEPS_THIS_RUN=2
    OUTPUT_LOG=log_compile-$SEQ_LEN.log
else
    STEPS_THIS_RUN=$total_steps
    OUTPUT_LOG=log_exe-$SEQ_LEN.log
fi

echo TP_DEGREE=$TP_DEGREE
echo USE_MIX_PRECISION=$USE_MIX_PRECISION
echo USE_ZERO_1=$USE_ZERO_1
echo GBS=$GBS
echo MBS=$MBS
echo TOTAL_STEPS=$TOTAL_STEPS
echo WARMUP_STEPS=$WARMUP_STEPS
echo LR=$LR
echo MODEL_PATH=$MODEL_PATH
echo DATA_PATH=$DATA_PATH
echo SEQ_LEN=$SEQ_LEN

echo EXTRA_ARGS=$EXTRA_ARGS
echo DP=$DP
echo ACC_STEPS=$ACC_STEPS
echo STEPS_THIS_RUN=$STEPS_THIS_RUN
echo OUTPUT_LOG=$OUTPUT_LOG

torchrun $DISTRIBUTED_ARGS \
    run_llama_nxd_ptl.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_PATH \
    --tensor_parallel_size $TP_DEGREE \
    --train_batch_size $MBS \
    --steps_this_run $STEPS_THIS_RUN\
    --max_steps $TOTAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --grad_accum_usteps $ACC_STEPS \
    --seq_len $SEQ_LEN \
    --use_sequence_parallel 1 \
    --use_selective_checkpoint 1 \
    --use_fp32_optimizer 1 \
    --use_zero1_optimizer 1 \
    --scheduler_type 'linear' \
    --num_layers $n_layers \
    --use_flash_attention=$use_flash_attention |& tee $OUTPUT_LOG

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
