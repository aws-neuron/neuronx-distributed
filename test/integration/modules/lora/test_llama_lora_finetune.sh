#!/bin/bash

#############################################
# Override transformers and Optimum-Neuron packages, can be removed once ON released changes in https://github.com/huggingface/optimum-neuron/pull/370
pip install git+https://github.com/huggingface/optimum-neuron.git 
pip install -U transformers==4.48.0 # reinstall transformers due to optimum neuron override
pip install --no-warn-conflicts nltk

#############################################
# User defined parameters and env vars

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export NEURON_CC_FLAGS="--model-type=transformer -O1 "
export NEURON_FUSE_SOFTMAX=1

# Async Runtime
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3

# HOST OOM
export MALLOC_ARENA_MAX=64

# TP degree
TP_DEGREE=32
# PP degree
PP_DEGREE=1
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=0
# 0: use pure DP; 1: use ZeRO-1
USE_ZERO_1=0
# global batch size
GBS=1
# micro batch size
MBS=1
# number of steps to run
TOTAL_STEPS=20
# number of epochs to run
TOTAL_EPOCHS=1
# warmup steps
WARMUP_STEPS=5
# learning rate
LR=5.0e-4
# model path
MODEL_PATH=$SCRIPT_DIR
# pretrained weight path
PRETRAINED_PATH=/dev/shm/llama3_model
# base model name
BASE_MODEL='meta-llama/Meta-Llama-3-8B'
# HF Token
HF_TOKEN=''
# sequence length
SEQ_LEN=4096

#############################################
PROCESSES_PER_NODE=32
export NUM_NEURONCORES=${PROCESSES_PER_NODE}
NODE_ID=0
WORLD_SIZE=$TP_DEGREE
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

export NEURON_RT_NUM_CORES=${PROCESSES_PER_NODE}
export NUM_NEURONCORES=$NEURON_RT_NUM_CORES
export TPU_NUM_DEVICES=$NEURON_RT_NUM_CORES
export TPU_CHIPS_PER_HOST_BOUNDS=$NEURON_RT_NUM_CORES
export NEURON_RT_ROOT_COMM_ID=localhost:48620

#############################################

EXTRA_ARGS=" "
if [ $USE_MIX_PRECISION -gt 0 ]; then
    EXTRA_ARGS+=" --use_mix_precision"
fi
if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero_1"
fi

DP=$(($NEURON_RT_NUM_CORES / $TP_DEGREE / $PP_DEGREE))
ACC_STEPS=$(($GBS / $MBS / $DP))

if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    STEPS_THIS_RUN=10
    OUTPUT_LOG=log_compile-$NODE_ID.log
else
    STEPS_THIS_RUN=$TOTAL_STEPS
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

export XLA_USE_BF16=1

torchrun $DISTRIBUTED_ARGS \
    tp_llama_hf_finetune_ptl.py \
    --model_path $MODEL_PATH \
    --model_name $BASE_MODEL \
    --hf_token $HF_TOKEN \
    --data_dir "databricks/databricks-dolly-15k" \
    --tensor_parallel_size $TP_DEGREE \
    --batch_size $MBS \
    --steps_this_run $STEPS_THIS_RUN \
    --max_steps $TOTAL_STEPS \
    --num_train_epochs $TOTAL_EPOCHS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --selective_checkpoint_enabled \
    --grad_accum_usteps $ACC_STEPS \
    --seq_len $SEQ_LEN \
    --pretrained_ckpt $PRETRAINED_PATH \
    --sequence_parallel_enabled \
    --task "open_qa" \
    --fuse_qkv 0 \
    $EXTRA_ARGS \
    --use_gpu_compatible_precision 0 \
    --enable_lora \
    --qkv_linear 1 \
    --kv_replicator 4 \

ret_val=${PIPESTATUS[0]}
echo ret_val=$ret_val
