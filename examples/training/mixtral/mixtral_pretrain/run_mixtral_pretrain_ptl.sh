#!/bin/bash

# User defined parameters and env vars
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Neuron Compiler Flags
# TODO: temporarily disable "--distribution-strategy=llm-training" because of a compiler bug. Ideally we should enable it for modular flow
# --enable-saturate-infinity: convert inf to max_float to avoid nan (e.g. in transpose)
export NEURON_CC_FLAGS="--model-type=transformer --retry_failed_compilation --enable-saturate-infinity"
export NEURON_FUSE_SOFTMAX=1

# Async Runtime
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=7

# HOST OOM
export MALLOC_ARENA_MAX=64

# Tensor parallel degree
: ${TP_DEGREE:=32}
# Pipeline parallel degree
: ${PP_DEGREE:=4}
# Expert parallel degree
: ${EP_DEGREE:=1}
# SP
SEQUENCE_PARALLEL_ENABLED=1
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=1
# 0: use pure DP; 1: use ZeRO-1
USE_ZERO_1=1
# global batch size
: ${GBS:=32}
# micro batch size
MBS=8
# Enable selective checkpointing in integration tests.
: ${SELECTIVE_CHECKPOINT_ENABLED:=1}
# number of steps to run
: ${TOTAL_STEPS:=22500}
# warmup steps
: ${WARMUP_STEPS:=4000}
# learning rate
LR=3.0e-4
# Minimum learning rate
MIN_LR=3.0e-5
# model path
MODEL_PATH=$SCRIPT_DIR/configs/8x7b_config.json
# data path
DATA_PATH="$HOME/examples_datasets/wikicorpus_llama2_tokenized_4k"
# sequence length
SEQ_LEN=4096
# capacity factor
CAPACITY_FACTOR=2.0
# Use meta init
META_DEVICE_INIT=1
#############################################

export NUM_NEURONCORES=32
NODE_ID=0
WORLD_SIZE=16
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

export NEURON_RT_NUM_CORES=$NUM_NEURONCORES
export TPU_NUM_DEVICES=$NEURON_RT_NUM_CORES
export TPU_CHIPS_PER_HOST_BOUNDS=$NEURON_RT_NUM_CORES

#############################################

EXTRA_ARGS=" "
if [ $META_DEVICE_INIT -gt 0 ]; then
    EXTRA_ARGS+=" --use_meta_device_init"
fi
if [ $USE_MIX_PRECISION -gt 0 ]; then
    EXTRA_ARGS+=" --use_mix_precision"
fi
if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero_1"
fi
if [ $SEQUENCE_PARALLEL_ENABLED -eq 1 ]; then
    EXTRA_ARGS+=" --sequence_parallel_enabled"
fi
if [ $SELECTIVE_CHECKPOINT_ENABLED -eq 1 ]; then
    EXTRA_ARGS+=" --selective_checkpoint_enabled"
fi

if [ $PP_DEGREE -gt 1 ]; then
    # Data paralell size
    DP=$(($NUM_NEURONCORES * $WORLD_SIZE / $TP_DEGREE / $PP_DEGREE))
    # Batch size per model replica
    BS=$(($GBS / $DP))
    # Number microbatches for pipeline execution
    # Setting same as BS so each microbatch contains a single datasample
    NUM_MICROBATCHES=$BS
    echo NUM_MICROBATCHES=$NUM_MICROBATCHES
    EXTRA_ARGS+=" --num_microbatches $NUM_MICROBATCHES"
else
    DP=$(($NEURON_RT_NUM_CORES * $WORLD_SIZE / $TP_DEGREE))
    ACC_STEPS=$(($GBS / $MBS / $DP))
    echo ACC_STEPS=$ACC_STEPS
    EXTRA_ARGS+=" --grad_accum_usteps $ACC_STEPS"
fi


if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    STEPS_THIS_RUN=2
    OUTPUT_LOG=log_compile-$NODE_ID.log
else
    STEPS_THIS_RUN=-1
    OUTPUT_LOG=log_exe-$NODE_ID.log
fi

echo TP_DEGREE=$TP_DEGREE
echo EP_DEGREE=$EP_DEGREE
echo SEQUENCE_PARALLEL_ENABLED=$SEQUENCE_PARALLEL_ENABLED
echo PP_DEGREE=$PP_DEGREE
echo USE_MIX_PRECISION=$USE_MIX_PRECISION
echo USE_ZERO_1=$USE_ZERO_1
echo GBS=$GBS
echo MBS=$MBS
echo TOTAL_STEPS=$TOTAL_STEPS
echo WARMUP_STEPS=$WARMUP_STEPS
echo LR=$LR
echo MIN_LR=$MIN_LR
echo MODEL_PATH=$MODEL_PATH
echo DATA_PATH=$DATA_PATH
echo SEQ_LEN=$SEQ_LEN
echo CAPACITY_FACTOR=$CAPACITY_FACTOR
echo EXTRA_ARGS=$EXTRA_ARGS
echo DP=$DP
echo STEPS_THIS_RUN=$STEPS_THIS_RUN
echo OUTPUT_LOG=$OUTPUT_LOG

torchrun $DISTRIBUTED_ARGS \
    run_mixtral_pretrain_ptl.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_PATH \
    --tensor_parallel_size $TP_DEGREE \
    --pipeline_parallel_size $PP_DEGREE \
    --expert_parallel_size $EP_DEGREE \
    --batch_size $MBS \
    --steps_this_run $STEPS_THIS_RUN\
    --max_steps $TOTAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --min_lr $MIN_LR \
    --seq_len $SEQ_LEN \
    --capacity_factor $CAPACITY_FACTOR \
    $EXTRA_ARGS |& tee $OUTPUT_LOG
