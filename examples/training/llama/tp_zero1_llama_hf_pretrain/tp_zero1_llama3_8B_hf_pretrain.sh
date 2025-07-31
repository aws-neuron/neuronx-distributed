#!/bin/bash

#############################################
# User defined parameters and env vars

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export NEURON_CC_FLAGS="--model-type transformer --cache_dir=$HOME/neuron_compile_cache/"
export NEURON_FUSE_SOFTMAX=1

# Async Runtime
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3

# HOST OOM
export MALLOC_ARENA_MAX=64

MODEL_SIZE="8B"
LLAMA_VERSION='3'
: ${LLAMA_CONFIG_VERSION:=3}

# TP degree
: ${TP_DEGREE:=32}
# CP degree
: ${CP_DEGREE:=1}
# Flash attention
: ${USE_FLASH_ATTENTION:=1}
# Num layers
: ${NUM_LAYERS:=32}
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=1
# 0: use pure DP; 1: use ZeRO-1
USE_ZERO_1=1
# global batch size
: ${GBS:=1024}
# micro batch size
MBS=1
# number of steps to run
TOTAL_STEPS=10000
# warmup steps
WARMUP_STEPS=100
# learning rate
LR=1.5e-4
# model path
MODEL_PATH=$SCRIPT_DIR/${MODEL_SIZE}_config_llama${LLAMA_CONFIG_VERSION}
# data path
DATA_PATH="$HOME/examples_datasets/wikicorpus_llama3_tokenized_8k" 
# sequence length
SEQ_LEN=8192

#############################################

export NUM_NEURONCORES=32
NODE_ID=0
WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"
METRICS_FILE="results.json"

if [ ! -z "$SLURM_NTASKS" ]; then
    WORLD_SIZE=$SLURM_NTASKS
    NODE_ID=$SLURM_NODEID
    MASTER_ADDRESS=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
    DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $WORLD_SIZE --node_rank $NODE_ID --master_addr $MASTER_ADDRESS --master_port 44000"
    if [ $NODE_ID -eq 0 ]; then
        echo "WORLD_SLURM_NTASKS=$WORLD_SIZE"
        echo "NODE_ID=$NODE_ID"
        echo "MASTER_ADDRESS=$MASTER_ADDRESS"
        echo "DISTRIBUTED_ARGS=$DISTRIBUTED_ARGS"
    fi
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
    sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620

    if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    STEPS_THIS_RUN=2
    OUTPUT_LOG=log_compile-$NODE_ID.log
    elif [ -v PERF_TEST ] && [ $PERF_TEST -gt 0 ]; then
        STEPS_THIS_RUN=${EARLY_EXIT_STEPS:-100}
        OUTPUT_LOG=log_exe-$NODE_ID.log
    else
        STEPS_THIS_RUN=-1
        OUTPUT_LOG=log_exe-$NODE_ID.log
    fi
elif [ -v OMPI_COMM_WORLD_RANK ]; then
    # Increase the fd limit for container
    ulimit -n 65535
    WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
    NODE_ID=$OMPI_COMM_WORLD_RANK
    NODELIST=$(/root/nodelist_helper.py)
    HOSTS=(${NODELIST//\ / })
    MASTER_ADDRESS=${HOSTS[0]}
    DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $WORLD_SIZE --node_rank $NODE_ID --master_addr $MASTER_ADDRESS --master_port 44000"
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
    export CCOM_SOCKET_IFNAME=eth0
    export FI_EFA_FORK_SAFE=1

    # Dataset is in shared location
    DATA_PATH="$SHARED_PATH_PREFIX/mars_data_set/examples_datasets/wikicorpus_llama3_tokenized_8k" 

    # Store metrics in shared location
    METRICS_FILE=$ARTIFACT_PATH/results.json
    mkdir -p $ARTIFACT_PATH

    JOB_ID=$POD_UID
    export EXPLICIT_LOGDIR=null
    LOG_PATH="$ARTIFACT_PATH/logs/$JOB_ID/$NODE_ID"
    mkdir -p $LOG_PATH

    if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
        STEPS_THIS_RUN=2
        OUTPUT_LOG="$LOG_PATH/log_compile-$NODE_ID.log"
    elif [ -v PERF_TEST ] && [ $PERF_TEST -gt 0 ]; then
        STEPS_THIS_RUN=100
        OUTPUT_LOG="$LOG_PATH/log_exe-$NODE_ID.log"
    else
        STEPS_THIS_RUN=-1
        OUTPUT_LOG="$LOG_PATH/log_exe-$NODE_ID.log"
    fi
fi

echo "WORLD_SLURM_NTASKS=$WORLD_SIZE"
echo "NODE_ID=$NODE_ID"
echo "MASTER_ADDRESS=$MASTER_ADDRESS"

export NEURON_RT_NUM_CORES=32
export NUM_NEURONCORES=$NEURON_RT_NUM_CORES
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

echo EARLY_EXIT_STEPS=$EARLY_EXIT_STEPS
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
    tp_zero1_llama_hf_pretrain.py \
    --metrics_file $METRICS_FILE \
    --model_path $MODEL_PATH \
    --data_dir $DATA_PATH \
    --tensor_parallel_size $TP_DEGREE \
    --context_parallel_size $CP_DEGREE \
    --batch_size $MBS \
    --steps_this_run $STEPS_THIS_RUN\
    --max_steps $TOTAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --grad_accum_usteps $ACC_STEPS \
    --seq_len $SEQ_LEN \
    --sequence_parallel_enabled \
    --selective_checkpoint_enabled \
    --logging_interval 1 \
    --qkv_linear \
    --kv_replicator 4 \
    --use_flash_attention $USE_FLASH_ATTENTION \
    --use_gpu_compatible_precision 1 \
    --num_layers $NUM_LAYERS \
    $EXTRA_ARGS |& tee $OUTPUT_LOG
exit ${PIPESTATUS[0]}
