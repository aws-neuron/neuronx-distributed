#!/bin/bash

# This shell script executes a program with an optional -m option.
#
# Usage:
#   ./run_mixtral_pretrain_ptl.sh [-m <model_config_path> -d <data_path> -s <steps_this_run>]
#
# Options:
#   -m <model_config_path>   Specify the path to the model configuration file.
#                            If this option is not provided, the script will
#                            use the default model configuration path for the
#                            8x7b configuration model.
#   -d <data_path>           Specify the path to the data directory.
#   -s <steps_this_run>      Specify the number of steps to run.
#
# Example:
#   ./run_mixtral_pretrain_ptl.sh -m /path/to/model/config.json
#
# If no -m option is provided, the script will default to:
#   /path/to/default/8x7b/config.json
#
# Ensure that the script has execute permissions. You can set the execute
# permission using the following command:
#   chmod +x run_mixtral_pretrain_ptl.sh
#

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
# Total number of nodes
: ${WORLD_SIZE:=16}
if [ ! -z "$SLURM_NTASKS" ]; then
    WORLD_SIZE=$SLURM_NTASKS
fi
export NUM_NEURONCORES=32
if [ $(( (NUM_NEURONCORES * WORLD_SIZE) % (TP_DEGREE * PP_DEGREE) )) -ne 0 ]; then
    echo "NUM_NEURONCORES [$NUM_NEURONCORES] * WORLD_SIZE [$WORLD_SIZE] must be divisible by TP_DEGREE [$TP_DEGREE] * PP_DEGREE [$PP_DEGREE]"
    exit 1
fi
# Data parallel degree
DP_DEGREE=$(( (NUM_NEURONCORES * WORLD_SIZE) / (TP_DEGREE * PP_DEGREE) ))

# SP
SEQUENCE_PARALLEL_ENABLED=1
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=1
# 0: use pure DP; 1: use ZeRO-1
USE_ZERO_1=1
# Check if DP_DEGREE is 1 and modify USE_ZERO_1 accordingly
if [ "$DP_DEGREE" -eq 1 ]; then
    echo "WARNING: DP_DEGREE is 1, setting USE_ZERO_1 to 0"
    USE_ZERO_1=0
fi

# global batch size
: ${GBS:=32}
# Micro batch size
: ${MICRO_BS:=1}
if [ $(( GBS % (DP_DEGREE * MICRO_BS) )) -ne 0 ]; then
    echo "GBS [$GBS] must be divisible by DP_DEGREE [$DP_DEGREE] * MICRO_BS [$MICRO_BS]"
    exit 1
fi
# Number of samples in a gradient accumulation step for a DP rank.
MINI_BS=$(( GBS / DP_DEGREE ))
# Number of micro batches for a gradient accumulation step.
NUM_MICROBATCHES=$(( MINI_BS / MICRO_BS ))
# Enable selective checkpointing in integration tests.
: ${SELECTIVE_CHECKPOINT_ENABLED:=0}
# number of steps to run
: ${TOTAL_STEPS:=22500}
# warmup steps
: ${WARMUP_STEPS:=4000}
# learning rate
LR=3.0e-4
# Minimum learning rate
MIN_LR=3.0e-5
# model path; takes in -m option argument to parse the model path
MODEL_PATH_ARG=""
# data path; takes in -d option argument to parse the data path
DATA_PATH="$HOME/examples_datasets/wikicorpus_llama2_tokenized_4k"
#steps this run; takes in -s option argument for early exit 
STEPS_THIS_RUN=-1
while getopts "m:d:s:" opt; do
  case $opt in
    m) MODEL_PATH_ARG="$OPTARG"
    ;;
    d) DATA_PATH="$OPTARG"
    ;;
    s) STEPS_THIS_RUN="$OPTARG"
    ;;
    \?) echo "Invalid option: -$OPTARG" >&2
    ;;
  esac
done



# sequence length
: ${SEQ_LEN:=4096}
# capacity factor
CAPACITY_FACTOR=2.0
# Use meta init
META_DEVICE_INIT=1
#############################################
NODE_ID=0
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"
if [ ! -z "$SLURM_NTASKS" ]; then
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
export NEURON_RT_DISABLE_EXECUTION_BARRIER=0
echo "NEURON_RT_DISABLE_EXECUTION_BARRIER=$NEURON_RT_DISABLE_EXECUTION_BARRIER"
#
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
else
    EXTRA_ARGS+=" --use_gpu_compatible_precision 0"
fi
if [ $SEQUENCE_PARALLEL_ENABLED -eq 1 ]; then
    EXTRA_ARGS+=" --sequence_parallel_enabled"
fi
if [ $SELECTIVE_CHECKPOINT_ENABLED -eq 1 ]; then
    EXTRA_ARGS+=" --selective_checkpoint_enabled"
fi

if [ $PP_DEGREE -gt 1 ]; then
    # Number microbatches for pipeline execution
    echo NUM_MICROBATCHES=$NUM_MICROBATCHES
    EXTRA_ARGS+=" --num_microbatches $NUM_MICROBATCHES"
    # PP>1 uses minibatch size as the dataloader batch size.
    EXTRA_ARGS+=" --batch_size $MINI_BS"
else
    ACC_STEPS=$NUM_MICROBATCHES
    echo ACC_STEPS=$ACC_STEPS
    EXTRA_ARGS+=" --grad_accum_usteps $ACC_STEPS"
    # PP=1 uses microbatch size as the dataloader batch size.
    EXTRA_ARGS+=" --batch_size $MICRO_BS"
fi


if [ -n $NEURON_EXTRACT_GRAPHS_ONLY ] && [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    STEPS_THIS_RUN=2
    OUTPUT_LOG=log_compile-$NODE_ID.log
else
    OUTPUT_LOG=log_exe-$NODE_ID.log
fi

DEFAULT_MODEL_PATH="$SCRIPT_DIR/configs/8x7b_config.json"

#sets model_path to 7b config if not specified
if [ -z "$MODEL_PATH_ARG" ]; then
    MODEL_PATH="$DEFAULT_MODEL_PATH"
else
  # If argument is passed, use it relative to SCRIPT_DIR
  if [[ "$MODEL_PATH_ARG" != /* ]]; then
    MODEL_PATH="$SCRIPT_DIR/configs/$MODEL_PATH_ARG"
  else
    MODEL_PATH="$MODEL_PATH_ARG"
  fi
fi

echo TP_DEGREE=$TP_DEGREE
echo EP_DEGREE=$EP_DEGREE
echo SEQUENCE_PARALLEL_ENABLED=$SEQUENCE_PARALLEL_ENABLED
echo PP_DEGREE=$PP_DEGREE
echo USE_MIX_PRECISION=$USE_MIX_PRECISION
echo USE_ZERO_1=$USE_ZERO_1
echo GBS=$GBS
echo MICRO_BS=$MICRO_BS
echo MINI_BS=$MINI_BS
echo TOTAL_STEPS=$TOTAL_STEPS
echo WARMUP_STEPS=$WARMUP_STEPS
echo LR=$LR
echo MIN_LR=$MIN_LR
echo MODEL_PATH=$MODEL_PATH
echo DATA_PATH=$DATA_PATH
echo SEQ_LEN=$SEQ_LEN
echo CAPACITY_FACTOR=$CAPACITY_FACTOR
echo EXTRA_ARGS=$EXTRA_ARGS
echo DP=$DP_DEGREE
echo STEPS_THIS_RUN=$STEPS_THIS_RUN
echo OUTPUT_LOG=$OUTPUT_LOG

torchrun $DISTRIBUTED_ARGS \
    run_mixtral_pretrain_ptl.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_PATH \
    --tensor_parallel_size $TP_DEGREE \
    --pipeline_parallel_size $PP_DEGREE \
    --expert_parallel_size $EP_DEGREE \
    --steps_this_run $STEPS_THIS_RUN\
    --max_steps $TOTAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --min_lr $MIN_LR \
    --seq_len $SEQ_LEN \
    --capacity_factor $CAPACITY_FACTOR \
    $EXTRA_ARGS |& tee $OUTPUT_LOG
