#!/bin/bash
set -ex

sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_PATH="$HOME/wikicorpus_datasets/wikicorpus_llama_v2_tokenized_4k"

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export NEURON_RT_NUM_CORES=32

export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
export NEURON_TRANSFER_WITH_STATIC_RING_OPS=""
export MALLOC_ARENA_MAX=128
export XLA_DOWNCAST_BF16=1
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export NEURON_CC_FLAGS="--model-type=transformer --enable-saturate-infinity"

PROCESSES_PER_NODE=32
WORLD_SIZE=1
NODEID=0
HOSTNAME=`hostname`
if [ -v SLURM_NTASKS ]; then
    # SLURM runs
    IPS=""
    for h in $(scontrol show hostname); do
        IPS="$IPS $(nslookup $h  | awk '/^Address: / { print $2 }')";
    done
    HOSTS=(${IPS//\ / })
    NODEID=$SLURM_NODEID
    NTASKS=$SLURM_NTASKS
    WORLD_SIZE=$SLURM_NTASKS
    JOB_ID=$SLURM_JOB_ID
    export NEMO_EXPM_VERSION=$SLURM_JOB_ID
    export EXPLICIT_LOGDIR=null
    LOG_PATH=logs/$SLURM_JOB_ID/$NODEID

    MASTER_ADDR=${HOSTS[0]}
    MASTER_PORT=44000
    NUM_NEURONCORES=$NEURON_RT_NUM_CORES
    DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
else
    DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE"
    LOG_PATH=logs
fi
mkdir -p $LOG_PATH
echo "Nodeinfo NODEID $NODEID hostname $HOSTNAME"
echo $DISTRIBUTED_ARGS

: ${GBS=4}
: ${SEQ_LEN=4096}
: ${PP_DEGREE=4}
TP_DEGREE=8
: ${NUM_LAYERS=4}
HIDDEN_SIZE=2048
DP=$(($NEURON_RT_NUM_CORES * $WORLD_SIZE / $TP_DEGREE / $PP_DEGREE))
BS=$(($GBS / $DP))


if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]; then
    max_steps=10
    compare_loss=0
else
    max_steps=50
    compare_loss=1
fi

EXTRA_ARGS=" "
if [ $META_DEVICE_INIT -gt 0 ]; then
    EXTRA_ARGS+=" --use_meta_device_init 1"
fi

if [[ "${ENABLE_CHECKPOINTING}" == "1" ]]; then
    echo "Checkpointing is enabled."

    if [ $XSER -gt 0 ]; then
        EXTRA_ARGS+=" --save_load_xser 1"
    else
        EXTRA_ARGS+=" --save_load_xser 0"
    fi

    if [ $ASYNC_CHECKPOINT_SAVING -gt 0 ]; then
        EXTRA_ARGS+=" --async_checkpoint_saving 1"
    fi
    # Check for CHECKPOINT_DIR
    if [[ -n "${CHECKPOINT_DIR}" ]]; then
        checkpoint_dir="${CHECKPOINT_DIR}"
    else
        checkpoint_dir="ckpt"
    fi
    echo "Checkpoint directory set to: ${checkpoint_dir}"
    EXTRA_ARGS+=" --checkpoint_dir $checkpoint_dir"

    # Checkpoint frequency
    if [[ -n "${CHECKPOINT_FREQ}" ]]; then
        if [[ "${CHECKPOINT_FREQ}" =~ ^[0-9]+$ ]]; then
            checkpoint_freq="${CHECKPOINT_FREQ}"
            echo "Checkpoint freq set to: ${CHECKPOINT_FREQ}"
        else
            echo "Error: CHECKPOINT_FREQ must be an integer."
            exit 1
        fi
    else
        echo "Setting CHECKPOINT_FREQ to be 10"
        checkpoint_freq="10"
    fi
    EXTRA_ARGS+=" --checkpoint_freq $checkpoint_freq"

    # Check for LOAD_CHECKPOINT_STEP
    if [[ -n "${LOAD_CHECKPOINT_STEP}" ]]; then
        if [[ "${LOAD_CHECKPOINT_STEP}" =~ ^[0-9]+$ ]]; then
            load_checkpoint_step="${LOAD_CHECKPOINT_STEP}"
            echo "Load checkpoint step set to: ${load_checkpoint_step}"
            EXTRA_ARGS+=" --loading_step $load_checkpoint_step" 
            LOG_PATH=$LOG_PATH/load
        else
            echo "Error: LOAD_CHECKPOINT_STEP must be an integer."
            exit 1
        fi
    else
        echo "LOAD_CHECKPOINT_STEP is unset. Training from scratch"
        LOG_PATH=$LOG_PATH/save
    fi
    
    EXTRA_ARGS+=" --num_kept_checkpoint 10"
else
    echo "Checkpointing is disabled or not configured."
fi

if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero1_optimizer 1"
    if [ $USE_MASTER_WEIGHT_IN_CKPT -gt 0 ]; then
        EXTRA_ARGS+=" --use_master_weight_in_ckpt 1"
        if [ $AVOID_SAVING_LOWER_PRECISION_WEIGHTS -gt 0 ]; then
            EXTRA_ARGS+=" --avoid_saving_lower_precision_weights 1"
        fi
    fi
fi
mkdir -p $LOG_PATH

torchrun $DISTRIBUTED_ARGS run_llama_nxd.py \
    --train_batch_size $BS \
    --training_dir $DATA_PATH \
    --training_config $SCRIPT_DIR/70B_config_llama2 \
    --max_steps $max_steps \
    --num_layer $NUM_LAYERS \
    --hidden_size $HIDDEN_SIZE \
    --seq_len $SEQ_LEN \
    --pipeline_parallel_size $PP_DEGREE \
    --tensor_parallel_size $TP_DEGREE \
    --num_microbatches $BS \
    --lr 0.00015 \
    --min_lr 1e-05 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --constant_steps 0 \
    --use_gpu_compatible_precision 0 \
    $EXTRA_ARGS |& tee $LOG_PATH/log

ret_val=${PIPESTATUS[0]}

if [ -v PERF_TEST ];
then
    echo "Performance test complete"
else
  if [ $ret_val -eq 0 ]; then
      success=1
  else
      success=0
  fi
fi

exit $ret_val
