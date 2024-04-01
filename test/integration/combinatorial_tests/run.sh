#!/bin/bash

## Parse inputs from file
experiment_file=$1
echo "experiment_file: $experiment_file"

# Read variable-value pairs from the file
while IFS='=' read -r variable value; do
  declare "$variable=$value"
done < $experiment_file

# Show the values read from the config file
echo "TP_DEGREE: $TP_DEGREE"
echo "NEURON_FLAGS: $NEURON_FLAGS"
echo "GBS: $GBS"
echo "MBS: $MBS"
echo "SEQUENCE_PARALLEL: $SEQUENCE_PARALLEL"
echo "SEQ_LEN: $SEQ_LEN"
echo "PIPELINE_PARALLEL: $PIPELINE_PARALLEL" # TODO - enable pp when running model
echo "USE_ZERO_1= $USE_ZERO_1" # 0: use pure DP; 1: use ZeRO-1
echo "USE_MIX_PRECISION=$USE_MIX_PRECISION" # 0: bf16; 1: mixed precision
echo "SELECTIVE_CHECKPOINT: $SELECTIVE_CHECKPOINT"

#############################################
# User defined parameters and env vars

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export NEURON_CC_FLAGS=$NEURON_FLAGS
export NEURON_FUSE_SOFTMAX=1

# Async Runtime
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3

# HOST OOM
export MALLOC_ARENA_MAX=64

# number of steps to run
TOTAL_STEPS=10000
# warmup steps
WARMUP_STEPS=10
# learning rate
LR=3.0e-4
# model path
MODEL_PATH=$SCRIPT_DIR
# data path
DATA_PATH="$HOME/wikicorpus_datasets/wikicorpus_llama_v2_tokenized_4k"

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
export NEURON_RT_ROOT_COMM_ID=localhost:48620

#############################################

EXTRA_ARGS=" "
if [ $USE_MIX_PRECISION -gt 0 ]; then
    EXTRA_ARGS+=" --use_mix_precision"
fi
if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero_1"
fi
if [ $SEQUENCE_PARALLEL -gt 0 ]; then
    EXTRA_ARGS+=" --sequence_parallel_enabled"
fi
if [ $SELECTIVE_CHECKPOINT -gt 0 ]; then
  EXTRA_ARGS+=" --selective_checkpoint_enabled"
fi

DP=$(($NEURON_RT_NUM_CORES * $WORLD_SIZE / $TP_DEGREE))
#DP=$(($NEURON_RT_NUM_CORES * $WORLD_SIZE / $TP_DEGREE / $PP_DEGREE)) #TODO - use once PP is enabled
ACC_STEPS=$(($GBS / $MBS / $DP))

if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    STEPS_THIS_RUN=2
    OUTPUT_LOG=log_compile-$NODE_ID.log
else
    #STEPS_THIS_RUN=5
    STEPS_THIS_RUN=50
    OUTPUT_LOG=log_exe-$NODE_ID.log
fi

echo USE_MIX_PRECISION=$USE_MIX_PRECISION
echo USE_ZERO_1=$USE_ZERO_1
echo TOTAL_STEPS=$TOTAL_STEPS
echo WARMUP_STEPS=$WARMUP_STEPS
echo LR=$LR
echo MODEL_PATH=$MODEL_PATH
echo DATA_PATH=$DATA_PATH

echo EXTRA_ARGS=$EXTRA_ARGS
echo DP=$DP
echo ACC_STEPS=$ACC_STEPS
echo STEPS_THIS_RUN=$STEPS_THIS_RUN
echo OUTPUT_LOG=$OUTPUT_LOG

torchrun $DISTRIBUTED_ARGS \
    $HOME/ktest/NeuronxDistributed/examples/training/llama2/tp_zero1_llama2_7b_hf_pretrain/tp_zero1_llama2_7b_hf_pretrain.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_PATH \
    --tensor_parallel_size $TP_DEGREE \
    --batch_size $MBS \
    --steps_this_run $STEPS_THIS_RUN\
    --max_steps $TOTAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --grad_accum_usteps $ACC_STEPS \
    --seq_len $SEQ_LEN \
    $EXTRA_ARGS |& tee $OUTPUT_LOG


# TODO update smoothed_weight factor and delta percentage based on experiments
if [ "$NEURON_EXTRACT_GRAPHS_ONLY" != "1" ]; then
    echo "run pretraining gpu comparison"
    python3 ./common/compare_gpu_trn1_metrics.py ~/gpu_benchmark/events.out.tfevents.1691458200.platform-queue-dy-platform-p4d24xlarge-5.7774.0 output/neuron_tblogs_*/events.out.* "step loss" --smoothed_weight=0.666 --delta_percentage=5.0 --comparison_start_step=0
    ret_val=$?
    echo $ret_val
    if [ $ret_val -eq 0 ]; then
        success=1
    else
        success=0
    fi
    dump_to_s3_update_json_scr=$SCRIPT_DIR/../../../../dump_to_s3_update_test_json.sh
    if [ -e $dump_to_s3_update_json_scr ]; then
        $dump_to_s3_update_json_scr $@ --key=inference_success --value=$success || echo "Unable to update test result JSON."
    else
        echo "WARNING: Script $dump_to_s3_update_json_scr not found. Not updating test result JSON."
    fi

    exit $ret_val
fi
