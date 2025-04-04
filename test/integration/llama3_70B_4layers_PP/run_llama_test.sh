#!/bin/bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000

MODEL_SIZE="70B"
LLAMA_VERSION='3'

## Added this
#export NEURON_FRAMEWORK_DEBUG=1

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=7
export MALLOC_ARENA_MAX=128
export XLA_DOWNCAST_BF16=1
export NEURON_CC_FLAGS="--model-type=transformer --cache_dir=$HOME/cache_dir_neuron/"

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
    DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
else
    DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE"
    LOG_PATH=logs
fi
mkdir -p $LOG_PATH
echo "Nodeinfo NODEID $NODEID hostname $HOSTNAME"
echo $DISTRIBUTED_ARGS

# Global batch size
: ${GBS:=4}
# Input sequence length
SEQ_LEN=8192
# Pipeline parallel degree
PP_DEGREE=4
# Tensor parallel degree
TP_DEGREE=8
: ${NUM_LAYERS=4}
# Data paralell size
DP=$(($PROCESSES_PER_NODE * $WORLD_SIZE / $TP_DEGREE / $PP_DEGREE))
# Batch size per model replica
BS=$(($GBS / $DP))
# Number microbatches for pipeline execution
# Setting same as BS so each microbatch contains a single datasample
NUM_MICROBATCHES=$BS
DATA_PATH="$HOME/examples_datasets/wikicorpus_llama3_tokenized_8k"


if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]; then
    max_steps=10
    checkpoint_freq=-1
else
    max_steps=50
    checkpoint_freq=30000
fi

EXTRA_ARGS=" "
if [ $META_DEVICE_INIT -gt 0 ]; then
    EXTRA_ARGS+=" --use_meta_device_init 1"
fi

torchrun $DISTRIBUTED_ARGS run_llama_nxd.py \
    --train_batch_size $BS \
    --use_meta_device_init 1 \
    --num_layer $NUM_LAYERS \
    --training_dir $DATA_PATH \
    --training_config $SCRIPT_DIR/${MODEL_SIZE}_config_llama${LLAMA_VERSION} \
    --max_steps $max_steps \
    --seq_len $SEQ_LEN \
    --pipeline_parallel_size $PP_DEGREE \
    --tensor_parallel_size $TP_DEGREE \
    --num_microbatches $NUM_MICROBATCHES \
    --lr 0.000015 \
    --min_lr 1e-06 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.1 \
    --warmup_steps 2000 \
    --constant_steps 0 \
    --use_zero1_optimizer 1 \
    --use_selective_checkpoint 1 \
    --qkv_linear 1 \
    --kv_replicator 4 \
    --pretrained_weight 0 \
    --save_load_xser 1 \
    --num_kept_checkpoint -1 \
    --use_flash_attention 1 \
    --loading_step -1 \
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

exit $ret_val