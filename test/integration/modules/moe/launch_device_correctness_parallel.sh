#!/bin/bash

set -e

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export NXD_DIR=/home/ubuntu/ktest/NeuronxDistributed
export UNIT_TEST_DIR=$NXD_DIR/test/unit_test/modules/moe

export PYTHONPATH=$NXD_DIR/src:$NXD_DIR:$UNIT_TEST_DIR:$PYTHONPATH
echo $PYTHONPATH

# new arg for include inference case, default to training
MODE=${6:-"training"}

if [[ "$MODE" == "training" ]]; then
    # training mode
    MASTER_ADDR_JOB=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
    NPROC_PER_NODE=32
    NNODES=${SLURM_NTASKS}
    NODE_RANK=${SLURM_NODEID}
    TEST_MODE="training"
elif [[ "$MODE" == "inference" ]]; then
    # inference mode
    MASTER_ADDR_JOB=$(hostname -I | awk '{print $1}')
    NPROC_PER_NODE=$1
    NNODES=1
    NODE_RANK=0
    TEST_MODE="inference"
else
    echo "Unknown mode: $MODE. Use 'training' or 'inference'"
    exit 1
fi

# prevents hanging during NCCL init
#sudo sysctl -w net.ipv4.ip_local_reserved_ports=48620

export OMP_NUM_THREADS=1
export FI_LOG_LEVEL=warn
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=${MASTER_ADDR_JOB} \
    --master_port=2020 \
    $SCRIPT_DIR/test_device_correctness_parallel.py \
    --test_tp_degree=$1 \
    --test_ep_degree=$2 \
    --token_shuffle_group_size=$3 \
    --test_mode=$TEST_MODE \
    --test_dtype=$5 \
    --zero1=$4
