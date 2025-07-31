#!/bin/bash

set -e

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export NXD_DIR=$(dirname $(dirname $(dirname $(dirname $SCRIPT_DIR))))
export UNIT_TEST_DIR=$NXD_DIR/test/unit_test/modules/moe

export PYTHONPATH=$NXD_DIR/src:$NXD_DIR:$UNIT_TEST_DIR:$PYTHONPATH
echo $PYTHONPATH

MASTER_ADDR_JOB=$(hostname -I | awk '{print $1}')
NPROC_PER_NODE=$1
NNODES=1
NODE_RANK=0

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
    $SCRIPT_DIR/test_module_performance.py \
    --test_tp_degree=$1 \
    --test_dtype=$2
