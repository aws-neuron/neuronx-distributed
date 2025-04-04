#!/bin/bash
set -x

# SOTA Config


export ENABLE_CHECKPOINTING=1
if [ -z "$NEURON_EXTRACT_GRAPHS_ONLY" ]; then
  export CHECKPOINT_FREQ="10"
  source ./run_llama_test.sh 
  if [ $? -ne 0 ]; then
    exit 1
  fi

  ### Performance/Regression Test: Mean Checkpointing Time
  if [ -v CKPT_PERF_TEST ]; then
    MEAN_CKPT_TIME_THRESHOLD="0.5"

    python3 analyze_ckpt_time.py $LOG_PATH/log $MEAN_CKPT_TIME_THRESHOLD
    if [ $? -ne 0 ]; then
      exit 1
    fi
  fi

  export LOAD_CHECKPOINT_STEP=20  # load
  source ./run_llama_test.sh
  if [ $? -ne 0 ]; then
    exit 1
  fi
  if [ $? -eq 0 ]; then
    success=1
  else
    success=0
  fi
  echo "success=$success"
else
  export CHECKPOINT_FREQ=5
  source ./run_llama_test.sh
  if [ $? -ne 0 ]; then
    exit 1
  fi
fi
