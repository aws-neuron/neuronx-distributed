which python
pkill python

export NXD_LOG_LEVEL=DEBUG
# export NXD_CPU_MODE=1
export NXD_CPU_MODE=0

export NEURON_CC_FLAGS="--model-type=transformer --enable-saturate-infinity --retry_failed_compilation --auto-cast=none "

exec_file=test_training_pp.py
torchrun --nnodes=1 --nproc-per-node=32 --master_port=1234 ${exec_file} --pp-size 4 2>&1 | tee ${exec_file}.log