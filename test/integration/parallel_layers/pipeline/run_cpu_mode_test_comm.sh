
export NXD_CPU_MODE=1
export NXD_LOG_LEVEL=DEBUG

exec_file="test_comm.py"

torchrun --nnodes=1 --nproc-per-node=8 ${exec_file} 2>&1 | tee ${exec_file}.log