export NXD_CPU_MODE=1
export NXD_LOG_LEVEL=DEBUG

torchrun --nnodes=1 --nproc-per-node=8 test_layers.py 2>&1 | tee test_layers.log