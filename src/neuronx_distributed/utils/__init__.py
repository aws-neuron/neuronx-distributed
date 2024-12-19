import os
import torch
import torch_xla.core.xla_model as xm

CPU_MODE = int(os.environ.get("NXD_CPU_MODE", "0")) > 0


def cpu_mode():
    return CPU_MODE


def mark_step():
    if not cpu_mode():
        xm.mark_step()


def master_print(msg):
    if cpu_mode() and torch.distributed.get_rank() == 0:
        print(msg)
    else:
        xm.master_print(msg)

def get_device():
    if cpu_mode():
        return torch.device("cpu")
    else:
        return xm.xla_device()