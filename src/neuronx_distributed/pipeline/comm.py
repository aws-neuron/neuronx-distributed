import pickle

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from ..parallel_layers import parallel_state
from ..parallel_layers.parallel_state import get_gloo_group, rmsg
from ..utils.logger import get_logger

logger = get_logger()

# max size of python object buffer to send between stages
MAX_LENGTH = 2**20  # 1M


"""
Isolate the CC graph to prevent hanging.
All workers in the replica group of a CC op should participate in load for the neff that contains that CC op.
A pipeline case where worker 1 would do a few forwards before starting to send
and then the subsequent workers in the pipeline would do a less forward but would do a recv first.
There appears a case, when the first worker would enter the steady state (forward+send) one step after other workers.
So in this case, since other workers are in steady state they are doing an infer,
whereas worker 1 which entered steady state 1 step later, would try to load its steady state graph and get stuck
"""


def send(tensor, send_next=True, tracing=False):
    if not tracing:
        xm.mark_step()
    if send_next:
        groups = parallel_state.get_next_rank_group(as_list=True)
    else:
        groups = parallel_state.get_prev_rank_group(as_list=True)
    logger.debug(rmsg(f"send with groups {groups}"))
    _ = xm.all_reduce(xm.REDUCE_SUM, tensor, groups=groups)
    if not tracing:
        xm.mark_step()


def recv_from(tensor_meta, recv_prev=True, tracing=False):
    if not tracing:
        xm.mark_step()
    tensor_recv_next = torch.zeros(
        tensor_meta.shape,
        requires_grad=tensor_meta.requires_grad,
        device=xm.xla_device(),
        dtype=tensor_meta.dtype,
    )
    if recv_prev:
        groups = parallel_state.get_prev_rank_group(as_list=True)
    else:
        groups = parallel_state.get_next_rank_group(as_list=True)
    logger.debug(rmsg(f"recv with groups {groups}"))
    tensor_recv_next = xm.all_reduce(xm.REDUCE_SUM, tensor_recv_next, groups=groups)
    if not tracing:
        xm.mark_step()
    return tensor_recv_next


"""
Eager send/recv for python objects, mainly used for get tensor shapes.
"""


def send_python_object(obj, send_next=True):
    gloo_group = get_gloo_group()
    if send_next:
        dst_rank = parallel_state.get_pipeline_model_parallel_next_rank()
    else:
        dst_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
    data = pickle.dumps(obj)
    data_length = len(data)
    # first 4 bytes will be data length
    data = data_length.to_bytes(4, "big") + data
    assert len(data) < MAX_LENGTH, "Sending python object larger than 1M is not supported"
    # Pad to MAX_LENGTH
    data += bytes(MAX_LENGTH - len(data))
    data = np.frombuffer(data, dtype=np.uint8)
    assert len(data) == MAX_LENGTH
    tensor = torch.from_numpy(data).cpu()
    torch.distributed.send(tensor, dst_rank, group=gloo_group)


def recv_python_object(recv_prev=True):
    gloo_group = get_gloo_group()
    if recv_prev:
        src_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
    else:
        src_rank = parallel_state.get_pipeline_model_parallel_next_rank()
    tensor = torch.zeros(MAX_LENGTH, dtype=torch.uint8, device="cpu")
    torch.distributed.recv(tensor, src=src_rank, group=gloo_group)
    data = tensor.cpu().numpy().tobytes()
    # get original length
    length = int.from_bytes(data[:4], "big")
    data = data[4 : length + 4]
    return pickle.loads(data)
