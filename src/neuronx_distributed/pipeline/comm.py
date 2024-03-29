import pickle
from collections import defaultdict

import numpy as np
import torch
import torch_xla.core.xla_model as xm

from ..parallel_layers import parallel_state
from ..parallel_layers.parallel_state import rmsg
from ..utils.logger import get_logger
from ..utils.serialization import compress_to_string, uncompress_from_string

logger = get_logger()

# max size of python object buffer to send between stages using gloo
MAX_LENGTH = 2**20  # 1M

# global dict to store the number of send/recvs using tcp store
kv_tag_send_count = defaultdict(int)
kv_tag_recv_count = defaultdict(int)

# max retry when getting from tcp store
# tested with 512 nodes cluster
MAX_RETRY = 3


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
    # PT XLA >= 2.1 requires_grad attr is not preserved after CC comm
    # TODO: check with XLA team
    tensor_recv_next.requires_grad_(tensor_meta.requires_grad)
    if not tracing:
        xm.mark_step()
    return tensor_recv_next


"""
Eager send/recv for python objects, mainly used for get tensor shapes.
"""


def send_python_object(obj, send_next=True, method="tcp"):
    if send_next:
        dst_rank = parallel_state.get_pipeline_model_parallel_next_rank()
    else:
        dst_rank = parallel_state.get_pipeline_model_parallel_prev_rank()

    if method == "gloo":
        _send_with_gloo_group(obj, dst_rank)
    else:
        _send_with_tcp_store(obj, dst_rank)


def _send_with_gloo_group(obj, dst_rank):
    """
    Convert the object into a torch cpu tensor and send it with the gloo group
    """
    gloo_group = parallel_state.get_pp_gloo_group()
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


def _send_with_tcp_store(obj, dst_rank):
    """
    Convert the object into a string and set it in the tcp store.
    Each send/recv has unique tcp store key, with a suffix of the send/recv count.
    The send/recv index should be sync on all ranks, i.e. for the same tag,
    the send/recv ranks should have the same count
    """
    global kv_tag_send_count
    tcp_store = parallel_state.get_tcp_store()
    obj_str = compress_to_string(obj)
    rank = torch.distributed.get_rank()
    tag = f"{rank}_to_{dst_rank}"
    kv_tag_send_count[tag] += 1
    tcp_store.set(f"{tag}_{kv_tag_send_count[tag]}", obj_str)


def recv_python_object(recv_prev=True, method="tcp"):
    if recv_prev:
        src_rank = parallel_state.get_pipeline_model_parallel_prev_rank()
    else:
        src_rank = parallel_state.get_pipeline_model_parallel_next_rank()

    if method == "gloo":
        return _recv_with_gloo_group(src_rank)
    else:
        return _recv_with_tcp_store(src_rank)


def _recv_with_gloo_group(src_rank):
    """
    Receive a torch cpu tensor and convert it back to python object
    """
    gloo_group = parallel_state.get_pp_gloo_group()
    tensor = torch.zeros(MAX_LENGTH, dtype=torch.uint8, device="cpu")
    torch.distributed.recv(tensor, src=src_rank, group=gloo_group)
    data = tensor.cpu().numpy().tobytes()
    # get original length
    length = int.from_bytes(data[:4], "big")
    data = data[4 : length + 4]
    return pickle.loads(data)


def _recv_with_tcp_store(src_rank):
    """
    Getting the object from tcp store and uncompress it back to python object
    """
    global kv_tag_recv_count
    tcp_store = parallel_state.get_tcp_store()
    rank = torch.distributed.get_rank()
    tag = f"{src_rank}_to_{rank}"
    kv_tag_recv_count[tag] += 1
    key = f"{tag}_{kv_tag_recv_count[tag]}"
    count = 0
    success = False
    # Sometimes it will timeout for large cluster
    while count < MAX_RETRY and not success:
        try:
            # Need to decode since tcp_store.get returns byte types
            obj_str = tcp_store.get(key).decode("utf-8")
            success = True
        except:
            count += 1
    if not success:
        raise RuntimeError(rmsg(f"Failed to receive object with key {key}"))
    # After received the object, delete the key
    tcp_store.delete_key(key)
    obj = uncompress_from_string(obj_str)
    return obj
