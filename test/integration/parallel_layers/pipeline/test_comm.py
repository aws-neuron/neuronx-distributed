# Third Party
import torch
import torch_xla.core.xla_model as xm

from neuronx_distributed.pipeline.comm import send, recv_from, send_python_object, recv_python_object
from neuronx_distributed.utils.serialization import TensorMeta
from neuronx_distributed.parallel_layers.parallel_state import initialize_model_parallel, \
    get_pipeline_model_parallel_rank, set_gloo_group
from neuronx_distributed.utils.serialization import TensorMeta, SerializationManager

class MyClass(object):
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        if isinstance(other, MyClass):
            return self.x == other.x
        return False
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
def test_send_and_recv():
    tensor_meta = TensorMeta(
            tensor_index=-1,
            dtype=torch.float32,
            shape=torch.Size([2,3]),
            requires_grad=False,
            device=None,
        )
    if get_pipeline_model_parallel_rank() == 0:
        a = torch.rand(2, 3, device = xm.xla_device())
        send(a)
        torch.save(a.cpu(), 'tensor.pt')
    elif get_pipeline_model_parallel_rank() < 7:
        recv_a = recv_from(tensor_meta)
        a = torch.load('tensor.pt', map_location=torch.device('cpu'))
        assert torch.equal(a, recv_a)
        send(recv_a)
    else:
        recv_a = recv_from(tensor_meta)
        a = torch.load('tensor.pt', map_location=torch.device('cpu'))
        assert torch.equal(a, recv_a)

def test_1f_1b_comm():
    forward_tensor_meta = TensorMeta(
            tensor_index=-1,
            dtype=torch.float32,
            shape=torch.Size([2,3]),
            requires_grad=False,
            device=None,
        )
    backward_tensor_meta = TensorMeta(
            tensor_index=-1,
            dtype=torch.float32,
            shape=torch.Size([1,2]),
            requires_grad=False,
            device=None,
        )
    # Testing 1F1B communication
    if get_pipeline_model_parallel_rank() == 0:
        forward = torch.rand(2, 3, device = xm.xla_device())
        send(forward)
        torch.save(forward.cpu(), 'forward.pt')
        recv_backward = recv_from(backward_tensor_meta, recv_prev=False)
        backward = torch.load('backward.pt', map_location=torch.device('cpu'))
        assert torch.equal(backward, recv_backward)
    elif get_pipeline_model_parallel_rank() < 7:
        recv_forward = recv_from(forward_tensor_meta)
        forward = torch.load('forward.pt', map_location=torch.device('cpu'))
        assert torch.equal(forward, recv_forward)
        send(recv_forward)
        recv_backward = recv_from(backward_tensor_meta, recv_prev=False)
        backward = torch.load('backward.pt', map_location=torch.device('cpu'))
        assert torch.equal(backward, recv_backward)
        send(recv_backward, send_next=False)
    else:
        recv_forward = recv_from(forward_tensor_meta)
        forward = torch.load('forward.pt', map_location=torch.device('cpu'))
        assert torch.equal(forward, recv_forward)
        backward = torch.rand(1, 2, device = xm.xla_device())
        send(backward, send_next=False)
        torch.save(backward.cpu(), 'backward.pt')

def test_send_and_recv_python_object():
    set_gloo_group()
    cls_type = MyClass(2)
    data = {"a": 1, "b": [torch.ones([2, 4]), torch.ones([2, 4]), (1, 2)], 
            "c" : (1, 2, torch.tensor(1.0)), "d": torch.zeros([2, 4]), "f": cls_type}
    s = SerializationManager()
    serialized, tx_list, tensor_meta = s.serialize(data)
    if get_pipeline_model_parallel_rank() == 0:
        send_python_object((serialized, tensor_meta))
    elif get_pipeline_model_parallel_rank() < 7:
        recv_serialized, recv_tensor_meta = recv_python_object()
        assert tensor_meta == recv_tensor_meta
        assert serialized == recv_serialized
        send_python_object((recv_serialized, recv_tensor_meta))
    else:
        recv_serialized, recv_tensor_meta = recv_python_object()
        assert tensor_meta == recv_tensor_meta
        assert serialized == recv_serialized

if __name__ == "__main__":
    torch.distributed.init_process_group("xla")
    initialize_model_parallel(1,8)
    test_send_and_recv()
    test_1f_1b_comm()
    test_send_and_recv_python_object()