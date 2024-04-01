import atexit
import os
import traceback
import argparse
import json
from datetime import datetime
# Third Party
import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

from neuronx_distributed.pipeline.comm import send, recv_from, send_python_object, recv_python_object
from neuronx_distributed.utils.serialization import TensorMeta
from neuronx_distributed.parallel_layers.parallel_state import initialize_model_parallel, \
    get_pipeline_model_parallel_rank, initialize_pp_gloo_groups
from neuronx_distributed.utils.serialization import TensorMeta, SerializationManager
from neuronx_distributed.parallel_layers.utils import requires_init_pg_override

datetime_str = str(datetime.now())

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--test_json",
        required=False,
        help="input json listing the test spec for network to compile",
    )
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument(
        "--s3_bucket",
        default="s3://ktf-test-runs/neuronx_distributed_parallel_layers/comm",
    )
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    with open(args.test_json, "r") as f:
        test_dict = json.load(f)
    return test_dict, S3_BUCKET_NAME, args


test_config, S3_BUCKET_NAME, args = parse_args()
results = {"inference_success": 1}

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
    def _test_send_and_recv():
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
            xm.mark_step()
        elif get_pipeline_model_parallel_rank() < 7:
            recv_a = recv_from(tensor_meta)
            send(recv_a)
            xm.mark_step()
            recv_a_cpu = recv_a.to(torch.device('cpu'))
            a = torch.load('tensor.pt', map_location=torch.device('cpu'))
            assert torch.equal(a, recv_a_cpu)
        else:
            recv_a = recv_from(tensor_meta)
            xm.mark_step()
            recv_a_cpu = recv_a.to(torch.device('cpu'))
            a = torch.load('tensor.pt', map_location=torch.device('cpu'))
            assert torch.equal(a, recv_a_cpu)

    global results
    try:
        _test_send_and_recv()
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise

def test_1f_1b_comm():
    def _test_1f_1b_comm():
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
            xm.mark_step()
            recv_backward_cpu = recv_backward.to(torch.device('cpu'))
            backward = torch.load('backward.pt', map_location=torch.device('cpu'))
            assert torch.equal(backward, recv_backward_cpu)
        elif get_pipeline_model_parallel_rank() < 7:
            recv_forward = recv_from(forward_tensor_meta)
            send(recv_forward)
            recv_backward = recv_from(backward_tensor_meta, recv_prev=False)
            send(recv_backward, send_next=False)
            xm.mark_step()
            recv_forward_cpu = recv_forward.to(torch.device('cpu'))
            forward = torch.load('forward.pt', map_location=torch.device('cpu'))
            assert torch.equal(forward, recv_forward_cpu)
            recv_backward_cpu = recv_backward.to(torch.device('cpu'))
            backward = torch.load('backward.pt', map_location=torch.device('cpu'))
            assert torch.equal(backward, recv_backward_cpu)
        else:
            recv_forward = recv_from(forward_tensor_meta)
            backward = torch.rand(1, 2, device = xm.xla_device())
            send(backward, send_next=False)
            torch.save(backward.cpu(), 'backward.pt')
            xm.mark_step()
            recv_forward_cpu = recv_forward.to(torch.device('cpu'))
            forward = torch.load('forward.pt', map_location=torch.device('cpu'))
            assert torch.equal(forward, recv_forward_cpu)

    global results
    try:
        _test_1f_1b_comm()
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise

def test_send_and_recv_python_object():
    def _test_send_and_recv_python_object():
        initialize_pp_gloo_groups()
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

    global results
    try:
        _test_send_and_recv_python_object()
    except:
        results["inference_success"] = 0
        print(traceback.format_exc())
        raise

def upload_to_s3():
    os.system(f'aws s3 cp --no-progress "{datetime_str}" {S3_BUCKET_NAME}')
    print(met.metrics_report())


def on_exit():
    upload_to_s3()
    for k in test_config:
        os.system(f"rm {args.test_json}")
        with open(args.test_json, "w") as f:
            json.dump({k: results}, f)

if __name__ == "__main__":
    if requires_init_pg_override():
        import torch_xla.experimental.pjrt_backend
        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    initialize_model_parallel(1,8)
    test_send_and_recv()
    test_1f_1b_comm()
    test_send_and_recv_python_object()
    atexit.register(on_exit)
