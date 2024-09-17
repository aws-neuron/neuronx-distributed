import argparse
import atexit
import json
import os
import random
from datetime import datetime
import shutil

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met

from neuronx_distributed.optimizer import NeuronZero1Optimizer
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.utils import is_pjrt_device
from neuronx_distributed.optimizer.zero_dcp_utils import get_dcp_aux_infos, save_optim_state_dict, load_optim_state_dict
from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu

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
        default="s3://ktf-test-runs/neuronx_distributed_parallel_layers/parallel_state",
    )
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    with open(args.test_json, "r") as f:
        test_dict = json.load(f)
    return test_dict, S3_BUCKET_NAME, args


test_config, S3_BUCKET_NAME, args = parse_args()
results = {"inference_success": 1}


def upload_to_s3():
    os.system(f'aws s3 cp --no-progress "{datetime_str}" {S3_BUCKET_NAME}')
    print(met.metrics_report())


def on_exit():
    upload_to_s3()
    for k in test_config:
        os.system(f"rm {args.test_json}")
        with open(args.test_json, "w") as f:
            json.dump({k: results}, f)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # test pad/unpad
        self.a = torch.nn.Parameter(torch.randn(10, 10))
        self.b = torch.nn.Parameter(torch.randn(15, 10))
        self.c = torch.nn.Parameter(torch.randn(16, 10))
        self.d = torch.nn.Parameter(torch.randn(20, 10))


def test_zero1_dcp():
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=1,
    )

    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = DummyModel()
    model.to(device=xm.xla_device())
    for p in model.parameters():
        p.grad = torch.clone(p.data) / 100

    optimizer = NeuronZero1Optimizer(
        model.parameters(),
        torch.optim.AdamW,
        lr=0.01,
        pin_layout=False,
        sharding_groups=parallel_state.get_data_parallel_group(as_list=True),
        grad_norm_groups=parallel_state.get_tensor_model_parallel_group(as_list=True),
        max_norm=1.0,
        grad_clipping=True,
    )
    xm.mark_step()

    # run step once to get states inited
    optimizer.step()
    optimizer.zero_grad()
    xm.mark_step()

    dcp_aux_infos = get_dcp_aux_infos(model, optimizer)

    if os.path.exists("ckpts"):
        load_optim_state_dict("ckpts", optimizer, dcp_aux_infos, False)
        xm.mark_step()
        xm.rendezvous("sync load 1")
        # test able to run step normally
        optimizer.step()
        optimizer.zero_grad()
        xm.mark_step()

    s0 = optimizer.state_dict()
    s0 = move_all_tensor_to_cpu(s0)
    save_optim_state_dict("ckpts", s0, dcp_aux_infos, False)
    xm.mark_step()
    xm.rendezvous("sync save 1")
    save_optim_state_dict("ckpts2", s0, dcp_aux_infos, False)
    xm.mark_step()
    xm.rendezvous("sync save 2")
    s1 = optimizer.state_dict()
    s1 = move_all_tensor_to_cpu(s1)
    load_optim_state_dict("ckpts", optimizer, dcp_aux_infos, False)
    xm.mark_step()
    xm.rendezvous("sync load 2")
    s2 = optimizer.state_dict()
    s2 = move_all_tensor_to_cpu(s2)
    torch.testing.assert_close(s1, s2, rtol=0, atol=0)
    shutil.rmtree("ckpts", ignore_errors=True)


if __name__ == "__main__":
    if is_pjrt_device():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")

    test_zero1_dcp()
    atexit.register(on_exit)
