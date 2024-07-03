import argparse
import atexit
import json
import os
import traceback
from datetime import datetime

import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from commons import print_separator

from neuronx_distributed.optimizer import NeuronZero1Optimizer
from neuronx_distributed.parallel_layers import parallel_state
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
    parser.add_argument("--s3_bucket", default="s3://ktf-test-runs/neuronx_distributed_parallel_layers/layers")
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    with open(args.test_json, "r") as f:
        test_dict = json.load(f)
    return test_dict, S3_BUCKET_NAME, args


test_config, S3_BUCKET_NAME, args = parse_args()
results = {"inference_success": 1}


def test_zero1_checkpoint():
    def _test_zero1_checkpoint():
        device = xm.xla_device()
        parallel_state.initialize_model_parallel(1)

        p = [torch.nn.Parameter(torch.randn(32, 32).to(xm.xla_device()))]
        p[0].grad = torch.randn(32, 32).to(xm.xla_device())
        opt = NeuronZero1Optimizer(
            p,
            torch.optim.SGD,
            lr=0.01,
            momentum=0.9,
            pin_layout=False,
            grad_clipping=False,
            sharding_groups=parallel_state.get_data_parallel_group(as_list=True),
            grad_norm_groups=parallel_state.get_tensor_model_parallel_group(as_list=True),
        )
        opt.step()
        s1 = opt.state_dict()
        opt.save_sharded_state_dict("/tmp/opt_ckpt")
        opt.load_sharded_state_dict("/tmp/opt_ckpt")
        s2 = opt.state_dict()
        torch.testing.assert_close(s1, s2, rtol=0, atol=0)

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test passed")
        del device

    global results
    try:
        _test_zero1_checkpoint()
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
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")
    print_separator("test zero1 checkpoint")
    test_zero1_checkpoint()
    atexit.register(on_exit)
