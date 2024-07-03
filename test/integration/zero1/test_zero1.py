import argparse
import atexit
import itertools
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from packaging import version
from utils import (
    RefOptimizer,
    destroy_gloo_groups,
    get_test_params,
    initialize_gloo_groups,
)

from neuronx_distributed.optimizer import NeuronZero1Optimizer
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.utils import is_pjrt_device

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


def get_test_result(opt, use_pp, model_dtype, optimizer_dtype, grad_clipping, max_norm, pin_layout, coalesce_cc):
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_zero1 = opt == NeuronZero1Optimizer
    device = "xla" if use_zero1 else None
    params = get_test_params(dtype=model_dtype, device=device)

    optim_inputs = {
        "params": params,
        "optimizer_class": torch.optim.AdamW,
        "optimizer_dtype": optimizer_dtype,
        "grad_clipping": grad_clipping,
        "max_norm": max_norm,
        "pin_layout": pin_layout,
        "sharding_groups": parallel_state.get_data_parallel_group(as_list=True),
        "grad_norm_groups": parallel_state.get_tensor_model_parallel_group(as_list=True),
        "lr": 1e-2,
    }
    if version.parse(torch.__version__) >= version.parse("2.0"):
        optim_inputs["coalesce_cc"] = coalesce_cc
    optimizer = opt(**optim_inputs)

    if use_zero1:
        xm.mark_step()

    res = []
    for _ in range(5):
        for p in params:
            p.grad = torch.clone(p) / 100
        optimizer.step()
        optimizer.zero_grad()
        if use_zero1:
            xm.mark_step()

        param_norm = torch.tensor(0.0).to(torch.double)
        for p in params:
            param_norm += p.detach().clone().cpu().to(torch.double).sum()
        param_norm /= len(params)
        res.append(param_norm)

    return res


def test_zero1(parallel_config, model_dtype, optimizer_dtype, grad_clipping, max_norm, pin_layout, coalesce_cc):
    # skip fp32 weights + bf16 states cases
    if model_dtype == torch.float32 and optimizer_dtype == torch.bfloat16:
        return
    if pin_layout:
        return
    # quick path, as `max_norm` has no effects when grad clipping is disabled
    if not grad_clipping and max_norm != 1.0:
        return

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=parallel_config["tp_degree"],
        pipeline_model_parallel_size=parallel_config["pp_degree"],
    )
    initialize_gloo_groups()

    use_pp = parallel_config["pp_degree"] > 1
    res_zero1 = get_test_result(
        NeuronZero1Optimizer,
        use_pp=use_pp,
        model_dtype=model_dtype,
        optimizer_dtype=optimizer_dtype,
        grad_clipping=grad_clipping,
        max_norm=max_norm,
        pin_layout=pin_layout,
        coalesce_cc=coalesce_cc,
    )
    res_ref = get_test_result(
        RefOptimizer,
        use_pp=use_pp,
        model_dtype=model_dtype,
        optimizer_dtype=optimizer_dtype,
        grad_clipping=grad_clipping,
        max_norm=max_norm,
        pin_layout=pin_layout,
        coalesce_cc=coalesce_cc,
    )
    if torch.distributed.get_rank() == 0:
        print(res_zero1)
        print(res_ref)
        for res0, res1 in zip(res_zero1, res_ref):
            error = 5e-2
            torch.testing.assert_close(res0, res1, rtol=error, atol=error)
        print(
            "test passed with: ",
            parallel_config,
            model_dtype,
            optimizer_dtype,
            grad_clipping,
            max_norm,
            pin_layout,
            coalesce_cc,
        )

    parallel_state.destroy_model_parallel()
    destroy_gloo_groups()


if __name__ == "__main__":
    if is_pjrt_device():
        import torch_xla.experimental.pjrt_backend  # noqa

        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")

    for (
        parallel_config,
        model_dtype,
        optimizer_dtype,
        grad_clipping,
        max_norm,
        pin_layout,
        coalesce_cc,
    ) in itertools.product(
        [{"tp_degree": 8, "pp_degree": 1}, {"tp_degree": 1, "pp_degree": 4}],
        [torch.float32, torch.bfloat16],
        [torch.float32, torch.bfloat16],
        [True, False],
        [1.0, 0.5, 0.2, 2.0],
        [True, False],
        [True, False],
    ):
        test_zero1(parallel_config, model_dtype, optimizer_dtype, grad_clipping, max_norm, pin_layout, coalesce_cc)
        xm.mark_step()
    atexit.register(on_exit)
