import argparse
import atexit
import itertools
from datetime import datetime

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
    set_seed,
)

from neuronx_distributed.optimizer import NeuronZero1Optimizer, NeuronEPZero1Optimizer
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.utils import requires_init_pg_override

datetime_str = str(datetime.now())


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument(
        "--s3_bucket",
        default="s3://ktf-test-runs/neuronx_distributed_parallel_layers/parallel_state",
    )
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    return S3_BUCKET_NAME, args


S3_BUCKET_NAME, args = parse_args()
results = {"inference_success": 1}


def on_exit():
    print(met.metrics_report())


def get_test_result(opt, use_ep, model_dtype, optimizer_dtype, grad_clipping, max_norm, pin_layout, coalesce_cc):
    use_zero1 = issubclass(opt, NeuronZero1Optimizer)
    device = "xla" if use_zero1 else None
    params = get_test_params(use_ep=use_ep, dtype=model_dtype, device=device)

    optim_inputs = {
        "optimizer_class": torch.optim.AdamW,
        "optimizer_dtype": optimizer_dtype,
        "grad_clipping": grad_clipping,
        "max_norm": max_norm,
        "pin_layout": pin_layout,
        "sharding_groups": parallel_state.get_data_parallel_replica_groups(),
        "grad_norm_groups": parallel_state.get_tensor_model_parallel_replica_groups(),
        "lr": 1e-2,
    }
    # coalesce_cc should not be passed for Torch 2.5
    if version.parse(torch.__version__) == version.parse("2.1"):
        optim_inputs["coalesce_cc"] = coalesce_cc
    optimizer = opt(params, **optim_inputs)

    if use_zero1:
        xm.mark_step()

    set_seed(1234 + parallel_state.get_data_parallel_rank())
    res = []
    for i in range(5):
        for p in params:
            grad = torch.randn_like(p, device="cpu") / 10
            p.grad = grad.to(p.device)
        optimizer.step()
        optimizer.zero_grad()
        if use_zero1:
            xm.mark_step()

        param_norm = torch.tensor(0.0).to(torch.double)
        for p in params:
            param_norm += p.detach().clone().cpu().norm(2) ** 2
        param_norm = torch.sqrt(param_norm)
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
        expert_model_parallel_size=parallel_config["ep_degree"],
    )
    initialize_gloo_groups()

    use_ep = parallel_config["ep_degree"] > 1
    res_zero1 = get_test_result(
        NeuronEPZero1Optimizer if use_ep else NeuronZero1Optimizer,
        use_ep=use_ep,
        model_dtype=model_dtype,
        optimizer_dtype=optimizer_dtype,
        grad_clipping=grad_clipping,
        max_norm=max_norm,
        pin_layout=pin_layout,
        coalesce_cc=coalesce_cc,
    )
    res_ref = get_test_result(
        RefOptimizer,
        use_ep=use_ep,
        model_dtype=model_dtype,
        optimizer_dtype=optimizer_dtype,
        grad_clipping=grad_clipping,
        max_norm=max_norm,
        pin_layout=pin_layout,
        coalesce_cc=coalesce_cc,
    )
    if torch.distributed.get_rank() == 0:
        print(
            "test with config:",
            parallel_config,
            model_dtype,
            optimizer_dtype,
            grad_clipping,
            max_norm,
            pin_layout,
            coalesce_cc,
        )
        print(res_zero1)
        print(res_ref)
        for res0, res1 in zip(res_zero1, res_ref):
            error = 2e-3
            torch.testing.assert_close(res0, res1, rtol=error, atol=error)
        print("test passed!")

    parallel_state.destroy_model_parallel()
    destroy_gloo_groups()


if __name__ == "__main__":
    if requires_init_pg_override():
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
        [
            {"tp_degree": 8, "pp_degree": 1, "ep_degree": 1},
            {"tp_degree": 1, "pp_degree": 4, "ep_degree": 1},
            {"tp_degree": 1, "pp_degree": 1, "ep_degree": 8},
        ],
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
