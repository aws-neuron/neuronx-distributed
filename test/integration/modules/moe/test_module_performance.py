import argparse
import dataclasses
import gc
import os
import time
import traceback
from enum import Enum

import torch

# Imports from MoE unit tests (for this import to succeed, test/unit_test/modules/moe must be added to PYTHONPATH)
import utils_testing as ut
from device_correctness_test_configs import get_neuron_cc_flags, get_model_config
from device_correctness_test_runner import get_model_outputs, print_rank0

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.utils import requires_init_pg_override

SEPARATOR = "-" * 70

# FP32 test tolerances
OUTPUT_TEST_TOLS_FP32 = {
    "atol": 5e-4,
    "rtol": 1e-2,
}
# BF16 test tolerances
OUTPUT_TEST_TOLS_BF16 = {
    "atol": 5e-2,
    "rtol": 1e-2,
}

class INSTANCE_TYPE(Enum):
    TRN2_48XLARGE = "trn2.48xlarge"


BANDWIDTH_PER_CORE_FOR_INSTANCE = {
    INSTANCE_TYPE.TRN2_48XLARGE: 46.4 * 1024 / 16 / 4,
}

BYTES_PER_DTYPE_MAP = {
    torch.bfloat16: 2,
    torch.float: 4,
}


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--s3_dir", required=False, help="location to upload all test artifacts")
    parser.add_argument("--s3_bucket", default="s3://ktf-test-runs/neuronx_distributed_modules/moe")
    parser.add_argument("--test_dtype", required=True, choices=["fp32", "bf16"], help="Either fp32 or bf16")
    allowed_tp_degrees = [2, 8, 16, 32]
    tp_degrees_help_msg = "One of 2, 8, 16, 32"
    if ut.get_platform_lnc() == 2:
        allowed_tp_degrees.append(64)
        tp_degrees_help_msg.join(", 64")
    parser.add_argument(
        "--test_tp_degree", required=True, type=int, choices=allowed_tp_degrees, help=tp_degrees_help_msg
    )
    args, leftovers = parser.parse_known_args()
    S3_BUCKET_NAME = args.s3_bucket
    test_dtype = torch.float32 if args.test_dtype == "fp32" else torch.bfloat16
    return S3_BUCKET_NAME, args, test_dtype, args.test_tp_degree


S3_BUCKET_NAME, args, TEST_DTYPE, TEST_TP_DEGREE = parse_args()
results = {"inference_success": 1}

# Set compiler flags before TRN enablement
os.environ["NEURON_CC_FLAGS"] = get_neuron_cc_flags(test_dtype=TEST_DTYPE)

# TRN enablement
import torch_xla.core.xla_model as xm  # noqa: E402


@dataclasses.dataclass
class ExptCfg(ut.ExptCfg):
    tp_degree: int = 0
    stack_size: int = 1
    return_router_logits: bool = True


def summarize_test(start_time, num_tests, failed):
    print_rank0(f"{SEPARATOR}\nRan {num_tests} tests in {round(time.time()-start_time, 1)}s\n\n")
    if failed == 0:
        print_rank0("OK\n\n")
    else:
        raise Exception(f"Failed {failed}/{num_tests} tests")


def calculate_mbu_for_moe(
        latency_in_ms,
        batch_size,
        tp_degree,
        num_local_experts,
        num_experts_per_tok,
        num_shared_experts,
        hidden_size,
        intermediate_size,
        dtype,
        stack_size,
        instance_type,
    ):
    """
    Calculate MBU (in the range  of [0, 1]) for MoE module.

    MBU = achieved memory bandwidth / total memory bandwidth
    achieved memory bandwidth = params loaded / latency [there is no KV cache in MoE module]
    params loaded in MoE layer = params loaded in routed experts + params loaded in shared experts
    params loaded in routed experts = 3 x hidden x intermediate x min(top_k x bs, num_experts)
    """
    assert instance_type in BANDWIDTH_PER_CORE_FOR_INSTANCE, f"Currently only supports {BANDWIDTH_PER_CORE_FOR_INSTANCE.keys()}"
    bandwidth_per_core = BANDWIDTH_PER_CORE_FOR_INSTANCE[instance_type] # single core mem bandwidth GBps
    total_avail_bandwidth = bandwidth_per_core * tp_degree # Mem bandwidth for all cores GBps

    assert dtype in BYTES_PER_DTYPE_MAP, f"Currently only supports {BYTES_PER_DTYPE_MAP.keys()}"
    bytes_per_dtype = BYTES_PER_DTYPE_MAP[dtype]

    experts = 3 * hidden_size * intermediate_size * min(num_local_experts, num_experts_per_tok * batch_size)
    shared_experts = 3 * hidden_size * intermediate_size * num_shared_experts
    total_bytes_read = bytes_per_dtype * (experts + shared_experts)

    actual_bandwidth_used = (total_bytes_read / (1024 * 1024 * 1024)) / (latency_in_ms / 1000 / stack_size)  # read bandwidth achieved (GB/s)
    mbu = actual_bandwidth_used / total_avail_bandwidth
    return mbu


def get_instance_type():
    fpath = '/sys/devices/virtual/dmi/id/product_name'
    try:
        with open(fpath, 'r') as f:
            fc = f.readline().strip()
    except IOError:
        raise RuntimeError('Unable to read platform target. If running on CPU, please supply \
        compiler argument target, with one of options trn1, inf2, trn1n, or trn2. Ex: \"--target trn1\"')
    return INSTANCE_TYPE(fc)


def get_test_configs(dtype, tp_degree):
    test_configs = []

    # Llama4 test cases
    test_cfg = {
        "dtype": dtype,
        "glu_mlp": True,
        "hidden_act": "silu",
        "implementation": "llama4",
        "capacity_factor": None,
        "num_iters": 1,
    }
    test_cfg["test_mode"] = "inference"
    test_configs.extend(
        [
            # Token-generation
            ExptCfg(
                seq_len=1,
                batch_size=1,
                tp_degree=tp_degree,
                early_expert_affinity_modulation=True,
                moe_fused_tkg_enabled=True,
                moe_fused_tkg_kernel_enabled=True,
                **get_model_config("llama4-100b"),
                **test_cfg
            ),
            ExptCfg(
                seq_len=1,
                batch_size=4,
                tp_degree=tp_degree,
                early_expert_affinity_modulation=True,
                moe_fused_tkg_enabled=True,
                moe_fused_tkg_kernel_enabled=True,
                **get_model_config("llama4-100b"),
                **test_cfg
            )
        ]
    )

    return test_configs


def run_module_accuracy_test(cfg: ExptCfg, output_tols):
    device = "xla"
    cfg = dataclasses.replace(cfg, device=device)  # Overwrite the device in the config
    cfg_fc = dataclasses.replace(
        cfg,
        moe_fused_tkg_kernel_enabled=False,
        router_topk_kernel_enabled=False,
        expert_mlp_kernel_enabled=False,
        shared_mlp_kernel_enabled=False,
    )
    cfg_nki = dataclasses.replace(cfg)
    token_shuffle_group_size = getattr(cfg, "token_shuffle_group_size", 1)
    assert cfg.test_mode == "inference", f"Unknown test_mode: {cfg.test_mode}"
    sequence_parallel_enabled = cfg.sequence_parallel_enabled
    assert cfg.sequence_parallel_enabled is False, "SP not supported in TKG"

    ut.nxd_init(tp_degree=cfg.tp_degree, ep_degree=1, token_shuffle_group_size=token_shuffle_group_size, seed=0)
    dp_size = parallel_state.get_data_parallel_size()
    dp_rank = parallel_state.get_data_parallel_rank()

    with torch.no_grad():
        for it in range(cfg.num_iters):
            print_rank0(f"iteration {it}")
            # Initialize model on trn
            ut.nxd_init(
                tp_degree=cfg.tp_degree, ep_degree=1, token_shuffle_group_size=token_shuffle_group_size, seed=it
            )
            model_fc = ut.initialize_neuron_model(cfg_fc)
            model_nki = ut.initialize_neuron_model(cfg_nki)
            # Match expert weights (same tp in both models, so we can copy the weights directly)
            with torch.no_grad():
                for (fc_name, fc_param), (nki_name, nki_param) in zip(
                    model_fc.named_parameters(), model_nki.named_parameters()
                ):
                    if "expert_mlps" in fc_name:
                        nki_param.copy_(fc_param)
                    elif "shared_experts" in fc_name:
                        nki_param.copy_(fc_param.transpose(0, 1).contiguous())
                xm.mark_step()

            model_fc.eval()
            model_nki.eval()
            # Initialize input, target, model
            # Input is BSH in inference
            ip_fc = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, dtype=cfg.dtype, device=device)
            ip_nki = ip_fc.detach().clone().to(device)

            ut.nxd_init(tp_degree=cfg.tp_degree, ep_degree=1, token_shuffle_group_size=token_shuffle_group_size, seed=it)
            # Get outputs and gradients from flat compiler
            router_logits_fc, op_fc, loss_fc, grad_norm_fc, grad_dict_cpu = get_model_outputs(
                cfg_fc,
                model_fc,
                None,
                ip_fc,
                torch.Tensor([0]),
                sequence_parallel_enabled,
                dp_size,
                dp_rank,
                token_shuffle_group_size,
                is_cpu=False,
            )

            ut.nxd_init(
                tp_degree=cfg.tp_degree, ep_degree=1, token_shuffle_group_size=token_shuffle_group_size, seed=it
            )
            router_logits_nki, op_nki, loss_nki, grad_norm_nki, grad_dict_nki = get_model_outputs(
                cfg_nki,
                model_nki,
                None,
                ip_nki,
                torch.Tensor([0]),
                sequence_parallel_enabled,
                dp_size,
                dp_rank,
                token_shuffle_group_size,
                is_cpu=False,
            )

            del ip_fc, ip_nki

            ut.check_tensors(router_logits_fc, router_logits_nki, **output_tols)
            del router_logits_fc, router_logits_nki

            ut.check_tensors(op_fc.detach(), op_nki.detach(), **output_tols)
            del op_fc, op_nki

            xm.mark_step()

    del model_fc, model_nki
    gc.collect()


def run_module_performance_test(cfg: ExptCfg):
    device = "xla"
    cfg = dataclasses.replace(cfg, device=device, return_router_logits=False, stack_size=10)  # Overwrite the device in the config
    cfg_fc = dataclasses.replace(
        cfg,
        moe_fused_tkg_kernel_enabled=False,
        router_topk_kernel_enabled=False,
        expert_mlp_kernel_enabled=False,
        shared_mlp_kernel_enabled=False
    )
    cfg_nki = cfg
    token_shuffle_group_size = getattr(cfg, "token_shuffle_group_size", 1)
    assert cfg.test_mode == "inference", f"Unknown test_mode: {cfg.test_mode}"
    sequence_parallel_enabled = cfg.sequence_parallel_enabled
    assert cfg.sequence_parallel_enabled is False, "SP not supported in TKG"
    instance_type = get_instance_type()

    ut.nxd_init(tp_degree=cfg.tp_degree, ep_degree=1, token_shuffle_group_size=token_shuffle_group_size, seed=0)
    dp_size = parallel_state.get_data_parallel_size()
    dp_rank = parallel_state.get_data_parallel_rank()

    # Record MBU for each iter
    mbu_fc = []
    mbu_nki = []

    with torch.no_grad():
        for it in range(cfg.num_iters + 1):
            print_rank0(f"iteration {it}")
            # Initialize model on trn
            ut.nxd_init(
                tp_degree=cfg.tp_degree, ep_degree=1, token_shuffle_group_size=token_shuffle_group_size, seed=it
            )
            model_fc = ut.initialize_neuron_model(cfg_fc)
            model_nki = ut.initialize_neuron_model(cfg_nki)

            model_fc.eval()
            model_nki.eval()
            # Initialize input, target, model
            # Input is BSH in inference
            ip_fc = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, dtype=cfg.dtype, device=device)
            ip_nki = ip_fc.detach().clone().to(device)

            ut.nxd_init(tp_degree=cfg.tp_degree, ep_degree=1, token_shuffle_group_size=token_shuffle_group_size, seed=it)
            # Get outputs and gradients from flat compiler
            start = time.perf_counter()
            router_logits_fc, op_fc, loss_fc, grad_norm_fc, grad_dict_cpu = get_model_outputs(
                cfg_fc,
                model_fc,
                None,
                ip_fc,
                torch.Tensor([0]),
                sequence_parallel_enabled,
                dp_size,
                dp_rank,
                token_shuffle_group_size,
                is_cpu=False,
            )
            latency = (time.perf_counter() - start) * 1000
            mbu = calculate_mbu_for_moe(
                latency_in_ms=latency,
                batch_size=cfg.batch_size,
                tp_degree=cfg.tp_degree,
                num_local_experts=cfg.num_experts, # EP = 1
                num_experts_per_tok=cfg.top_k,
                num_shared_experts=cfg.num_shared_experts,
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                dtype=cfg.dtype,
                stack_size=cfg.stack_size,
                instance_type=instance_type,
            )
            if iter > 1:
                mbu_fc.append(mbu)
                print_rank0(f"Flat compiler latency in ms: {latency}")
                print_rank0(f"MBU for flat compiler: {mbu}")

            ut.nxd_init(
                tp_degree=cfg.tp_degree, ep_degree=1, token_shuffle_group_size=token_shuffle_group_size, seed=it
            )
            start = time.perf_counter()
            router_logits_nki, op_nki, loss_nki, grad_norm_nki, grad_dict_nki = get_model_outputs(
                cfg_nki,
                model_nki,
                None,
                ip_nki,
                torch.Tensor([0]),
                sequence_parallel_enabled,
                dp_size,
                dp_rank,
                token_shuffle_group_size,
                is_cpu=False,
            )
            latency = (time.perf_counter() - start) * 1000
            mbu = calculate_mbu_for_moe(
                latency_in_ms=latency,
                batch_size=cfg.batch_size,
                tp_degree=cfg.tp_degree,
                num_local_experts=cfg.num_experts, # EP = 1
                num_experts_per_tok=cfg.top_k,
                num_shared_experts=cfg.num_shared_experts,
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size,
                dtype=cfg.dtype,
                stack_size=cfg.stack_size,
                instance_type=instance_type,
            )
            if it > 0:
                mbu_nki.append(mbu)
                print_rank0(f"Kernel latency in ms: {latency}")
                print_rank0(f"MBU for kernel: {mbu}")

            del ip_fc, ip_nki

            xm.mark_step()

    print_rank0(f"Average MBU for flat compiler: {sum(mbu_fc) / len(mbu_fc)}")
    print_rank0(f"Average MBU for kernel: {sum(mbu_nki) / len(mbu_nki)}")
    del model_fc, model_nki
    gc.collect()


def test_moe_layer_accuracy_and_performance():
    if TEST_DTYPE == torch.float32:
        output_test_tols = OUTPUT_TEST_TOLS_FP32
    elif TEST_DTYPE == torch.bfloat16:
        output_test_tols = OUTPUT_TEST_TOLS_BF16
    else:
        raise ValueError(f"Unknown TEST_DTYPE: {str(TEST_DTYPE)}")

    def _test_moe_layer_module_accuracy_and_performance():
        test_configs = get_test_configs(dtype=TEST_DTYPE, tp_degree=TEST_TP_DEGREE)
        start_time = time.time()
        failed = 0
        for i, cfg in enumerate(test_configs):
            print_rank0(f"Running test {i+1}/{len(test_configs)}: {str(cfg)}")
            try:
                run_module_accuracy_test(cfg, output_test_tols)
                run_module_performance_test(cfg)
                print_rank0("ok\n")
            except Exception:
                print("Failed test")
                print(traceback.format_exc())
                failed += 1
        summarize_test(start_time, len(test_configs), failed)

    global results
    try:
        _test_moe_layer_module_accuracy_and_performance()
    except Exception:
        results["inference_success"] = 0
        print_rank0(traceback.format_exc())
        raise


if __name__ == "__main__":
    if requires_init_pg_override():
        import torch_xla.experimental.pjrt_backend  # noqa
        torch.distributed.init_process_group("xla", init_method="pjrt://")
    else:
        torch.distributed.init_process_group("xla")

    print_rank0(f"Running MoE layer performance test, test_dtype={str(TEST_DTYPE)}")
    test_moe_layer_accuracy_and_performance()
