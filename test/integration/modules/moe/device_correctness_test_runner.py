import dataclasses

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm  # TRN enablement

# Imports from MoE unit tests (for this import to succeed, test/unit_test/modules/moe must be added to PYTHONPATH)
import utils_testing as ut

from neuronx_distributed.modules.moe import MoESequenceParallelMode
from neuronx_distributed.parallel_layers import mappings, parallel_state, random

STATE_KEYS = {
    "_TENSOR_MODEL_PARALLEL_GROUP",
    "_TENSOR_MODEL_PARALLEL_GROUP_SPMD",
    "_PIPELINE_MODEL_PARALLEL_GROUP",
    "_PIPELINE_GLOBAL_RANKS",
    "_PIPELINE_MODEL_PARALLEL_GROUP_SPMD",
    "_NEXT_RANK_GROUP_SPMD",
    "_PREV_RANK_GROUP_SPMD",
    "_NEXT_RANK_GROUP",
    "_PREV_RANK_GROUP",
    "_DATA_PARALLEL_GROUP",
    "_DATA_PARALLEL_GROUP_SPMD",
    "_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE",
    "_MPU_TENSOR_MODEL_PARALLEL_RANK",
}

PARALLEL_STATE_MAP = {}


def get_model_outputs(cfg, model, ip, target, sequence_parallel_enabled):
    assert model.is_test is False
    if model.return_router_logits:
        op, _ = model(ip)
    else:
        (op,) = model(ip)

    if cfg.test_mode == "training":
        if sequence_parallel_enabled:
            op_full = mappings.gather_from_sequence_parallel_region(op, to_model_parallel=False)
        else:
            op_full = op
        op_full = op_full.view(-1, cfg.hidden_size)
        loss = F.nll_loss(op_full, target)
        del op_full
        loss.backward()
        grad_dict = ut.get_model_grads_dict(model)
        return op, loss, grad_dict
    else:
        return op, torch.Tensor([0]), {}


def nxd_init(tp_degree, ep_degree, seed):
    assert ep_degree == 1

    world_size = torch.distributed.get_world_size()
    parallel_state_key = f"{world_size}_{tp_degree}_{ep_degree}"

    def _save_parallel_state(key):
        state = {}
        for attr in STATE_KEYS:
            state[attr] = parallel_state.__dict__[attr]
        PARALLEL_STATE_MAP[key] = state

    def _load_parallel_state(key):
        for k, v in PARALLEL_STATE_MAP[key].items():
            parallel_state.__dict__[k] = v

    if parallel_state_key in PARALLEL_STATE_MAP:
        _load_parallel_state(parallel_state_key)
    else:
        parallel_state.destroy_model_parallel()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_degree,
            pipeline_model_parallel_size=1,
        )
        _save_parallel_state(parallel_state_key)

    # Set seed
    random.model_parallel_xla_manual_seed(seed)


def run_device_correctness_test(cfg, output_tols, grad_tols):
    device = "xla"
    cfg_trn = dataclasses.replace(cfg, device=device)  # Overwrite the device in the config
    tp_degree = getattr(cfg, "tp_degree", 1)
    ep_degree = getattr(cfg, "ep_degree", 1)
    assert cfg.test_mode in {"training", "inference"}, f"Unknown test_mode: {cfg.test_mode}"
    sequence_parallel_mode = getattr(cfg, "sequence_parallel_mode", MoESequenceParallelMode.NO_SP)
    sequence_parallel_enabled = cfg.test_mode == "training" and sequence_parallel_mode != MoESequenceParallelMode.NO_SP

    # Initialize model on cpu and trn
    nxd_init(tp_degree=1, ep_degree=1, seed=0)
    model_cpu = ut.initialize_neuron_model(cfg)
    nxd_init(tp_degree=tp_degree, ep_degree=ep_degree, seed=0)
    model_trn = ut.initialize_neuron_model(cfg_trn)
    if cfg.test_mode == "training":
        model_cpu.train()
        model_trn.train()
        # Set sinkhorn_iterations=0, because small precision errors can cause differences in routing decisions
        model_cpu.router.sinkhorn_iterations = 0
        model_trn.router.sinkhorn_iterations = 0
        grad_ctx_mgr = torch.enable_grad
    else:
        model_cpu.eval()
        model_trn.eval()
        grad_ctx_mgr = torch.no_grad

    with grad_ctx_mgr():
        for it in range(cfg.num_iters):
            # Init NxD with tp_degree=1 and ep_degree=1, for running on cpu model
            nxd_init(tp_degree=1, ep_degree=1, seed=it)

            # Initialize input, target, model on cpu
            if cfg.test_mode == "training":
                # Input is SBH in training
                ip_cpu = torch.randn(cfg.seq_len, cfg.batch_size, cfg.hidden_size, dtype=cfg.dtype).detach()
            else:
                # Input is BSH in inference
                ip_cpu = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, dtype=cfg.dtype).detach()
            ip_trn_full = ip_cpu.detach().to(device)
            target_cpu = torch.randint(
                0, cfg.hidden_size - 1, (cfg.seq_len * cfg.batch_size,), dtype=torch.long
            ).detach()

            # torch.topk behavior is different on cpu and device in the case of ties.
            # This causes mismatches in expert assignment for the TopK tests in bf16.
            if cfg.dtype == torch.bfloat16 and cfg.implementation == "topk":
                # Set is_test=True to return expert_index
                model_cpu.is_test = True
                model_trn.is_test = True

                # Simulate dropping of tokens in input where the expert assignments are not matching on cpu and device
                with torch.no_grad():
                    router_logits_cpu, expert_index_cpu = model_cpu(ip_cpu)[-2:]
                    expert_index_trn = model_trn(ip_trn_full)[-1]
                    expert_mismatch_indices = set(torch.where(expert_index_cpu != expert_index_trn.cpu())[0].tolist())
                    if len(expert_mismatch_indices) > 0:
                        # Check that mismatches only happen when the (top_k+1) router logits are non-unique
                        for mismatch_idx in expert_mismatch_indices:
                            router_logits_idx = router_logits_cpu[mismatch_idx]
                            topk_logits, _ = torch.topk(router_logits_idx, min(cfg.top_k + 1, cfg.num_experts))
                            assert len(topk_logits) != len(torch.unique(topk_logits)), str(topk_logits)
                        # Update the input tensor to mask tokens where there is an expert assignment mismatch
                        ip_cpu = ut.drop_tokens_in_tensor(ip_cpu, expert_mismatch_indices)
                        ip_trn_full = ip_cpu.detach().to(device)

                # Reset is_test
                model_cpu.is_test = False
                model_trn.is_test = False

            # Get outputs and gradients from cpu
            op_cpu, loss_cpu, grad_dict_cpu = get_model_outputs(
                cfg, model_cpu, ip_cpu, target_cpu, sequence_parallel_enabled
            )

            # Re-init NxD with actual TP degree
            nxd_init(tp_degree=tp_degree, ep_degree=ep_degree, seed=it)

            # Get sharded input for rank (for sequence parallel)
            tp_size = parallel_state.get_tensor_model_parallel_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            if sequence_parallel_enabled:
                ip_trn = mappings.scatter_to_sequence_parallel_region(ip_trn_full)
            else:
                ip_trn = ip_trn_full

            # Get outputs and gradients from trn, using the same input and target
            target_trn = target_cpu.clone().detach().to(device)
            op_trn, loss_trn, grad_dict_trn = get_model_outputs(
                cfg, model_trn, ip_trn, target_trn, sequence_parallel_enabled
            )
            xm.mark_step()  # TRN enablement

            del ip_cpu, ip_trn_full, ip_trn, target_cpu, target_trn

            # Compare output
            if sequence_parallel_enabled:
                # Compare with only output shard belonging to the TP rank
                op_cpu = torch.tensor_split(op_cpu, tp_degree, dim=0)[tp_rank]
            ut.check_tensors(op_cpu.detach(), op_trn.detach(), **output_tols, additional_msg=f"Iteration {it}")
            del op_cpu, op_trn

            # Compare loss
            ut.check_tensors(loss_cpu.detach(), loss_trn.detach(), **output_tols)
            del loss_cpu, loss_trn

            # Check gradients on each tp_rank
            assert set(grad_dict_cpu.keys()) == set(grad_dict_trn.keys())
            for key in sorted(grad_dict_cpu):
                grad_dict_cpu[key] = grad_dict_cpu[key].detach()
                if grad_dict_cpu[key].shape == grad_dict_trn[key].shape:
                    key_grad_for_rank = grad_dict_cpu[key]
                else:
                    if "gate_up_proj" in key:
                        gate_proj_grad, up_proj_grad = torch.tensor_split(grad_dict_cpu[key], 2, dim=2)
                        gate_proj_grad_for_rank = torch.tensor_split(gate_proj_grad, tp_size, dim=2)[tp_rank]
                        up_proj_grad_for_rank = torch.tensor_split(up_proj_grad, tp_size, dim=2)[tp_rank]
                        key_grad_for_rank = torch.cat([gate_proj_grad_for_rank, up_proj_grad_for_rank], dim=2)
                    elif "up_proj" in key:
                        key_grad_for_rank = torch.tensor_split(grad_dict_cpu[key], tp_size, dim=2)[tp_rank]
                    elif "down_proj" in key:
                        key_grad_for_rank = torch.tensor_split(grad_dict_cpu[key], tp_size, dim=1)[tp_rank]
                    else:
                        raise Exception(
                            f"Unexpected shapes for key: {key}, {grad_dict_cpu[key].shape}, {grad_dict_trn[key].shape}"
                        )

                additional_msg = f"Iteration {it} \nKey: {key}"
                ut.check_tensors(
                    key_grad_for_rank, grad_dict_trn[key].detach(), **grad_tols, additional_msg=additional_msg
                )

            del grad_dict_cpu, grad_dict_trn

            model_cpu.zero_grad(set_to_none=True)
            model_trn.zero_grad(set_to_none=True)
