import dataclasses
import gc
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch_xla.core.xla_model as xm  # TRN enablement

# Imports from MoE unit tests (for this import to succeed, test/unit_test/modules/moe must be added to PYTHONPATH)
import utils_testing as ut

from neuronx_distributed.parallel_layers import mappings, parallel_state, random

def get_model_outputs(cfg, model, optimizer, ip, target, sequence_parallel_enabled, dp_size, reduce_gradients=False, serialize_dp=False):
    assert model.is_test is False

    # for cpu, sequentially run each data-parallel shard and accummulate grads
    # so that the token dropping pattern is the same as trn
    ip_chunks = split_ip_into_chunks(ip, dp_size, serialize_dp, cfg.test_mode)
    outputs = []

    for ip in ip_chunks:
        if model.return_router_logits:
            op, _ = model(ip)
        else:
            (op,) = model(ip)
        outputs.append(op)

    batch_dim = 1 if cfg.test_mode == "training" else 0
    op = torch.cat(outputs, dim=batch_dim)

    if cfg.test_mode == "training":
        if sequence_parallel_enabled:
            op_full = mappings.gather_from_sequence_parallel_region(op, to_model_parallel=False)
        else:
            op_full = op
        op_full = op_full.view(-1, cfg.hidden_size)
        loss = F.nll_loss(op_full, target)
        del op_full
        loss.backward()

        # prevents runtime errors when running back-to-back unit tests with cross-node ep
        xm.mark_step()

        loss = reduce_gradients_and_losses_and_take_step(optimizer, loss, reduce_gradients)
        grad_dict = ut.get_model_grads_dict(model)

        return op, loss, grad_dict
    else:
        assert cfg.test_mode == "inference"
        return op, torch.Tensor([0]), {}

def reduce_gradients_and_losses_and_take_step(optimizer, loss, reduce_gradients):
    optimizer.step()
    if not reduce_gradients:
        return loss

    # reduce loss
    edp_group = parallel_state.get_expert_data_parallel_group(as_list=True)
    emp_group = parallel_state.get_expert_model_parallel_group(as_list=True)
    dp_size = parallel_state.get_data_parallel_size()

    loss /= dp_size
    xm.all_reduce("sum", [loss], groups=edp_group)
    xm.all_reduce("sum", [loss], groups=emp_group)
    return loss


def split_ip_into_chunks(ip, dp_size, serialize_dp, test_mode):
    # inference input is already sharded by dp
    if test_mode == 'inference' or not serialize_dp:
        return [ip]

    batch_dim = 1 if test_mode == "training" else 0
    split_tensor = torch.tensor_split(ip, dp_size, dim=batch_dim)
    return [t.contiguous() for t in split_tensor]


def shard_batch(tensor, cfg, dp_size, dp_rank, test_mode):
    assert tensor.dim() < 4 and tensor.dim() > 0
    shape = list(tensor.shape)
    if test_mode == "training":
        tensor = tensor.reshape(cfg.seq_len, dp_size*cfg.batch_size, -1)
        tensor = tensor.narrow(1, dp_rank*cfg.batch_size, cfg.batch_size)

        if len(shape) > 2:
            shape[1] //= dp_size
        else:
            shape[0] //= dp_size
        return tensor.reshape(*shape)
    else:
        return tensor


def _get_slice_for_rank(tensor, sharding_info, split_dims=None):
    tp_rank, tp_size, ep_rank, ep_size = sharding_info
    for dim in split_dims:
        rank, size = (tp_rank, tp_size) if dim > 0 else (ep_rank, ep_size)
        tensor = torch.tensor_split(tensor, size, dim=dim)[rank]
    return tensor

def _slice_and_compare_tensors(cpu_dict, trn_dict, sharding_info, it, **tols):
    assert set(cpu_dict.keys()) == set(trn_dict.keys())
    for key in sorted(cpu_dict):
        cpu_dict[key] = cpu_dict[key].detach()
        if cpu_dict[key].shape == trn_dict[key].shape:
            key_tensor_for_rank = cpu_dict[key]
        else:
            if "gate_up_proj" in key:
                gate_proj_tensor, up_proj_tensor = torch.tensor_split(cpu_dict[key], 2, dim=2)
                gate_proj_tensor_for_rank = _get_slice_for_rank(gate_proj_tensor, sharding_info, split_dims=(0, 2))
                up_proj_tensor_for_rank = _get_slice_for_rank(up_proj_tensor, sharding_info, split_dims=(0, 2))
                key_tensor_for_rank = torch.cat([gate_proj_tensor_for_rank, up_proj_tensor_for_rank], dim=2)
            elif "up_proj" in key:
                key_tensor_for_rank = _get_slice_for_rank(cpu_dict[key], sharding_info, split_dims=(0, 2))
            elif "down_proj" in key:
                key_tensor_for_rank = _get_slice_for_rank(cpu_dict[key], sharding_info, split_dims=(0, 1))
            else:
                raise Exception(
                    f"Unexpected shapes for key: {key}, {cpu_dict[key].shape}, {trn_dict[key].shape}"
                )

        additional_msg = f"Iteration {it} \nKey: {key}"

        ut.check_tensors(
            key_tensor_for_rank, trn_dict[key].detach(), **tols, additional_msg=additional_msg
        )

def run_device_correctness_test(cfg, output_tols, grad_tols):
    device = "xla"
    cfg_trn = dataclasses.replace(cfg, device=device)  # Overwrite the device in the config
    tp_degree = getattr(cfg, "tp_degree", 1)
    ep_degree = getattr(cfg, "ep_degree", 1)
    assert cfg.test_mode in {"training", "inference"}, f"Unknown test_mode: {cfg.test_mode}"
    sequence_parallel_enabled = cfg.sequence_parallel_enabled
    if sequence_parallel_enabled:
        assert cfg.test_mode == "training"

    ut.nxd_init(tp_degree=tp_degree, ep_degree=ep_degree, seed=0)
    # using non-zero learning rate for zero-1 so that we can do an end-to-end test
    lr = cfg_trn.lr if cfg_trn.zero1 else 0.0
    grad_clipping = True
    dp_size = parallel_state.get_data_parallel_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    ep_size = parallel_state.get_expert_model_parallel_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    if cfg.test_mode == "training":
        grad_ctx_mgr = torch.enable_grad
    else:
        grad_ctx_mgr = torch.no_grad

    with grad_ctx_mgr():
        for it in range(cfg.num_iters):
            print(f"iteration {it}")
            # Initialize model on cpu and trn
            ut.nxd_init(tp_degree=1, ep_degree=1, seed=it)
            model_cpu = ut.initialize_neuron_model(cfg)
            ut.nxd_init(tp_degree=tp_degree, ep_degree=ep_degree, seed=it)
            model_trn = ut.initialize_neuron_model(cfg_trn)
            ut.match_expert_weights(model_trn, model_cpu, cfg.glu_mlp)

            if cfg.test_mode == "training":
                model_cpu.train()
                model_trn.train()
                # Set sinkhorn_iterations=0, because small precision errors can cause differences in routing decisions
                model_cpu.router.sinkhorn_iterations = 0
                model_trn.router.sinkhorn_iterations = 0
            else:
                model_cpu.eval()
                model_trn.eval()
            optimizer_cpu = ut.initialize_neuron_optimizer(model_cpu, grad_clipping=grad_clipping, override_grad_reduction=True, zero1=False, lr=lr)
            optimizer_trn = ut.initialize_neuron_optimizer(model_trn, grad_clipping=grad_clipping, zero1=cfg_trn.zero1, lr=lr)
            # Init NxD with tp_degree=1 and ep_degree=1, for running on cpu model
            ut.nxd_init(tp_degree=1, ep_degree=1, seed=it)

            # Initialize input, target, model on cpu
            if cfg.test_mode == "training":
                # Input is SBH in training
                ip_cpu = torch.randn(cfg.seq_len, cfg.batch_size * dp_size, cfg.hidden_size, dtype=cfg.dtype).detach()
                target_cpu = torch.randint(
                0, cfg.hidden_size - 1, (cfg.seq_len * cfg.batch_size * dp_size,), dtype=torch.long).detach()
            else:
                # Input is BSH in inference
                ip_cpu = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, dtype=cfg.dtype).detach()
                target_cpu = torch.randint(
                0, cfg.hidden_size - 1, (cfg.seq_len * cfg.batch_size,), dtype=torch.long).detach()
            ip_trn_full = ip_cpu.detach().to(device)
            ip_trn_full = shard_batch(ip_trn_full, cfg, dp_size, dp_rank, cfg.test_mode)

            # torch.topk behavior is different on cpu and device in the case of ties.
            # This causes mismatches in expert assignment for the TopK tests in bf16.
            if cfg.dtype == torch.bfloat16 and cfg.implementation == "topk":
                # Set is_test=True to return expert_index
                model_cpu.is_test = True
                model_trn.is_test = True

                # Simulate dropping of tokens in input where the expert assignments are not matching on cpu and device
                with torch.no_grad():
                    router_logits_cpu, expert_index_cpu = model_cpu(ip_cpu)[-2:]
                    ut.nxd_init(tp_degree=tp_degree, ep_degree=ep_degree, seed=it)
                    if sequence_parallel_enabled:
                        ip_trn = mappings.scatter_to_sequence_parallel_region(ip_trn_full)
                    else:
                        ip_trn = ip_trn_full
                    expert_index_trn = model_trn(ip_trn)[-1]
                    ut.nxd_init(tp_degree=1, ep_degree=1, seed=it)
                    local_ip_cpu = shard_batch(ip_cpu, cfg, dp_size, dp_rank, cfg.test_mode)
                    local_expert_index_cpu = shard_batch(expert_index_cpu, cfg, dp_size, dp_rank, cfg.test_mode)
                    local_router_logits_cpu = shard_batch(router_logits_cpu, cfg, dp_size, dp_rank, cfg.test_mode)
                    expert_mismatch_indices = set(torch.where(local_expert_index_cpu != expert_index_trn.cpu())[0].tolist())
                    if len(expert_mismatch_indices) > 0:
                        # Check that mismatches only happen when the (top_k+1) router logits are non-unique
                        for mismatch_idx in expert_mismatch_indices:
                            router_logits_idx = local_router_logits_cpu[mismatch_idx]
                            topk_logits, _ = torch.topk(router_logits_idx, min(cfg.top_k + 1, cfg.num_experts))
                            assert len(topk_logits) != len(torch.unique(topk_logits)), str(topk_logits)
                        # Update the input tensor to mask tokens where there is an expert assignment mismatch
                        # Modifying local_ip_cpu also modifies ip_cpu since they share underlying memory
                        local_ip_cpu = ut.drop_tokens_in_tensor(local_ip_cpu, expert_mismatch_indices)
                        ip_trn_full = ip_cpu.detach().to(device)
                        ip_trn_full = shard_batch(ip_trn_full, cfg, dp_size, dp_rank, cfg.test_mode)
                # Reset is_test
                model_cpu.is_test = False
                model_trn.is_test = False

            sharding_info = (tp_rank, tp_size, ep_rank, ep_size)

            # Get outputs and gradients from cpu
            op_cpu, loss_cpu, grad_dict_cpu = get_model_outputs(
                cfg, model_cpu, optimizer_cpu, ip_cpu, target_cpu, sequence_parallel_enabled, dp_size, serialize_dp=True
            )

            # Re-init NxD with actual TP degree
            ut.nxd_init(tp_degree=tp_degree, ep_degree=ep_degree, seed=it)

            # Get sharded input for rank (for sequence parallel)
            if sequence_parallel_enabled:
                ip_trn = mappings.scatter_to_sequence_parallel_region(ip_trn_full)
            else:
                ip_trn = ip_trn_full

            # Get outputs and gradients from trn, using the same input and target
            target_trn = target_cpu.clone().detach().to(device)

            # Data-parallel sharding
            target_trn = shard_batch(target_trn, cfg, dp_size, dp_rank, cfg.test_mode)

            op_trn, loss_trn, grad_dict_trn = get_model_outputs(
                cfg, model_trn, optimizer_trn, ip_trn, target_trn, sequence_parallel_enabled, dp_size, reduce_gradients=True
            )
            xm.mark_step()  # TRN enablement

            del ip_cpu, ip_trn_full, ip_trn, target_cpu, target_trn

            # Compare output
            if sequence_parallel_enabled:
                # Compare with only output shard belonging to the TP rank
                op_cpu = torch.tensor_split(op_cpu, tp_degree, dim=0)[tp_rank]

            if cfg.test_mode == "training":
                batch_dim = 1
                op_cpu = op_cpu.narrow(batch_dim, dp_rank*cfg.batch_size, cfg.batch_size)
            ut.check_tensors(op_cpu.detach(), op_trn.detach(), **output_tols, additional_msg=f"Iteration {it}")
            del op_cpu, op_trn

            # Compare loss
            ut.check_tensors(loss_cpu.detach(), loss_trn.detach(), **output_tols)
            del loss_cpu, loss_trn

            if not cfg_trn.zero1:
                # Check gradients on each rank
                _slice_and_compare_tensors(grad_dict_cpu, grad_dict_trn, sharding_info, it, **grad_tols)
                del grad_dict_cpu, grad_dict_trn
            else:
                # if zero1 is enabled then directly compare updated parameters, not the gradients may not match because the true gradients used is private in zero1 optimizer
                trn_parameters = {n: p for n, p in model_trn.named_parameters()}
                cpu_parameters = {n: p for n, p in model_cpu.named_parameters()}
                param_tols = {k: cfg_trn.lr * v for k, v in grad_tols.items()}
                _slice_and_compare_tensors(cpu_parameters, trn_parameters, sharding_info, it, **param_tols)
                del cpu_parameters, trn_parameters, grad_dict_cpu, grad_dict_trn

            optimizer_cpu.zero_grad(set_to_none=True)
            optimizer_trn.zero_grad(set_to_none=True)
            xm.mark_step()

    del model_cpu, model_trn
    gc.collect()
