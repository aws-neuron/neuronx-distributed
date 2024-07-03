import math
import random
from dataclasses import dataclass

import torch

from neuronx_distributed.modules.moe import (
    ACT2FN,
    ExpertMLPsCapacityFactor,
    MoE,
    MoESequenceParallelMode,
    RouterSinkhorn,
    RouterTopK,
)

ALL_ACTIVATIONS = sorted(list(ACT2FN.keys()))


@dataclass
class ExptCfg:
    # Class which encapsulates the experiment config parameters
    seq_len: int
    batch_size: int
    hidden_size: int
    num_experts: int
    capacity_factor: float
    dtype: torch.dtype
    glu_mlp: bool
    expert_mlps_permute_strategy: str  # Either 'matmul' or 'index'
    implementation: str  # "sbase" or "topk"
    intermediate_size: int = None
    hidden_act: str = "silu"  # One of ACT2FN
    device: str = "cpu"
    top_k: int = 1


@dataclass
class ExptCfgCorrectness(ExptCfg):
    num_iters: int = 10


def get_random_activations(num, seed=None):
    if seed is not None:
        random.seed(seed)
    return random.sample(ALL_ACTIVATIONS, num)


def filter_valid_expt_configs(expt_configs):
    valid_expt_configs = []

    for cfg in expt_configs:
        # OPTIMIZED_SP_MATMUL mode does not apply to the the 'index' permute strategy
        sequence_parallel_mode = MoESequenceParallelMode[getattr(cfg, "sequence_parallel_mode", MoESequenceParallelMode.NO_SP)]
        if (
            cfg.expert_mlps_permute_strategy == "index"
            and sequence_parallel_mode == MoESequenceParallelMode.OPTIMIZED_SP_MATMUL
        ):
            continue

        # OPTIMIZED_SP_MATMUL is not supported for inference due to SPMD restriction
        test_mode = getattr(cfg, "test_mode", "training")
        if test_mode == "inference" and sequence_parallel_mode == MoESequenceParallelMode.OPTIMIZED_SP_MATMUL:
            continue

        valid_expt_configs.append(cfg)

    return valid_expt_configs


class StackedModel(torch.nn.Module):
    def __init__(self, stack_size, cfg, return_router_logits):
        super().__init__()
        self.cfg = cfg
        self.stacks = torch.nn.ModuleList()
        self.return_router_logits = return_router_logits

        for i in range(stack_size):
            neuron_model = initialize_neuron_model(cfg, seed=i)
            self.stacks.append(neuron_model)

    def forward(self, hidden_states):
        ip = hidden_states
        router_logits_list = []
        for layer in self.stacks:
            op, router_logits = layer(ip)
            if self.return_router_logits:
                router_logits_list.append(router_logits)
            ip = op

        if self.return_router_logits:
            all_router_logits = torch.cat(router_logits_list, dim=0)
            return op, all_router_logits
        else:
            return op


def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (testcase_func.__name__, "_".join(str(x) for x in param.args))


def check_tensors(t1, t2, atol, rtol, additional_msg=None):
    msg = lambda s: "%s\n%s" % (s, additional_msg) if additional_msg is not None else s
    torch.testing.assert_close(t1, t2, atol=atol, rtol=rtol, msg=msg, check_device=False, check_dtype=True)


def drop_tokens_in_tensor(tensor, dropped_token_indices):
    # Simulate the dropping of tokens at the given indices in the given tensor (shape: (S,B,H))
    # dropped_token_indices contains indices corresponding to the token dimension (T), which is obtained by flattening S and B
    batch_size = tensor.shape[1]
    for dropped_token_idx in dropped_token_indices:
        # Convert token_idx to the (S, B) dimensions
        seq_idx, batch_idx = dropped_token_idx // batch_size, dropped_token_idx % batch_size
        tensor[seq_idx, batch_idx, :] = 0
    return tensor


def get_model_grads_dict(model):
    return {param_name: param.grad for param_name, param in model.named_parameters()}


def get_expert_capacity(cfg):
    expert_capacity = math.ceil(cfg.seq_len * cfg.batch_size * cfg.capacity_factor / cfg.num_experts)
    expert_capacity = min(expert_capacity, cfg.seq_len * cfg.batch_size)
    return expert_capacity


def get_intermediate_size(cfg):
    if hasattr(cfg, "intermediate_size") and cfg.intermediate_size:
        intermediate_size = cfg.intermediate_size
    elif cfg.glu_mlp:
        intermediate_size = int(8 / 3 * cfg.hidden_size)
    else:
        intermediate_size = int(4 * cfg.hidden_size)
    return intermediate_size


def initialize_neuron_model(cfg, seed=0):
    """
    Create a Neuron model, as specified in the config.
    """

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    sequence_parallel_mode = getattr(cfg, "sequence_parallel_mode", MoESequenceParallelMode.NO_SP)

    # Initialize router
    router_args = dict(
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        hidden_size=cfg.hidden_size,
        sequence_parallel_mode=sequence_parallel_mode,
        dtype=cfg.dtype,
        device=torch.device("cpu"),
    )
    if cfg.implementation == "sbase":
        router = RouterSinkhorn(**router_args)
    elif cfg.implementation == "topk":
        router = RouterTopK(**router_args)
    else:
        raise AssertionError(f"Unknown implementation: {cfg.implementation}")

    # Initialize expert_mlps
    expert_mlps_args = dict(
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        hidden_size=cfg.hidden_size,
        intermediate_size=get_intermediate_size(cfg),
        hidden_act=cfg.hidden_act,
        capacity_factor=cfg.capacity_factor,
        init_method=torch.nn.init.kaiming_uniform_,
        output_layer_init_method=torch.nn.init.kaiming_uniform_,
        glu_mlp=cfg.glu_mlp,
        sequence_parallel_mode=sequence_parallel_mode,
        permute_strategy=cfg.expert_mlps_permute_strategy,
        dtype=cfg.dtype,
        device=torch.device("cpu"),
    )

    if cfg.implementation == "topk":
        # Hardcode normalize_top_k_affinities=False and then overwrite it later (Workaround for when testing top_k == 1)
        expert_mlps_args.update({"normalize_top_k_affinities": False})

    # Initialize ExpertMLPs
    expert_mlps = ExpertMLPsCapacityFactor(**expert_mlps_args)
    # Workaround for when testing top_k=1 with topk
    if cfg.implementation == "topk":
        expert_mlps.normalize_top_k_affinities = True

    # Initialize model
    neuron_model = MoE(
        router=router,
        expert_mlps=expert_mlps,
        # Always return router logits in testing
        return_router_logits=True,
        sequence_parallel_mode=sequence_parallel_mode,
    )

    # Move model to required device
    neuron_model = neuron_model.to(device=cfg.device)

    return neuron_model
