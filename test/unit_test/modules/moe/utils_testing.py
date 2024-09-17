import math
import random
from dataclasses import dataclass
import functools

import torch
from torch.optim import Adam, SGD

from neuronx_distributed.optimizer import NeuronZero1Optimizer, NeuronEPZero1Optimizer
from neuronx_distributed.modules.moe import (
    ACT2FN,
    ExpertMLPs,
    MoE,
    RouterSinkhorn,
    RouterTopK,
)
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers import random as nxd_random
from neuronx_distributed.utils.model_utils import move_model_to_device
from neuronx_distributed.trainer.optimizer import NxDOptimizer
import torch_xla.core.xla_model as xm

ALL_ACTIVATIONS = sorted(list(ACT2FN.keys()))

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
    "_EXPERT_MODEL_PARALLEL_GROUP",
    "_EXPERT_MODEL_PARALLEL_GROUP_SPMD",
    "_EXP_DATA_PARALLEL_GROUP",
    "_EXP_DATA_PARALLEL_GROUP_SPMD",
    "_DATA_PARALLEL_GROUP",
    "_DATA_PARALLEL_GROUP_SPMD",
    "_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE",
    "_MPU_TENSOR_MODEL_PARALLEL_RANK",
}

PARALLEL_STATE_MAP = {}

def nxd_init(tp_degree, ep_degree, seed):

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
            expert_model_parallel_size=ep_degree,
            pipeline_model_parallel_size=1,
        )
        _save_parallel_state(parallel_state_key)

    # Set seed
    nxd_random.model_parallel_xla_manual_seed(seed)


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
    test_mode: str  # Either 'training' or 'inference'
    implementation: str  # "sbase" or "topk"
    intermediate_size: int = None
    hidden_act: str = "silu"  # One of ACT2FN
    device: str = "cpu"
    top_k: int = 1
    zero1: bool = False
    sequence_parallel_enabled: bool = False
    num_iters: int = 10
    lr: float = 0.1


def get_random_activations(num, seed=None):
    if seed is not None:
        random.seed(seed)
    return random.sample(ALL_ACTIVATIONS, num)


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
    def msg(s):
        return "%s\n%s" % (s, additional_msg) if additional_msg is not None else s
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
    total_tokens = cfg.seq_len * cfg.batch_size
    if cfg.capacity_factor is None:
        return total_tokens
    else:
        expert_capacity = math.ceil(total_tokens * cfg.capacity_factor / cfg.num_experts)
        expert_capacity = min(expert_capacity, total_tokens)
        return expert_capacity


def get_intermediate_size(cfg):
    if hasattr(cfg, "intermediate_size") and cfg.intermediate_size:
        intermediate_size = cfg.intermediate_size
    elif cfg.glu_mlp:
        intermediate_size = int(8 / 3 * cfg.hidden_size)
    else:
        intermediate_size = int(4 * cfg.hidden_size)
    return intermediate_size


def match_expert_weights(model_trn, model_cpu, glu_mlp):
    """
    Copy expert weights from the CPU model to the TRN model. This is necessary
    under expert parallelism because NxD weight initialization currently does not
    take expert parallelism into account.
    """

    module = model_cpu.expert_mlps.mlp_op.gate_up_proj if glu_mlp else model_cpu.expert_mlps.mlp_op.up_proj
    num_experts = module.weight.shape[0]
    ep_degree = parallel_state.get_expert_model_parallel_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    tp_degree = parallel_state.get_tensor_model_parallel_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    num_local_experts = num_experts // ep_degree
    expert_dim = 0
    input_dim = 1
    output_dim = 2

    with torch.no_grad():
        for (cpu_name, cpu_param), (trn_name, trn_param) in zip(model_cpu.named_parameters(), model_trn.named_parameters()):
            if "gate_up_proj" in cpu_name:
                _, input_size, output_size = cpu_param.shape
                stride = 2 if glu_mlp else 1
                local_output_size = output_size // tp_degree // stride
                single_output_size = output_size // stride
                weight_slice = cpu_param.narrow(expert_dim, num_local_experts * ep_rank, num_local_experts)
                gate_weight_slice = weight_slice.narrow(output_dim, local_output_size * tp_rank, local_output_size)
                up_weight_slice = weight_slice.narrow(output_dim, local_output_size * tp_rank + single_output_size, local_output_size)
                gate_up_weight_slice = torch.cat((gate_weight_slice, up_weight_slice), dim=output_dim)
                trn_param.copy_(gate_up_weight_slice.contiguous())
            elif "up_proj" in cpu_name:
                _, input_size, output_size = cpu_param.shape
                stride = 1
                local_output_size = output_size // tp_degree // stride
                weight_slice = cpu_param.narrow(expert_dim, num_local_experts * ep_rank, num_local_experts)
                up_weight_slice = weight_slice.narrow(output_dim, local_output_size * tp_rank, local_output_size)
                trn_param.copy_(up_weight_slice.contiguous())
            elif "down_proj" in cpu_name:
                _, input_size, output_size = cpu_param.shape
                local_input_size = input_size // tp_degree
                weight_slice = cpu_param.narrow(expert_dim, num_local_experts * ep_rank, num_local_experts)
                weight_slice = weight_slice.narrow(input_dim, local_input_size * tp_rank, local_input_size)
                trn_param.copy_(weight_slice.contiguous())

    xm.mark_step()

def initialize_neuron_optimizer(model, override_grad_reduction=True, sequence_parallel=False, grad_clipping=False, zero1=False, lr=0.0, optimizer=None):
    optimizer_config = {"zero_one_enabled": zero1, "grad_clipping": grad_clipping}

    if not zero1:
        optimizer_config["max_grad_norm"] = 1.0

    # MoE parameters are not in sequence parallel
    nxd_config = {"optimizer_config": optimizer_config, "sequence_parallel": sequence_parallel}

    def dummy_fetch_grads(self, *args, **kwargs):
        return [], []

    base_optimizer_cls = Adam if optimizer == "adam" else SGD

    if zero1:
        ep_enabled = parallel_state.get_expert_model_parallel_size() > 1
        zero1_optimizer_cls = NeuronEPZero1Optimizer if ep_enabled else NeuronZero1Optimizer

        optimizer = zero1_optimizer_cls(
            [p for p in model.parameters()],
            base_optimizer_cls,
            grad_clipping=optimizer_config["grad_clipping"],
            pin_layout=False,
            grad_norm_groups=parallel_state.get_tensor_model_parallel_group(as_list=True),
            max_norm=1.0,
            lr=lr,
        )

    else:
        optimizer = base_optimizer_cls(model.parameters(), lr=lr)

    nxd_opt = NxDOptimizer(optimizer, nxd_config)
    if override_grad_reduction:
        nxd_opt._fetch_gradients = functools.partial(dummy_fetch_grads, self=nxd_opt)
    return nxd_opt


def initialize_neuron_model(cfg, seed=0):
    """
    Create a Neuron model, as specified in the config.
    """

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Initialize router
    router_args = dict(
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        hidden_size=cfg.hidden_size,
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
        dtype=cfg.dtype,
        device=torch.device("cpu"),
    )

    if cfg.implementation == "topk":
        # Hardcode normalize_top_k_affinities=False and then overwrite it later (Workaround for when testing top_k == 1)
        expert_mlps_args.update({"normalize_top_k_affinities": False})

    # Initialize ExpertMLPs
    expert_mlps = ExpertMLPs(**expert_mlps_args)
    # Workaround for when testing top_k=1 with topk
    if cfg.implementation == "topk":
        expert_mlps.normalize_top_k_affinities = True
    # Enable selective loading for unit tests
    expert_mlps.SELECTIVE_LOADING_THRESHOLD = 1.0

    # Initialize model
    neuron_model = MoE(
        router=router,
        expert_mlps=expert_mlps,
        # Always return router logits in testing
        return_router_logits=True,
        sequence_parallel_enabled=cfg.sequence_parallel_enabled,
    )

    # Move model to required device
    move_model_to_device(neuron_model, cfg.device)

    return neuron_model
