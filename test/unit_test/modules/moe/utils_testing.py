import functools
import math
import random
import dataclasses
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch_neuronx
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from neuronx_distributed.modules.moe import (
    ACT2FN,
    ExpertMLPs,
    MoE,
    RouterSinkhorn,
    RouterTopK,
)
from neuronx_distributed.modules.moe.blockwise import BlockShardStrategy
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.model_utils import DEFAULT_BLOCK_SIZE
from neuronx_distributed.modules.moe.moe_configs import (
    BlockwiseMatmulConfig,
    MoEFusedTKGConfig,
    RoutedExpertsMLPOpsConfig,
)
from neuronx_distributed.modules.moe.moe_fused_tkg import MoEFusedTKG
from neuronx_distributed.modules.moe.shared_experts import SharedExperts
from neuronx_distributed.optimizer import NeuronEPZero1Optimizer, NeuronZero1Optimizer
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers import random as nxd_random
from neuronx_distributed.quantization.quantize import convert
from neuronx_distributed.quantization.quantization_config import get_default_expert_wise_per_channel_custom_qconfig_dict
from neuronx_distributed.trainer.optimizer import NxDOptimizer
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.utils.model_utils import get_platform_lnc, move_model_to_device
from neuronxcc.nki._private_kernels.blockwise_mm import SkipMode
from torch import nn
from torch.optim import SGD, Adam

logger = get_logger()

ALL_ACTIVATIONS = sorted(list(ACT2FN.keys()))

STATE_KEYS = {
    "_TENSOR_MODEL_PARALLEL_GROUP",
    "_TENSOR_MODEL_PARALLEL_GROUP_SPMD",
    "_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE",
    "_MPU_TENSOR_MODEL_PARALLEL_RANK",

    "_PIPELINE_MODEL_PARALLEL_GROUP",
    "_PIPELINE_MODEL_PARALLEL_GROUP_SPMD",
    "_PIPELINE_GLOBAL_RANKS",

    "_NEXT_RANK_GROUP",
    "_NEXT_RANK_GROUP_SPMD",
    "_PREV_RANK_GROUP",
    "_PREV_RANK_GROUP_SPMD",

    "_EXPERT_MODEL_PARALLEL_GROUP",
    "_EXPERT_MODEL_PARALLEL_GROUP_SPMD",
    "_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE",
    "_MPU_EXPERT_MODEL_PARALLEL_RANK",

    "_EXP_DATA_PARALLEL_GROUP",
    "_EXP_DATA_PARALLEL_GROUP_SPMD",

    "_DATA_PARALLEL_GROUP",
    "_DATA_PARALLEL_GROUP_SPMD",

    "_TOKEN_SHUFFLE_GROUP",
    "_TOKEN_SHUFFLE_GROUP_SPMD",
    "_TOKEN_SHUFFLE_GROUP_SIZE",

    "_KV_SHARED_GROUP",
    "_KV_SHARED_GROUP_SPMD",
    "_KV_SHARED_GROUP_SIZE",

    "PP_GROUP_PG_GLOO",
}

PARALLEL_STATE_MAP = {}


def print_rank0(s):
    if xr.global_ordinal() == 0:
        print(s)


def nxd_init(tp_degree, ep_degree, token_shuffle_group_size, seed):

    world_size = torch.distributed.get_world_size()
    parallel_state_key = f"{world_size}_{tp_degree}_{ep_degree}_{token_shuffle_group_size}"

    def _save_parallel_state(key):
        state = {}
        for attr in STATE_KEYS:
            if attr in parallel_state.__dict__:
                state[attr] = parallel_state.__dict__[attr]
            else:
                raise ValueError(f"Unknown key: {attr}")
        PARALLEL_STATE_MAP[key] = state

    def _load_parallel_state(key):
        for k, v in PARALLEL_STATE_MAP[key].items():
            if k in parallel_state.__dict__:
                parallel_state.__dict__[k] = v
            else:
                raise ValueError(f"Unknown key: {k}")

    if parallel_state_key in PARALLEL_STATE_MAP:
        _load_parallel_state(parallel_state_key)
    else:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_token_shuffle_group()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_degree,
            expert_model_parallel_size=ep_degree,
            pipeline_model_parallel_size=1,
            lnc_size=get_platform_lnc(),
        )
        parallel_state.initialize_token_shuffle_group(token_shuffle_group_size)
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
    capacity_factor: Union[float|None]
    dtype: torch.dtype
    glu_mlp: bool
    test_mode: str  # Either 'training' or 'inference'
    implementation: str  # "sbase" or "topk"
    test_module: str = None
    intermediate_size: int = None
    hidden_act: str = "silu"  # One of ACT2FN
    device: str = "cpu"
    top_k: int = 1
    block_size: int = DEFAULT_BLOCK_SIZE
    zero1: bool = False
    sequence_parallel_enabled: bool = False
    num_iters: int = 10
    lr: float = 0.1
    use_torch_block_wise: bool = False
    normalize_top_k_affinities: bool = False
    parallelize_token_to_block_mapping: bool = True
    use_expert_mlps_v2: bool = False
    skip_dma: SkipMode = SkipMode(False, False)
    use_block_parallel: bool=False
    early_expert_affinity_modulation: bool = False
    block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.HI_LO
    enable_spmd_rank: bool = False
    always_augment_inputs_for_blockwise_matmul: bool = False
    optimized_block_to_token_mapping: bool = True
    num_shared_experts: int = 0
    fused_gate_up_shared_expert: bool = False
    shared_experts_sequence_parallel_enabled: bool = False
    rms_norm_eps: int = 1e-5
    moe_fused_tkg_enabled: bool = False
    quantized: bool = False
    moe_fused_tkg_kernel_enabled: Optional[bool] = None
    router_topk_kernel_enabled: Optional[bool] = None
    expert_mlp_kernel_enabled: Optional[bool] = None
    shared_mlp_kernel_enabled: Optional[bool] = None
    stack_size: int = 1

def get_random_activations(num, seed=None):
    if seed is not None:
        random.seed(seed)
    return random.sample(ALL_ACTIVATIONS, num)

class LlamaRMSNormV2(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Critical difference with LlamaRMSNorm: We multiply in full precision and then convert
        # to the target data type instead of converting hidden_states to the target data type and
        # then multiplying in full precision.
        output = self.weight * hidden_states
        return output.to(input_dtype)

class StackedModel(torch.nn.Module):
    def __init__(self, stack_size, cfg, return_router_logits):
        super().__init__()
        self.cfg = cfg
        self.stacks = torch.nn.ModuleList()
        self.return_router_logits = return_router_logits

        cfg_one_layer = dataclasses.replace(cfg, stack_size=1)
        for i in range(stack_size):
            neuron_model = initialize_neuron_model(cfg_one_layer, seed=i, move_to_device=False)
            self.stacks.append(neuron_model)

    def forward(self, hidden_states):
        ip = hidden_states
        router_logits_list = []
        for layer in self.stacks:
            op = layer(ip)[0]
            if self.return_router_logits:
                router_logits = layer(ip)[1]
                router_logits_list.append(router_logits)
            ip = op

        return_op = (op,)
        if self.return_router_logits:
            all_router_logits = torch.cat(router_logits_list, dim=0)
            return_op += (all_router_logits,)

        return return_op


def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (testcase_func.__name__, "_".join(str(x) for x in param.args))


def check_tensors(t1: torch.Tensor, t2: torch.Tensor, atol, rtol, additional_msg=None):
    def msg(s):
        return "%s\n%s" % (s, additional_msg) if additional_msg is not None else s
    try:
        t1 = t1.detach().cpu()
        t2 = t2.detach().cpu()
        torch_neuronx.testing.assert_close(t1, t2, atol=atol, rtol=rtol)
    except AssertionError as e:
        logger.error(parallel_state.rmsg("tensor comparison failed"))
        t1 = t1.detach().cpu()
        t2 = t2.detach().cpu()
        logger.info(parallel_state.rmsg(f"{t1=}"))
        logger.info(parallel_state.rmsg(f"{t2=}"))
        # Compute the absolute difference
        abs_diff = torch.abs(t1 - t2)

        # Compute the denominator for relative difference
        # Use torch.where to handle zeros in 'b'
        epsilon = 1e-12  # Small value to avoid division by zero
        denominator = torch.where(t2 != 0, torch.abs(t2), epsilon)

        # Compute the relative difference
        relative_diff = abs_diff / denominator

        # Unravel the index to get the multi-dimensional indices
        max_abs_idx_unravel = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
        max_rel_idx_unravel = np.unravel_index(relative_diff.argmax(), relative_diff.shape)
        # Print the results
        logger.info(parallel_state.rmsg(f"{abs_diff.max()=} {t1[max_abs_idx_unravel]=} {t2[max_abs_idx_unravel]=}"))
        logger.info(parallel_state.rmsg(f"{relative_diff.max()=} {t1[max_rel_idx_unravel]=} {t2[max_rel_idx_unravel]=}"))
        raise e


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


def match_expert_weights(model_trn, model_cpu, glu_mlp, moe_fused_tkg_enabled, run_shared_expert_in_sp=False):
    """
    Copy expert weights from the CPU model to the TRN model. This is necessary
    under expert parallelism because NxD weight initialization currently does not
    take expert parallelism into account.
    """

    if moe_fused_tkg_enabled:
        module = model_cpu.moe.expert_mlps.mlp_op.gate_up_proj if glu_mlp else model_cpu.moe.expert_mlps.mlp_op.up_proj
    else:
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

    # Record trn params that does not exist in cpu params
    missing_keys = []
    with torch.no_grad():
        cpu_params_dict = dict(model_cpu.named_parameters())
        for trn_name, trn_param in model_trn.named_parameters():
            if trn_name not in cpu_params_dict:
                missing_keys.append(trn_name)
                continue
            # Find corresponding parameter in CPU model
            cpu_param = cpu_params_dict[trn_name]
            if "shared_experts" in trn_name and run_shared_expert_in_sp:
                 trn_param.copy_(cpu_param.contiguous())

            elif "expert_mlp" in trn_name:
                if "gate_up_proj" in trn_name:
                    _, input_size, output_size = cpu_param.shape
                    stride = 2 if glu_mlp else 1
                    local_output_size = output_size // tp_degree // stride
                    single_output_size = output_size // stride
                    weight_slice = cpu_param.narrow(expert_dim, num_local_experts * ep_rank, num_local_experts)
                    gate_weight_slice = weight_slice.narrow(output_dim, local_output_size * tp_rank, local_output_size)
                    up_weight_slice = weight_slice.narrow(
                        output_dim, local_output_size * tp_rank + single_output_size, local_output_size
                    )
                    gate_up_weight_slice = torch.cat((gate_weight_slice, up_weight_slice), dim=output_dim)
                    trn_param.copy_(gate_up_weight_slice.contiguous())

                elif "up_proj" in trn_name:
                    _, input_size, output_size = cpu_param.shape
                    stride = 1
                    local_output_size = output_size // tp_degree // stride
                    weight_slice = cpu_param.narrow(expert_dim, num_local_experts * ep_rank, num_local_experts)
                    up_weight_slice = weight_slice.narrow(output_dim, local_output_size * tp_rank, local_output_size)
                    trn_param.copy_(up_weight_slice.contiguous())

                elif "gate_proj" in trn_name:
                    _, input_size, output_size = cpu_param.shape
                    stride = 1
                    local_output_size = output_size // tp_degree // stride
                    weight_slice = cpu_param.narrow(expert_dim, num_local_experts * ep_rank, num_local_experts)
                    up_weight_slice = weight_slice.narrow(output_dim, local_output_size * tp_rank, local_output_size)

                elif "down_proj" in trn_name:
                    _, input_size, output_size = cpu_param.shape
                    local_input_size = input_size // tp_degree
                    weight_slice = cpu_param.narrow(expert_dim, num_local_experts * ep_rank, num_local_experts)
                    weight_slice = weight_slice.narrow(input_dim, local_input_size * tp_rank, local_input_size)
                    trn_param.copy_(weight_slice.contiguous())

        logger.info(parallel_state.rmsg(f"missing key: {missing_keys}"))
        global_rank = torch.distributed.get_rank()
        spmd_rank = torch.tensor([global_rank], dtype=torch.int32)

        # local_indices is only computed when EP is enabled
        if ep_degree > 1:
            curr_expert_rank = parallel_state.get_expert_parallel_rank_from_global_rank(
                rank=global_rank, expert_parallel_group=parallel_state.get_expert_model_parallel_group())
            curr_expert_indices = parallel_state.get_experts_for_expert_parallel_rank(
                curr_expert_rank,
                total_number_of_experts=num_experts,
                expert_model_parallel_size=ep_degree,
            )
            local_indices = torch.tensor([curr_expert_indices], dtype=torch.int32)

        for trn_name, trn_param in model_trn.named_parameters():
            if trn_name.endswith("spmd_rank.rank"):
                trn_param.copy_(spmd_rank.contiguous())

            if trn_name.endswith("experts.spmd_rank.rank"):
                trn_param.copy_(spmd_rank.contiguous())

            if trn_name.endswith("expert_mlps.spmd_rank.local_expert_indices") and ep_degree > 1:
                trn_param.copy_(local_indices.contiguous())

    xm.mark_step()

def token_shuffle_single_core(hidden_states: torch.Tensor, cfg, dp_size, permutation_index=None):
    """
    CPU implementation of token shuffle/unshuffle. Use single CPU to simulate a distributed backend.
    permutation_index:
        None means permute, and return permutation_index; otherwise, unpermute according to permutation_index.
    """
    sp_degree = cfg.tp_degree
    sharded_seq_len = hidden_states.shape[0] // sp_degree
    hidden_size = hidden_states.shape[-1]
    hidden_states = hidden_states.reshape(sp_degree, sharded_seq_len, dp_size, cfg.batch_size, hidden_size)
    hidden_states = hidden_states.transpose(1, 2)

    # permute
    if permutation_index is None:
        hidden_states = hidden_states.reshape(sp_degree, dp_size, sharded_seq_len * cfg.batch_size, hidden_size)

        seed = 42
        # add mark_step() to cut graph because birsim cannot verify rand()/argsort() op
        xm.mark_step()
        torch.manual_seed(seed)
        xm.set_rng_state(seed)
        new_permutation_index = torch.argsort(torch.rand(sharded_seq_len * cfg.batch_size, device=xm.xla_device()))
        xm.mark_step()
        new_permutation_index = new_permutation_index.cpu()
        for s in range(sp_degree):
            for d in range(dp_size):
                hidden_states[s, d, :, :] = hidden_states[s, d, new_permutation_index, :]

    # all to all
    hidden_states = hidden_states.reshape(
        sp_degree, dp_size // cfg.ep_degree, cfg.ep_degree, sharded_seq_len * cfg.batch_size, hidden_size
    )
    shuffled_hidden_states = torch.zeros_like(hidden_states)
    chunk_size = sharded_seq_len * cfg.batch_size // cfg.ep_degree
    for s in range(sp_degree):
        for i in range(dp_size // cfg.ep_degree):
            for expert_idx in range(cfg.ep_degree):
                for chunk_idx in range(cfg.ep_degree):
                    shuffled_hidden_states[
                        s, i, expert_idx, chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size, :
                    ] = hidden_states[s, i, chunk_idx, expert_idx * chunk_size : (expert_idx + 1) * chunk_size, :]

    # unpermute
    if permutation_index is not None:
        shuffled_hidden_states = shuffled_hidden_states.reshape(
            sp_degree, dp_size, sharded_seq_len * cfg.batch_size, hidden_size
        )
        unpermuted_shuffled_hidden_states = shuffled_hidden_states.clone()
        for s in range(sp_degree):
            for d in range(dp_size):
                unpermuted_shuffled_hidden_states[s, d, permutation_index, :] = shuffled_hidden_states[s, d, :, :]
        shuffled_hidden_states = unpermuted_shuffled_hidden_states

    shuffled_hidden_states = shuffled_hidden_states.reshape(
        sp_degree, dp_size, sharded_seq_len, cfg.batch_size, hidden_size
    )
    shuffled_hidden_states = shuffled_hidden_states.transpose(1, 2)
    shuffled_hidden_states = shuffled_hidden_states.reshape(
        sp_degree * sharded_seq_len, dp_size * cfg.batch_size, hidden_size
    )

    if permutation_index is None:
        return shuffled_hidden_states, new_permutation_index
    return shuffled_hidden_states


def initialize_neuron_optimizer(
    model,
    override_grad_reduction=True,
    sequence_parallel=False,
    grad_clipping=False,
    zero1=False,
    lr=0.0,
    optimizer=None,
):
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
            grad_norm_groups=parallel_state.get_tensor_model_parallel_replica_groups(),
            max_norm=1.0,
            lr=lr,
        )

    else:
        optimizer = base_optimizer_cls(model.parameters(), lr=lr)

    nxd_opt = NxDOptimizer(optimizer, nxd_config)
    if override_grad_reduction:
        nxd_opt._fetch_gradients = functools.partial(dummy_fetch_grads, self=nxd_opt)
    return nxd_opt

def get_router_act_fn(cfg: ExptCfg):
    if cfg.implementation == "topk":
        return "softmax"
    if cfg.implementation == "sbase" or cfg.implementation == "llama4":
        return "sigmoid"
    raise AssertionError(f"Unknown supported implementation for get router activation function: {cfg.implementation}")

def initialize_neuron_model(cfg, seed=0, move_to_device=True):
    """
    Create a Neuron model, as specified in the config.
    """

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    return_router_logits = getattr(cfg, "return_router_logits", True)
    return_expert_index = getattr(cfg, "return_expert_index", False)

    if cfg.sequence_parallel_enabled:
        assert cfg.seq_len > 1, "SP cannot be enabled for token-gen"

    if cfg.test_mode == "training" and cfg.sequence_parallel_enabled:
        sequence_dimension = 0  # SBH
        # Disable router in SP for training
        router_sequence_parallel_enabled = False
    else:
        sequence_dimension = 1  # BSH
        router_sequence_parallel_enabled = cfg.sequence_parallel_enabled

    # Initialize router
    router_args = dict(
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        hidden_size=cfg.hidden_size,
        sequence_parallel_enabled=router_sequence_parallel_enabled,
        sequence_dimension=sequence_dimension,
        dtype=cfg.dtype,
        act_fn=get_router_act_fn(cfg),
        device=torch.device("cpu"),
    )
    if cfg.implementation == "sbase":
        router = RouterSinkhorn(**router_args)
    elif cfg.implementation == "topk" or cfg.implementation == "llama4":
        router = RouterTopK(**router_args)
    else:
        raise AssertionError(f"Unknown implementation: {cfg.implementation}")

    expert_mlps = init_expert_mlps(cfg)
    shared_experts = init_shared_experts(cfg)
    # Initialize model
    neuron_model = MoE(
        router=router,
        expert_mlps=expert_mlps,
        return_router_logits=return_router_logits,
        return_expert_index=return_expert_index,
        sequence_parallel_enabled=cfg.sequence_parallel_enabled,
        sequence_dimension=sequence_dimension,
        shared_experts=shared_experts,
        token_shuffle_group_size=parallel_state.get_token_shuffle_group_size(),
        token_shuffle_seed=42,
    )

    if cfg.moe_fused_tkg_enabled is True:
        moe = neuron_model
        post_attention_layernorm = LlamaRMSNormV2(hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps)
        moe_fused_tkg_config = MoEFusedTKGConfig(
            quantized=cfg.quantized,
            moe_fused_kernel_enabled=cfg.moe_fused_tkg_kernel_enabled,
            router_topk_kernel_enabled=cfg.router_topk_kernel_enabled,
            expert_mlp_kernel_enabled=cfg.expert_mlp_kernel_enabled,
            shared_mlp_kernel_enabled=cfg.shared_mlp_kernel_enabled,
        )
        neuron_model = MoEFusedTKG(
            config=moe_fused_tkg_config,
            post_attention_layernorm=post_attention_layernorm,
            moe=moe,
            return_expert_index=return_expert_index,
            return_router_logits=return_router_logits,
        )

    stack_size = getattr(cfg, "stack_size", 1)
    if stack_size > 1:
        neuron_model = StackedModel(stack_size=stack_size, cfg=cfg, return_router_logits=return_router_logits)

    # Move model to required device
    if move_to_device:
        move_model_to_device(neuron_model, cfg.device)

    return neuron_model


def init_expert_mlps(cfg:ExptCfg):

    # Initialize ExpertMLPs
    if not cfg.use_expert_mlps_v2:
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
            use_torch_block_wise=cfg.use_torch_block_wise,
            device=torch.device("cpu"),
            logical_nc_config=get_platform_lnc(),
            parallelize_token_to_block_mapping=cfg.parallelize_token_to_block_mapping,
            skip_dma_token=cfg.skip_dma.skip_token,
            skip_dma_weight=cfg.skip_dma.skip_weight,
            use_block_parallel=cfg.use_block_parallel,
            early_expert_affinity_modulation=cfg.early_expert_affinity_modulation,
            block_sharding_strategy=cfg.block_sharding_strategy,
            enable_spmd_rank=cfg.enable_spmd_rank,
        )

        if cfg.implementation == "topk":
            # Hardcode normalize_top_k_affinities=False and then overwrite it later (Workaround for when testing top_k == 1)
            expert_mlps_args.update({"normalize_top_k_affinities": False})
        expert_mlps = ExpertMLPs(**expert_mlps_args)
    else:
        routed_experts_mlp_config = RoutedExpertsMLPOpsConfig(
            num_experts=cfg.num_experts,
            hidden_size=cfg.hidden_size,
            intermediate_size=get_intermediate_size(cfg),
            glu_mlp=cfg.glu_mlp,
            normalize_top_k_affinities=cfg.normalize_top_k_affinities,
            early_expert_affinity_modulation=cfg.early_expert_affinity_modulation,
            input_layer_init_method=torch.nn.init.kaiming_uniform_,
            output_layer_init_method=torch.nn.init.kaiming_uniform_,
            capacity_factor=cfg.capacity_factor,
            top_k=cfg.top_k,
            hidden_act=cfg.hidden_act,
            enable_spmd_rank=cfg.enable_spmd_rank,
        )

        blockwise_matmul_config = BlockwiseMatmulConfig.from_kwargs(
            logical_nc_config=get_platform_lnc(),
            block_size=cfg.block_size,
            use_torch_block_wise=cfg.use_torch_block_wise,
            blockwise_nki_autograd_cls=None,
            parallelize_token_to_block_mapping=cfg.parallelize_token_to_block_mapping,
            early_expert_affinity_modulation=cfg.early_expert_affinity_modulation,
            optimized_block_to_token_mapping=cfg.optimized_block_to_token_mapping,
            use_block_parallel=cfg.use_block_parallel,
            always_augment_inputs_for_blockwise_matmul=cfg.always_augment_inputs_for_blockwise_matmul,
            block_sharding_strategy=cfg.block_sharding_strategy,
            skip_dma_token=cfg.skip_dma.skip_token,
            skip_dma_weight=cfg.skip_dma.skip_weight,
        )

        expert_mlps = ExpertMLPsV2(
            routed_experts_mlp_config=routed_experts_mlp_config,
            blockwise_matmul_config=blockwise_matmul_config,
            dtype=cfg.dtype,
            return_bias=False,
        )

    if cfg.quantized:
        q_config = get_default_expert_wise_per_channel_custom_qconfig_dict()
        expert_mlps = convert(expert_mlps, q_config=q_config, inplace=True)

    # Workaround for when testing top_k=1 with topk
    if cfg.implementation == "topk":
        expert_mlps.routed_experts_mlp_config.normalize_top_k_affinities = True

    return expert_mlps

def init_shared_experts(cfg:ExptCfg):
    if cfg.num_shared_experts == 0:
        return None
    if cfg.moe_fused_tkg_enabled:
        transpose_weights = cfg.moe_fused_tkg_kernel_enabled or (
            cfg.moe_fused_tkg_kernel_enabled is None and cfg.device == "xla"
        ) or (cfg.moe_fused_tkg_kernel_enabled is False and cfg.shared_mlp_kernel_enabled is None and cfg.device == "xla")
    else:
        transpose_weights = False
    return SharedExperts(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_shared_experts=cfg.num_shared_experts,
        fused_gate_up_projection=cfg.fused_gate_up_shared_expert,
        hidden_act=cfg.hidden_act,
        dtype=cfg.dtype,
        sequence_parallel_enabled=cfg.shared_experts_sequence_parallel_enabled,
        transpose_weights=transpose_weights,
    )
