import logging
import importlib
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import torch
from torch.distributed import ProcessGroup

from neuronx_distributed.modules.moe.model_utils import (
    ACTFunc,
    DEFAULT_HIDDEN_ACT_SCALING_FACTOR,
    DEFAULT_LNC_SIZE,
    DEFAULT_SELECTIVE_LOADING_THRESHOLD,
    get_kernel_activation_func_id,
    GLUType,
)
from neuronx_distributed.modules.moe.moe_configs import MoEFusedTKGConfig
from neuronx_distributed.parallel_layers import mappings, parallel_state, utils
from neuronx_distributed.modules.moe.routing import RouterBase
from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPsV2
from neuronx_distributed.modules.moe.shared_experts import SharedExperts


def _import_beta2_moe():
    """Import Beta 2 kernels from nkilib and return registry dict."""
    beta2_paths = [
        ("nkilib.core.moe_block.moe_block_tkg", "nkilib.core.utils.common_types"),
        ("nkilib_src.nkilib.core.moe_block.moe_block_tkg", "nkilib_src.nkilib.core.utils.common_types"),
    ]

    for kernel_path, types_path in beta2_paths:
        try:
            kernel_module = importlib.import_module(kernel_path)
            types_module = importlib.import_module(types_path)
            import nki.language as nl

            return {
                "nl": nl,
                "moe_block_tkg": getattr(kernel_module, "moe_block_tkg"),
                "RouterActFnType": types_module.RouterActFnType,
                "ActFnType": types_module.ActFnType,
                "ExpertAffinityScaleMode": types_module.ExpertAffinityScaleMode,
            }
        except (ImportError, AttributeError):
            continue

    raise ImportError("Beta 2 NKI kernels not available. Ensure nkilib is installed.")

_registry = _import_beta2_moe()
nl = _registry["nl"]
moe_block_tkg_kernel = _registry["moe_block_tkg"]

ExpertAffinityScaleMode = _registry["ExpertAffinityScaleMode"]
ActFnType = _registry["ActFnType"]
RouterActFnType = _registry["RouterActFnType"]

logger = logging.getLogger("Neuron")

ROUTER_ACT_FN_MAPPING = {
    "sigmoid": RouterActFnType.SIGMOID,
    "softmax": RouterActFnType.SOFTMAX,
}

def _convert_torch_dtype_to_nki_dtype(dtype: torch.dtype):
    TORCH_NKI_DTYPE_MAP = {
        torch.float16: nl.float16,
        torch.bfloat16: nl.bfloat16,
        torch.float32: nl.float32,
    }

    assert dtype in TORCH_NKI_DTYPE_MAP.keys(), f"expected dtype in {TORCH_NKI_DTYPE_MAP.keys()}, got {dtype=}"
    return TORCH_NKI_DTYPE_MAP[dtype]

def _post_create_quantized_module_hook(layer):
    # This is a workaround in order to avoid loading weights for MoEFusedTKG during tracing.
    # Quantized modules implement these functions so they will always attempt to load these
    # weights. We need to delete these methods to avoid loading.
    delattr(layer.scale, "get_tensor_from_state_dict")
    delattr(layer.weight, "get_tensor_from_state_dict")
    delattr(layer.weight, "set_tensor_to_state_dict")
    if layer.bias is not None:
        delattr(layer.bias, "get_tensor_from_state_dict")
        delattr(layer.bias, "set_tensor_to_state_dict")

class MoEFusedTKG(torch.nn.Module):
    """
    Fused MoE module for token generation.
    """

    def __init__(
        self,
        router: RouterBase,
        expert_mlps: ExpertMLPsV2,
        config: MoEFusedTKGConfig,
        sequence_dimension: int,
        shared_experts: Optional[SharedExperts] = None,
        post_attention_layernorm: Optional[torch.nn.Module] = None,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        logical_nc_config: int = DEFAULT_LNC_SIZE,
        return_router_logits: bool = False,
        return_expert_index: bool = False,
    ):
        """
        Arguments:
            router: RouterBase object.
            expert_mlps: ExpertMLPsV2 object.
            config: Module config object.
            sequence_dimension: Sequence dimension of the input.
            shared_experts: SharedExperts object, optional.
            post_attention_layernorm: Optional. If passed, forward method will forward hidden_states through
                                      post_attention_layernorm before moe.
            logical_nc_config: LNC size (1 or 2).
            return_router_logits: Whether to return router logits in addition to final hidden states.
            return_expert_index: Whether to return the expert index from router in the forward pass
        """
        super().__init__()
        self.post_attention_layernorm = post_attention_layernorm
        self.router = router
        self.expert_mlps = expert_mlps
        self.shared_experts = shared_experts
        self.config = config
        self.return_router_logits = return_router_logits
        self.return_expert_index = return_expert_index

        routed_experts_mlp_config = self.expert_mlps.routed_experts_mlp_config
        self.hidden_size = routed_experts_mlp_config.hidden_size

        self.num_experts_per_tok = routed_experts_mlp_config.top_k
        self.hidden_act = routed_experts_mlp_config.hidden_act
        self.sequence_dimension = sequence_dimension

        expert_model_parallel_size = self.expert_mlps.moe_expert_model_parallel_group.size()
        self.ep_enabled = expert_model_parallel_size > 1
        self.num_local_experts = utils.divide(routed_experts_mlp_config.num_experts, expert_model_parallel_size)
        self.logical_nc_config = logical_nc_config

        if self.config.quantized:
            # This is a workaround in order to avoid loading weights for MoEFusedTKG during tracing.
            setattr(self.expert_mlps.mlp_op.gate_up_proj, "post_create_quantized_module_hook", _post_create_quantized_module_hook)
            setattr(self.expert_mlps.mlp_op.down_proj, "post_create_quantized_module_hook", _post_create_quantized_module_hook)

    def _can_use_nki_kernel(self, kernel_type, hidden_states):
        """
        Check if the moe_fused NKI kernel can be used based on configuration and conditions.
        """
        assert kernel_type == "moe_fused"

        # Check explicit user setting
        enabled = getattr(self.config, f"{kernel_type}_kernel_enabled", None)
        if enabled is not None:
            return enabled

        if hidden_states.device.type == "cpu":
            logger.info(f"Conditions not met for running {kernel_type} NKI kernel: cannot run on cpu")
            return False

        if not self.expert_mlps.routed_experts_mlp_config.glu_mlp:
            logger.info(f"Conditions not met for {kernel_type} NKI kernel: disabling GLU not supported")
            return False

        batch_dimension = 1 - self.sequence_dimension  # hidden states are [B, S, H] or [S, B, H]
        if hidden_states.shape[batch_dimension] > 64:
            logger.info(f"Conditions not met for {kernel_type} NKI kernel: bs > 64 not yet supported")
            return False

        glu_type = GLUType.validate(self.expert_mlps.routed_experts_mlp_config.glu_type)
        if glu_type == GLUType.SWIGLU and self.expert_mlps.routed_experts_mlp_config.hidden_act_scaling_factor != DEFAULT_HIDDEN_ACT_SCALING_FACTOR:
            logger.info(f"Conditions not met for {kernel_type} NKI kernel: NKI kernel only supports scaling factor = 1.702 for SWIGLU")
            return False

        if moe_block_tkg_kernel is None:
            logger.info("Failed to load MoE Fused NKI kernel")
            return False

        if self.shared_experts is not None:
            logger.info(f"Conditions not met for {kernel_type} NKI kernel: shared experts not yet supported")
            return False

        return True

    def _can_use_fused_residual_add(self, hidden_states):
        """
        Currently, no fused kernels called by MoEFusedTKG support fused residual add.
        """
        return False

    def _slice_shared_experts_weights(self):
        """
        When sequence parallel is enabled for shared experts, their weights will be replicated on each core.
        In TKG, they will still run TP hence the weights need to be sliced.
        :return:
        """
        shared_experts = self.shared_experts
        if shared_experts is None:
            shared_experts_gate_proj_weight = None
            shared_experts_up_proj_weight = None
            shared_experts_down_proj_weight = None
        elif shared_experts.sequence_parallel_enabled:
            # slicing transposed weights
            shared_experts_up_proj_weight = shared_experts.up_proj.weight[:, shared_experts.get_split_indices(shared_experts.up_proj.weight, 1)]
            shared_experts_gate_proj_weight = shared_experts.gate_proj.weight[:, shared_experts.get_split_indices(shared_experts.gate_proj.weight, 1)]
            shared_experts_down_proj_weight = shared_experts.down_proj.weight[shared_experts.get_split_indices(shared_experts.down_proj.weight, 0), :]
        else:
            shared_experts_up_proj_weight = shared_experts.up_proj.weight
            shared_experts_gate_proj_weight = shared_experts.gate_proj.weight
            shared_experts_down_proj_weight = shared_experts.down_proj.weight
        return shared_experts_gate_proj_weight, shared_experts_up_proj_weight, shared_experts_down_proj_weight

    def _router_topk(self, hidden_states):
        """
        Args:
            hidden_states: [B, S, H] or [S, B, H]

        Returns:
            router_logits: [T, E]
            expert_affinities: [T, E]
            expert_index: [T, k], int32
        """
        logger.info("Running RouterTopK without kernel")
        router_logits, expert_affinities, expert_index = self.router(hidden_states)
        return router_logits, expert_affinities, expert_index

    def _expert_mlp(self, hidden_states, expert_affinities, expert_index):
        """
        Args:
            hidden_states: [S, B, H] or [B, S, H]
            expert_affinities: [T, E]
            expert_index: [T, k], int32 or uint32

        Returns:
            output: [T, H]
        """
        # hidden_states: (S, B, H) or (B, S, H) -> (T, H)
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states_shape[-1])

        logger.info("Running ExpertMLP without kernel")
        seq_len = hidden_states_shape[self.sequence_dimension]
        output = self.expert_mlps(
            hidden_states=hidden_states,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            seq_len=seq_len,
        )

        # output: (T, H) -> (S, B, H) or (B, S, H)
        output = output.view(hidden_states_shape)

        return output

    def _shared_mlp(self, hidden_states):
        """
        Args:
            hidden_states: [B, S, H] or [S, B, H]

        Returns:
            shared_output: original shape
        """
        logger.info("Running SharedMLP without kernel")
        seq_len = hidden_states.shape[self.sequence_dimension]

        if not self.shared_experts.transpose_weights:
            return self.shared_experts(hidden_states, seq_len)

        # With transposed weights, gate/up are RowParallelLinear(input_is_parallel=True)
        # which expects pre-scattered input. The fallback path has full input, so we
        # compute manually using the raw weight partitions to produce a correct partial sum.
        gate_w, up_w, down_w = self._slice_shared_experts_weights()
        gate = torch.matmul(hidden_states, gate_w)
        up = torch.matmul(hidden_states, up_w)
        intermediate = self.shared_experts.act_fn(gate) * up
        return torch.matmul(intermediate, down_w)

    def _moe_fused_tkg_kernel(self, hidden_states, residual=None):
        """
        Args:
            hidden_states: [B, S, H] or [S, B, H]
            residual (optional): None TODO CR

        Returns:
            output: original shape
            router_logits: [B*S, E]
        """
        hidden_states_shape = hidden_states.shape
        router_mm_dtype = _convert_torch_dtype_to_nki_dtype(self.config.router_mm_dtype)
        if self.expert_mlps.routed_experts_mlp_config.early_expert_affinity_modulation:
            expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE
        else:
            expert_affinities_scaling_mode = ExpertAffinityScaleMode.POST_SCALE
        local_rank = self.expert_mlps.spmd_rank.get_rank()
        # TODO: make this compatible with hybrid sharding, current issue is moe_tensor_model_parallel_group will be the tensor_model_parallel_group used in CTE
        local_ep_rank = local_rank // self.expert_mlps.moe_tensor_model_parallel_group.size()
        grid = self.logical_nc_config
        shared_experts_gate_proj_weight, shared_experts_up_proj_weight, shared_experts_down_proj_weight = self._slice_shared_experts_weights()
        
        # Use .data to avoid NKI tracing issues with nn.Parameter
        def get_data(t):
            return t.data if t is not None and hasattr(t, 'data') else t
        
        # router_mm_dtype must match router_weights dtype
        router_mm_dtype = _convert_torch_dtype_to_nki_dtype(self.router.weight_T.dtype)

        common_args = dict(
            inp=get_data(hidden_states),  # [B, S, H]
            gamma=get_data(self.post_attention_layernorm.weight.unsqueeze(0)),  # [1, H]
            router_weights=get_data(self.router.weight_T),  # [H, E]
            shared_expert_gate_w=get_data(shared_experts_gate_proj_weight),  # [H, I]
            shared_expert_up_w=get_data(shared_experts_up_proj_weight),  # [H, I]
            shared_expert_down_w=get_data(shared_experts_down_proj_weight),  # [I, H]
            expert_gate_up_weights=get_data(self.expert_mlps.mlp_op.gate_up_proj.weight.view(
                self.num_local_experts, self.hidden_size, 2, -1
            )),  # [E, H, 2, I]
            expert_down_weights=get_data(self.expert_mlps.mlp_op.down_proj.weight),  # [E, I, H]
            expert_gate_up_weights_scale=(
                get_data(self.expert_mlps.mlp_op.gate_up_proj.scale.view(self.num_local_experts, 2, -1))
                if self.config.quantized else None
            ),  # [E, 2, I]
            expert_down_weights_scale=(
                get_data(self.expert_mlps.mlp_op.down_proj.scale.view(self.num_local_experts, -1))
                if self.config.quantized else None
            ),  # [E, H]
            eps=self.post_attention_layernorm.variance_epsilon,
            top_k=self.num_experts_per_tok,
            router_act_fn=ROUTER_ACT_FN_MAPPING[self.router.act_fn],
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            router_mm_dtype=router_mm_dtype,
        )

        # this is a temporary check that can be removed once release compiler supports
        # `hidden_actual` kwarg in the moe tkg kernel calls
        if self.expert_mlps.routed_experts_mlp_config.hidden_size_actual is not None:
            common_args["hidden_actual"] = self.expert_mlps.routed_experts_mlp_config.hidden_size_actual

        total_tokens = hidden_states_shape[0] * hidden_states_shape[1]
        perc_experts_loaded = total_tokens * self.num_experts_per_tok / self.num_local_experts

        kernel_call = moe_block_tkg_kernel
        is_all_expert = perc_experts_loaded >= DEFAULT_SELECTIVE_LOADING_THRESHOLD
        if is_all_expert:
            logger.info("Percentage of experts loaded >= selective loading threshold, run forward all experts kernel")
        else:
            logger.info("Run selective loading kernel")

        if kernel_call:
            # run kernels that's compatible with clamp, bias, non-shared experts, SWIGLU
            routed_experts_mlp_config = self.expert_mlps.routed_experts_mlp_config
            kernel_activation_func_id = get_kernel_activation_func_id(
                ACTFunc.validate(routed_experts_mlp_config.hidden_act),
                routed_experts_mlp_config.glu_type
            )
            # pass args not in the original interface as kwargs to ensure compatibility with different compiler versions
            optional_kwargs = {}
            if routed_experts_mlp_config.gate_clamp_upper_limit is not None:
                optional_kwargs["gate_clamp_upper_limit"] = routed_experts_mlp_config.gate_clamp_upper_limit
            if routed_experts_mlp_config.gate_clamp_lower_limit is not None:
                optional_kwargs["gate_clamp_lower_limit"] = routed_experts_mlp_config.gate_clamp_lower_limit
            if routed_experts_mlp_config.up_clamp_upper_limit is not None:
                optional_kwargs["up_clamp_upper_limit"] = routed_experts_mlp_config.up_clamp_upper_limit
            if routed_experts_mlp_config.up_clamp_lower_limit is not None:
                optional_kwargs["up_clamp_lower_limit"] = routed_experts_mlp_config.up_clamp_lower_limit
            
            if is_all_expert:
                optional_kwargs["rank_id"] = get_data(local_ep_rank.reshape(1, 1))
            out, router_logits = kernel_call[grid](
                **common_args,
                router_bias=get_data(self.router.linear_router.bias) if self.router.bias else None,
                expert_gate_up_bias=get_data(self.expert_mlps.mlp_op.gate_up_proj.bias.view(self.num_local_experts, 2, -1)) if routed_experts_mlp_config.bias else None,
                expert_down_bias=get_data(self.expert_mlps.mlp_op.down_proj.bias) if routed_experts_mlp_config.bias else None,
                shared_expert_gate_bias=None,  # kernel only supports None
                shared_expert_up_bias=None,  # kernel only supports None
                shared_expert_down_bias=None,  # kernel only supports None
                router_pre_norm=not self.router.apply_act_fn_over_topk,
                hidden_act_fn=ActFnType(kernel_activation_func_id),
                hidden_act_scale_factor=None,  # kernel only supports None
                hidden_act_bias=None,  # kernel only supports None
                norm_topk_prob=self.config.norm_topk_prob,
                is_all_expert=is_all_expert,
                **optional_kwargs,
            )

        return out.view(hidden_states_shape), router_logits.to(hidden_states.dtype)

    def forward(self, hidden_states, residual=None):
        """
        Forward through MoE TKG mega-kernel if conditions are satisfied. Otherwise forward through
        flat compiler / individual kernels.

        Args:
            hidden_states: [B, S, H] or [S, B, H]
            residual (optional): [B, S, H] or [S, B, H]

        Returns:
            output: original shape
            router_logits: [B*S, E]
            residual (optional): residual tensor of same shape as output
        """

        if not self._can_use_fused_residual_add(hidden_states) and residual is not None:
            hidden_states = hidden_states + residual
            residual = hidden_states.clone()

        if self._can_use_nki_kernel("moe_fused", hidden_states):
            logger.info("Running MoE Fused NKI kernel")
            if not self._can_use_fused_residual_add(hidden_states) or residual is None:
                output, router_logits = self._moe_fused_tkg_kernel(hidden_states)
            else:
                output, router_logits, residual = self._moe_fused_tkg_kernel(hidden_states, residual=residual)

            if self.return_expert_index:
                # return_expert_index not supported in kernel, return emtpy tensor to match with cte tracing when return_expert_index is set to True
                expert_index = torch.empty((self.expert_mlps.routed_experts_mlp_config.num_experts, self.num_experts_per_tok), device=hidden_states.device, dtype=torch.long)
        else:
            if self.post_attention_layernorm is not None:
                # we don't have individual kernel support for RMSNorm
                logger.info("Running RMSNorm without kernel")
                hidden_states = self.post_attention_layernorm(hidden_states)
            # RMSNorm is optional, so hidden_states must maintain same shape after RMSNorm

            router_logits, expert_affinities, expert_index = self._router_topk(hidden_states)
            expert_affinities = mappings.copy_to_tensor_model_parallel_region(
                expert_affinities
            )

            shared_output = 0
            if self.shared_experts:
                shared_output = self._shared_mlp(hidden_states)
            output = self._expert_mlp(
                hidden_states=hidden_states,
                expert_affinities=expert_affinities,
                expert_index=expert_index
            )
            output = output + shared_output

        # Delayed All-Reduce
        output = mappings.reduce_from_tensor_model_parallel_region(
            output, process_group=parallel_state.get_world_group()
        )

        return_op = (output,)
        if self.return_router_logits:
            return_op += (router_logits,)
        if self.return_expert_index:
            return_op += (expert_index,)
        if residual is not None:
            return_op += (residual,)

        return return_op

    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        pass
