import math
import logging
from typing import Optional
from importlib import import_module

import nki
import nki.language as nl
import torch
from torch.distributed import ProcessGroup

from neuronx_distributed.modules.moe.model_utils import (
    ACTFunc,
    DEFAULT_LNC_SIZE,
    DEFAULT_SELECTIVE_LOADING_THRESHOLD,
    get_kernel_activation_func_id,
)
from neuronx_distributed.modules.moe.moe_configs import MoEFusedTKGConfig
from neuronx_distributed.modules.moe.routing import RouterBase
from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPsV2
from neuronx_distributed.modules.moe.shared_experts import SharedExperts
from neuronx_distributed.modules.moe.moe_fused_tkg import MoEFusedTKG

def _import_beta2_moe():
    """Try importing Beta 2 kernels from nkilib."""
    beta2_paths = [
        ("nkilib.core.moe_block.moe_block_tkg", "nkilib.core.utils.common_types"),
        ("nkilib_src.nkilib.core.moe_block.moe_block_tkg", "nkilib_src.nkilib.core.utils.common_types"),
    ]

    for kernel_path, types_path in beta2_paths:
        try:
            kernel_module = import_module(kernel_path)
            types_module = import_module(types_path)

            # Import TensorView from the same nkilib path
            tensor_view_path = kernel_path.rsplit(".", 2)[0] + ".utils.tensor_view"
            tensor_view_module = import_module(tensor_view_path)

            return {
                "nl": nl,
                "moe_block_tkg": getattr(kernel_module, "moe_block_tkg"),
                "RouterActFnType": types_module.RouterActFnType,
                "ActFnType": types_module.ActFnType,
                "ExpertAffinityScaleMode": types_module.ExpertAffinityScaleMode,
                "TensorView": tensor_view_module.TensorView,
                "float4_e2m1fn_x4": nl.float4_e2m1fn_x4,
            }
        except (ImportError, AttributeError):
            continue

    raise ImportError("Beta 2 NKI kernels not available. Ensure nkilib is installed.")

_registry = _import_beta2_moe()
moe_block_tkg_kernel = _registry["moe_block_tkg"]

ExpertAffinityScaleMode = _registry["ExpertAffinityScaleMode"]
ActFnType = _registry["ActFnType"]
RouterActFnType = _registry["RouterActFnType"]

TensorView = _registry["TensorView"]
float4_e2m1fn_x4 = _registry["float4_e2m1fn_x4"]

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

@nki.jit
def mxfp4_moe_block_tkg_wrapper(
    inp: nl.ndarray,
    gamma: nl.ndarray,
    router_weights: nl.ndarray,
    expert_gate_up_weights: nl.ndarray,
    expert_down_weights: nl.ndarray,
    shared_expert_gate_w: Optional[nl.ndarray] = None,
    shared_expert_up_w: Optional[nl.ndarray] = None,
    shared_expert_down_w: Optional[nl.ndarray] = None,
    expert_gate_up_weights_scale: Optional[nl.ndarray] = None,
    expert_down_weights_scale: Optional[nl.ndarray] = None,
    router_bias: Optional[nl.ndarray] = None,
    expert_gate_up_bias: Optional[nl.ndarray] = None,
    expert_down_bias: Optional[nl.ndarray] = None,
    shared_expert_gate_bias: Optional[nl.ndarray] = None,
    shared_expert_up_bias: Optional[nl.ndarray] = None,
    shared_expert_down_bias: Optional[nl.ndarray] = None,
    eps: float = 1e-6,
    top_k: int = 1,
    router_act_fn = RouterActFnType.SIGMOID,
    router_pre_norm: bool = True,
    norm_topk_prob = False,
    expert_affinities_scaling_mode = ExpertAffinityScaleMode.NO_SCALE,
    hidden_act_fn = ActFnType.SiLU,
    hidden_act_scale_factor: Optional[float] = None,
    hidden_act_bias: Optional[float] = None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    router_mm_dtype = nl.bfloat16,
    hidden_actual: Optional[int] = None,
    skip_router_logits: bool = False,
    is_all_expert: bool = False,
    rank_id: Optional[nl.ndarray] = None,
    residual: Optional[nl.ndarray] = None,
):
    """
    Wrapper kernel for MoE TKG fused kernel that bitcasts FP4 weights to float4x4 NKI dtype prior to calling the kernel
    """
    expert_gate_up_weights = TensorView(expert_gate_up_weights).reinterpret_cast(float4_e2m1fn_x4)
    expert_down_weights = TensorView(expert_down_weights).reinterpret_cast(float4_e2m1fn_x4)
    return moe_block_tkg_kernel(
        inp=inp,
        gamma=gamma,
        router_weights=router_weights,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        shared_expert_gate_w=shared_expert_gate_w,
        shared_expert_up_w=shared_expert_up_w,
        shared_expert_down_w=shared_expert_down_w,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        router_bias=router_bias,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        shared_expert_gate_bias=shared_expert_gate_bias,
        shared_expert_up_bias=shared_expert_up_bias,
        shared_expert_down_bias=shared_expert_down_bias,
        eps=eps,
        top_k=top_k,
        router_act_fn=router_act_fn,
        router_pre_norm=router_pre_norm,
        norm_topk_prob=norm_topk_prob,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        hidden_act_fn=hidden_act_fn,
        hidden_act_scale_factor=hidden_act_scale_factor,
        hidden_act_bias=hidden_act_bias,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        router_mm_dtype=router_mm_dtype,
        hidden_actual=hidden_actual,
        skip_router_logits=skip_router_logits,
        is_all_expert=is_all_expert,
        rank_id=rank_id,
        residual=residual,
    )

_moe_tkg_mx_wrapper = mxfp4_moe_block_tkg_wrapper

class MoEFusedTKGMX(MoEFusedTKG):
    """
    Fused MoE module for token generation using MXFP4 weights
    This requires a series of layout transforms on the weights, controlled by config flag `is_mxfp4_compute`
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
        super().__init__(router, expert_mlps, config, sequence_dimension, shared_experts, post_attention_layernorm, tensor_model_parallel_group, logical_nc_config, return_router_logits, return_expert_index)
        logger.info("Selected MXFP4 variant of MoE Fused TKG")

    def _prepare_kernel_inputs(self):
        router_mm_dtype = _convert_torch_dtype_to_nki_dtype(self.config.router_mm_dtype)
        routed_experts_mlp_config = self.expert_mlps.routed_experts_mlp_config
        if routed_experts_mlp_config.early_expert_affinity_modulation:
            expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE
        else:
            expert_affinities_scaling_mode = ExpertAffinityScaleMode.POST_SCALE
        
        kernel_activation_func_id = get_kernel_activation_func_id(
            ACTFunc.validate(routed_experts_mlp_config.hidden_act),
            routed_experts_mlp_config.glu_type
        )
        hidden_size = self.expert_mlps.mlp_op.gate_up_proj.input_size
        assert hidden_size % 512 == 0, f"Hidden size must be divisible by 512, got {hidden_size}"

        intermediate_size_per_partition = self.expert_mlps.mlp_op.down_proj.input_size_per_partition
        assert intermediate_size_per_partition % 4 == 0, f"Intermediate size must be divisible by 4, got {intermediate_size_per_partition}"
        num_I_TP_blocks = math.ceil(intermediate_size_per_partition / 512.0)
        assert intermediate_size_per_partition % num_I_TP_blocks == 0, f"{intermediate_size_per_partition=} must be divisible by {num_I_TP_blocks=}"
        I_TP_block_size = intermediate_size_per_partition // num_I_TP_blocks
        gate_up_weights_bias = self.expert_mlps.mlp_op.gate_up_proj.bias
        gate_up_weights_bias = gate_up_weights_bias.view(self.num_local_experts, I_TP_block_size // 4, 2, num_I_TP_blocks, 4)
        
        # run kernels that's compatible with clamp, bias, non-shared experts, SWIGLU
        optional_kwargs = {}
        # pass args not in the original interface as kwargs to ensure compatibility with different compiler versions
        if routed_experts_mlp_config.gate_clamp_upper_limit is not None:
            optional_kwargs["gate_clamp_upper_limit"] = routed_experts_mlp_config.gate_clamp_upper_limit
        if routed_experts_mlp_config.gate_clamp_lower_limit is not None:
            optional_kwargs["gate_clamp_lower_limit"] = routed_experts_mlp_config.gate_clamp_lower_limit
        if routed_experts_mlp_config.up_clamp_upper_limit is not None:
            optional_kwargs["up_clamp_upper_limit"] = routed_experts_mlp_config.up_clamp_upper_limit
        if routed_experts_mlp_config.up_clamp_lower_limit is not None:
            optional_kwargs["up_clamp_lower_limit"] = routed_experts_mlp_config.up_clamp_lower_limit
        if self.router.bias:
            optional_kwargs["router_bias"] = self.router.linear_router.bias.unsqueeze(0)

        return dict(
            router_mm_dtype=router_mm_dtype,
            kernel_activation_func_id=kernel_activation_func_id,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            down_weights=self.expert_mlps.mlp_op.down_proj.weight,
            down_weights_scale=self.expert_mlps.mlp_op.down_proj.scale,
            gate_up_weights=self.expert_mlps.mlp_op.gate_up_proj.weight,
            gate_up_weights_scale=self.expert_mlps.mlp_op.gate_up_proj.scale,
            gate_up_weights_bias=gate_up_weights_bias,
            down_weights_bias=self.expert_mlps.mlp_op.down_proj.bias,
            router_weights=self.router.weight_T,
            **optional_kwargs,
        )

    def _should_use_all_expert(self, hidden_states):
        """
        Helper function to determine whether to use selective loading or all experts algorithm for MoE compute. Selective loading is
        generally more performant when B * S * topk < E, and all experts is more performant when B * S * topk > E.
        """

        hidden_states_shape = hidden_states.shape
        total_tokens = hidden_states_shape[0] * hidden_states_shape[1]
        perc_experts_loaded = total_tokens * self.num_experts_per_tok / self.num_local_experts
        return perc_experts_loaded >= DEFAULT_SELECTIVE_LOADING_THRESHOLD

    def _can_use_fused_residual_add(self, hidden_states):
        """
        Helper function to determine whether we can use fused residual add feature inside fused TKG kernel. Currently
        fused residual add is only supported in recent versions of all expert kernel when MXFP weights are used.
        """

        batch_x_seq = hidden_states.shape[0] * hidden_states.shape[1]
        _has_fused_residual_add_support = batch_x_seq >= 256
        logger.info(f"Residual add support: {_has_fused_residual_add_support=} ({batch_x_seq=})")
        return self._should_use_all_expert(hidden_states) and _has_fused_residual_add_support

    def _moe_fused_tkg_kernel(self, hidden_states, residual=None):
        """
        Args:
            hidden_states: [B, S, H] or [S, B, H]
            residual (optional): [B, S, H] or [S, B, H]

        Returns:
            output: original shape
            router_logits: [B*S, E]
            residual (optional): same shape as output
        """
        hidden_states_shape = hidden_states.shape
        local_rank = self.expert_mlps.spmd_rank.get_rank()
        # TODO: make this compatible with hybrid sharding, current issue is moe_tensor_model_parallel_group will be the tensor_model_parallel_group used in CTE
        local_ep_rank = local_rank // self.expert_mlps.moe_tensor_model_parallel_group.size()
        shared_experts_gate_proj_weight, shared_experts_up_proj_weight, shared_experts_down_proj_weight = self._slice_shared_experts_weights()
        prepared_kernel_inputs = self._prepare_kernel_inputs()
        grid = self.logical_nc_config

        # this is a temporary check that can be removed once release compiler supports
        # `hidden_actual` kwarg in the moe tkg kernel calls
        routed_experts_mlp_config = self.expert_mlps.routed_experts_mlp_config
        if routed_experts_mlp_config.hidden_size_actual is not None:
            prepared_kernel_inputs["hidden_actual"] = self.expert_mlps.routed_experts_mlp_config.hidden_size_actual

        is_all_expert = self._should_use_all_expert(hidden_states)
        
        if _moe_tkg_mx_wrapper is not None:
            logger.info(f"Using moe_block_tkg kernel (is_all_expert={is_all_expert})")
            
            def get_data(t):
                return t.data if t is not None and hasattr(t, 'data') else t
            
            kernel_kwargs = dict(
                inp=get_data(hidden_states),
                gamma=get_data(self.post_attention_layernorm.weight.unsqueeze(0)),
                router_weights=get_data(prepared_kernel_inputs["router_weights"]),
                shared_expert_gate_w=get_data(shared_experts_gate_proj_weight),
                shared_expert_up_w=get_data(shared_experts_up_proj_weight),
                shared_expert_down_w=get_data(shared_experts_down_proj_weight),
                expert_gate_up_weights=get_data(prepared_kernel_inputs["gate_up_weights"]),
                expert_down_weights=get_data(prepared_kernel_inputs["down_weights"]),
                expert_gate_up_weights_scale=get_data(prepared_kernel_inputs["gate_up_weights_scale"]),
                expert_down_weights_scale=get_data(prepared_kernel_inputs["down_weights_scale"]),
                eps=self.post_attention_layernorm.variance_epsilon,
                top_k=self.num_experts_per_tok,
                router_act_fn=ROUTER_ACT_FN_MAPPING[self.router.act_fn],
                expert_affinities_scaling_mode=prepared_kernel_inputs["expert_affinities_scaling_mode"],
                router_mm_dtype=prepared_kernel_inputs["router_mm_dtype"],
                router_bias=get_data(prepared_kernel_inputs.get("router_bias")),
                expert_gate_up_bias=get_data(prepared_kernel_inputs.get("gate_up_weights_bias")),
                expert_down_bias=get_data(prepared_kernel_inputs.get("down_weights_bias")),
                shared_expert_gate_bias=None,  # kernel only supports None
                shared_expert_up_bias=None,  # kernel only supports None
                shared_expert_down_bias=None,  # kernel only supports None
                router_pre_norm=not self.router.apply_act_fn_over_topk,
                hidden_act_fn=ActFnType(prepared_kernel_inputs["kernel_activation_func_id"]),
                hidden_act_scale_factor=None,  # kernel only supports None
                hidden_act_bias=None,  # kernel only supports None
                norm_topk_prob=self.config.norm_topk_prob,
                gate_clamp_upper_limit=prepared_kernel_inputs.get("gate_clamp_upper_limit"),
                gate_clamp_lower_limit=prepared_kernel_inputs.get("gate_clamp_lower_limit"), 
                up_clamp_upper_limit=prepared_kernel_inputs.get("up_clamp_upper_limit"),
                up_clamp_lower_limit=prepared_kernel_inputs.get("up_clamp_lower_limit"),
                hidden_actual=prepared_kernel_inputs.get("hidden_actual"),
                is_all_expert=is_all_expert,
                rank_id=get_data(local_ep_rank.reshape(1, 1)) if is_all_expert else None,
                residual=get_data(residual) if is_all_expert else None,
            )
            # Utilize fused residual add when possible. Residual add is computed in parent class forward call if fused residual add is not available.
            if self._can_use_fused_residual_add(hidden_states) and residual is not None:
                out, router_logits, residual = _moe_tkg_mx_wrapper[grid](**kernel_kwargs)
                return out.view(hidden_states_shape), router_logits.to(hidden_states.dtype), residual.view(hidden_states_shape)
            else:
                out, router_logits = _moe_tkg_mx_wrapper[grid](**kernel_kwargs)
                return out.view(hidden_states_shape), router_logits.to(hidden_states.dtype)

        raise RuntimeError("Unable to select a MoE fused TKG kernel to run")
