import math
import logging
import warnings
from typing import Optional

from neuronxcc import nki
import neuronxcc.nki.language as nl
import torch
from torch.distributed import ProcessGroup
from neuronxcc.nki.language import nc

from neuronx_distributed.modules.moe.model_utils import (
    ACTFunc,
    DEFAULT_LNC_SIZE,
    DEFAULT_SELECTIVE_LOADING_THRESHOLD,
    get_kernel_activation_func_id,
)
from neuronx_distributed.modules.moe.moe_configs import MoEFusedTKGConfig
from neuronx_distributed.modules.moe.nki_import import NKIImport, import_nki
from neuronx_distributed.modules.moe.routing import RouterBase
from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPsV2
from neuronx_distributed.modules.moe.shared_experts import SharedExperts
from neuronx_distributed.modules.moe.moe_fused_tkg import MoEFusedTKG, ROUTER_ACT_FN_MAPPING


logger = logging.getLogger("Neuron")


def initialize_nki_components() -> dict:
    """
    Initialize all NKI components.

    Returns:
        dict: Mapping of component names to their imported values
    """
    imports = {
        "moe_token_gen_selective_load_kernel": NKIImport("moe_token_gen_selective_load_kernel", module_name="moe_token_gen"),
        "nki_expert_mlp_tkg_isa_kernel": NKIImport("nki_expert_mlp_tkg_isa_kernel", module_name="mlp_tkg.expert_mlp_tkg_isa"),
        "float4_e2m1fn_x4": NKIImport("float4_e2m1fn_x4", module_name="private_api"),
        "router_act_fn_type": NKIImport("RouterActFnType"),
        "affinity_scale_mode": NKIImport("ExpertAffinityScaleMode"),
        "act_fn_type": NKIImport("ActFnType"),
    }

    components = {}
    for name, config in imports.items():
        component, error = import_nki(config)
        if error:
            warnings.warn(f"Warning: {error}")
        components[name] = component

    return components

nki_components = initialize_nki_components()

ExpertAffinityScaleMode = nki_components["affinity_scale_mode"]
RouterActFnType = nki_components["router_act_fn_type"]
ActFnType = nki_components["act_fn_type"]
float4_e2m1fn_x4 = nki_components["float4_e2m1fn_x4"]

_nki_expert_mlp_tkg_isa_kernel_call = nki_components["nki_expert_mlp_tkg_isa_kernel"]
_moe_token_gen_selective_load_kernel_call = nki_components["moe_token_gen_selective_load_kernel"]


@nki.compiler.skip_middle_end_transformations
@nki.jit(show_compiler_tb=True, debug_kernel=True, experimental_flags='skip-non-top-level-shared-hbm-check')
def mxfp4_nki_expert_mlp_tkg_isa_kernel_wrapper(
    inp: nl.ndarray,
    gate_up_weights: nl.ndarray,
    down_weights: nl.ndarray,
    expert_affinities: nl.ndarray,
    expert_index: nl.ndarray,
    is_all_expert: bool,
    gate_up_weights_scale: Optional[nl.ndarray],
    down_weights_scale: Optional[nl.ndarray],
    gate_up_weights_bias: Optional[nl.ndarray],
    down_weights_bias: Optional[nl.ndarray],
    expert_affinities_scaling_mode = ExpertAffinityScaleMode.NO_SCALE, 
    act_fn = ActFnType.SiLU,
    output_in_sbuf: bool = False,
    lhs_rhs_swap: bool = True,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    base_addr: int = 0,
):
    """
    Wrapper kernel for expert mlp TKG kernel that bitcasts FP4 weights to float4x4 NKI dtype prior to calling the kernel
    """
    gate_up_weights = gate_up_weights.view(float4_e2m1fn_x4)
    down_weights = down_weights.view(float4_e2m1fn_x4)
    return _nki_expert_mlp_tkg_isa_kernel_call(
        inp=inp,
        gate_up_weights=gate_up_weights,
        down_weights=down_weights,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        is_all_expert=is_all_expert,
        gate_up_weights_scale=gate_up_weights_scale,
        down_weights_scale=down_weights_scale,
        gate_up_weights_bias=gate_up_weights_bias,
        down_weights_bias=down_weights_bias,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        act_fn=act_fn,
        output_in_sbuf=output_in_sbuf,
        lhs_rhs_swap=lhs_rhs_swap,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        base_addr=base_addr,
    )

@nki.compiler.skip_middle_end_transformations
@nki.jit(show_compiler_tb=True, debug_kernel=True, experimental_flags='skip-non-top-level-shared-hbm-check')
def mxfp4_moe_token_gen_selective_load_kernel_wrapper(
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
    use_router_topk_nki_kernel: bool = False
):
    """
    Wrapper kernel for MoE TKG fused kernel that bitcasts FP4 weights to float4x4 NKI dtype prior to calling the kernel
    """
    expert_gate_up_weights = expert_gate_up_weights.view(float4_e2m1fn_x4)
    expert_down_weights = expert_down_weights.view(float4_e2m1fn_x4)
    return _moe_token_gen_selective_load_kernel_call(
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
        use_router_topk_nki_kernel=use_router_topk_nki_kernel,
    )

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

        input_size_per_partition = self.expert_mlps.mlp_op.down_proj.input_size_per_partition
        assert input_size_per_partition % 4 == 0, f"Intermediate size must be divisible by 4, got {input_size_per_partition}"
        num_I_TP_blocks = math.ceil(input_size_per_partition / 512.0)
        gate_up_weights_bias = self.expert_mlps.mlp_op.gate_up_proj.bias
        gate_up_weights_bias = gate_up_weights_bias.view(self.num_local_experts, input_size_per_partition // 4, 2, num_I_TP_blocks, 4)
        
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

        return dict(
            kernel_activation_func_id=kernel_activation_func_id,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            down_weights=self.expert_mlps.mlp_op.down_proj.weight,
            down_weights_scale=self.expert_mlps.mlp_op.down_proj.scale,
            gate_up_weights=self.expert_mlps.mlp_op.gate_up_proj.weight,
            gate_up_weights_scale=self.expert_mlps.mlp_op.gate_up_proj.scale,
            gate_up_weights_bias=gate_up_weights_bias,
            down_weights_bias=self.expert_mlps.mlp_op.down_proj.bias,
            **optional_kwargs,
        )

    def _moe_fused_tkg_kernel(self, hidden_states):
        hidden_states_shape = hidden_states.shape
        local_rank = self.expert_mlps.spmd_rank.get_rank()
        # TODO: make this compatible with hybrid sharding, current issue is moe_tensor_model_parallel_group will be the tensor_model_parallel_group used in CTE
        local_ep_rank = local_rank // self.expert_mlps.moe_tensor_model_parallel_group.size()
        shared_experts_gate_proj_weight, shared_experts_up_proj_weight, shared_experts_down_proj_weight = self._slice_shared_experts_weights()
        prepared_kernel_inputs = self._prepare_kernel_inputs()
        grid = (nc(self.logical_nc_config),)

        # this is a temporary check that can be removed once release compiler supports
        # `hidden_actual` kwarg in the moe tkg kernel calls
        routed_experts_mlp_config = self.expert_mlps.routed_experts_mlp_config
        if routed_experts_mlp_config.hidden_size_actual is not None:
            prepared_kernel_inputs["hidden_actual"] = self.expert_mlps.routed_experts_mlp_config.hidden_size_actual

        total_tokens = hidden_states_shape[0] * hidden_states_shape[1]
        perc_experts_loaded = total_tokens * self.num_experts_per_tok / self.num_local_experts

        kernel_call = None
        if (perc_experts_loaded >= DEFAULT_SELECTIVE_LOADING_THRESHOLD):
            logger.info("Percentage of experts loaded >= selective loading threshold, run forward all experts fused megakernel")
            raise NotImplementedError("Forward all experts fused megakernel not integrated yet for MXFP4")
            prepared_kernel_inputs["rank_id"] = local_ep_rank.reshape(1, 1)
        elif (perc_experts_loaded < DEFAULT_SELECTIVE_LOADING_THRESHOLD and self.shared_experts is None):
            logger.info("Run MXFP4 selective loading fused megakernel: _moe_token_gen_selective_load_kernel_nki_call")
            # kernel_call = mxfp4_moe_token_gen_selective_load_kernel_wrapper
            kernel_call = mxfp4_moe_token_gen_selective_load_kernel_wrapper

        out, router_logits = kernel_call[grid](
            inp=hidden_states,
            gamma=self.post_attention_layernorm.weight.unsqueeze(0),
            router_weights=self.router.weight_T,
            shared_expert_gate_w=shared_experts_gate_proj_weight,
            shared_expert_up_w=shared_experts_up_proj_weight,
            shared_expert_down_w=shared_experts_down_proj_weight,
            expert_gate_up_weights=prepared_kernel_inputs["gate_up_weights"],  # [E, 128, 2, H/512, I_TP]
            expert_down_weights=prepared_kernel_inputs["down_weights"],  # [E, I_TP//4, num_I_TP_blocks, H]
            expert_gate_up_weights_scale=prepared_kernel_inputs["gate_up_weights_scale"],  # [E, 128/8, 2, H/512, I_TP]
            expert_down_weights_scale=prepared_kernel_inputs["down_weights_scale"],  # [E, I_TP//4, num_I_TP_blocks, H]
            eps=self.post_attention_layernorm.variance_epsilon,
            top_k=self.num_experts_per_tok,
            router_act_fn=ROUTER_ACT_FN_MAPPING[self.router.act_fn],
            expert_affinities_scaling_mode=prepared_kernel_inputs["expert_affinities_scaling_mode"],
            router_mm_dtype=nl.float32,
            router_bias=self.router.linear_router.bias if self.router.bias else None,
            expert_gate_up_bias=prepared_kernel_inputs["gate_up_weights_bias"] if routed_experts_mlp_config.bias else None,
            expert_down_bias=prepared_kernel_inputs["down_weights_bias"] if routed_experts_mlp_config.bias else None,
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
        )

        return out.view(hidden_states_shape), router_logits.to(hidden_states.dtype)

    def _expert_mlp(self, hidden_states, expert_affinities, expert_index):
        """
        Args:
            hidden_states: [S, B, H] or [B, S, H]
            expert_affinities: [T, E]
            expert_index: [T, k], int32 or uint32

        Returns:
            output: [T, H]
        """
        logger.info("Running MXFP4 ExpertMLP NKI kernel")
        # hidden_states: (S, B, H) or (B, S, H) -> (T, H)
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states_shape[-1])

        prepared_kernel_inputs = self._prepare_kernel_inputs()
        grid = (nc(self.logical_nc_config),)

        output = mxfp4_nki_expert_mlp_tkg_isa_kernel_wrapper[grid](
            inp=hidden_states,
            gate_up_weights=prepared_kernel_inputs["gate_up_weights"],
            down_weights=prepared_kernel_inputs["down_weights"], 
            expert_affinities=expert_affinities,
            expert_index=expert_index.to(torch.int32),
            is_all_expert=False,
            gate_up_weights_scale=prepared_kernel_inputs["gate_up_weights_scale"],
            down_weights_scale=prepared_kernel_inputs["down_weights_scale"],
            gate_up_weights_bias=prepared_kernel_inputs["gate_up_weights_bias"],
            down_weights_bias=prepared_kernel_inputs["down_weights_bias"],
            expert_affinities_scaling_mode=prepared_kernel_inputs["expert_affinities_scaling_mode"],
            act_fn=ActFnType(prepared_kernel_inputs["kernel_activation_func_id"]),
            gate_clamp_upper_limit=prepared_kernel_inputs.get("gate_clamp_upper_limit"),
            gate_clamp_lower_limit=prepared_kernel_inputs.get("gate_clamp_lower_limit"), 
            up_clamp_upper_limit=prepared_kernel_inputs.get("up_clamp_upper_limit"),
            up_clamp_lower_limit=prepared_kernel_inputs.get("up_clamp_lower_limit")
        )
        # output: (T, H) -> (S, B, H) or (B, S, H)
        output = output.view(hidden_states_shape)
        return output