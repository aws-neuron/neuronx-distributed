import logging
import warnings
from typing import Any, Dict, Optional

import neuronxcc.nki.language as nl
import torch
from torch.distributed import ProcessGroup
from neuronxcc import nki
from neuronxcc.nki.language import nc

from neuronx_distributed.modules.moe.model_utils import (
    ACTFunc,
    DEFAULT_HIDDEN_ACT_SCALING_FACTOR,
    DEFAULT_LNC_SIZE,
    DEFAULT_SELECTIVE_LOADING_THRESHOLD,
    get_kernel_activation_func_id,
    GLUType,
)
from neuronx_distributed.modules.moe.moe_configs import MoEFusedTKGConfig
from neuronx_distributed.modules.moe.nki_import import NKIImport, import_nki
from neuronx_distributed.parallel_layers import mappings, parallel_state, utils
from neuronx_distributed.modules.moe.routing import RouterBase
from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPsV2
from neuronx_distributed.modules.moe.shared_experts import SharedExperts


def initialize_nki_components() -> dict:
    """
    Initialize all NKI components.

    Returns:
        dict: Mapping of component names to their imported values
    """
    imports = {
        "router_topk": NKIImport("router_topk_isa_kernel", module_name="router_topk", nki_jit_type="use_nki_jit_decorator"),
        "mlp": NKIImport("mlp_isa_kernel", module_name="mlp", nki_jit_type="use_nki_jit_decorator"),
        "quant_mlp": NKIImport("quant_mlp_isa_kernel", module_name="mlp", nki_jit_type="use_nki_jit_decorator"),
        "expert_mlps": NKIImport("expert_mlps_isa_inline_kernel", module_name="expert_mlps"),
        "moe_token_gen_selective_loading": NKIImport("moe_token_gen_kernel", module_name="moe_token_gen"),
        "moe_token_gen_forward_all_experts": NKIImport("moe_token_gen_all_experts_kernel", module_name="moe_token_gen"),
        "affinity_scale_mode": NKIImport("ExpertAffinityScaleMode"),
        "act_fn_type": NKIImport("ActFnType"),
        "router_act_fn_type": NKIImport("RouterActFnType"),
    }

    components = {}
    for name, config in imports.items():
        component, error = import_nki(config)
        if error:
            warnings.warn(f"Warning: {error}")
        components[name] = component

    return components


# Initialize all components
nki_components = initialize_nki_components()

_router_topk_nki_call = nki_components["router_topk"]
_mlp_nki_call = nki_components["mlp"]
_quant_mlp_nki_call = nki_components["quant_mlp"]
_moe_tkg_selective_loading_nki_call = nki_components["moe_token_gen_selective_loading"]
_moe_tkg_forward_all_experts_nki_call = nki_components["moe_token_gen_forward_all_experts"]
ExpertAffinityScaleMode = nki_components["affinity_scale_mode"]
ActFnType = nki_components["act_fn_type"]
RouterActFnType = nki_components["router_act_fn_type"]


logger = logging.getLogger("Neuron")

ACT_FN_MAPPING = {"silu": ActFnType.SiLU, "gelu": ActFnType.GELU}

ROUTER_ACT_FN_MAPPING = {
    "sigmoid": RouterActFnType.SIGMOID,
    "softmax": RouterActFnType.SOFTMAX,
}

def expert_isa_kernel_wrapper(
    inp: nl.ndarray,
    gate_up_weights: nl.ndarray,
    down_weights: nl.ndarray,
    expert_affinities: nl.ndarray,
    expert_index: nl.ndarray,
    expert_affinities_scaling_mode = ExpertAffinityScaleMode.NO_SCALE,
    enable_kernel_fusion: bool = False,
) -> nl.ndarray:
    T, K = expert_index.shape
    _, _, H = down_weights.shape
    out = nl.ndarray((T, H), dtype=inp.dtype, buffer=nl.shared_hbm)
    nki_components["expert_mlps"](
        inp,  # [T, H]
        gate_up_weights,  # [E, H, 2, I]
        down_weights,  # [E, I, H]
        expert_affinities,  # [T, E]
        expert_index,  # [T, k]
        out,
        expert_affinities_scaling_mode,
        enable_kernel_fusion,
    )  # [T, H]
    return out

def _post_create_quantized_module_hook(layer):
    # This is a workaround in order to avoid loading weights for MoEFusedTKG during tracing.
    # Quantized modules implement these functions so they will always attempt to load these
    # weights. We need to delete these methods to avoid loading.
    delattr(layer.scale, "get_tensor_from_state_dict")
    delattr(layer.weight, "get_tensor_from_state_dict")
    delattr(layer.weight, "set_tensor_to_state_dict")

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
        Check if a specific NKI kernel can be used based on configuration and conditions.
        """
        assert kernel_type in ["moe_fused", "router_topk", "expert_mlp", "shared_mlp"]
        # Check explicit user setting
        enabled = getattr(self.config, f"{kernel_type}_kernel_enabled", None)
        if enabled is not None:
            return enabled

        if hidden_states.device.type == "cpu":
            logger.info(f"Conditions not met for running {kernel_type} NKI kernel: cannot run on cpu")
            return False

        if kernel_type in ["moe_fused", "expert_mlp"]:
            if not self.expert_mlps.routed_experts_mlp_config.glu_mlp:
                logger.info(f"Conditions not met for {kernel_type} NKI kernel: disabling GLU not supported")
                return False
            if self.expert_mlps.routed_experts_mlp_config.normalize_top_k_affinities:
                logger.info(f"Conditions not met for {kernel_type} NKI kernel: normalize top k affinities not supported")
                return False

        if kernel_type in ["shared_mlp"]:
            if self.shared_experts is None:
                logger.info(f"Conditions not met for {kernel_type} NKI kernel: module does not contain shared experts")
                return False

        if kernel_type == "moe_fused":
            batch_dimension = 1 - self.sequence_dimension  # hidden states are [B, S, H] or [S, B, H]
            if hidden_states.shape[batch_dimension] > 32:
                logger.info(f"Conditions not met for {kernel_type} NKI kernel: bs > 32 not yet supported")
                return False
            batch_size = hidden_states.shape[batch_dimension]
            seq_len = hidden_states.shape[self.sequence_dimension]
            total_tokens = batch_size * seq_len
            perc_experts_loaded = total_tokens * self.num_experts_per_tok / self.num_local_experts
            if perc_experts_loaded >= DEFAULT_SELECTIVE_LOADING_THRESHOLD:
                logger.info(f"perc_experts_loaded={perc_experts_loaded} >= DEFAULT_SELECTIVE_LOADING_THRESHOLD={DEFAULT_HIDDEN_ACT_SCALING_FACTOR}")
                glu_type = GLUType.validate(self.expert_mlps.routed_experts_mlp_config.glu_type)
                if glu_type == GLUType.SWIGLU and self.expert_mlps.routed_experts_mlp_config.hidden_act_scaling_factor != DEFAULT_HIDDEN_ACT_SCALING_FACTOR:
                    logger.info(f"Conditions not met for {kernel_type} NKI kernel: NKI kernel only supports scaling factor = 1.702 for SWIGLU")
                    return False
                kernel_call = _moe_tkg_forward_all_experts_nki_call
            else:
                logger.info(f"perc_experts_loaded={perc_experts_loaded} < DEFAULT_SELECTIVE_LOADING_THRESHOLD={DEFAULT_HIDDEN_ACT_SCALING_FACTOR}")
                # selective loading kernel does not support optional shared experts at the moment
                if self.shared_experts is None:
                    logger.info(f"Conditions not met for {kernel_type} NKI kernel: module does not contain shared experts")
                    return False
                if self.ep_enabled:
                    logger.info(f"Conditions not met for {kernel_type} NKI kernel: EP not supported")
                    return False
                kernel_call = _moe_tkg_selective_loading_nki_call
            if kernel_call is None:
                logger.info("Failed to load MoE Fused NKI kernel")
                return False
        elif kernel_type == "expert_mlp" and nki_components["expert_mlps"] is None:
            logger.info("Failed to load ExpertMLP NKI kernel")
            return False

        return True

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
        # Get the router_logits, expert_affinities and expert_index from the router
        if self._can_use_nki_kernel("router_topk", hidden_states):
            logger.info("Running RouterTopK NKI kernel")
            hidden_states = hidden_states.view(-1, self.hidden_size).transpose(
                0, 1
            )  # flatten and transpose from original [B, S, H] shape
            _, total_tokens = hidden_states.shape
            router_logits = torch.zeros(
                total_tokens,
                self.num_local_experts,
                device=hidden_states.device,
                dtype=torch.float32,
            )  # [T, E]
            expert_affinities = torch.zeros(
                total_tokens,
                self.num_local_experts,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )  # [T, E]
            expert_index = torch.zeros(
                total_tokens,
                self.num_experts_per_tok,
                device=hidden_states.device,
                dtype=torch.int64,
            )  # [T, k]

            grid = (nc(self.logical_nc_config),)
            _router_topk_nki_call[grid](
                x=hidden_states,  # input tensor, [H, T]
                w=self.router.weight_T,  # weight tensor, [H, E]
                k=self.num_experts_per_tok,
                router_logits=router_logits,
                expert_affinities=expert_affinities,
                expert_index=expert_index,
                act_fn=ROUTER_ACT_FN_MAPPING[self.router.act_fn],
                shard_on_hidden=True,
            )
            router_logits = router_logits.to(hidden_states.dtype)
        else:
            logger.info("Running RouterTopK without kernel")
            router_logits, expert_affinities, expert_index = self.router(
                hidden_states
            )
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

        if self._can_use_nki_kernel("expert_mlp", hidden_states):
            logger.info("Running ExpertMLP NKI kernel")
            grid = (nc(self.logical_nc_config),)
            if self.expert_mlps.routed_experts_mlp_config.early_expert_affinity_modulation:
                expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE
            else:
                expert_affinities_scaling_mode = ExpertAffinityScaleMode.POST_SCALE

            # Reshape gate_up_proj_weight to (E, H, 2, I) as expected by the kernel
            gate_up_weights = self.expert_mlps.mlp_op.gate_up_proj.weight  # [E, H, 2I]
            gate_up_weights = gate_up_weights.view(
                self.num_local_experts, self.hidden_size, 2, -1
            )

            _expert_mlp_nki_call = nki.jit(platform_target="trn2")(expert_isa_kernel_wrapper)
            output = _expert_mlp_nki_call[grid](
                inp=hidden_states,  # [T, H]
                gate_up_weights=gate_up_weights,  # [E, H, 2, I]
                down_weights=self.expert_mlps.mlp_op.down_proj.weight,  # [E, I, H]
                expert_affinities=expert_affinities,  # [T, E]
                expert_index=expert_index,  # [T, k]
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                enable_kernel_fusion=False,
            )  # [T, H]
        else:
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
        if self._can_use_nki_kernel("shared_mlp", hidden_states):
            logger.info("Running SharedMLP NKI kernel")
            hidden_states_shape = hidden_states.shape
            out = torch.zeros(
                hidden_states.size(), device=hidden_states.device, dtype=hidden_states.dtype
            )
            grid = (nc(self.logical_nc_config),)
            shared_experts_gate_proj_weight, shared_experts_up_proj_weight, shared_experts_down_proj_weight = self._slice_shared_experts_weights()
            common_args = dict(
                    hidden=hidden_states,  # [B, S, H] or [S, B, H]
                    ln_w=torch.zeros(
                        1, self.hidden_size, device=hidden_states.device
                    ),  # dummy tensor
                    gate_w=shared_experts_gate_proj_weight,  # [H, I]
                    up_w=shared_experts_up_proj_weight,  # [H, I]
                    down_w=shared_experts_down_proj_weight,  # [I, H]
                    out=out,
                    kernel_name="MLP",
                    fused_rmsnorm=False,
                    act_fn=ACT_FN_MAPPING[self.hidden_act],
            )
            if self.config.quantized is True:
                _quant_mlp_nki_call[grid](
                    **common_args,
                    gate_w_scale=self.shared_experts.gate_proj.scale,
                    up_w_scale=self.shared_experts.up_proj.scale,
                    down_w_scale=self.shared_experts.down_proj.scale,
                )
            else:
                _mlp_nki_call[grid](**common_args)
            shared_output = out.view(hidden_states_shape)
        else:
            logger.info("Running SharedMLP without kernel")
            hidden_states_shape = hidden_states.shape
            seq_len = hidden_states_shape[self.sequence_dimension]
            shared_output = self.shared_experts(hidden_states, seq_len)

        return shared_output

    def _moe_fused_tkg_kernel(self, hidden_states):
        """
        Args:
            hidden_states: [B, S, H] or [S, B, H]

        Returns:
            output: original shape
        """
        hidden_states_shape = hidden_states.shape
        if self.expert_mlps.routed_experts_mlp_config.early_expert_affinity_modulation:
            expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE
        else:
            expert_affinities_scaling_mode = ExpertAffinityScaleMode.POST_SCALE

        grid = (nc(self.logical_nc_config),)
        shared_experts_gate_proj_weight, shared_experts_up_proj_weight, shared_experts_down_proj_weight = self._slice_shared_experts_weights()
        common_args = dict(
            inp=hidden_states,  # [B, S, H]
            gamma=self.post_attention_layernorm.weight.unsqueeze(0),  # [1, H]
            router_weights=self.router.weight_T,  # [H, E]
            shared_expert_gate_w=shared_experts_gate_proj_weight,  # [H, I]
            shared_expert_up_w=shared_experts_up_proj_weight,  # [H, I]
            shared_expert_down_w=shared_experts_down_proj_weight,  # [I, H]
            expert_gate_up_weights=self.expert_mlps.mlp_op.gate_up_proj.weight.view(
                self.num_local_experts, self.hidden_size, 2, -1
            ),  # [E, H, 2, I]
            expert_down_weights=self.expert_mlps.mlp_op.down_proj.weight,  # [E, I, H]
            expert_gate_up_weights_scale=(
                self.expert_mlps.mlp_op.gate_up_proj.scale.view(self.num_local_experts, 2, -1)
                if self.config.quantized else None
            ),  # [E, 2, I]
            expert_down_weights_scale=(
                self.expert_mlps.mlp_op.down_proj.scale.view(self.num_local_experts, -1)
                if self.config.quantized else None
            ),  # [E, H]
            eps=self.post_attention_layernorm.variance_epsilon,
            top_k=self.num_experts_per_tok,
            router_act_fn=ROUTER_ACT_FN_MAPPING[self.router.act_fn],
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            router_mm_dtype=nl.float32,
        )

        total_tokens = hidden_states_shape[0] * hidden_states_shape[1]
        perc_experts_loaded = total_tokens * self.num_experts_per_tok / self.num_local_experts
        if perc_experts_loaded >= DEFAULT_SELECTIVE_LOADING_THRESHOLD:
            logger.info("Percentage of experts loaded >= selective loading threshod, run forward all experts kernel")
            routed_experts_mlp_config = self.expert_mlps.routed_experts_mlp_config
            kernel_activation_func_id = get_kernel_activation_func_id(
                ACTFunc.validate(routed_experts_mlp_config.hidden_act),
                routed_experts_mlp_config.glu_type
            )
            logger.info(self.expert_mlps.spmd_rank.get_rank().item())
            out, router_logits = _moe_tkg_forward_all_experts_nki_call[grid](
                **common_args,
                router_bias=self.router.linear_router.bias if self.router.bias else None,
                expert_gate_up_bias=self.expert_mlps.mlp_op.gate_up_proj.bias if routed_experts_mlp_config.bias else None,
                expert_down_bias=self.expert_mlps.mlp_op.down_proj.bias if routed_experts_mlp_config.bias else None,
                shared_expert_gate_bias=None,  # kernel only supports None
                shared_expert_up_bias=None,  # kernel only supports None
                shared_expert_down_bias=None,  # kernel only supports None
                router_pre_norm=True,
                hidden_act_fn=kernel_activation_func_id,
                hidden_act_scale_factor=None,  # kernel only supports None
                hidden_act_bias=None,  # kernel only supports None
                rank_id=self.expert_mlps.spmd_rank.get_rank().item(),
            )
        else:
            logger.info("Percentage of experts loaded < selective loading threshod, run seletive loading kernel")
            out, router_logits = _moe_tkg_selective_loading_nki_call[grid](
                **common_args
            )

        return out.view(hidden_states_shape), router_logits.to(hidden_states.dtype)

    def forward(self, hidden_states):
        """
        Forward through MoE TKG mega-kernel if conditions are satisfied. Otherwise forward through
        flat compiler / individual kernels.

        Conditions for MoE TKG mega-kernel:
        - batch_size <= 32
        - must use RMSNorm, RouterTopK, ExpertMLPs, SharedExperts

        Args:
            hidden_states: [B, S, H] or [S, B, H]

        Returns:
            output: original shape
        """
        if self._can_use_nki_kernel("moe_fused", hidden_states):
            logger.info("Running MoE Fused NKI kernel")
            output, router_logits = self._moe_fused_tkg_kernel(hidden_states)

            if self.return_expert_index:
                expert_index = None  # can't return expert_index from fused kernel
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

        return return_op

    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        pass
