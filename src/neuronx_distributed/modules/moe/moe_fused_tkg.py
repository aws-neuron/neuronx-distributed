import logging
import warnings

import neuronxcc.nki.language as nl
import torch
from neuronxcc import nki
from neuronxcc.nki.language import nc

from neuronx_distributed.modules.moe import MoE
from neuronx_distributed.modules.moe.model_utils import DEFAULT_LNC_SIZE
from neuronx_distributed.modules.moe.moe_configs import MoEFusedTKGConfig
from neuronx_distributed.modules.moe.nki_import import NKIImport, import_nki
from neuronx_distributed.parallel_layers import mappings


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
        "moe_token_gen": NKIImport("moe_token_gen_kernel", module_name="moe_token_gen"),
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
_moe_tkg_nki_call = nki_components["moe_token_gen"]
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


class MoEFusedTKG(torch.nn.Module):
    """
    Fused MoE module for token generation.
    """

    def __init__(
        self,
        moe: MoE,
        config: MoEFusedTKGConfig,
        post_attention_layernorm: torch.nn.Module = None,
        logical_nc_config: int = DEFAULT_LNC_SIZE,
        return_router_logits: bool = False,
        return_expert_index: bool = False,
    ):
        """
        Arguments:
            moe: MoE module object.
            config: Module config object.
            post_attention_layernorm: Optional. If passed, forward method will forward hidden_states through
                                      post_attention_layernorm before moe.
            logical_nc_config: LNC size (1 or 2).
            return_router_logits: Whether to return router logits in addition to final hidden states.
            return_expert_index: Whether to return the expert index from router in the forward pass
        """
        super().__init__()
        self.post_attention_layernorm = post_attention_layernorm
        self.moe = moe
        self.config = config
        self.return_router_logits = return_router_logits
        self.return_expert_index = return_expert_index

        routed_experts_mlp_config = self.moe.expert_mlps.routed_experts_mlp_config
        self.hidden_size = routed_experts_mlp_config.hidden_size
        self.num_local_experts = routed_experts_mlp_config.num_experts
        self.num_experts_per_tok = routed_experts_mlp_config.top_k
        self.hidden_act = routed_experts_mlp_config.hidden_act
        self.sequence_dimension = self.moe.sequence_dimension

        self.logical_nc_config = logical_nc_config

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
            if not self.moe.expert_mlps.routed_experts_mlp_config.glu_mlp:
                logger.info(f"Conditions not met for {kernel_type} NKI kernel: disabling GLU not supported")
                return False
            if self.moe.expert_mlps.routed_experts_mlp_config.normalize_top_k_affinities:
                logger.info(f"Conditions not met for {kernel_type} NKI kernel: normalize top k affinities not supported")
                return False

        if kernel_type in ["moe_fused", "shared_mlp"]:
            if self.moe.shared_experts is None:
                logger.info(f"Conditions not met for {kernel_type} NKI kernel: module does not contain shared experts")
                return False

        if kernel_type == "moe_fused":
            if _moe_tkg_nki_call is None:
                logger.info("Failed to load MoE Fused NKI kernel")
                return False
            if hidden_states.shape[1 - self.sequence_dimension] > 4:
                logger.info(f"Conditions not met for {kernel_type} NKI kernel: bs > 4 not supported")
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
        shared_experts = self.moe.shared_experts
        if shared_experts.sequence_parallel_enabled:
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
                w=self.moe.router.linear_router.weight.transpose(
                    0, 1
                ),  # weight tensor, [H, E]
                k=self.num_experts_per_tok,
                router_logits=router_logits,
                expert_affinities=expert_affinities,
                expert_index=expert_index,
                act_fn=ROUTER_ACT_FN_MAPPING[self.moe.router.act_fn],
                shard_on_hidden=True,
            )
            router_logits = router_logits.to(hidden_states.dtype)
        else:
            logger.info("Running RouterTopK without kernel")
            router_logits, expert_affinities, expert_index = self.moe.router(
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
            if self.moe.expert_mlps.routed_experts_mlp_config.early_expert_affinity_modulation:
                expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE
            else:
                expert_affinities_scaling_mode = ExpertAffinityScaleMode.POST_SCALE

            # Reshape gate_up_proj_weight to (E, H, 2, I) as expected by the kernel
            gate_up_weights = self.moe.expert_mlps.mlp_op.gate_up_proj.weight  # [E, H, 2I]
            gate_up_weights = gate_up_weights.view(
                self.num_local_experts, self.hidden_size, 2, -1
            )

            _expert_mlp_nki_call = nki.jit(platform_target="trn2")(expert_isa_kernel_wrapper)
            output = _expert_mlp_nki_call[grid](
                inp=hidden_states,  # [T, H]
                gate_up_weights=gate_up_weights,  # [E, H, 2, I]
                down_weights=self.moe.expert_mlps.mlp_op.down_proj.weight,  # [E, I, H]
                expert_affinities=expert_affinities,  # [T, E]
                expert_index=expert_index,  # [T, k]
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                enable_kernel_fusion=False,
            )  # [T, H]
        else:
            logger.info("Running ExpertMLP without kernel")
            seq_len = hidden_states_shape[self.sequence_dimension]
            output = self.moe.expert_mlps(
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
                    gate_w_scale=self.moe.shared_experts.gate_proj.scale,
                    up_w_scale=self.moe.shared_experts.up_proj.scale,
                    down_w_scale=self.moe.shared_experts.down_proj.scale,
                )
            else:
                _mlp_nki_call[grid](**common_args)
            shared_output = out.view(hidden_states_shape)
        else:
            logger.info("Running SharedMLP without kernel")
            hidden_states_shape = hidden_states.shape
            seq_len = hidden_states_shape[self.sequence_dimension]
            shared_output = self.moe.shared_experts(hidden_states, seq_len)

        return shared_output

    def _moe_fused_tkg_kernel(self, hidden_states):
        """
        Args:
            hidden_states: [B, S, H] or [S, B, H]

        Returns:
            output: original shape
        """
        hidden_states_shape = hidden_states.shape
        if self.moe.expert_mlps.routed_experts_mlp_config.early_expert_affinity_modulation:
            expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE
        else:
            expert_affinities_scaling_mode = ExpertAffinityScaleMode.POST_SCALE

        grid = (nc(self.logical_nc_config),)
        shared_experts_gate_proj_weight, shared_experts_up_proj_weight, shared_experts_down_proj_weight = self._slice_shared_experts_weights()
        out, router_logits = _moe_tkg_nki_call[grid](
            inp=hidden_states,  # [B, S, H]
            gamma=self.post_attention_layernorm.weight.unsqueeze(0),  # [1, H]
            router_weights=self.moe.router.linear_router.weight.transpose(
                0, 1
            ),  # [H, E]
            shared_expert_gate_w=shared_experts_gate_proj_weight,  # [H, I]
            shared_expert_up_w=shared_experts_up_proj_weight,  # [H, I]
            shared_expert_down_w=shared_experts_down_proj_weight,  # [I, H]
            expert_gate_up_weights=self.moe.expert_mlps.mlp_op.gate_up_proj.weight.view(
                self.num_local_experts, self.hidden_size, 2, -1
            ),  # [E, H, 2, I]
            expert_down_weights=self.moe.expert_mlps.mlp_op.down_proj.weight,  # [E, I, H]
            expert_gate_up_weights_scale=(
                self.moe.expert_mlps.mlp_op.gate_up_proj.scale.view(self.num_local_experts, 2, -1)
                if self.config.quantized else None
            ),  # [E, 2, I]
            expert_down_weights_scale=(
                self.moe.expert_mlps.mlp_op.down_proj.scale.view(self.num_local_experts, -1)
                if self.config.quantized else None
            ),  # [E, H]
            eps=self.post_attention_layernorm.variance_epsilon,
            top_k=self.num_experts_per_tok,
            router_act_fn=ROUTER_ACT_FN_MAPPING[self.moe.router.act_fn],
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            router_mm_dtype=nl.float32,
        )
        return out.view(hidden_states_shape), router_logits.to(hidden_states.dtype)

    def forward(self, hidden_states):
        """
        Forward through MoE TKG mega-kernel if conditions are satisfied. Otherwise forward through
        flat compiler / individual kernels.

        Conditions for MoE TKG mega-kernel:
        - batch_size <= 4
        - must use RMSNorm, RouterTopK, ExpertMLPs, SharedExperts

        Args:
            hidden_states: [B, S, H] or [S, B, H]

        Returns:
            output: original shape
        """
        if self._can_use_nki_kernel("moe_fused", hidden_states):
            logger.info("Running MoE Fused NKI kernel")
            output, router_logits = self._moe_fused_tkg_kernel(hidden_states)
            output = mappings.reduce_from_tensor_model_parallel_region(
                output, process_group=self.moe.tensor_parallel_group
            )
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
            if self.moe.shared_experts:
                shared_output = self._shared_mlp(hidden_states)
            output = self._expert_mlp(
                hidden_states=hidden_states,
                expert_affinities=expert_affinities,
                expert_index=expert_index
            )
            output = output + shared_output

            # Delayed All-Reduce
            output = mappings.reduce_from_tensor_model_parallel_region(
                output, process_group=self.moe.tensor_parallel_group
            )

        return_op = (output,)
        if self.return_router_logits:
            return_op += (router_logits,)
        if self.return_expert_index:
            return_op += (expert_index,)

        return return_op
