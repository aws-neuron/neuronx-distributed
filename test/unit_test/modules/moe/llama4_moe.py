import torch
from torch import nn
from transformers.models.llama4 import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe, Llama4TextRMSNorm

from . import utils_testing as ut  # noqa: E402

class Llama4FusedMoe(Llama4TextMoe):
    def __init__(self, config, moe_fused_tkg_enabled=False):
        super().__init__(config)
        if moe_fused_tkg_enabled:
            self.rmsnorm = Llama4TextRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.rmsnorm = None

        if moe_fused_tkg_enabled is True:
            self.moe_fused_tkg = MoEFused(
                config=config,
                rmsnorm=self.rmsnorm,
                router=self.router,
                shared_experts=self.shared_expert,
                expert_mlps=self.experts,
            )

    def forward(self, hidden_states):
        if self.rmsnorm:
            hidden_states = self.rmsnorm(hidden_states)
        return super().forward(hidden_states)

class MoEFused(nn.Module):
    def __init__(self, config, router, expert_mlps, shared_experts=None, rmsnorm=None):
        super().__init__()
        self.config = config
        self.router = router
        self.expert_mlps = expert_mlps
        self.shared_experts = shared_experts
        self.rmsnorm = rmsnorm


def initialize_llama4_text_moe(cfg: ut.ExptCfg, seed=5):
    torch.manual_seed(seed)
    llama4_config = Llama4TextConfig()
    llama4_config.hidden_size = cfg.hidden_size
    llama4_config.intermediate_size = ut.get_intermediate_size(cfg)
    llama4_config.num_local_experts = cfg.num_experts
    llama4_config.num_experts_per_tok = cfg.top_k
    llama4_config.torch_dtype = cfg.dtype
    llama4_config.rms_norm_eps = cfg.rms_norm_eps
    llama4_moe = Llama4FusedMoe(llama4_config, cfg.moe_fused_tkg_enabled)

    # Move model to required device
    llama4_moe = llama4_moe.to(device=cfg.device)
    return llama4_moe


def convert_llama4_moe_to_neuron_state_dict(llama4_state_dict, cfg):
    """
    Helper function which returns the model weights from the LLama4 MoE model in a state dictionary compatible with the stucture of the neuron MoE model.
    This function implements workarounds for this.
    """
    neuron_state_dict = {}
    neuron_state_dict["router.linear_router.weight"] = llama4_state_dict["router.weight"].clone().detach()
    neuron_state_dict["expert_mlps.mlp_op.gate_up_proj.weight"] = llama4_state_dict["experts.gate_up_proj"].clone().detach()
    neuron_state_dict["expert_mlps.mlp_op.down_proj.weight"] = llama4_state_dict["experts.down_proj"].clone().detach()
    neuron_state_dict["shared_experts.gate_proj.weight"] = llama4_state_dict["shared_expert.gate_proj.weight"].clone().detach()
    neuron_state_dict["shared_experts.up_proj.weight"] = llama4_state_dict["shared_expert.up_proj.weight"].clone().detach()
    neuron_state_dict["shared_experts.down_proj.weight"] = llama4_state_dict["shared_expert.down_proj.weight"].clone().detach()

    if cfg.moe_fused_tkg_enabled:
        neuron_state_dict["rmsnorm.weight"] = llama4_state_dict["rmsnorm.weight"].clone().detach()
        neuron_state_dict["router.weight_T"] = llama4_state_dict["router.weight"].T.clone().detach()
        # HF knows to reuse the same weights for TKG module but we need to explicitly load these weights in our implementation
        neuron_state_dict["moe_fused_tkg.router.weight_T"] = llama4_state_dict["router.weight"].T.clone().detach()
        neuron_state_dict["moe_fused_tkg.post_attention_layernorm.weight"] = llama4_state_dict["rmsnorm.weight"].clone().detach()
        neuron_state_dict["moe_fused_tkg.router.linear_router.weight"] = llama4_state_dict["router.weight"].clone().detach()
        neuron_state_dict["moe_fused_tkg.expert_mlps.mlp_op.gate_up_proj.weight"] = llama4_state_dict["experts.gate_up_proj"].clone().detach()
        neuron_state_dict["moe_fused_tkg.expert_mlps.mlp_op.down_proj.weight"] = llama4_state_dict["experts.down_proj"].clone().detach()
        neuron_state_dict["moe_fused_tkg.shared_experts.gate_proj.weight"] = llama4_state_dict["shared_expert.gate_proj.weight"].clone().detach()
        neuron_state_dict["moe_fused_tkg.shared_experts.up_proj.weight"] = llama4_state_dict["shared_expert.up_proj.weight"].clone().detach()
        neuron_state_dict["moe_fused_tkg.shared_experts.down_proj.weight"] = llama4_state_dict["shared_expert.down_proj.weight"].clone().detach()
    return neuron_state_dict
