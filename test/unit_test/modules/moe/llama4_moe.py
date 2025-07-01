import torch
from transformers.models.llama4 import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe

from . import utils_testing as ut  # noqa: E402

def initialize_llama4_text_moe(cfg: ut.ExptCfg, seed=5):
    torch.manual_seed(seed)
    llama4_config = Llama4TextConfig()
    llama4_config.hidden_size = cfg.hidden_size
    llama4_config.intermediate_size = ut.get_intermediate_size(cfg)
    llama4_config.num_local_experts = cfg.num_experts
    llama4_config.num_experts_per_tok = cfg.top_k
    llama4_config.torch_dtype = cfg.dtype
    llama4_moe = Llama4TextMoe(llama4_config)
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
    return neuron_state_dict