import torch
import transformers

# check version of transformers library
from packaging import version

transformers_ver = transformers.__version__
if version.parse(transformers_ver) < version.parse("4.36.0"):
    assert False, f"transformers library version is {transformers_ver}. Minimum required is 4.36.0"

from transformers.models.mixtral.configuration_mixtral import MixtralConfig  # noqa: E402
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock  # noqa: E402

from . import utils_testing as ut  # noqa: E402


def initialize_mixtral_model(cfg, seed=5):
    assert cfg.implementation == "topk"
    assert cfg.glu_mlp is True, "Mixtral implementation available only for GLU MLP"

    mixtral_config = MixtralConfig()
    mixtral_config.hidden_size = cfg.hidden_size
    mixtral_config.intermediate_size = ut.get_intermediate_size(cfg)
    mixtral_config.num_local_experts = cfg.num_experts
    mixtral_config.num_experts_per_tok = cfg.top_k

    mixtral_model = MixtralSparseMoeBlock(mixtral_config)

    # Move model to required device
    mixtral_model = mixtral_model.to(device=cfg.device)
    return mixtral_model


def convert_mixtral_to_neuron_state_dict(mixtral_state_dict, cfg):
    """
    Helper function which returns the model weights from the Mixtral MoE model in a state dictionary compatible with the stucture of the neuron MoE model.

    Note that the gradients of the mixtral experts are None if no token is assigned to them.
    This function implements workarounds for this.
    """

    assert cfg.glu_mlp is True, "Only GPU MLP is supported for Mixtral Top-K model"

    neuron_state_dict = {}
    # Copy router weights
    neuron_state_dict["router.linear_router.weight"] = mixtral_state_dict["gate.weight"].clone().detach()

    intermediate_size, hidden_size = ut.get_intermediate_size(cfg), cfg.hidden_size
    device = cfg.device

    # copy the MLP parameters
    gate_up_proj = torch.empty(cfg.num_experts, hidden_size, 2 * intermediate_size, device=device)
    for e in range(cfg.num_experts):
        # Copy gate_proj and up_proj after concatenation
        gate_proj_weights = mixtral_state_dict[f"experts.{e}.w1.weight"].T
        up_proj_weights = mixtral_state_dict[f"experts.{e}.w3.weight"].T
        if gate_proj_weights is None:
            assert up_proj_weights is None
        else:
            gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
            gate_up_proj_weights = torch.cat([gate_proj_weights, up_proj_weights], dim=1)
            gate_up_proj_slice.copy_(gate_up_proj_weights)
    neuron_state_dict["expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

    down_proj = torch.empty(cfg.num_experts, intermediate_size, hidden_size, device=device)
    for e in range(cfg.num_experts):
        # Copy down_proj
        down_proj_weights = mixtral_state_dict[f"experts.{e}.w2.weight"].T
        if down_proj_weights is not None:
            down_proj_slice = torch.narrow(down_proj, 0, e, 1)
            down_proj_slice.copy_(down_proj_weights)
    neuron_state_dict["expert_mlps.mlp_op.down_proj.weight"] = down_proj

    return neuron_state_dict
