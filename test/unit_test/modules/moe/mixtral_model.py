import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F

# check version of transformers library
from packaging import version

transformers_ver = transformers.__version__
if version.parse(transformers_ver) < version.parse("4.36.0"):
    assert False, f"transformers library version is {transformers_ver}. Minimum required is 4.36.0"

from transformers.models.mixtral.configuration_mixtral import MixtralConfig  # noqa: E402
from transformers.activations import ACT2FN # noqa: E402
from . import utils_testing as ut  # noqa: E402

# add new features for hf 
class extendMixtralConfig(MixtralConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.early_expert_affinity_modulation = kwargs.pop("early_expert_affinity_modulation", False)  
        

def initialize_mixtral_model(cfg, seed=5):
    assert cfg.implementation == "topk"
    assert cfg.glu_mlp is True, "Mixtral implementation available only for GLU MLP"

    mixtral_config = extendMixtralConfig()
    mixtral_config.hidden_size = cfg.hidden_size
    mixtral_config.intermediate_size = ut.get_intermediate_size(cfg)
    mixtral_config.num_local_experts = cfg.num_experts
    mixtral_config.num_experts_per_tok = cfg.top_k
    mixtral_config.early_expert_affinity_modulation = cfg.early_expert_affinity_modulation

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

class MixtralSparseMoeBlock(nn.Module):
    """
    The original implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters not enabled
        self.jitter_noise = 0
        # adding early_expert_affinity_modulation
        self.early_expert_affinity_modulation = config.early_expert_affinity_modulation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            # prescaling before expertMLP
            if self.early_expert_affinity_modulation:
                current_state = (hidden_states[None, top_x].reshape(-1, hidden_dim) * routing_weights[top_x, idx, None])
                current_hidden_states = expert_layer(current_state)
            else:
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

# incldude the MixtralBLockSparseTop2MLP since training use a older version of transformers
class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states
