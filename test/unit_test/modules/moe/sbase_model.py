import torch

from neuronx_distributed.modules.moe import ACT2FN, RouterSinkhorn

from . import utils_testing as ut


class MLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, glu_mlp):
        super().__init__()
        self.glu_mlp = glu_mlp
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)
        if glu_mlp:
            self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        assert hidden_act in ACT2FN
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        if self.glu_mlp:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)), None
        else:
            return self.down_proj(self.act_fn(self.up_proj(x))), None


class SBaseMoE(torch.nn.Module):
    """
    Top-1 MoE with Sinkhorn based expert routing.
    Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/transformer.py
    """

    is_test = False

    def __init__(self, hidden_size, intermediate_size, num_experts, hidden_act, glu_mlp):
        super().__init__()
        self.router = torch.nn.Linear(hidden_size, num_experts, bias=False)
        self.num_experts = num_experts
        self.experts = torch.nn.ModuleList(
            [MLP(hidden_size, intermediate_size, hidden_act, glu_mlp) for _ in range(num_experts)]
        )

    def forward(self, hidden_states):
        hidden_shape = hidden_states.shape

        route = self.router(hidden_states).view(-1, self.num_experts)
        if self.training:
            with torch.no_grad():
                sinkroute = RouterSinkhorn._sinkhorn(
                    route.detach().to(dtype=torch.float32), num_iters=RouterSinkhorn.DEFAULT_SINKHORN_ITERS
                )
                _, max_ind = torch.max(sinkroute, dim=1)
            route = torch.sigmoid(route)
            max_prob = route[torch.arange(route.size(0)), max_ind]
        else:
            route = torch.sigmoid(route)
            max_prob, max_ind = torch.max(route, dim=1)

        max_prob = torch.unsqueeze(max_prob, 1)
        hidden_states = hidden_states.view(-1, hidden_shape[-1])

        output_total = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        for expert_num, expert in enumerate(self.experts):
            local_indices = (max_ind == expert_num).nonzero()
            hidden = hidden_states[local_indices, :]
            output, output_bias = expert(hidden)
            output_total[local_indices, :] = output

        output_total = output_total * max_prob
        output_total = output_total.view(hidden_shape)

        if self.is_test:
            return output_total, None, max_ind.unsqueeze(1)
        else:
            return output_total, None  # no bias for MLP


def initialize_sbase_model(cfg, seed=5):
    assert cfg.implementation == "sbase"

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    sbase_model = SBaseMoE(
        hidden_size=cfg.hidden_size,
        intermediate_size=ut.get_intermediate_size(cfg),
        num_experts=cfg.num_experts,
        hidden_act=cfg.hidden_act,
        glu_mlp=cfg.glu_mlp,
    )
    # Move model to required device
    sbase_model = sbase_model.to(device=cfg.device, dtype=cfg.dtype)
    return sbase_model


def convert_sbase_to_neuron_state_dict(sbase_state_dict, cfg):
    """
    Helper function which returns the model weights from the sbase MoE model in a state dictionary compatible with the stucture of the neuron MoE model.
    """

    neuron_state_dict = {}
    # Copy router weights
    neuron_state_dict["router.linear_router.weight"] = sbase_state_dict["router.weight"].clone().detach()

    intermediate_size, hidden_size = sbase_state_dict["experts.0.up_proj.weight"].shape
    device = sbase_state_dict["experts.0.up_proj.weight"].device

    # copy the MLP parameters
    if cfg.glu_mlp:
        gate_up_proj = torch.empty(cfg.num_experts, hidden_size, 2 * intermediate_size, device=device, dtype=cfg.dtype)
        for e in range(cfg.num_experts):
            # Copy gate_proj and up_proj after concatenation
            gate_proj_weights = sbase_state_dict[f"experts.{e}.gate_proj.weight"].T
            up_proj_weights = sbase_state_dict[f"experts.{e}.up_proj.weight"].T
            gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
            gate_up_proj_weights = torch.cat([gate_proj_weights, up_proj_weights], dim=1)
            gate_up_proj_slice.copy_(gate_up_proj_weights)
        neuron_state_dict["expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj
    else:
        up_proj = torch.empty(cfg.num_experts, hidden_size, intermediate_size, device=device, dtype=cfg.dtype)
        for e in range(cfg.num_experts):
            # Copy up_proj
            up_proj_weights = sbase_state_dict[f"experts.{e}.up_proj.weight"].T
            up_proj_slice = torch.narrow(up_proj, 0, e, 1)
            up_proj_slice.copy_(up_proj_weights)
        neuron_state_dict["expert_mlps.mlp_op.up_proj.weight"] = up_proj

    down_proj = torch.empty(cfg.num_experts, intermediate_size, hidden_size, device=device, dtype=cfg.dtype)
    for e in range(cfg.num_experts):
        # Copy down_proj
        down_proj_weights = sbase_state_dict[f"experts.{e}.down_proj.weight"].T
        down_proj_slice = torch.narrow(down_proj, 0, e, 1)
        down_proj_slice.copy_(down_proj_weights)
    neuron_state_dict["expert_mlps.mlp_op.down_proj.weight"] = down_proj

    return neuron_state_dict
