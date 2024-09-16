import torch

from neuronx_distributed.modules.moe import expert_mlps, routing
from neuronx_distributed.parallel_layers import mappings, parallel_state


class MoE(torch.nn.Module):
    """Module which implements a Mixture-of-Experts layer.

    Example usage (Mixtral model configuration):
    ```
        # Mixtral specific configuration
        num_experts = 8
        top_k = 2
        hidden_size = 4096
        intermediate_size = 14336
        glu_mlp = True
        hidden_act = "silu"
        normalize_top_k_affinities = True

        # Other configurations
        capacity_factor = None   # Full capacity with no token dropping
        init_method = lambda weight: torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        output_layer_init_method = lambda weight: torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        sequence_parallel_enabled = True

        # Initialize router
        router = routing.RouterTopK(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
        )

        # Initialize expert_mlps
        expert_mlps = expert_mlps.ExpertMLPs(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            glu_mlp=glu_mlp,
            capacity_factor=capacity_factor,
            normalize_top_k_affinities=normalize_top_k_affinities,
            init_method=init_method,
            output_layer_init_method=init_method,
        )

        # Initial moe_layer
        moe_layer = MoE(
            router=router,
            expert_mlps=expert_mlps,
            return_router_logits=True,  # Required downstream for the load balancing loss function
            sequence_parallel_enabled=sequence_parallel_enabled,
        )
    ```

    Arguments:
        router: Determines expert routing for input tokens
        expert_mlps: Obtains the output of the MoE layer by passing tokens through the chosen experts
        sequence_parallel_enabled: Whether the model is running in sequence parallel or not
        return_router_logits: Whether to return the router logits in the forward pass
    """

    # Flag used in testing. Should not be used in production.
    is_test = False

    def __init__(
        self,
        router: routing.RouterBase,
        expert_mlps: expert_mlps.ExpertMLPs,
        sequence_parallel_enabled: bool = False,
        return_router_logits: bool = False,
    ):
        super().__init__()

        for attr in ["num_experts", "top_k", "hidden_size"]:
            if getattr(router, attr) != getattr(expert_mlps, attr):
                raise ValueError("Inconsistent {attr} across the router and expert_mlps")

        self.router = router
        self.expert_mlps = expert_mlps
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.return_router_logits = return_router_logits
        self.ep_enabled = parallel_state.get_expert_model_parallel_size() > 1

    def forward(self, hidden_states):
        """Forward pass of the MoE layer.

        Common nomenclature:
            S: Sequence length, B: Batch size, H: Hidden Size
            S': Sequence length (when the input is in SP)
            T: Tokens = S * B (token dimension obtained by flattening S and B)

        Arguments:
            hidden_states: Input tensor of shape (S, B, H) or (S', B, H) in training, (B, S, H) in inference.

        Returns:
            output: Output tensor of the same shape as hidden_states, containing the output of the MoE layer.
            bias: (Optional) Returned if expert_mlps.return_bias is True. Currently bias is not supported for the MoE layer.
            router_logits: (Optional) Tensor of shape (T, E) containing the router logits for each token.
                                      Returned if self.return_router_logits is True.
            expert_index: (Optional) Tensor of shape (T, E) containing the experts assigned to each token.
                                     Returned if self.is_test is True.
        """

        # hidden_states: (S, B, H) or (S', B, H) in training, (B, S, H) in inference

        if not self.training:
            # Sequence parallelism is only supported for training
            assert self.sequence_parallel_enabled is False, "SP is not currently supported for inference"

        if self.sequence_parallel_enabled:
            # All-Gather the hidden_states to exit sequence parallel
            # full_hidden_states: (S', B, H) -> (S, B, H)
            full_hidden_states = mappings.gather_from_sequence_parallel_region(hidden_states, to_model_parallel=False)
        else:
            full_hidden_states = hidden_states

        # full_hidden_states: (S, B, H) in training, (B, S, H) in inference
        full_hidden_states_shape = full_hidden_states.shape
        seq_len = full_hidden_states_shape[0] if self.training else full_hidden_states_shape[1]

        # full_hidden_states: (T, H)
        full_hidden_states = full_hidden_states.view(-1, full_hidden_states.shape[-1])

        # Get the router_logits, expert_affinities and expert_index from the router
        # router_logits: (T, E), expert_affinities: (T, E), expert_index: (T, top_k)
        router_logits, expert_affinities, expert_index = self.router(full_hidden_states)

        # Get the output from the ExpertMLPs: (T, H)
        output = self.expert_mlps(
            hidden_states=full_hidden_states,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            seq_len=seq_len,
        )

        # output: (S, B, H) in training, (B, S, H) in inference
        output = output.view(full_hidden_states_shape)

        if self.sequence_parallel_enabled and self.ep_enabled:
            # Reduction is done earlier in the case of EP
            output = mappings.scatter_to_sequence_parallel_region(output)
        elif self.sequence_parallel_enabled:
            # Reduce-scatter back to sequence parallel (as the hidden_states were in sequence parallel)
            # output: (S, B, H) -> (S', B, H)
            output = mappings.reduce_scatter_to_sequence_parallel_region(output)
        elif not self.ep_enabled:
            # output: (S, B, H) in training, (B, S, H) in inference
            output = mappings.reduce_from_tensor_model_parallel_region(output)

        return_op = (output,)
        if self.expert_mlps.return_bias:
            return_op += (None,)
        if self.return_router_logits:
            return_op += (router_logits,)
        if self.is_test:
            return_op += (expert_index,)

        return return_op
