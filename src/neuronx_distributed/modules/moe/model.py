from typing import Optional

import torch
from torch.distributed import ProcessGroup

from neuronx_distributed.modules.moe import expert_mlps, routing, token_shuffling
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
        token_shuffle_group_size: Size of token shuffling group. If size=1, token shuffling is disabled.
        1 <= token_shuffle_group_size <= dp_size.
        token_shuffle_seed: Seed for token shuffling. If None, a random seed is used.
    """

    # Flag used in testing. Should not be used in production.
    is_test = False

    def __init__(
        self,
        router: routing.RouterBase,
        expert_mlps: expert_mlps.ExpertMLPs,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        return_router_logits: bool = False,
        token_shuffle_group_size: int = 1,  # disable token shuffle by default
        token_shuffle_seed=None,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
    ):
        super().__init__()

        for attr in ["num_experts", "top_k", "hidden_size"]:
            if getattr(router, attr) != getattr(expert_mlps, attr):
                raise ValueError("Inconsistent {attr} across the router and expert_mlps")

        if router.sequence_parallel_enabled:
            if not sequence_parallel_enabled:
                raise ValueError("MoE layer must have SP enabled to run router in SP")
            if router.sequence_dimension != sequence_dimension:
                raise ValueError(f"Inconsistent sequence_dimension across MoE and router modules, {router.sequence_dimension} != {sequence_dimension}")

        self.router = router
        self.expert_mlps = expert_mlps

        self.sequence_parallel_enabled = sequence_parallel_enabled
        if sequence_dimension is None:
            # Default to 0
            sequence_dimension = 0
        self.sequence_dimension = sequence_dimension

        self.return_router_logits = return_router_logits
        self.ep_enabled = parallel_state.get_expert_model_parallel_size() > 1
        self.token_shuffle_group_size = token_shuffle_group_size
        self.token_shuffle_seed = token_shuffle_seed
        self.shuffle_permutation = None
        if self.token_shuffle_group_size > 1:
            parallel_state.initialize_token_shuffle_group(self.token_shuffle_group_size)

        self.tensor_parallel_group = tensor_model_parallel_group if \
            tensor_model_parallel_group is not None else parallel_state.get_tensor_model_parallel_group()

    def forward(self, hidden_states):
        """Forward pass of the MoE layer.

        Common nomenclature:
            S: Sequence length, B: Batch size, H: Hidden Size
            S': Sequence length (when the input is in SP)
            T: Tokens = S * B (token dimension obtained by flattening S and B)

        Layout of input hidden_states:
            - In the training flow,
                With SP enabled  : (S', B, H)
                With SP disabled : (B, S, H)
            - In the inference flow,
                With SP enabled  : (B, S', H)
                With SP disabled : (B, S, H)

        Arguments:
            hidden_states: Input tensor (of shape as described above)

        Returns:
            output: Output tensor of the same shape as hidden_states, containing the output of the MoE layer.
            bias: (Optional) Returned if expert_mlps.return_bias is True. Currently bias is not supported for the MoE layer.
            router_logits: (Optional) Tensor of shape (T, E) containing the router logits for each token.
                                      Returned if self.return_router_logits is True.
            expert_index: (Optional) Tensor of shape (T, E) containing the experts assigned to each token.
                                     Returned if self.is_test is True.
        """

        if self.token_shuffle_group_size > 1:
            hidden_states, shuffle_permutation = token_shuffling.token_shuffle(
                hidden_states, seed=self.token_shuffle_seed
            )
            if self.token_shuffle_seed is not None:
                # store for debugging purpose, which is when user specifies a seed
                self.shuffle_permutation = shuffle_permutation

        if self.sequence_parallel_enabled:
            # All-Gather the hidden_states to exit sequence parallel
            # full_hidden_states: (S', B, H) -> (S, B, H)
            full_hidden_states = mappings.gather_from_sequence_parallel_region(
                hidden_states,
                sequence_dimension=self.sequence_dimension,
                to_model_parallel=False,
                process_group=self.tensor_parallel_group,
            )
        else:
            full_hidden_states = hidden_states

        # full_hidden_states: (S, B, H) or (B, S, H)
        full_hidden_states_shape = full_hidden_states.shape
        seq_len = full_hidden_states_shape[self.sequence_dimension]

        # Get the router_logits, expert_affinities and expert_index from the router
        # router_logits: (T, E), expert_affinities: (T, E), expert_index: (T, top_k)
        if self.router.sequence_parallel_enabled:
            router_logits, expert_affinities, expert_index = self.router(hidden_states)
        else:
            router_logits, expert_affinities, expert_index = self.router(full_hidden_states)

        if not self.ep_enabled:
            # All-Reduce expert_affinities gradients in backward pass, to account for delayed output All-Reduce
            expert_affinities = mappings.copy_to_tensor_model_parallel_region(expert_affinities)

        # full_hidden_states: (S, B, H) or (B, S, H) -> (T, H)
        full_hidden_states = full_hidden_states.reshape(-1, full_hidden_states_shape[-1])

        # Get the output from the ExpertMLPs
        output = self.expert_mlps(
            hidden_states=full_hidden_states,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            seq_len=seq_len,
        )

        # output: (T, H) -> (S, B, H) or (B, S, H)
        output = output.view(full_hidden_states_shape)

        if self.sequence_parallel_enabled and self.ep_enabled:
            # Reduction is done earlier in the case of EP
            output = mappings.scatter_to_sequence_parallel_region(
                output, self.sequence_dimension, process_group=self.tensor_parallel_group,
            )
        elif self.sequence_parallel_enabled:
            # Delayed reduce-scatter back to sequence parallel (as the hidden_states were in SP)
            output = mappings.reduce_scatter_to_sequence_parallel_region(
                output, self.sequence_dimension, process_group=self.tensor_parallel_group,
            )
        elif not self.ep_enabled:
            # Delayed All-Reduce
            output = mappings.reduce_from_tensor_model_parallel_region(
                output, process_group=self.tensor_parallel_group
            )

        if self.token_shuffle_group_size > 1:
            output = token_shuffling.token_unshuffle(output, shuffle_permutation)

        return_op = (output,)
        if self.expert_mlps.return_bias:
            return_op += (None,)
        if self.return_router_logits:
            return_op += (router_logits,)
        if self.is_test:
            return_op += (expert_index,)

        return return_op
