import torch
from typing import Union

from neuronx_distributed.modules.moe import expert_mlps, routing
from neuronx_distributed.modules.moe.model_utils import MoESequenceParallelMode
from neuronx_distributed.parallel_layers import mappings


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
        capacity_factor = 4.0   # Full capacity with no token dropping, set to num_experts/top_k
        sequence_parallel_mode = MoESequenceParallelMode.EXIT_SP_ON_ENTRY
        permute_strategy = "matmul"
        init_method = lambda weight: torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        output_layer_init_method = lambda weight: torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # Initialize router
        router = routing.RouterTopK(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            sequence_parallel_mode=sequence_parallel_mode,
        )

        # Initialize expert_mlps
        expert_mlps_cf = expert_mlps.ExpertMLPsCapacityFactor(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            capacity_factor=capacity_factor,
            init_method=init_method,
            output_layer_init_method=init_method,
            glu_mlp=glu_mlp,
            sequence_parallel_mode=sequence_parallel_mode,
            permute_strategy=permute_strategy,
            normalize_top_k_affinities=normalize_top_k_affinities,
        )

        # Initial moe_layer
        moe_layer = MoE(
            router=router,
            expert_mlps=expert_mlps_cf,
            return_router_logits=True,  # Required downstream for the load balancing loss function
            sequence_parallel_mode=sequence_parallel_mode,
        )
    ```

    Due to difference between training and inference on SP, SP is implementated differently for them in MoE.
    Note that the NO_SP mode is equivalent to the EXIT_SP_ON_ENTRY mode under the inference assumptions. (There
    are no additional collectives for EXIT_SP_ON_ENTRY).

    Arguments:
        router: Determines expert routing for input tokens
        expert_mlps: Obtains the output of the MoE layer by passing tokens through the chosen experts
        return_router_logits: Whether to return the router logits in the forward pass
        sequence_parallel_mode: SP mode being used for the MoE layer.
    """

    # Flag used in testing. Should not be used in production.
    is_test = False

    def __init__(
        self,
        router: routing.RouterBase,
        expert_mlps: expert_mlps.ExpertMLPsBase,
        sequence_parallel_mode: Union[str, MoESequenceParallelMode],
        return_router_logits: bool = False,
    ):
        super().__init__()

        if sequence_parallel_mode not in MoESequenceParallelMode.__members__:
            raise TypeError(f"Unknown sequence_parallel_mode: {sequence_parallel_mode}")
        if len({sequence_parallel_mode, router.sequence_parallel_mode, expert_mlps.sequence_parallel_mode}) > 1:
            raise ValueError("Inconsistent SP modes across router, expert_mlps and MoE modules")
        for attr in ["num_experts", "top_k", "hidden_size"]:
            if getattr(router, attr) != getattr(expert_mlps, attr):
                raise ValueError("Inconsistent {attr} across the router and expert_mlps")

        self.router = router
        self.expert_mlps = expert_mlps
        self.sequence_parallel_mode = MoESequenceParallelMode[sequence_parallel_mode]
        self.return_router_logits = return_router_logits

    def forward(self, hidden_states):
        """Forward pass of the MoE layer.

        Common nomenclature:
            S: Sequence length, B: Batch size, H: Hidden Size
            S': Sequence length (when the input is in SP)
            T: Tokens = S * B (token dimension obtained by flattening S and B)

        Arguments:
            hidden_states: Input tensor of shape (S, B, H) or (S', B, H)

        Returns:
            output: Output tensor of the same shape as hidden_states, containing the output of the MoE layer.
            bias: (Optional) Returned if expert_mlps.return_bias is True. Currently bias is not supported for the MoE layer.
            router_logits: (Optional) Tensor of shape (T, E) containing the router logits for each token.
                                      Returned if self.return_router_logits is True.
            expert_index: (Optional) Tensor of shape (T, E) containing the experts assigned to each token.
                                     Returned if self.is_test is True.
        """

        # Sequence parallelism is supported for training, but not for inference, so we need different branches for them.
        # However, we may still want to run the MoE module in a particular SP mode, which requires the collective
        # operations to be adjusted (compared to the base class).
        if self.training:
            output, router_logits, expert_index = self.forward_for_training(hidden_states)
        else:
            output, router_logits, expert_index = self.forward_for_inference(hidden_states)

        return_op = (output,)
        if self.expert_mlps.return_bias:
            return_op += (None,)
        if self.return_router_logits:
            return_op += (router_logits,)
        if self.is_test:
            return_op += (expert_index,)

        return return_op

    def forward_for_training(self, hidden_states):
        if self.sequence_parallel_mode in {
            MoESequenceParallelMode.EXIT_SP_ON_ENTRY,
            MoESequenceParallelMode.EXIT_SP_ON_ENTRY_DELAY_MLP_AR,
        }:
            # All-Gather the hidden_states to exit sequence parallel
            # hidden_states: (S', B, H) -> (S, B, H)
            hidden_states = mappings.gather_from_sequence_parallel_region(hidden_states, to_model_parallel=False)

        # Get the router_logits, expert_affinities and expert_index from the router
        # router_logits: (T, E), expert_affinities: (T, E), expert_index: (T, top_k)
        router_logits, expert_affinities, expert_index = self.router(hidden_states)

        # Get the output from the ExpertMLPs: (S, B, H)
        output = self.expert_mlps(hidden_states, expert_affinities, expert_index)

        if self.sequence_parallel_mode == MoESequenceParallelMode.EXIT_SP_ON_ENTRY:
            # Scatter back to sequence parallel (as the hidden_states were in sequence parallel)
            # output: (S, B, H) -> (S', B, H)
            output = mappings.scatter_to_sequence_parallel_region(output)

        if self.sequence_parallel_mode == MoESequenceParallelMode.EXIT_SP_ON_ENTRY_DELAY_MLP_AR:
            # Reduce-scatter back to sequence parallel (as the hidden_states were in sequence parallel)
            # output: (S, B, H) -> (S', B, H)
            output = mappings.reduce_scatter_to_sequence_parallel_region(output)

        return output, router_logits, expert_index

    def forward_for_inference(self, hidden_states):
        """
        The collective ops for inference differ from training because the rest of the model does not support
        sequence parallelism in inference. Moreover, the input in inference is (B, S, H) instead of (S, B, H), 
        which leads to differences in scatter/gather operations for SP. 
        The implementation differences are summarized as follows: 
        1. EXIT_SP_ON_ENTRY is equivalent to NO_SP because there are no additional collectives (input is not in SP). 
        2. The reduce-scatter used in training for EXIT_SP_ON_ENTRY_DELAY_MLP_AR is modified to an all-reduce. 
        3. In OPTIMIZED_SP_MATMUL, 
            a. We run the router on the entire sequence (avoiding the all-gather of router logits). 
            b. We scatter/gather the sequence dimension to enter/exit SP before/after ExpertMLPs. 
            c. Note that OPTIMIZED_SP_MATMUL is not an SPMD workload, and is therefore not supported currently for inference. 

        We run in SP mode only for context encoding, and not for token generation (since sequence length is 1).
        """

        assert self.sequence_parallel_mode != MoESequenceParallelMode.OPTIMIZED_SP_MATMUL, "OPTIMIZED_SP_MATMUL is unsupported for inference" 

        seq_len, _, _ = hidden_states.shape
        is_context_encoding = seq_len > 1

        # Get the router_logits, expert_affinities and expert_index from the router
        # router_logits: (T, E), expert_affinities: (T, E), expert_index: (T, top_k)
        router_logits, expert_affinities, expert_index = self.router(hidden_states)

        if is_context_encoding and self.sequence_parallel_mode == MoESequenceParallelMode.OPTIMIZED_SP_MATMUL:
            # Scatter the sequence dimension to enter SP
            # hidden_states: (B, S, H) -> (B, S', H)
            hidden_states = mappings.scatter_input_channels_to_tensor_model_parallel_region(hidden_states)

        # Get the output from the ExpertMLPs: (B, S, H)
        output = self.expert_mlps(hidden_states, expert_affinities, expert_index)

        if self.sequence_parallel_mode == MoESequenceParallelMode.EXIT_SP_ON_ENTRY_DELAY_MLP_AR:
            # Perform delayed all-reduce (required since the MLP is in TP)
            output = mappings.reduce_from_tensor_model_parallel_region(output)

        if is_context_encoding and self.sequence_parallel_mode == MoESequenceParallelMode.OPTIMIZED_SP_MATMUL:
            # output: (B, S', H) -> (B, S', H)
            output = mappings.gather_from_tensor_model_parallel_region_second_dim(output)

        return output, router_logits, expert_index
