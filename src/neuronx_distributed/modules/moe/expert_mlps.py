import math
from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.nn.functional as F

from neuronx_distributed.modules.moe.model_utils import ACT2FN, MoESequenceParallelMode
from neuronx_distributed.modules.moe.moe_parallel_layers import (
    ExpertFusedColumnParallelLinear,
    ExpertFusedRowParallelLinear,
)
from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.utils.tensor_utils import cumsum


class ExpertMLPsBase(torch.nn.Module, ABC):
    """Base class for ExpertMLPs, which are used for obtaining the output from passing the token hidden states through the assigned expert(s).

    This class is used to set common initialization parameters, and define the function signature of the forward pass of child classes.

    Arguments:
        num_experts: Total number of experts.
        hidden_size: Hidden dimension.
        intermediate_size: Intermediate dimension used in the MLPs.
        hidden_act: Activation function. See ACT2FN for supported activations.
        capacity_factor: Hyperparameter which controls the expert capacity, and determines the rate of token dropping.
        init_method: Function used for initializing the gate and up projection linear layer weights.
        output_layer_init_method:Function used for initializing the down projection linear layer weights.
        glu_mlp: Whether to use the Gated Linear Unit in the MLP. If True, then a combination of gate and up projection is performed in the MLP.
                 Otherwise, a simple up projection is performed.
        sequence_parallel_mode: SP mode being used for the MoE layer.
        permute_strategy: Specifies how to perform the token permute and un-permute. Must be one of 'matmul' or 'index.
        top_k: Number of experts activated per token. Should be less than or equal to num_experts.
        normalize_top_k_affinities: Whether to normalize the affinities of the chosen experts before combining with the MLP outputs.
                                    Should be used only with top_k > 1.
        return_bias: Whether to return the bias in the forward pass. Currently not supported.
        dtype: Datatype for the layer weights.
        device: Device for the layer weights.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        glu_mlp: bool,
        sequence_parallel_mode: Union[str, MoESequenceParallelMode],
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        self.num_experts = num_experts
        if not (0 < top_k <= num_experts):
            raise ValueError(f"Invalid top_k={top_k} for num_experts={num_experts}")
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        if hidden_act not in ACT2FN:
            raise ValueError(f"Unknown activation: {hidden_act} ; Supported: {list(ACT2FN.keys())}")
        self.act_fn = ACT2FN[hidden_act]
        self.glu_mlp = glu_mlp

        if sequence_parallel_mode not in MoESequenceParallelMode.__members__:
            raise TypeError(f"Unknown sequence_parallel_mode: {sequence_parallel_mode}")
        self.sequence_parallel_mode = MoESequenceParallelMode[sequence_parallel_mode]

        self.dtype = dtype
        self.device = device

    @abstractmethod
    def forward(self, hidden_states, expert_affinities, expert_index):
        """Forward pass of the ExpertMLPs.

        This function should internally account for whether the hidden_states are in SP or not, and return the output accordingly,
        i.e. the output should be in SP iff the hidden_states are in SP.

        Common nomenclature:
            S: Sequence length, B: Batch size, H: Hidden Size
            S': Sequence length (when the input is in SP)
            T: Tokens = S * B (token dimension obtained by flattening S and B)
            T': Tokens (when the input is in SP) = S' * B

        Arguments:
            hidden_states: Tensor of shape (S, B, H) or (S', B, H).
            expert_affinities: Tensor of shape (T, E), containing the normalized affinities of each token for each expert.
            expert_index: Tensor of shape (T, top_k), containing the 'chosen' experts for each token.

        Returns:
            output: Output tensor of the same shape as hidden_states, obtained by passing each token through its assigned experts,
                    combined with the corresponding expert affinities.
        """


class ExpertMLPsCapacityFactor(ExpertMLPsBase):
    """ExpertMLPs where each expert has a fixed 'expert capacity', i.e. maximum number of tokens that it can process.
    This is necessary for maintaining static shapes in the compilation graph, but may lead to dropped tokens in the computation.

    Arguments:
        capacity_factor: Hyperparameter which controls the expert capacity, and determines the rate of token dropping.
        init_method: Function used for initializing the gate and up projection linear layer weights.
        output_layer_init_method:Function used for initializing the down projection linear layer weights.
        permute_strategy: Specifies how to perform the token permute and un-permute. Must be one of 'matmul' or 'index.
        normalize_top_k_affinities: Whether to normalize the affinities of the chosen experts before combining with the MLP outputs.
                                    Should be used only with top_k > 1.
        return_bias: Whether to return the bias in the forward pass. Currently not supported.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        capacity_factor: float,
        glu_mlp: bool,
        sequence_parallel_mode: Union[str, MoESequenceParallelMode],
        permute_strategy: str,
        normalize_top_k_affinities: bool = False,
        return_bias: bool = False,
        init_method: torch.nn.init = torch.nn.init.kaiming_uniform_,
        output_layer_init_method: torch.nn.init = torch.nn.init.kaiming_uniform_,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            glu_mlp=glu_mlp,
            sequence_parallel_mode=sequence_parallel_mode,
            dtype=dtype,
            device=device,
        )

        self.capacity_factor = capacity_factor

        if permute_strategy not in {"matmul", "index"}:
            raise ValueError(f"Unknown permute_strategy: {permute_strategy}")
        if self.sequence_parallel_mode == MoESequenceParallelMode.OPTIMIZED_SP_MATMUL and permute_strategy != "matmul":
            raise ValueError("SP mode OPTIMIZED_SP_MATMUL can only be used with the 'matmul' permute strategy")
        self.permute_strategy = permute_strategy

        if normalize_top_k_affinities and top_k == 1:
            raise ValueError("top_k must be greater than 1 for normalizing top-k expert affinities")
        self.normalize_top_k_affinities = normalize_top_k_affinities

        if return_bias:
            raise NotImplementedError("bias is currently unsupported for MoE")
        self.return_bias = return_bias

        # Define the layers for expert MLP operations
        if self.glu_mlp:
            # Combine the gate and up projections into a single large tensor multiplication for efficiency
            self.gate_up_proj = ExpertFusedColumnParallelLinear(
                num_experts=num_experts,
                input_size=hidden_size,
                output_size=2 * intermediate_size,
                gather_output=False,
                dtype=dtype,
                device=device,
                stride=2,
                init_method=init_method,
            )
        else:
            self.up_proj = ExpertFusedColumnParallelLinear(
                num_experts=num_experts,
                input_size=hidden_size,
                output_size=intermediate_size,
                gather_output=False,
                dtype=dtype,
                device=device,
                init_method=init_method,
            )

        self.down_proj = ExpertFusedRowParallelLinear(
            num_experts=num_experts,
            input_size=intermediate_size,
            output_size=hidden_size,
            input_is_parallel=True,
            dtype=dtype,
            device=device,
            init_method=output_layer_init_method,
            sequence_parallel_mode=self.sequence_parallel_mode,
        )

    def mlp_op(self, expert_aligned_hidden_states):
        """Helper function which performs the expert MLP computations for the given hidden states.

        Common nomenclature:
            E: Total number of experts
            C: Expert capacity
            H: Hidden Size

        Arguments:
            expert_aligned_hidden_states: Input tensor of shape (E, C, H) containing the token hidden states for each expert.

        Returns:
            expert_aligned_output: Output tensor of shape (E, C, H) obtained after the gate/up projection + activation + down
                                   projection operations.
        """

        if self.glu_mlp:
            # gate_up_proj_op: (E, C, H) @ (E, H, 2I) -> (E, C, 2I)
            gate_up_proj_op = self.gate_up_proj(expert_aligned_hidden_states)
            # Split into gate_proj and up_proj, both (E, C, I)
            gate_proj_op, up_proj_op = torch.tensor_split(gate_up_proj_op, 2, dim=2)
            # intermediate_op: (E, C, I)
            intermediate_op = self.act_fn(gate_proj_op) * up_proj_op
        else:
            # up_proj_op: (E, C, H) @ (E, H, I) -> (E, C, I)
            up_proj_op = self.up_proj(expert_aligned_hidden_states)
            # intermediate_op: (E, C, I)
            intermediate_op = self.act_fn(up_proj_op)

        # down projection: (E, C, I) @ (E, I, H) -> (E, C, H)
        expert_aligned_output = self.down_proj(intermediate_op)
        return expert_aligned_output

    def compute_position_in_expert(self, expert_index, total_tokens):
        """Helper function used for computing the expert capacity, expert mask and position in expert,
        corresponding to the input expert_index.

        Arguments:
            expert_index: Tensor of shape (T, top_k), containing the 'chosen' experts for each token.
            total_tokens: Integer specifying the number of input tokens to the forward() function

        Returns:
            expert_capacity: Integer indicating the capacity of each expert
            expert_mask: top_k-hot tensor of shape (T, E), computed using expert_index
            position_in_expert: Tensor of shape (T, E), specifying the position of a given token within each chosen expert.
        """

        # compute expert capacity C = (total_tokens * top_k * Cf) / E
        expert_capacity = math.ceil(total_tokens * self.top_k * self.capacity_factor / self.num_experts)
        # expert_capacity can be upper bounded by total number of tokens, for the case when every token is routed to an expert
        expert_capacity = min(expert_capacity, total_tokens)

        # Construct expert_mask from expert_index, using efficient version of one-hot encoding for xla device
        # Perform operation in float64 to prevent precision issues due to auto-downcasting to bf16
        # (Use float dtype to perform computations in the vector engine for efficiency)
        # expert_mask: top_k-hot encoded expert assignment per token -> (T, E)
        expert_mask = torch.zeros(
            expert_index.shape[0], self.num_experts, device=expert_index.device, dtype=torch.float64
        )
        expert_num_idx_arr = torch.arange(self.num_experts, device=expert_index.device, dtype=torch.float64)
        for e in range(self.top_k):
            expert_mask += (expert_index[:, e].unsqueeze(1) == expert_num_idx_arr).to(torch.float64)

        # Compute the position of each token in experts, by a cumulative sum over the T dimension
        # position in expert: (T, E)
        position_in_expert = cumsum(expert_mask)

        # Update expert_mask by accounting for capacity factor (i.e. tokens exceeding capacity are dropped)
        expert_mask.masked_fill_(torch.gt(position_in_expert, expert_capacity), 0)

        # Mask out those positions which exceed capacity
        position_in_expert.masked_fill_(torch.eq(expert_mask, 0), 0)

        return expert_capacity, expert_mask, position_in_expert

    def forward(self, hidden_states, expert_affinities, expert_index):
        """Lightweight wrapper which directs the computation to the forward function for the required permute_strategy.

        Common nomenclature:
            S: Sequence length, B: Batch size, H: Hidden Size
            S': Sequence length (when the input is in SP)
            T: Tokens = S * B (token dimension obtained by flattening S and B)
            E: Total number of experts
            C: Expert capacity
        """

        # hidden_states: (S, B, H) in training, (B, S, H) in inference
        # expert_affinities: (T, E)
        # expert_index: (T, top_k)

        # In token generation mode if running inference with seq_len = 1
        is_token_gen = (not self.training) and (hidden_states.shape[1] == 1)

        if is_token_gen or self.capacity_factor >= self.num_experts / self.top_k:
            # Token generation or Training/Context encoding with full capacity (no tokens dropped)
            return self.forward_full_capacity(hidden_states, expert_affinities, expert_index)
        elif self.permute_strategy == "matmul":
            return self.forward_permute_matmul(hidden_states, expert_affinities, expert_index)
        else:
            return self.forward_permute_index(hidden_states, expert_affinities, expert_index)

    def forward_full_capacity(self, hidden_states, expert_affinities, expert_index):
        """Forward pass where all tokens are computed by all experts. 
        This is equivalent to running 'matmul' or 'index' with full capacity (i.e. no token dropping), but
        by avoiding the permute/unpermute overhead. 
        """
        
        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_shape[-1])  # (T, H)

        # Pass all tokens through all experts
        # gate_up_proj: 1TH @ EHI -> ETI
        # down_proj: ETI @ EIH -> ETH 
        mlp_output = self.mlp_op(hidden_states.unsqueeze(0))

        # expert_mask: (T, E)  (top_k-hot encoded)
        expert_mask = torch.zeros(
            expert_index.shape[0], self.num_experts, device=expert_index.device, dtype=torch.float64
        )
        expert_num_idx_arr = torch.arange(self.num_experts, device=expert_index.device, dtype=torch.float64)
        for e in range(self.top_k):
            expert_mask += (expert_index[:, e].unsqueeze(1) == expert_num_idx_arr).to(torch.float64)

        # expert_affinities_masked: (T, E)
        expert_affinities_masked = expert_affinities.to(dtype=hidden_states.dtype).masked_fill(
            torch.eq(expert_mask, 0), 0
        )
        if self.normalize_top_k_affinities:
            # Normalize the affinities across the chosen experts
            expert_affinities_masked = F.normalize(expert_affinities_masked, p=1.0, dim=1)

        # Scale by output affinity
        output = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype)
        for e in range(self.num_experts):
            # TH * T1 -> TH
            output += mlp_output[e] * expert_affinities_masked[:, e].unsqueeze(1)

        # Reshape output to original hidden_shape
        output = output.view(hidden_shape)
        return output

    def forward_permute_matmul(self, hidden_states, expert_affinities, expert_index):
        """Forward pass of the 'matmul' permute strategy, which uses matrix-multiplication to permute and un-permute the tokens."""

        hidden_shape = hidden_states.shape
        seq_len = hidden_shape[0]
        hidden_states = hidden_states.view(-1, hidden_shape[-1])  # (T, H) or (T', H)

        # Due to different SP setup between training and inference, we have to implement SP for them differently here.
        # For inference, we only perform sequence parallelism when it is context encoding, because we partition on
        # the sequence dimension which can't be partitioned during token generation.
        is_context_encoding = not self.training and seq_len > 1

        if self.sequence_parallel_mode == MoESequenceParallelMode.OPTIMIZED_SP_MATMUL:
            if self.training or is_context_encoding:
                # hidden_states: (T', H)
                total_tokens = hidden_states.shape[0] * mappings.get_tensor_model_parallel_size()
            else:
                # hidden_states: (T, H)
                total_tokens = hidden_states.shape[0]
        else:
            # hidden_states: (T, H)
            total_tokens = hidden_states.shape[0]

        assert total_tokens == expert_affinities.shape[0]
        assert total_tokens == expert_index.shape[0]

        # Compute expert_capacity, expert_mask and position_in_expert
        expert_capacity, expert_mask, position_in_expert = self.compute_position_in_expert(expert_index, total_tokens)

        if self.sequence_parallel_mode == MoESequenceParallelMode.OPTIMIZED_SP_MATMUL:
            if self.training or is_context_encoding:
                # Obtain the position_in_expert corresponding to just the tokens at this rank
                # position_in_expert: (T, E) -> (T', E)
                position_in_expert = mappings._split_along_first_dim(position_in_expert)

        # position_mask: one-hot encode position_in_expert (T, E) into (T, E, C)
        # Perform operation in float64 to prevent precision issues due to auto-downcasting to bf16
        # (Use float dtype to perform computations in the vector engine for efficiency)
        expert_capacity_idx_arr = torch.arange(expert_capacity + 1, device=hidden_states.device, dtype=torch.float64)
        position_mask = (position_in_expert.unsqueeze(2) == expert_capacity_idx_arr).to(hidden_states.dtype)
        # Account for 1-indexing of position_in_expert
        position_mask = position_mask[:, :, 1:]

        # expert_aligned_hidden_states: (T, E, C) @ (T, H) -> (E, C, H) or (T', E, C) @ (T', H) -> (E, C, H)
        expert_aligned_hidden_states = torch.einsum("tec,th->ech", position_mask, hidden_states)

        if self.sequence_parallel_mode == MoESequenceParallelMode.OPTIMIZED_SP_MATMUL:
            if self.training or is_context_encoding:
                # All-reduce across ranks, since expert_aligned_hidden_states was computed in SP (i.e. using the T' tokens at each rank)
                expert_aligned_hidden_states = mappings.reduce_from_tensor_model_parallel_region(
                    expert_aligned_hidden_states
                )

        # Perform MLP operations
        # expert_aligned_output: (E, C, H)
        expert_aligned_output = self.mlp_op(expert_aligned_hidden_states)

        # Apply expert_mask obtain the affinities for the chosen experts
        # expert_affinities_masked -> (T, E)
        expert_affinities_masked = expert_affinities.to(dtype=hidden_states.dtype).masked_fill(
            torch.eq(expert_mask, 0), 0
        )
        if self.normalize_top_k_affinities:
            # Normalize the affinities across the chosen experts
            expert_affinities_masked = F.normalize(expert_affinities_masked, p=1.0, dim=1)

        if self.sequence_parallel_mode == MoESequenceParallelMode.OPTIMIZED_SP_MATMUL:
            if self.training or is_context_encoding:
                # Obtain the expert affinities corresponding to just the tokens at this rank
                # expert_affinities_masked: (T, E) -> (T', E)
                expert_affinities_masked = mappings.scatter_to_sequence_parallel_region(expert_affinities_masked)
                # Since the einsum operation below (with position_mask_with_affinities) is computed in SP (i.e. using T'),
                # we need an all-reduce in the backward pass to obtain the complete gradients for expert_aligned_output.
                expert_aligned_output = mappings.copy_to_tensor_model_parallel_region(expert_aligned_output)

        # position_mask_with_affinities: (T, E, C) * (T, E, 1) -> (T, E, C) or (T', E, C) * (T', E, 1) -> (T', E, C)
        position_mask_with_affinities = position_mask * expert_affinities_masked.unsqueeze(-1)

        # Unpermute and scale output with expert affinities
        # output: (T, E, C) @ (E, C, H) -> (T, H) or (T', E, C) @ (E, C, H) -> (T', H)
        output = torch.einsum("tec,ech->th", position_mask_with_affinities, expert_aligned_output)

        # Reshape output to original hidden_shape
        output = output.view(hidden_shape)
        return output

    def forward_permute_index(self, hidden_states, expert_affinities, expert_index):
        """Forward pass of the 'index' permute strategy, which uses indexing-based lookups to permute and un-permute the tokens."""

        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_shape[-1])  # (T, H)
        total_tokens = hidden_states.shape[0]

        # Compute expert_capacity, expert_mask and position_in_expert
        expert_capacity, expert_mask, position_in_expert = self.compute_position_in_expert(expert_index, total_tokens)

        # Add expert offset to the position_in_expert
        # Perform operation in float64 to prevent precision issues due to auto-downcasting to bf16
        # expert_index_offsets: (E, )
        expert_index_offsets = (
            torch.arange(self.num_experts, device=hidden_states.device, dtype=torch.float64) * expert_capacity
        )
        # position_in_expert_with_offset: (T, E)
        position_in_expert_with_offset = position_in_expert + expert_index_offsets
        position_in_expert_with_offset.masked_fill_(torch.eq(expert_mask, 0), 0)

        # Determine the index (with offset) of each token
        # total_tokens_idx: (T, 1)
        total_tokens_idx = torch.arange(total_tokens, device=hidden_states.device, dtype=torch.long).unsqueeze(1)
        # token_permutation_idx: (T, top_k)
        token_permutation_idx = position_in_expert_with_offset[total_tokens_idx, expert_index].to(dtype=torch.long)
        # token_permutation_idx is 1-indexed (contains 0 for dropped tokens)

        # token_assignments: (C*E+1, )
        # Account for the 1-indexed token_permutation_idx by adding an extra row
        token_assignments = torch.zeros(
            expert_capacity * self.num_experts + 1, device=hidden_states.device, dtype=torch.long
        )
        # Perform a broadcasted assignment to map to token_idx
        token_assignments[token_permutation_idx] = total_tokens_idx + 1
        # Drop the first row (which was added to account for the 1-indexed token_permutation_idx)
        token_assignments = token_assignments[1:]
        # token_assignments: (E, C)
        token_assignments = token_assignments.view(self.num_experts, expert_capacity)
        # token_assignments is 1-indexed (contains 0 for dropped tokens)

        # Convert token_permutation_idx and token_assignments to be 0-indexed
        token_permutation_idx = token_permutation_idx - 1
        token_assignments = token_assignments - 1
        # They now contain '-1' for dropped tokens, enforce non-negative indices to avoid potential runtime OOB
        zero_tensor = torch.zeros(1, device=hidden_states.device, dtype=torch.long)
        token_permutation_idx = torch.max(token_permutation_idx, zero_tensor)
        token_assignments = torch.max(token_assignments, zero_tensor)
        # Indexing using these will result in the first token (index 0) being loaded in place of dropped tokens
        # However, the output from these will get masked out in the affinity scaling step

        # Permute hidden_states using token_assignments to get expert_aligned_hidden_states
        # expert_aligned_hidden_states: (E, C, H)
        expert_aligned_hidden_states = hidden_states[token_assignments, :]

        # Perform MLP operations
        # expert_aligned_output: (E, C, H)
        expert_aligned_output = self.mlp_op(expert_aligned_hidden_states)

        # convert back (E, C, H) into (C*E, H)
        permuted_output = expert_aligned_output.view(expert_capacity * self.num_experts, hidden_shape[2])

        # Apply expert_mask obtain the affinities for the chosen experts
        # expert_affinities_masked -> (T, E)
        expert_affinities_masked = expert_affinities.to(dtype=hidden_states.dtype).masked_fill(
            torch.eq(expert_mask, 0), 0
        )
        if self.normalize_top_k_affinities:
            # Normalize the affinities across the chosen experts
            expert_affinities_masked = F.normalize(expert_affinities_masked, p=1.0, dim=1)

        # output: (T, H)
        output = torch.zeros(total_tokens, hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype)
        for k in range(self.top_k):
            # Unpermute output from the kth chosen expert for each token using token_permutation_idx
            output_k = permuted_output[token_permutation_idx[:, k]]
            expert_affinities_k = expert_affinities_masked[total_tokens_idx, expert_index[:, k].unsqueeze(1)]
            # Multiplying with the expert_affinities masks out the output of dropped tokens
            # (T, H) * (T, 1)
            output += output_k * expert_affinities_k

        # Reshape output to original hidden_shape
        output = output.view(hidden_shape)
        return output
