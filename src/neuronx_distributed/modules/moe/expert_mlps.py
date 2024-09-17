import math
from typing import Union, Optional, Callable, Any

import torch
import torch.nn.functional as F

from neuronx_distributed.modules.moe.experts import Experts
from neuronx_distributed.modules.moe.model_utils import ACT2FN
from neuronx_distributed.utils.tensor_utils import cumsum
from neuronx_distributed.parallel_layers.parallel_state import get_expert_model_parallel_size


class ExpertMLPs(torch.nn.Module):
    """Class which obtains the output from passing the token hidden states through the assigned expert(s).

    Arguments:
        num_experts: Total number of experts.
        top_k: Number of experts activated per token. Should be less than or equal to num_experts.
        hidden_size: Hidden dimension.
        intermediate_size: Intermediate dimension used in the MLPs.
        hidden_act: Activation function. See ACT2FN for supported activations.
        glu_mlp: Whether to use the Gated Linear Unit in the MLP. If True, then a combination of gate and up projection is performed in the MLP.
                 Otherwise, a simple up projection is performed.
        capacity_factor: Hyperparameter which controls the expert capacity, and determines the rate of token dropping.
                         If None, then assumed to be running with 'full capacity' (i.e. no tokens dropped).
        normalize_top_k_affinities: Whether to normalize the affinities of the chosen experts before combining with the MLP outputs.
                                    Should be used only with top_k > 1.
        return_bias: Whether to return the bias in the forward pass. Currently not supported.
        init_method: Function used for initializing the gate and up projection linear layer weights.
        output_layer_init_method: Function used for initializing the down projection linear layer weights.
        dtype: Datatype for the layer weights.
        device: Device for the layer weights.
    """

    # Used to determine when to use selective loading for token generation. See forward() for more details.
    SELECTIVE_LOADING_THRESHOLD = 1.0

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        glu_mlp: bool,
        capacity_factor: Union[None, float],
        normalize_top_k_affinities: bool = False,
        return_bias: bool = False,
        init_method: Optional[Callable[..., Any]] = torch.nn.init.kaiming_uniform_,
        output_layer_init_method: Optional[Callable[..., Any]] = torch.nn.init.kaiming_uniform_,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
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

        if capacity_factor is None or capacity_factor >= num_experts / top_k:
            capacity_factor = None  # Denotes full capacity
        self.capacity_factor = capacity_factor

        if normalize_top_k_affinities and top_k == 1:
            raise ValueError("top_k must be greater than 1 for normalizing top-k expert affinities")
        self.normalize_top_k_affinities = normalize_top_k_affinities

        if return_bias:
            raise NotImplementedError("bias is currently unsupported for MoE")
        self.return_bias = return_bias

        self.mlp_op = Experts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            glu=glu_mlp,
            activation_fn=self.act_fn,
            dtype=dtype,
            device=device,
            init_method=output_layer_init_method,
        )

        self.dtype = dtype
        self.device = device

    def get_expert_mask(self, expert_index):
        """Helper function which computes top_k-hot encoded expert_mask from the given expert_index.

        Arguments:
            expert_index: Tensor of shape (T, top_k), containing the 'chosen' experts for each token.
        Returns:
            expert_mask: Tensor of shape (T, E), containing top_k-hot encoded experts for each token derived from
                         expert_index.
        """

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

        return expert_mask

    def get_expert_affinities_masked(self, expert_affinities, expert_mask):
        """Helper function which computes the masked expert_affinities by selecting the chosen experts for each token,
        and normalizes the affinities if needed.

        Arguments:
            expert_affinities: Tensor of shape (T, E), containing the normalized affinities of each token for each expert.
            expert_mask: Tensor of shape (T, E), containing top_k-hot encoded experts for each token derived from
                         expert_index.
        Returns:
            expert_affinities_masked: Tensor of shape (T, E) containing the affinities of just the chosen experts for
                                      each token (after normalization if required).
        """

        # Apply expert_mask obtain the affinities for the chosen experts
        # expert_affinities_masked -> (T, E)
        expert_affinities_masked = expert_affinities.masked_fill(torch.eq(expert_mask, 0), 0)
        if self.normalize_top_k_affinities:
            # Normalize the affinities across the chosen experts
            expert_affinities_masked = F.normalize(expert_affinities_masked, p=1.0, dim=1)

        return expert_affinities_masked

    def forward_all_experts(self, hidden_states, expert_affinities, expert_index):
        """Forward pass where all tokens are computed by all experts.
        This is equivalent to running forward_capacity_factor with full capacity (i.e. no token dropping), but
        by avoiding the permute/unpermute overhead.
        """

        if get_expert_model_parallel_size() > 1:
            raise NotImplementedError("Expert parallelism is not supported without capacity factor.")

        # expert_mask: (T, E)
        expert_mask = self.get_expert_mask(expert_index)
        # expert_affinities_masked: (T, E)
        expert_affinities_masked = self.get_expert_affinities_masked(expert_affinities, expert_mask)

        # Pass all tokens through all experts
        # gate_up_proj: (1, T, H) @ (E, H, I) -> (E, T, I)
        # down_proj: (E, T, I) @ (E, I, H) -> (E, T, H)
        mlp_output = self.mlp_op(hidden_states.unsqueeze(0))

        # TODO: Modify to use multiplication + torch.sum instead
        # Scale by output affinity
        output = torch.zeros(
            hidden_states.shape[0], hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        for e in range(self.num_experts):
            # TH * T1 -> TH
            output += mlp_output[e] * expert_affinities_masked[:, e].unsqueeze(1)

        return output

    def forward_capacity_factor(self, hidden_states, expert_affinities, expert_index):
        """Forward pass for performing Expert MLP computations, where each expert has a fixed 'expert capacity',
        i.e. maximum number of tokens that it can process. This is necessary for maintaining static shapes in the
        compilation graph, but may lead to dropped tokens in the computation.

        Expert capacity C is defined as:
            C = min(total_tokens, (total_tokens * top_k * capacity_factor) / num_experts)
        Note that when capacity_factor >= num_experts / top_k, C = total_tokens (i.e. each expert can hold all
        input tokens, and therefore no tokens are dropped).
        """

        total_tokens = hidden_states.shape[0]

        # compute expert capacity C = (total_tokens * top_k * Cf) / E
        expert_capacity = math.ceil(total_tokens * self.top_k * self.capacity_factor / self.num_experts)
        # expert_capacity can be upper bounded by total number of tokens, for the case when every token is routed to an expert
        expert_capacity = min(expert_capacity, total_tokens)

        # expert_mask: (T, E)
        expert_mask = self.get_expert_mask(expert_index)

        # Compute the position of each token in experts, by a cumulative sum over the T dimension
        # position in expert: (T, E)
        position_in_expert = cumsum(expert_mask)

        # Update expert_mask by accounting for capacity factor (i.e. tokens exceeding capacity are dropped)
        expert_mask.masked_fill_(torch.gt(position_in_expert, expert_capacity), 0)

        # expert_affinities_masked: (T, E)
        expert_affinities_masked = self.get_expert_affinities_masked(expert_affinities, expert_mask)

        # Add expert offset to the position_in_expert
        # Perform operation in float64 to prevent precision issues due to auto-downcasting to bf16
        # expert_index_offsets: (E, )
        expert_index_offsets = (
            torch.arange(self.num_experts, device=hidden_states.device, dtype=torch.float64) * expert_capacity
        )
        # position_in_expert_with_offset: (T, E)
        position_in_expert_with_offset = position_in_expert + expert_index_offsets
        # Mask out those positions which exceed capacity
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
        permuted_output = expert_aligned_output.view(expert_capacity * self.num_experts, -1)

        # TODO: Modify to use multiplication + torch.sum instead
        # output: (T, H)
        output = torch.zeros(
            total_tokens, hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        for k in range(self.top_k):
            # Unpermute output from the kth chosen expert for each token using token_permutation_idx
            output_k = permuted_output[token_permutation_idx[:, k]]
            expert_affinities_k = expert_affinities_masked[total_tokens_idx, expert_index[:, k].unsqueeze(1)]
            # Multiplying with the expert_affinities masks out the output of dropped tokens
            # (T, H) * (T, 1)
            output += output_k * expert_affinities_k

        return output

    def forward_selective_loading(self, hidden_states, expert_affinities, expert_index):
        """Forward pass which selectively loads only the experts chosen for each input token, during token generation."""

        # hidden_states: (T, H)
        # expert_affinities: (T, E)
        # expert_index: (T, top_k)

        T = hidden_states.shape[0]

        # chosen_expert_affinities: (T, top_k)
        chosen_expert_affinities = expert_affinities[
            torch.arange(T, device=hidden_states.device).unsqueeze(1), expert_index
        ]
        if self.normalize_top_k_affinities:
            # Normalize the affinities across the chosen experts
            chosen_expert_affinities = F.normalize(chosen_expert_affinities, p=1.0, dim=1)

        output_list = []
        for t in range(T):
            # gate_up_proj: (1, 1, H) @ (top_k, H, I) -> (top_k, 1, I)
            # down_proj: (top_k, 1, I) @ (top_k, I, H) -> (top_k, 1, H)
            mlp_output_t = self.mlp_op(hidden_states[t].unsqueeze(0).unsqueeze(1), expert_indices=expert_index[t])
            # output_t: sum((top_k, H) * (top_k, 1), dim=0) -> H
            output_t = torch.sum(mlp_output_t.squeeze(1) * chosen_expert_affinities[t].unsqueeze(1), dim=0)
            output_list.append(output_t)

        # output: (T, H)
        output = torch.stack(output_list, dim=0)

        return output

    def forward(self, hidden_states, expert_affinities, expert_index, seq_len):
        """Forward pass of the ExpertMLPs.

        For training:
        1. If capacity_factor is None (full capacity), run forward_all_experts().
        2. Else run forward_capacity_factor().

        For inference:
        1. If context encoding:
            a. If capacity_factor is None (full capacity), run forward_all_experts().
            b. Else run forward_capacity_factor().
        2. Else (token generation):
            Run forward_selective_loading() or forward_all_experts() depending on the following logic.
            Let T be the total number of tokens. Using selective loading, T*top_k experts will be loaded.
            If (T*top_k/num_experts) is less than SELECTIVE_LOADING_THRESHOLD, then we use selective loading.
            Otherwise, we use forward_all_experts (for better performance).

        Note on the SELECTIVE_LOADING_THRESHOLD:
        This parameter determines when forward_selective_loading is used for token-gen (in favor of
        forward_all_experts), and should be a float <= 1.

        Common nomenclature:
            S: Sequence length, B: Batch size, H: Hidden Size
            T: Tokens = S * B (token dimension obtained by flattening S and B)

        Arguments:
            hidden_states: Tensor of shape (T, H).
            expert_affinities: Tensor of shape (T, E), containing the normalized affinities of each token for each expert.
            expert_index: Tensor of shape (T, top_k), containing the 'chosen' experts for each token.
            seq_len: Sequence length S. Used to infer context encoding vs token generation in inference.

        Returns:
            output: Output tensor of the same shape as hidden_states, obtained by passing each token through its assigned experts,
                    combined with the corresponding expert affinities.
        """

        if self.training:
            # Training flow
            if self.capacity_factor is None:
                return self.forward_all_experts(hidden_states, expert_affinities, expert_index)
            else:
                return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
        else:
            # Inference flow
            if seq_len > 1:
                # Context encoding
                if self.capacity_factor is None:
                    return self.forward_all_experts(hidden_states, expert_affinities, expert_index)
                else:
                    return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
            else:
                if get_expert_model_parallel_size() > 1:
                    raise NotImplementedError("Expert parallelism is not supported in token generation.")

                # Token generation
                perc_experts_loaded = hidden_states.shape[0] * self.top_k / self.num_experts
                if perc_experts_loaded >= self.SELECTIVE_LOADING_THRESHOLD:
                    return self.forward_all_experts(hidden_states, expert_affinities, expert_index)
                else:
                    return self.forward_selective_loading(hidden_states, expert_affinities, expert_index)
