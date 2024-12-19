import math
from typing import Union, Optional, Callable, Any

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from torch.distributed import ProcessGroup

from neuronx_distributed.modules.moe.experts import Experts
from neuronx_distributed.modules.moe.model_utils import ACT2FN
from neuronx_distributed.modules.moe.blockwise import (
    can_use_blockwise_matmul_nki,
    blockwise_matmul_nki_func,
    TorchBlockwiseTraining,
)
from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.utils.tensor_utils import cumsum
from neuronx_distributed.parallel_layers.parallel_state import (
    get_expert_model_parallel_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_size,
    rmsg,
)
from neuronx_distributed.utils.logger import get_logger
logger = get_logger()


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

    DEFAULT_BLOCK_SIZE = 512

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        glu_mlp: bool,
        capacity_factor: Union[None, float],
        block_size: Union[None, int] = None,
        normalize_top_k_affinities: bool = False,
        return_bias: bool = False,
        init_method: Optional[Callable[..., Any]] = torch.nn.init.kaiming_uniform_,
        output_layer_init_method: Optional[Callable[..., Any]] = torch.nn.init.kaiming_uniform_,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        enable_spmd_rank: bool = False,  # spmd_rank will be removed once we support ReplicaID (P87857655)
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

        self.block_size = block_size if block_size else self.DEFAULT_BLOCK_SIZE

        if normalize_top_k_affinities and top_k == 1:
            raise ValueError("top_k must be greater than 1 for normalizing top-k expert affinities")
        self.normalize_top_k_affinities = normalize_top_k_affinities

        if return_bias:
            raise NotImplementedError("bias is currently unsupported for MoE")
        self.return_bias = return_bias

        self.tensor_parallel_group = tensor_model_parallel_group if \
            tensor_model_parallel_group is not None else get_tensor_model_parallel_group()

        self.enable_spmd_rank = enable_spmd_rank
        if self.enable_spmd_rank:
            # spmd_rank will be removed once we support ReplicaID (P87857655)
            self.spmd_rank = SPMDRank(world_size=parallel_state.get_world_group().size())

        self.mlp_op = Experts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            glu=glu_mlp,
            activation_fn=self.act_fn,
            dtype=dtype,
            device=device,
            input_layer_init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            tensor_model_parallel_group=self.tensor_parallel_group,
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

        # Scale by expert affinity and combine output
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
        # position_in_expert: (T, E)
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
        # Apply expert_mask
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

    def forward_blockwise(self, hidden_states, expert_affinities, expert_index):
        """
        Forward pass which implements the blockwise matmul approach for expert computations without
        dropping tokens.

        The tokens are assembled into fixed sized 'blocks', each of which is assigned to a single expert.
        The outputs across blocks are un-permuted and combined to obtain the output corresponding to
        each input token.

        The 'block size', i.e. the number of tokens in each block, is a hyper-parameter that must be chosen
        carefully. Larger block sizes result in better hardware utilization, but may also lead to large
        padding overheads (especially at smaller sequence lengths).
        """
        total_tokens, hidden_size = hidden_states.shape

        # num_blocks N = CEIL(((T*top_k)-(E-1))/ B) + (E-1)
        # N is the number of blocks needed in the worst case distribution of tokens to experts. Intuition below.
        # With top_k = 1, let E-1 experts be assigned 1 token each. They require 1 block each, and (E-1) total. The remaining
        # tokens are all assigned the same experts, and require CEIL((T-(E-1))/B) blocks. The formula holds for top_k > 1 also.
        num_blocks = math.ceil((total_tokens * self.top_k - (self.num_experts - 1)) / self.block_size) + self.num_experts - 1
        # Handle case where T*top_k is smaller than E. We will need atmost T*top_k blocks.
        num_blocks = min(num_blocks, total_tokens * self.top_k)

        # expert_mask: (T, E)
        expert_mask = self.get_expert_mask(expert_index)
        # expert_affinities_masked: (T, E)
        expert_affinities_masked = self.get_expert_affinities_masked(expert_affinities, expert_mask)

        # tokens_per_expert: (E, )
        tokens_per_expert = torch.sum(expert_mask, dim=0)
        # blocks_per_expert: (E, )
        blocks_per_expert = ((tokens_per_expert + self.block_size - 1) // self.block_size).to(dtype=torch.long)
        # block_to_expert_idx: (N, 1)
        # The simplest way to do this is to use repeat_interleave after padding blocks_per_expert with unassigned blocks.
        # But this op is not lowered to xla with vector 'repeats', so we use the equivalent implementation below.
        num_blocks_idx = torch.arange(num_blocks, device=hidden_states.device, dtype=torch.long)  # (N, )
        cumulative_blocks_per_expert = cumsum(blocks_per_expert.unsqueeze(1)).squeeze(1)  # (E, )
        block_to_expert_idx = torch.sum(num_blocks_idx.unsqueeze(1) >= cumulative_blocks_per_expert[:-1], dim=1).to(torch.long).unsqueeze(1)

        # Imagine all the tokens in blocks laid out in a linear array: b0t0, b0t1,..., b1t0, b1t1,...
        # For each (token, expert) pair that needs to be computed, calculate the block position (index) in this array
        # block_position_indices: (T, E)
        block_position_indices = cumsum(expert_mask)
        # Tokens assigned to a given expert are assembled in consecutive blocks (and will have consecutive positions)
        # The block position for a token for an expert depends on the number of blocks assigned to all previous experts.
        # Compute and add this offset for each expert
        expert_block_offsets = cumulative_blocks_per_expert * self.block_size
        block_position_indices[:, 1:] += expert_block_offsets[:-1]
        # Apply expert_mask
        block_position_indices = block_position_indices.masked_fill(torch.eq(expert_mask, 0), 0).to(dtype=torch.long)

        # Invert block_position_indices to obtain block_position_to_token_idx
        # block_position_to_token_idx: (N*B+1,)
        # Initialize with -1 indices (which will remain as the indices of padding tokens)
        # block_position_indices is 1-indexed, add extra row to account for this
        block_position_to_token_idx = -1 * torch.ones(num_blocks * self.block_size + 1, device=hidden_states.device, dtype=torch.long)

        if total_tokens % get_tensor_model_parallel_size() != 0:
            # Pad block_position_indices
            num_pad = (-total_tokens) % get_tensor_model_parallel_size()
            block_position_indices = F.pad(block_position_indices, (0, 0, 0, num_pad))
            tokens_idx = torch.arange(total_tokens + num_pad, device=hidden_states.device, dtype=torch.long)
        else:
            tokens_idx = torch.arange(total_tokens, device=hidden_states.device, dtype=torch.long)

        # Distribute computation by splitting block_position_indices and tokens_idx across TP ranks
        # The same block_position_indices and tokens_idx is present at all ranks, so we can use any of MIN/MAX/AVG to as the reduce operation
        if self.enable_spmd_rank:
            # use rank information is available at runtime in inference
            # get tp_rank from global rank
            # note: we use `get_tensor_model_parallel_group()` here to parallelize within a single node
            #       but may get replicated in multi-node case
            tp_rank = torch.remainder(self.spmd_rank.get_rank(), get_tensor_model_parallel_size())
            block_position_indices = mappings.scatter_to_process_group_spmd(
                block_position_indices, partition_dim=0, rank=tp_rank, process_group=get_tensor_model_parallel_group(),
            )
            tokens_idx = mappings.scatter_to_process_group_spmd(
                tokens_idx, partition_dim=0, rank=tp_rank, process_group=get_tensor_model_parallel_group(),
            )
            # Assemble block_position_to_token_idx using chunk of block_position_indices and tokens_idx at each TP rank
            # This generates small DMA transfers because of discontinuous writes, and benefits from distributing across TP ranks
            block_position_to_token_idx[block_position_indices] = tokens_idx.unsqueeze(1)
            # Accumulate results across TP ranks (use MAX to correctly account for the -1 index initialization)
            block_position_to_token_idx = mappings._reduce(
                block_position_to_token_idx, computation=xm.REDUCE_MAX, process_group=get_tensor_model_parallel_group(),
            )
        else:
            # To avoid rank-specific computations, we do a reduce-scatter instead of a simple split
            block_position_indices = mappings._reduce_scatter_along_first_dim(
                block_position_indices, computation=xm.REDUCE_MIN, process_group=self.tensor_parallel_group,
            )
            tokens_idx = mappings._reduce_scatter_along_first_dim(
                tokens_idx, computation=xm.REDUCE_MIN, process_group=self.tensor_parallel_group,
            )
            # Assemble block_position_to_token_idx using chunk of block_position_indices and tokens_idx at each TP rank
            # This generates small DMA transfers because of discontinuous writes, and benefits from distributing across TP ranks
            block_position_to_token_idx[block_position_indices] = tokens_idx.unsqueeze(1)
            # Accumulate results across TP ranks (use MAX to correctly account for the -1 index initialization)
            block_position_to_token_idx = mappings._reduce(
                block_position_to_token_idx, computation=xm.REDUCE_MAX, process_group=self.tensor_parallel_group,
            )

        # block_position_to_token_idx is a flattened array that contains the mapping of token indices for each block
        # block_position_to_token_idx[0] accounts for the 1-indexed block_position_indices, and can be disregarded.
        # block_position_to_token_idx contains -1 as the index of 'padding' tokens

        if can_use_blockwise_matmul_nki(
            hidden_size=hidden_size,
            intermediate_size_tp=self.mlp_op.down_proj.weight.shape[1],
            block_size=self.block_size,
            glu_mlp=self.glu_mlp,
            device=hidden_states.device,
        ):
            return blockwise_matmul_nki_func(
                hidden_states=hidden_states,
                expert_affinities_masked=expert_affinities_masked,
                gate_up_proj_weight=self.mlp_op.gate_up_proj.weight,
                down_proj_weight=self.mlp_op.down_proj.weight,
                block_size=self.block_size,
                block_position_to_token_idx=block_position_to_token_idx,
                block_to_expert_idx=block_to_expert_idx,
                gate_up_proj_scale=getattr(self.mlp_op.gate_up_proj, "scale", None),
                down_proj_scale=getattr(self.mlp_op.down_proj, "scale", None),
            )
        elif self.training:
            # we split training/inference torch blockwise because training backward pass implements a simplified version of mlp_op.
            return TorchBlockwiseTraining.apply(
                hidden_states,
                expert_affinities_masked,
                block_position_to_token_idx,
                block_to_expert_idx.squeeze(-1),
                self.mlp_op.gate_up_proj.weight,
                self.mlp_op.down_proj.weight,
            )
        else:
            return self.torch_blockwise_matmul_inference(
                num_blocks=num_blocks,
                hidden_states=hidden_states,
                expert_affinities_masked=expert_affinities_masked,
                block_position_to_token_idx=block_position_to_token_idx,
                block_to_expert_idx=block_to_expert_idx,
            )

    def torch_blockwise_matmul_inference(
        self,
        num_blocks,
        hidden_states,
        expert_affinities_masked,
        block_position_to_token_idx,
        block_to_expert_idx,
    ):
        """
        PyTorch implementation of the blockwise matmul.

        This is used when running on GPU, or when the blockwise NKI kernel is not compatible with the model
        configuration.
        """
        total_tokens, hidden_size = hidden_states.shape

        # Add extra row for output of padding tokens (i.e. tokens which have -1 index)
        output = torch.zeros(total_tokens + 1, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)

        # block_to_token_indices: (N, B)
        block_to_token_indices = block_position_to_token_idx[1:].view(num_blocks, self.block_size)

        for block_idx in range(num_blocks):
            block_token_indices = block_to_token_indices[block_idx]
            block_expert_idx = block_to_expert_idx[block_idx]

            # block_hidden_states: (1, B, H)
            block_hidden_states = hidden_states[block_token_indices].unsqueeze(0)
            # block_mlp_output: (B, H)
            block_mlp_output = self.mlp_op(block_hidden_states, expert_indices=block_expert_idx).squeeze(0)
            # block_output: (B, H)
            block_output = block_mlp_output * expert_affinities_masked[block_token_indices, block_expert_idx].unsqueeze(
                1
            )
            # Update the tokens computed by the block
            output[block_token_indices] += block_output

        # Drop the last row
        output = output[:total_tokens, :]

        return output

    def forward(self, hidden_states, expert_affinities, expert_index, seq_len):
        """Forward pass of the ExpertMLPs.

        For training:
        1. If capacity_factor is None (full capacity), run forward_all_experts().
        2. Else if capacity_factor is smaller than or equal to zero, run forward_blockwise().
        3. Else run forward_capacity_factor().

        For inference:
        1. If token generation (seq len == 1):
            Run forward_selective_loading() or forward_all_experts() depending on the following logic.
            Let T be the total number of tokens. Using selective loading, T*top_k experts will be loaded.
            If (T*top_k/num_experts) is less than SELECTIVE_LOADING_THRESHOLD, then we use selective loading.
            Otherwise, we use forward_all_experts (for better performance).
        2. Else (seq len > 1):
            a. If capacity factor is not None, run forward_capacity_factor().
            b. If the number of tokens is very small such that the number of experts loaded is less
               than SELECTIVE_LOADING_THRESHOLD, use selective loading. This is useful for the speculation use case.
            c. If number of tokens * top_k is less than the block size, run forward_all_experts().
            c. Otherwise, run forward_blockwise().

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
            elif self.capacity_factor <= 0:
                if self.act_fn != F.silu:
                    logger.info(rmsg(f"{self.act_fn=}. Blockwise training only supports SiLU activation, falling back to full capacity"))
                    return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
                if not self.glu_mlp:
                    logger.info(rmsg(f"{self.glu_mlp=}. Blockwise training only supports glu_mlp=True, falling back to full capacity"))
                    return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
                return self.forward_blockwise(hidden_states, expert_affinities, expert_index)
            else:
                return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
        else:
            # Inference flow
            if seq_len == 1:
                # Token generation
                if get_expert_model_parallel_size() > 1:
                    raise NotImplementedError("Expert parallelism is not supported in token generation.")

                perc_experts_loaded = hidden_states.shape[0] * self.top_k / self.num_experts
                if perc_experts_loaded >= self.SELECTIVE_LOADING_THRESHOLD:
                    return self.forward_all_experts(hidden_states, expert_affinities, expert_index)
                else:
                    return self.forward_selective_loading(hidden_states, expert_affinities, expert_index)
            else:
                # Context Encoding / Speculative Decoding
                if self.capacity_factor is None:
                    perc_experts_loaded = hidden_states.shape[0] * self.top_k / self.num_experts
                    if perc_experts_loaded < self.SELECTIVE_LOADING_THRESHOLD:
                        # Use selective loading for small speculation lengths
                        return self.forward_selective_loading(hidden_states, expert_affinities, expert_index)
                    elif hidden_states.shape[0] * self.top_k < self.block_size:
                        # Use all experts for large speculation lengths, and small context encoding prompt sizes
                        # (more efficient to run all_experts instead of blockwise - equivalent in FLOPs, lower memory bandwidth usage)
                        return self.forward_all_experts(hidden_states, expert_affinities, expert_index)
                    else:
                        # Use blockwise for dropless context encoding
                        return self.forward_blockwise(hidden_states, expert_affinities, expert_index)
                else:
                    return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
