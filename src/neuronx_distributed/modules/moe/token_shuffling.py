"""
Worker randomly sends an equal number of tokens to each other worker randomly through all to all collective.

This is because tokens within each worker’s training sequence are highly correlated with each other, and we want to shuffling to achieve better load balancing across experts.
This happens before expert routing.
Citation: BASE Layers: Simplifying Training of Large, Sparse Models (https://arxiv.org/pdf/2103.16716)
"""
import torch
from torch import Tensor
from typing import Tuple, Sequence
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers import mappings, parallel_state
from neuronx_distributed.utils.logger import get_logger

logger = get_logger()


def _all_to_all_for_token_shuffle(x: Tensor, split_dim: int, concat_dim: int) -> Tensor:
    sf_size = parallel_state.get_token_shuffle_group_size()
    if sf_size == 1:  # bypass if there's no token permutation.
        return x

    return xm.all_to_all(
        x,
        split_dimension=split_dim,
        concat_dimension=concat_dim,
        split_count=sf_size,
        groups=parallel_state.get_token_shuffle_replica_groups(),
        pin_layout=False,
    )


class _AllToAllForTokenShuffle(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_: Tensor, split_dim: int, concat_dim: int) -> Tensor:
        return _all_to_all_for_token_shuffle(
            input_,
            split_dim=split_dim,
            concat_dim=concat_dim,
        )

    @staticmethod
    def forward(ctx, input_: Tensor, split_dim: int, concat_dim: int) -> Tensor:
        ctx.split_dim = split_dim
        ctx.concat_dim = concat_dim
        return _all_to_all_for_token_shuffle(
            input_,
            split_dim=split_dim,
            concat_dim=concat_dim,
        )

    @staticmethod
    def backward(ctx, *grad_outputs: Sequence[Tensor]) -> Tuple[Tensor, None, None]:
        # all2all as before but with concat/split dims inverted.
        grad_output: torch.Tensor = grad_outputs[0]
        grad_input = _all_to_all_for_token_shuffle(
            grad_output,
            split_dim=ctx.concat_dim,
            concat_dim=ctx.split_dim,
        )
        return grad_input, None, None


def token_shuffle(hidden_states, seed=None):
    """
    Shuffle the hidden states across the token shuffle group.

    Arguments:
        hidden_states: assumed in [b,s,h] or [s,b,h] shape. works either way.
        if_random_permute: if random permute before all to all. Else use identity permutation.
    """

    shape = hidden_states.shape
    s0, s1, h = shape

    hidden_states = hidden_states.reshape(-1, h)

    if seed is not None:
        # rand() at cpu to get the same random number
        torch.manual_seed(seed)
        xm.set_rng_state(seed)
    permutation = torch.argsort(torch.rand(s0 * s1, device=hidden_states.device))

    permuted_states = hidden_states[permutation]
    hidden_states = all_to_all_for_shuffle(permuted_states)

    return hidden_states.reshape(*shape), permutation


def all_to_all_for_shuffle(hidden_states, input_is_sequence_parallel=True):
    if not input_is_sequence_parallel:
        hidden_states = mappings._ScatterToSequenceParallelRegion.apply(hidden_states, 0)

    hidden_states_shuffled = _AllToAllForTokenShuffle.apply(hidden_states, 0, 0)

    if not input_is_sequence_parallel:
        hidden_states_shuffled = mappings._GatherFromSequenceParallelRegion.apply(hidden_states_shuffled, 0, False)

    return hidden_states_shuffled


def token_unshuffle(permuted_states, permutation):
    # hidden_states: output of the MoE layer after re-combination
    # permutation: 1D tensor of indices
    # shape: original shape of hidden_states
    shape = permuted_states.shape
    h = shape[-1]
    permuted_states = permuted_states.reshape(-1, h)

    # all-to-all is self-inverse
    permuted_states = all_to_all_for_shuffle(permuted_states)
    unpermuted_states = permuted_states.clone()
    unpermuted_states[permutation] = permuted_states

    unpermuted_states = unpermuted_states.view(*shape)

    return unpermuted_states
