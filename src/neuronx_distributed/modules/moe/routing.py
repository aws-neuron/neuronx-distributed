from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import torch
from torch.distributed import ProcessGroup
import torch.nn.functional as F

from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_group, get_world_group, get_expert_model_parallel_size


class RouterBase(torch.nn.Module, ABC):
    """Base class for various routing strategies used in MoE.

    Arguments:
        num_experts: Total number of experts.
        top_k: Number of experts activated per token. Should be less than or equal to num_experts.
        hidden_size: Hidden dimension of the input sequence.
        act_fn: Activation used to obtain expert affinities from router logits. One of 'sigmoid' or 'softmax'.
        dtype: Datatype for the layer weights.
        device: Device for the layer weights.
        jitter_eps: Random noise factor for input perturbation.
        store_transposed_weights: Whether to register transposed router weights in the buffer.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        act_fn: str,
        sequence_parallel_enabled: bool,
        sequence_dimension: Optional[int],
        dtype: torch.dtype,
        device: torch.device,
        bias: bool = False,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        jitter_eps: float = 0.0,
        store_transposed_weights: bool = False,
        apply_act_fn_over_topk: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        if not (0 < top_k <= num_experts):
            raise ValueError(f"Invalid top_k={top_k} for num_experts={num_experts}")
        self.top_k = top_k
        self.hidden_size = hidden_size
        if act_fn not in {"sigmoid", "softmax"}:
            raise ValueError("act_fn must be either 'sigmoid' or 'softmax'")
        self.act_fn = act_fn

        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.sequence_parallel_enabled and sequence_dimension is None:
            # Default to 0
            sequence_dimension = 0
        self.sequence_dimension = sequence_dimension

        self.dtype = dtype
        self.device = device
        self.bias = bias
        self.jitter_eps = jitter_eps
        self.store_transposed_weights = store_transposed_weights
        self.apply_act_fn_over_topk = apply_act_fn_over_topk

        # TODO: Refactor with expert MLPv2 design to include parallel groups as mandatory arg
        if get_expert_model_parallel_size() > 1:
            self.tensor_parallel_group = get_world_group()
        else:
            self.tensor_parallel_group = tensor_model_parallel_group if \
                tensor_model_parallel_group is not None else get_tensor_model_parallel_group()

        # Create router
        self.linear_router = torch.nn.Linear(hidden_size, num_experts, dtype=dtype, device=device, bias=bias)
        if self.store_transposed_weights:
            self.weight_T = torch.nn.Parameter(self.linear_router.weight.detach().T.clone())
        setattr(self.linear_router.weight, "sequence_parallel_enabled", sequence_parallel_enabled)

    def _if_training_gather_for_sp(self):
        """Determines when to gather from sequence parallel region based on mode.
            - training: gather router logits before activation
            - inference: delayed gather expert affinities mask and expert index
        """
        return self.sequence_parallel_enabled and self.training

    def get_router_logits(self, hidden_states):
        """
        Returns the router logits.
        Note that this function always returns the logits corresponding to the full hidden states, by handling
        sequence parallelism internally.
        """
        original_hidden_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(dtype=self.linear_router.weight.dtype)
        # Add noise to the input tensor by applying jitter.
        if self.jitter_eps != 0.0 and self.training:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_eps, 1.0 + self.jitter_eps
            )

        # router_logits: (*, H) @ (H, E) -> (*, E)
        router_logits = self.linear_router(hidden_states)

        if self._if_training_gather_for_sp():
            # Gather the router_logits across ranks
            router_logits = mappings.gather_from_sequence_parallel_region(
                router_logits,
                sequence_dimension=self.sequence_dimension,
                to_model_parallel=False,
                process_group=self.tensor_parallel_group,
            )

        # Flatten S and B to T dimension
        router_logits = router_logits.view(-1, self.num_experts)
        hidden_states = hidden_states.to(dtype=original_hidden_dtype)
        return router_logits

    def apply_activation_fn(self, weights):
        # Perform activation in fp64 to prevent auto-downcasting of operation to bf16, for numerical accuracy
        # expert_affinities: (T, E)
        if self.act_fn == "sigmoid":
            expert_affinities = torch.sigmoid(weights.to(dtype=torch.float64))
        elif self.act_fn == "softmax":
            expert_affinities = F.softmax(weights, dim=1, dtype=torch.float64)
        else:
            raise ValueError("act_fn must be either 'sigmoid' or 'softmax'")

        return expert_affinities

    @abstractmethod
    def forward(self, hidden_states):
        """Forward pass of the router.

        Common nomenclature:
            S: Sequence length, B: Batch size, H: Hidden Size
            T: Tokens = S * B (token dimension obtained by flattening S and B)

        Arguments:
            hidden_states: Input tensor of shape (S, B, H) or (B, S, H)
                           If self.sequence_parallel_enabled is True, then hidden_states is assumed to be sharded in SP.

        Returns:
            router_logits: Tensor of shape (T, E) containing the router logits for each token for each expert.
            expert_affinities: Tensor of shape (T, E), containing the normalized affinities of each token for each expert.
            expert_index: Tensor of shape (T, top_k), containing the 'chosen' experts for each token.
                          During training, the expert_index may be different from the expert with the maximum affinity for a token
                          (if the router performs any token balancing, e.g. Sinkhorn).
        """

    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        if self.store_transposed_weights:
            original_key = prefix.removesuffix("router.weight") + "router.linear_router.weight"
            transposed_key = prefix.removesuffix("router.weight") + "router.weight_T"
            model_state_dict[transposed_key] = model_state_dict[original_key].detach().transpose(0, 1).clone()


class RouterTopK(RouterBase):
    """Class which implements top-K expert routing for tokens.
    Since this router does not perform any token balancing, it should be used with the load_balancing_loss_func during training.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        bias: bool = False,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        act_fn="softmax",
        apply_act_fn_over_topk: bool = False,
        jitter_eps: float = 0.0,
        store_transposed_weights: bool = False,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            act_fn=act_fn,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sequence_dimension=sequence_dimension,
            dtype=dtype,
            device=device,
            bias=bias,
            tensor_model_parallel_group=tensor_model_parallel_group,
            jitter_eps=jitter_eps,
            store_transposed_weights=store_transposed_weights,
            apply_act_fn_over_topk=apply_act_fn_over_topk,
        )

    def forward(self, hidden_states):
        # Get router_logits and expert_affinities
        router_logits = self.get_router_logits(hidden_states)
        if self.apply_act_fn_over_topk:
            expert_affinities = torch.zeros_like(router_logits, dtype=torch.float64)
            topk_weights, expert_index = torch.topk(router_logits.to(torch.float64), self.top_k, dim=1)
            topk_affinities = self.apply_activation_fn(topk_weights)
            expert_affinities = expert_affinities.scatter_(1, expert_index, topk_affinities)
        else:
            expert_affinities = self.apply_activation_fn(router_logits)
            # For each token, get the top_k experts
            # expert_index: (T, top_k)
            _, expert_index = torch.topk(router_logits, self.top_k)

        # Cast to required dtype
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)
        expert_index = expert_index.detach().to(dtype=torch.long)

        return router_logits, expert_affinities, expert_index


class RouterSinkhorn(RouterBase):
    """Class which implements top-1 expert routing with Sinkhorn-based token balancing during training.

    It is important to note that the Sinkhorn algorithm is run for a constant number of iterations (and not necessarily until convergence).

    Arguments:
        sinkhorn_iterations: (Optional) Number of iterations of Sinkhorn to run for token balancing during training.
                                        If not specified, defaults to DEFAULT_SINKHORN_ITERS.
        sinkhorn_tol: (Optional) If specified, then the error is enforced to be less than the given tolerance value after running Sinkhorn.
    """

    DEFAULT_SINKHORN_ITERS = 30

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        act_fn: str = "sigmoid",
        sequence_parallel_enabled: bool = False,
        sequence_dimension: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        sinkhorn_iterations: Optional[int] = None,
        sinkhorn_tol: Optional[float] = None,
    ):
        if top_k != 1:
            raise NotImplementedError("RouterSinkhorn only supports Top-1 routing")

        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            act_fn=act_fn,
            sequence_parallel_enabled=sequence_parallel_enabled,
            sequence_dimension=sequence_dimension,
            dtype=dtype,
            device=device,
        )

        self.sinkhorn_iterations = (
            sinkhorn_iterations if sinkhorn_iterations is not None else self.DEFAULT_SINKHORN_ITERS
        )
        self.sinkhorn_tol = sinkhorn_tol

    def forward(self, hidden_states):
        # Get router_logits and expert_affinities
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)
        # Cast to required dtype
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)

        with torch.no_grad():
            if self.training:
                # Run Sinkhorn for token balancing in fp32 (to account for the discontinuous nature of the routing function, i.e. high
                # precision error in logits resulting in a large number of misrouted tokens, which cause more sudden degradations in output)
                sinkroute = self._sinkhorn(
                    router_logits.detach().to(dtype=torch.float32), num_iters=self.sinkhorn_iterations
                )
            else:
                sinkroute = router_logits.detach()

            # For each token, get the top-1 expert based on the Sinkhorn adjusted router logits
            # expert_index: (T, 1)
            expert_index = torch.argmax(sinkroute, dim=1, keepdim=True)

        expert_index = expert_index.to(dtype=torch.long)
        return router_logits, expert_affinities, expert_index

    @staticmethod
    def _sinkhorn(cost, num_iters, tol=None):
        """
        Sinkhorn implementation based on Megatron-LM, but with constant number of iterations (to ensure static compilation graph).

        Note that the number of iterations should be chosen carefully to ensure proper token balancing (as the algorithm is not
        run till convergence).

        Arguments:
            cost: 2D tensor containing the 'cost' values to be balanced.
            num_iters: The number of iterations to run the algorithm.
            tol: If specified, then the error is enforced to be less than the given tolerance value.
        """

        if num_iters == 0:
            return cost

        cost = torch.exp(cost)
        d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
        d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

        eps = 1e-8
        error = 1e9
        d1_old = d1
        for i in range(num_iters):
            d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
            d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
            error = torch.mean(torch.abs(d1_old - d1))
            d1_old = d1

        if tol is not None:
            assert float(error) < tol, f"Sinkhorn error {float(error)} exceeds tolerance {tol}"
        return d1 * cost * d0.unsqueeze(1)

class GroupLimitedRouter(RouterBase):
    """
    Implements top-K expert routing using the no auxiliary loss method from DeepSeekV3.

    The topk selection method selects the top-k experts based on computed scores using Sigmoid gate.
    This involves adding a bias term to the scores, grouping the scores, selecting the top groups,
    masking out scores for experts not in the top groups, then selecting the top-k experts based on
    the masked scores.

    Args:
        num_experts (int): Number of experts in the model
        top_k (int): Number of experts to route to for each token
        hidden_size (int): Size of the hidden layer
        n_group (int): Number of expert groups
        topk_group (int): Number of top groups to select
        sequence_parallel_enabled (bool): Whether sequence parallelism is enabled
        sequence_dimension (Optional[int]): Dimension for sequence parallelism
        dtype (torch.dtype): Data type for computations
        device (torch.device): Device to run computations on
        tensor_model_parallel_group (Optional[ProcessGroup]): Process group for tensor parallelism
        jitter_eps (float): Jitter epsilon for noise addition
    """

    def __init__(
            self,
            num_experts: int,
            top_k: int,
            hidden_size: int,
            n_group: int,
            topk_group: int,
            sequence_parallel_enabled: bool = False,
            sequence_dimension: Optional[int] = None,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cpu"),
            tensor_model_parallel_group: Optional[ProcessGroup] = None,
            jitter_eps: float = 0.0,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            act_fn="sigmoid",
            sequence_parallel_enabled=sequence_parallel_enabled,
            sequence_dimension=sequence_dimension,
            dtype=dtype,
            device=device,
            tensor_model_parallel_group=tensor_model_parallel_group,
            jitter_eps=jitter_eps,
        )
        self.n_group = n_group
        self.topk_group = topk_group

    def forward(self, hidden_states):
        """
        Forward pass of the router.

        Args:
            hidden_states: Input tensor to be routed

        Returns:
            tuple: (router_logits, expert_weights, topk_idx)
                - router_logits: Raw routing scores
                - expert_weights: Weights for each selected expert
                - topk_idx: Indices of selected experts
        """
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)
        # Cast to required dtype
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)

        topk_idx, expert_weights = self.noaux_tc_top_k(expert_affinities)
        topk_idx = topk_idx.detach().to(dtype=torch.long)

        return router_logits, expert_weights, topk_idx

    def noaux_tc_top_k(self, scores):
        """
        Performs top-k selection using the no auxiliary loss method.

        Args:
            scores (torch.Tensor): Expert scores of shape (batch_size, num_experts)

        Returns:
            tuple: (topk_idx, scores)
                - topk_idx: Indices of selected top-k experts
                - scores: Original expert scores
        """
        batch_size, num_experts = scores.shape
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        group_scores = self._calculate_group_scores(scores_for_choice, batch_size)
        group_idx = torch.topk(group_scores, k=self.topk_group)[1]
        group_mask = self._create_group_mask(group_scores, group_idx)
        score_mask = self._expand_group_mask(group_mask, batch_size)
        masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

        _, topk_idx = torch.topk(masked_scores, k=self.top_k)
        return topk_idx, scores

    def _calculate_group_scores(self, scores, batch_size):
        """
        Calculates scores for each group of experts.

        Args:
            scores (torch.Tensor): Expert scores
            batch_size (int): Batch size

        Returns:
            torch.Tensor: Group scores
        """
        return torch.topk(scores.view(batch_size, self.n_group, -1), k=2)[0].sum(dim=-1)

    def _create_group_mask(self, group_scores, group_idx):
        """
        Creates a mask for selected expert groups.

        Args:
            group_scores (torch.Tensor): Scores for each group
            group_idx (torch.Tensor): Indices of selected groups

        Returns:
            torch.Tensor: Binary mask for selected groups
        """
        return torch.scatter(
            input=torch.zeros_like(group_scores),
            dim=1,
            index=group_idx,
            src=torch.ones_like(group_scores)
        )

    def _expand_group_mask(self, group_mask, batch_size):
        """
        Expands the group mask to expert-level granularity.

        Args:
            group_mask (torch.Tensor): Mask for selected groups
            batch_size (int): Batch size

        Returns:
            torch.Tensor: Expanded mask for individual experts
        """
        return group_mask.unsqueeze(-1).expand(
            batch_size, self.n_group, self.num_experts // self.n_group
        ).reshape(batch_size, -1)
