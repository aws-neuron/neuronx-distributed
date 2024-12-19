from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch.distributed import ProcessGroup
import torch.nn.functional as F

from neuronx_distributed.modules.moe.moe_parallel_layers import LinearRouter
from neuronx_distributed.parallel_layers import mappings


class RouterBase(torch.nn.Module, ABC):
    """Base class for various routing strategies used in MoE.

    Arguments:
        num_experts: Total number of experts.
        top_k: Number of experts activated per token. Should be less than or equal to num_experts.
        hidden_size: Hidden dimension of the input sequence.
        act_fn: Activation used to obtain expert affinities from router logits. One of 'sigmoid' or 'softmax'.
        dtype: Datatype for the layer weights.
        device: Device for the layer weights.
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
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
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

        # Create router
        self.linear_router = LinearRouter(
            input_size=hidden_size,
            output_size=num_experts,
            dtype=dtype,
            device=device,
            tensor_model_parallel_group=tensor_model_parallel_group,
        )

    def get_router_logits_and_expert_affinities(self, hidden_states):
        """
        Returns the router logits and expert affinities.
        Note that this function always returns the logits and affinities corresponding to the full hidden states, by handling 
        sequence parallelism internally.
        """

        # router_logits: (*, H) @ (H, E) -> (*, E)            
        router_logits = self.linear_router(hidden_states)

        if self.sequence_parallel_enabled:
            assert not self.training, "Router in SP is currently supported only for inference"
            # Gather the router_logits across ranks
            router_logits = mappings.gather_from_sequence_parallel_region(
                router_logits,
                sequence_dimension=self.sequence_dimension,
                to_model_parallel=False,
            )

        # Flatten S and B to T dimension
        router_logits = router_logits.view(-1, self.num_experts)

        # Perform activation in fp64 to prevent auto-downcasting of operation to bf16, for numerical accuracy
        # expert_affinities: (T, E)
        if self.act_fn == "sigmoid":
            expert_affinities = torch.sigmoid(router_logits.to(dtype=torch.float64))
        elif self.act_fn == "softmax":
            expert_affinities = F.softmax(router_logits, dim=1, dtype=torch.float64)
        else:
            raise ValueError("act_fn must be either 'sigmoid' or 'softmax'")

        # Cast to required dtype
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)
        return router_logits, expert_affinities

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
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            act_fn="softmax",  # Always use softmax activation for TopK router
            sequence_parallel_enabled=sequence_parallel_enabled,
            sequence_dimension=sequence_dimension,
            dtype=dtype,
            device=device,
            tensor_model_parallel_group=tensor_model_parallel_group,
        )

    def forward(self, hidden_states):
        # Get router_logits and expert_affinities
        router_logits, expert_affinities = self.get_router_logits_and_expert_affinities(hidden_states)

        # For each token, get the top_k experts
        # expert_index: (T, top_k)
        _, expert_index = torch.topk(router_logits, self.top_k)
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
        router_logits, expert_affinities = self.get_router_logits_and_expert_affinities(hidden_states)

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
