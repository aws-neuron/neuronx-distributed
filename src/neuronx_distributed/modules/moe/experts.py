from typing import Callable, Optional

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.nn import Module

from neuronx_distributed.modules.moe.moe_parallel_layers import (
    ExpertFusedColumnParallelLinear,
    ExpertFusedRowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    enter_expert_parallel_region,
    exit_expert_parallel_region,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_expert_model_parallel_size,
    get_expert_model_parallel_group,
    get_tensor_model_parallel_group,
)
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.modules.moe.model_utils import GLUType
logger = get_logger()

class Experts(Module):
    """Module which performs the expert MLP computations for the given hidden states.
    Expert Parallelism (EP), if enabled, is applied through scatter-gather optimization
    across TP ranks.

    Arguments:
        num_experts: Total number of experts in the model
        hidden_size: Size of the hidden dimension
        intermediate_size: Size of the intermediate dimension in the MLP
        glu: Whether to use Gated Linear Units
        activation_fn: Activation function to use
        dtype: Data type for the module parameters
        device: Device to place the module parameters on
        bias: Whether to include bias terms in linear layers
        glu_type: Type of GLU to use (GLU or SWIGLU)
        hidden_act_scaling_factor: Scaling factor for hidden activations
        hidden_act_bias: Bias term added to the linear layer in hidden activation
        input_layer_init_method: Initialization method for input layers
        output_layer_init_method: Initialization method for output layers
        is_prefill: Whether this is used in prefill phase
        tensor_model_parallel_group: Process group for tensor parallelism
        expert_model_parallel_group: Process group for expert parallelism
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        glu: bool,
        activation_fn: Callable[[Tensor], Tensor],
        dtype: torch.dtype,
        device: torch.device,
        bias: bool = False,
        glu_type: Optional[GLUType] = GLUType.GLU,
        hidden_act_scaling_factor: float = 1.,
        hidden_act_bias: float = 0.,
        input_layer_init_method=None,
        output_layer_init_method=None,
        is_prefill = True,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        expert_model_parallel_group: Optional[ProcessGroup] = None,
    ) -> None:
        super().__init__()

        self._glu = glu
        if self._glu:
            glu_type = GLUType.validate(glu_type)
        self._glu_type = glu_type
        self._activation_fn = activation_fn
        self.num_experts = num_experts
        self.hidden_act_scaling_factor = hidden_act_scaling_factor
        self.hidden_act_bias = hidden_act_bias
        # todo: we can also generalize expert-parallel group
        self.tensor_parallel_group = tensor_model_parallel_group if \
            tensor_model_parallel_group is not None else get_tensor_model_parallel_group()
        self.expert_model_parallel_group = expert_model_parallel_group if \
            expert_model_parallel_group is not None else get_expert_model_parallel_group()

        if self._glu:
            self.gate_up_proj = ExpertFusedColumnParallelLinear(
                # we pass the global number of experts. the linear layer will itself
                # decide to initialize a subset of them if EP is applied.
                num_experts=num_experts,
                input_size=hidden_size,
                # we fuse up and gate projections to a single matmul. Later on in code
                # we'll split the resulting output to yield up and gate matrices.
                output_size=intermediate_size * 2,
                bias=bias,
                dtype=dtype,
                device=device,
                stride=2,
                init_method=input_layer_init_method,
                tensor_model_parallel_group=self.tensor_parallel_group,
                expert_model_parallel_group=self.expert_model_parallel_group,
                is_prefill=is_prefill,
            )
        else:
            self.up_proj = ExpertFusedColumnParallelLinear(
                # we pass the global number of experts. the linear layer will itself
                # decide to initialize a subset of them if EP is applied.
                num_experts=num_experts,
                input_size=hidden_size,
                output_size=intermediate_size,
                bias=bias,
                dtype=dtype,
                device=device,
                init_method=input_layer_init_method,
                tensor_model_parallel_group=self.tensor_parallel_group,
                expert_model_parallel_group=self.expert_model_parallel_group,
                is_prefill=is_prefill,
            )

        self.down_proj = ExpertFusedRowParallelLinear(
            # we pass the global number of experts. the linear layer will itself
            # decide to initialize a subset of them if EP is applied.
            num_experts=num_experts,
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            reduce_output=False,
            dtype=dtype,
            device=device,
            init_method=output_layer_init_method,
            tensor_model_parallel_group=self.tensor_parallel_group,
            expert_model_parallel_group=self.expert_model_parallel_group,
            is_prefill=is_prefill,
        )

    def forward(self, hidden_states: Tensor, expert_indices: Optional[Tensor] = None) -> Tensor:
        """
        Common nomenclature:
            E: Total number of experts, C: Expert capacity, H: Hidden Size

        If expert_indices is None, then the mlp_op is computed on all E experts.
        If specified, then the mlp_op is computed only on the experts specified.

        Let num_experts_computed E' = E if expert_indices is None else expert_indices.shape[0]

        Arguments:
            hidden_states: Input tensor containing the token hidden states.
                           Its shape must be broadcastable with (E', C, H).
            expert_indices: (Optional) 1D Tensor containing the indices of experts to perform the mlp_op on.
        Returns:
            output: Output tensor of shape (E', C, H) obtained after the gate/up projection
                            + activation + down projection operations.
        """

        if expert_indices is not None and get_expert_model_parallel_size() > 1:
            raise ValueError("Selective expert loading is not supported with expert parallelism.")

        # Verify shapes
        assert len(hidden_states.shape) == 3
        num_experts_computed = self.num_experts if expert_indices is None else expert_indices.shape[0]
        assert hidden_states.shape[0] in {1, num_experts_computed}

        e, c, h = hidden_states.shape

        # Apply scatter-gather optimization in EP only when the number of tokens
        # are divisible by TP. Note that this will exclude the token-generation scenario.
        # Also in training, performance will be much worse with EP if the local expert
        # capacity is not divisible by TP degree.
        # num_tokens_divisible_by_tp = c % get_tensor_model_parallel_size() == 0

        if self.expert_model_parallel_group.size() > 1 and self.training:
            # (e, c, h) -> (e/ep, ep, c, h)
            dispatched_hidden_states = enter_expert_parallel_region(
                hidden_states,
                # temporarily disable scatter_gather optimization
                scatter_gather=False,
                # scatter_gather=num_tokens_divisible_by_tp,
            )
        else:
            dispatched_hidden_states = hidden_states.view(e, 1, c, h)

        if self._glu:
            # (e/ep, ep, c, 2i/tp)
            intermediate_states = self.gate_up_proj.forward(dispatched_hidden_states, expert_indices=expert_indices)
        else:
            # (e/ep, ep, c, i/tp)
            intermediate_states = self.up_proj.forward(dispatched_hidden_states, expert_indices=expert_indices)

        # (e/ep, ep, c, i/tp)
        intermediate_states = self._activation(intermediate_states)

        # (e/ep, ep, c, h)
        projected_states = self.down_proj.forward(intermediate_states, expert_indices=expert_indices)

        if self.expert_model_parallel_group.size() > 1 and self.training:
            # (e/ep, ep, c, h) -> (e, c, h)
            output = exit_expert_parallel_region(
                projected_states,
                # temporarily disable scatter_gather optimization
                scatter_gather=False,
                # scatter_gather=num_tokens_divisible_by_tp,
            )
        else:
            output = projected_states.squeeze(1)

        return output

    def _activation(self, x: Tensor) -> Tensor:
        if self._glu:
            if self._glu_type == GLUType.GLU:
                gate, up = torch.chunk(x, chunks=2, dim=-1)
                return self._activation_fn(self.hidden_act_scaling_factor * gate) * (up + self.hidden_act_bias)
            elif self._glu_type == GLUType.SWIGLU:
                gate, up = torch.chunk(x, chunks=2, dim=-1)
                gate = gate * self._activation_fn(self.hidden_act_scaling_factor * gate)
                out = gate * (up + self.hidden_act_bias)
                return out
            else:
                raise NotImplementedError(f"Only supports glu_type='glu', 'swiglu', got {self._glu_type}")
        else:
            return self._activation_fn(x)
