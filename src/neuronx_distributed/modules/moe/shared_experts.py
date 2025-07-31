import torch
from torch import nn, Tensor
from typing import Optional, Dict, Any
from torch.distributed import ProcessGroup

from neuronx_distributed.modules.moe.model_utils import ACT2FN, create_spmd_ranks
from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear, parallel_state
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_group
from neuronx_distributed.parallel_layers.utils import indices_split_along_dim

weight_cache: Dict[str, Any] = {}

def _get_weight_from_state_dict(prefix: str, state_dict: Dict[str, Any]) -> torch.Tensor:
    if prefix in weight_cache:
        return weight_cache[prefix]

    if (prefix + "weight") in state_dict:
        transposed_weight = state_dict[prefix + "weight"].t()
        weight_cache[prefix] = transposed_weight
        return transposed_weight

    else:
        raise RuntimeError(f"Cannot find {(prefix + 'weight')} in the state_dict")


def _set_weight_to_state_dict(
    prefix: str, tensor: torch.Tensor, state_dict: Dict[str, Any]
) -> None:
    if (prefix + "weight") in state_dict:
        state_dict[prefix + "weight"] = tensor.t()
    else:
        raise RuntimeError(f"Cannot find {(prefix + 'weight')} in the state_dict")


def transpose_parallel_linear_layer(parallel_layer):
    """
    This function clones and transposes a ColumnParallelLinear or RowParallelLinear
    The attributes are also cloned and partition_dim is updated
    """
    orig_attrs = vars(parallel_layer)
    new_layer = torch.nn.Parameter(parallel_layer.clone().T, requires_grad=False)
    new_layer.__dict__.update(orig_attrs)
    # flip the partition_dim from 0->1 or 1->0
    setattr(new_layer, "partition_dim", 1 - getattr(new_layer, "partition_dim"))
    setattr(new_layer, "get_tensor_from_state_dict", _get_weight_from_state_dict)
    setattr(new_layer, "set_tensor_to_state_dict", _set_weight_to_state_dict)
    return new_layer


class SharedExperts(nn.Module):
    """
    Implementation of shared experts for mixture of experts architecture.

    This module implements a shared expert layer that consists of gate projection,
    up projection, and down projection with activation functions and parallel
    processing capabilities. It supports both fused and separate gate/up projections.

    Attributes:
        hidden_size (int): Size of the input and output hidden states
        intermediate_size (int): Size of the intermediate representations
        num_shared_experts (int): Number of shared expert networks
        act_fn (callable): Activation function to use
        dtype (torch.dtype): Data type for the model parameters
        reduce_dtype (torch.dtype): Data type for reduction operations
        fused_gate_up_projection (bool): Whether to use fused gate and up projections
        sequence_parallel_enabled (bool): Whether to enable sequence parallelism
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_shared_experts: int,
        hidden_act: str,
        dtype: torch.dtype = torch.float32,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        reduce_dtype: torch.dtype = torch.float32,
        fused_gate_up_projection: bool = False,
        sequence_parallel_enabled: bool = False,
        transpose_weights: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_shared_experts = num_shared_experts
        self.act_fn = ACT2FN[hidden_act]
        self.dtype = dtype

        self.reduce_dtype = reduce_dtype
        self.fused_gate_up_projection = fused_gate_up_projection
        self.transpose_weights = transpose_weights
        # initialized at the beginning
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.world_size = parallel_state.get_world_group().size()
        if sequence_parallel_enabled:
            self.spmd_rank = SPMDRank(world_size=self.world_size)
            # replicating weights on each core to run in sequence parallel for context encoding
            tp_group_mesh = [
                list(range(self.world_size))[i: i + 1]
                for i in range(0, self.world_size)
            ]
            self.tensor_parallel_group = torch.distributed.new_group(None, pg_options={
                "xla_pg_options": {"mesh": tp_group_mesh}})
        else:
            self.tensor_parallel_group = tensor_model_parallel_group or get_tensor_model_parallel_group()
        self._initialize_parallel_layers()

    def preshard_hook(self, model_state_dict: Dict[str, Any], prefix: str) -> None:
        if self.sequence_parallel_enabled :
            prefix = prefix.removesuffix("weight")
            create_spmd_ranks(
                model_state_dict=model_state_dict,
                prefix=prefix,
                world_size=self.world_size,
            )

    def _initialize_parallel_layers(self):
        """
        Initialize the linear projections for the expert layer.

        Sets up either fused or separate gate and up projections, along with
        the down projection, using tensor parallel linear layers.
        """
        expert_dim = self.intermediate_size * self.num_shared_experts
        common_args = {
            "dtype": self.dtype,
            "pad": not self.training,
            "tensor_model_parallel_group": self.tensor_parallel_group,
        }

        if self.fused_gate_up_projection:
            self.gate_up_proj = self._create_column_parallel_linear(
                output_size=expert_dim * 2,
                stride=2,
                **common_args
            )
        else:
            self.gate_proj = self._create_column_parallel_linear(
                output_size=expert_dim,
                **common_args
            )
            self.up_proj = self._create_column_parallel_linear(
                output_size=expert_dim,
                **common_args
            )

        self.down_proj = self._create_row_parallel_linear(
            input_size=expert_dim,
            **common_args
        )

        if self.transpose_weights:
            assert not self.fused_gate_up_projection
            self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
            self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
            self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)

    def _create_column_parallel_linear(self, output_size: int, stride: int = 1, **kwargs):
        """
        Create a column parallel linear layer.
        Args:
            output_size: Size of the output dimension
            stride: Stride for the linear transformation
            **kwargs: Additional arguments for ColumnParallelLinear
        Returns:
            ColumnParallelLinear: The initialized parallel linear layer
        """
        return ColumnParallelLinear(
            self.hidden_size,
            output_size,
            stride=stride,
            bias=False,
            gather_output=False,
            **kwargs
        )

    def _create_row_parallel_linear(self, input_size: int, **kwargs):
        """
        Create a row parallel linear layer.
        Args:
            input_size: Size of the input dimension
            **kwargs: Additional arguments for RowParallelLinear
        Returns:
            RowParallelLinear: The initialized parallel linear layer
        """
        return RowParallelLinear(
            input_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_output=False,
            reduce_dtype=self.reduce_dtype,
            **kwargs
        )

    def forward(self, x: Tensor, seq_len: int) -> Tensor:
        # Handle token generation vs context encoding forward passes
        if seq_len == 1 and self.sequence_parallel_enabled:
            return self._forward_token_gen_replicated_weights(x)
        else:
            return self._forward(x)

    def _forward_token_gen_replicated_weights(self, x: Tensor) -> Tensor:
        """Handles token gen with replicated weights which are used to run Shared Experts in SP for context encoding.
           Here the weights need to be sliced based on the current rank.
        """
        if self.fused_gate_up_projection:
            intermediate_states = self._fused_activation(
                self.gate_up_proj(x, self.get_split_indices(self.gate_up_proj.weight, dim=0))
            )
        else:
            gate = self.gate_proj(x, self.get_split_indices(self.gate_proj.weight, dim=0))
            up = self.up_proj(x, self.get_split_indices(self.up_proj.weight, dim=0))
            intermediate_states = self.act_fn(gate) * up

        return self.down_proj(
            intermediate_states,
            self.get_split_indices(self.down_proj.weight, dim=1)
        )

    def _forward(self, x: Tensor) -> Tensor:
        """Handle forward pass for pure TP (CTE + TKG) or SP flow (CTE)"""
        if self.fused_gate_up_projection:
            intermediate_states = self._fused_activation(self.gate_up_proj(x))
        else:
            intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(intermediate_states)

    def get_split_indices(self, weight: Tensor, dim: int) -> Tensor:
        """Helper method to get split indices for weight slicing"""
        return indices_split_along_dim(
            weight,
            dim,
            rank=self.spmd_rank.rank.data,
            num_partitions=parallel_state.get_world_group().size()
        )

    def _fused_activation(self, x: Tensor) -> Tensor:
        """
        Apply fused activation function to the concatenated gate and up projections.
        Args:
            x: Input tensor containing concatenated gate and up projections
        Returns:
            Tensor: Result of gate activation multiplied by up projection
        """
        gate, up = torch.chunk(x, chunks=2, dim=-1)
        return self.act_fn(gate) * up
