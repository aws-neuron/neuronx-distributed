import torch
from torch import nn, Tensor
from typing import Optional, Dict, Any, Union, Tuple
from torch.distributed import ProcessGroup

from neuronx_distributed.modules.moe.model_utils import ACT2FN, create_spmd_ranks
from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear, parallel_state
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_group
from neuronx_distributed.parallel_layers.utils import indices_split_along_dim

weight_cache: Dict[str, Any] = {}


class ColumnParallelLinearTransposed(ColumnParallelLinear):
    def forward(self, input: torch.Tensor, slice_indices: Optional[torch.Tensor] = None) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self._check_pad_false_for_training()

        input_parallel = self._cpl_maybe_input_copy_to_tp_region(input)
        weight = self.weight[slice_indices, :] if slice_indices is not None else self.weight
        # Matrix multiply.
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=weight.t(),
            bias=None,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            autograd_func_class=self.autograd_func_class,
            process_group=self.tensor_parallel_group,
            reduce_dtype=self.reduce_dtype,
        )

        output = self._cpl_maybe_gather_output(output_parallel)

        if self.skip_bias_add:
            return output, self.bias
        output = (output + self.bias) if self.bias is not None else output

        return output


class RowParallelLinearTransposed(RowParallelLinear):
    def forward(self, input_: torch.Tensor, slice_indices: Optional[torch.Tensor] = None) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        self._check_pad_false_for_training()

        # Set up backprop all-reduce.
        input_parallel = self._rpl_maybe_scatter_input(input_)
        weight = self.weight[:, slice_indices] if slice_indices is not None else self.weight
        # Matrix multiply.
        output_ = self._forward_impl(
            input=input_parallel,
            weight=weight.t(),
            bias=None,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
            sequence_dimension=self.sequence_dimension,
            autograd_func_class=self.autograd_func_class,
            process_group=self.tensor_parallel_group,
            reduce_dtype = self.reduce_dtype,
        )

        output_ = self._rpl_maybe_reduce_output(output_)

        if self.skip_bias_add:
            return output_, self.bias
        output = (output_ + self.bias) if self.bias is not None else output_
        return output


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
        if self.transpose_weights:
            assert not self.fused_gate_up_projection
            for suffix in ["weight", "scale"]:
                if prefix.endswith(suffix):
                    prefix = prefix.removesuffix("weight")
                    model_state_dict[prefix + "gate_proj.weight"] = model_state_dict[prefix + "gate_proj.weight"].T
                    model_state_dict[prefix + "up_proj.weight"] = model_state_dict[prefix + "up_proj.weight"].T
                    model_state_dict[prefix + "down_proj.weight"] = model_state_dict[prefix + "down_proj.weight"].T
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
            self.down_proj = self._create_row_parallel_linear(
                input_size=expert_dim,
                **common_args
            )
        else:
            if self.transpose_weights:
                self.gate_proj = self._create_row_parallel_linear(
                    input_size=expert_dim,
                    **common_args
                )
                self.up_proj = self._create_row_parallel_linear(
                    input_size=expert_dim,
                    **common_args
                )
                self.down_proj = self._create_column_parallel_linear(
                    output_size=expert_dim,
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
        if self.transpose_weights:
            return ColumnParallelLinearTransposed(
                self.hidden_size,
                output_size,
                stride=stride,
                bias=False,
                gather_output=False,
                **kwargs
            )
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
        if self.transpose_weights:
            return RowParallelLinearTransposed(
                input_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                reduce_output=False,
                reduce_dtype=self.reduce_dtype,
                **kwargs
            )
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
