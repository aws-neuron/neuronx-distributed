import torch
from torch import nn, Tensor
from typing import Optional
from torch.distributed import ProcessGroup

from neuronx_distributed.modules.moe.model_utils import ACT2FN
from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers.mappings import get_tensor_model_parallel_group

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
        tensor_parallel_group (ProcessGroup): Process group for tensor parallelism
        reduce_dtype (torch.dtype): Data type for reduction operations
        fused_gate_up_projection (bool): Whether to use fused gate and up projections
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
    ):
        """
        Initialize the SharedExperts module.

        Args:
            hidden_size: Size of the input and output hidden states
            intermediate_size: Size of the intermediate representations
            num_shared_experts: Number of shared expert networks
            hidden_act: Activation function identifier string
            dtype: Data type for the model parameters
            tensor_model_parallel_group: Process group for tensor parallelism
            reduce_dtype: Data type for reduction operations
            fused_gate_up_projection: Whether to use fused gate and up projections
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_shared_experts = num_shared_experts
        self.act_fn = ACT2FN[hidden_act]
        self.dtype = dtype
        self.tensor_parallel_group = tensor_model_parallel_group or get_tensor_model_parallel_group()
        self.reduce_dtype = reduce_dtype
        self.fused_gate_up_projection = fused_gate_up_projection

        self._initialize_projections()

    def _initialize_projections(self):
        """
        Initialize the linear projections for the expert layer.

        Sets up either fused or separate gate and up projections, along with
        the down projection, using tensor parallel linear layers.
        """
        expert_dim = self.intermediate_size * self.num_shared_experts
        common_args = {
            "dtype": self.dtype,
            "pad": True,
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SharedExperts module.
        Args:
            x: Input tensor of shape [batch_size, sequence_length, hidden_size]
        Returns:
            Tensor: Output tensor of shape [batch_size, sequence_length, hidden_size]
        """
        if self.fused_gate_up_projection:
            intermediate_states = self._fused_activation(self.gate_up_proj(x))
        else:
            intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(intermediate_states)

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
