import warnings
import torch
import torch_xla.core.xla_model as xm
import torch.nn.functional as F

from neuronx_distributed.kernels.kernel_utils import torch_to_nki_dtype
from neuronx_distributed.modules.moe.model_utils import DEFAULT_PADDING_VALUE, DEFAULT_BLOCK_SIZE
from neuronx_distributed.modules.moe.moe_configs import BlockwiseMatmulConfig
from neuronx_distributed.modules.moe.nki_import import NKIImport, import_nki
from neuronx_distributed.parallel_layers import mappings
from neuronxcc.nki.compiler.backends.neuron.dimensions import VNC
from neuronx_distributed.utils.model_utils import LogicalNCConfig
from torch_neuronx.xla_impl.base import xla_call

from dataclasses import dataclass
from typing import Tuple, Any, Optional


def initialize_nki_components() -> dict:
    """
    Initialize all NKI components.

    Returns:
        dict: Mapping of component names to their imported values
    """
    imports = {
        "blockwise_mm": NKIImport("blockwise_mm", module_name="blockwise_mm", nki_jit_type="use_nki_jit_decorator"),
        "blockwise_mm_baseline_shard_hidden": NKIImport("blockwise_mm_baseline_shard_hidden", module_name="blockwise_mm", nki_jit_type="use_nki_jit_decorator"),
        "blockwise_mm_baseline_block_parallel": NKIImport("blockwise_mm_baseline_block_parallel", module_name="blockwise_matmul", nki_jit_type="use_nki_jit_decorator"),
        "blockwise_mm_baseline_block_parallel_allocated": NKIImport("blockwise_mm_baseline_block_parallel_allocated", module_name="blockwise_matmul", nki_jit_type="use_jit_decorator"),
        "blockwise_mm_bwd": NKIImport("blockwise_mm_bwd", module_name="blockwise_mm_bwd", nki_jit_type="use_nki_jit_decorator"),
        "blockwise_mm_bwd_baseline_shard_hidden": NKIImport("blockwise_mm_bwd_baseline_shard_hidden", module_name="blockwise_mm_bwd", nki_jit_type="use_nki_jit_decorator"),
        "blockwise_mm_baseline": NKIImport("blockwise_mm_baseline", module_name="blockwise_mm", nki_jit_type="use_nki_jit_decorator"),
        "check_compatibility": NKIImport("check_blockwise_mm_kernel_compatibility", module_name="blockwise_mm"),
        "block_shard_strategy": NKIImport("BlockShardStrategy", module_name="blockwise_mm"),
        "affinity_scale_mode": NKIImport("ExpertAffinityScaleMode"),
        "skip_mode": NKIImport("SkipMode", module_name="blockwise_mm"),
    }

    components = {}
    for name, config in imports.items():
        component, error = import_nki(config)
        if error:
            warnings.warn(f"Warning: {error}")
        components[name] = component

    return components


# Initialize all components
nki_components = initialize_nki_components()

# Assign to module-level variables
_blockwise_mm_nki_call = nki_components["blockwise_mm"]
_blockwise_mm_baseline_shard_hidden_nki_call = nki_components["blockwise_mm_baseline_shard_hidden"]
_blockwise_mm_baseline_block_parallel_nki_call = nki_components["blockwise_mm_baseline_block_parallel"]
_blockwise_mm_baseline_block_parallel_allocated_nki_call = nki_components["blockwise_mm_baseline_block_parallel_allocated"]
_blockwise_mm_nki_bwd_call = nki_components["blockwise_mm_bwd"]
_blockwise_mm_bwd_baseline_shard_hidden_nki_call = nki_components["blockwise_mm_bwd_baseline_shard_hidden"]
_blockwise_mm_training_nki_call = nki_components["blockwise_mm_baseline"]
check_blockwise_mm_kernel_compatibility = nki_components["check_compatibility"]
BlockShardStrategy = nki_components["block_shard_strategy"]
ExpertAffinityScaleMode = nki_components["affinity_scale_mode"]
SkipMode = nki_components["skip_mode"]


def dynamic_slice_3D(tensor, start0, start1, start2, size0, size1, size2):
    @xla_call
    def _dynamic_slice(tensor, start0, start1, start2):
        return tensor.dynamic_slice([start0, start1, start2], [size0, size1, size2])

    return _dynamic_slice(tensor, start0, start1, start2)


def dynamic_slice_2D(tensor, start0, start1, size0, size1):
    @xla_call
    def _dynamic_slice(tensor, start0, start1):
        return tensor.dynamic_slice([start0, start1], [size0, size1])

    return _dynamic_slice(tensor, start0, start1)


def dynamic_slice_1D(tensor, start0, size0):
    @xla_call
    def _dynamic_slice(tensor, start0):
        return tensor.dynamic_slice([start0], [size0])

    return _dynamic_slice(tensor, start0)

@dataclass
class BlockwiseMatmulArgs:
    """Dataclass to hold all possible arguments for blockwise matmul operations."""
    # Input tensors
    hidden_states: torch.Tensor
    expert_affinities_masked: torch.Tensor
    
    # MLP weights
    gate_up_proj_weight: torch.Tensor
    down_proj_weight: torch.Tensor
    
    # Block related parameters
    token_position_to_id: torch.Tensor
    block_to_expert: torch.Tensor
    block_size: int = DEFAULT_BLOCK_SIZE
    
    # scales for quantization
    gate_up_proj_scale: Optional[torch.Tensor] = None
    down_proj_scale: Optional[torch.Tensor] = None

    # Output tensors
    output: Optional[torch.Tensor] = None
    gate_up_activations_T: Optional[torch.Tensor] = None
    down_activations: Optional[torch.Tensor] = None
    
    # Meta parameters
    skip_dma: Any = SkipMode(False, False)
    is_tensor_update_accumulating: bool = False
    expert_affinities_scaling_mode: Any = ExpertAffinityScaleMode.POST_SCALE
    block_sharding_strategy: Any = BlockShardStrategy.HI_LO
    dtype: torch.dtype = torch.bfloat16

def _call_training_kernel(args: BlockwiseMatmulArgs):
    """Call the training kernel for blockwise matmul."""
    _blockwise_mm_training_nki_call(
        hidden_states=args.hidden_states,
        expert_affinities_masked=args.expert_affinities_masked,
        gate_up_proj_weight=args.gate_up_proj_weight,
        down_proj_weight=args.down_proj_weight,
        block_size=args.block_size,
        token_position_to_id=args.token_position_to_id.to(dtype=torch.int32),
        block_to_expert=args.block_to_expert.to(dtype=torch.int32),
        output=args.output,
        gate_up_activations_T=args.gate_up_activations_T,
        down_activations=args.down_activations,
        is_tensor_update_accumulating=args.is_tensor_update_accumulating,
    )
    return args.output, args.gate_up_activations_T, args.down_activations


def _call_training_shard_hidden_kernel(args: BlockwiseMatmulArgs):
    """Call the training shard hidden kernel for blockwise matmul."""
    _blockwise_mm_baseline_shard_hidden_nki_call[VNC(2)](
        # Inputs
        hidden_states=args.hidden_states,
        expert_affinities_masked=args.expert_affinities_masked,
        # MLP weights
        gate_up_proj_weight=args.gate_up_proj_weight,
        down_proj_weight=args.down_proj_weight,
        # Block related
        block_size=args.block_size,
        token_position_to_id=args.token_position_to_id.to(dtype=torch.int32),
        block_to_expert=args.block_to_expert.to(dtype=torch.int32),
        # Output
        output=args.output,
        # Meta parameters
        gate_up_activations_T=args.gate_up_activations_T,
        down_activations=args.down_activations,
        skip_dma=args.skip_dma,
        is_tensor_update_accumulating=args.is_tensor_update_accumulating,
        expert_affinities_scaling_mode=args.expert_affinities_scaling_mode
    )
    return args.output, args.gate_up_activations_T, args.down_activations


def _call_block_parallel_allocated_kernel(args: BlockwiseMatmulArgs):
    """Call the block parallel allocated kernel for blockwise matmul with quantization."""

    down_proj_scale = args.down_proj_scale
    if down_proj_scale is not None:
        assert args.block_size in [128, 256], "Block size must be 128 or 256 for blockwise matmul Kernel with quantized checkpoints"
        e, h = down_proj_scale.shape[0], down_proj_scale.shape[-1]
        down_proj_scale = down_proj_scale.view(e,1,-1,128)
        down_proj_scale = down_proj_scale.transpose(2,3)
        down_proj_scale = down_proj_scale.view(e,1,h)

    return _blockwise_mm_baseline_block_parallel_allocated_nki_call[VNC(2)](
        hidden_states=args.hidden_states,
        expert_affinities_masked=args.expert_affinities_masked,
        gate_up_proj_weight=args.gate_up_proj_weight,
        down_proj_weight=args.down_proj_weight,
        gate_up_proj_scale=args.gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        block_size=args.block_size,
        token_position_to_id=args.token_position_to_id.to(dtype=torch.int32),
        block_to_expert=args.block_to_expert.to(dtype=torch.int32),
        skip_dma=args.skip_dma,
        is_tensor_update_accumulating=args.is_tensor_update_accumulating,
        expert_affinities_scaling_mode=args.expert_affinities_scaling_mode,
        block_sharding_strategy=args.block_sharding_strategy,
        compute_dtype = torch_to_nki_dtype(args.dtype),
    )


def _call_block_parallel_kernel(args: BlockwiseMatmulArgs):
    """Call the block parallel kernel for blockwise matmul."""
    _blockwise_mm_baseline_block_parallel_nki_call[VNC(2)](
        hidden_states=args.hidden_states,
        expert_affinities_masked=args.expert_affinities_masked,
        gate_up_proj_weight=args.gate_up_proj_weight,
        down_proj_weight=args.down_proj_weight,
        block_size=args.block_size,
        token_position_to_id=args.token_position_to_id.to(dtype=torch.int32),
        block_to_expert=args.block_to_expert.to(dtype=torch.int32),
        output=args.output,
        skip_dma=args.skip_dma,
        is_tensor_update_accumulating=args.is_tensor_update_accumulating,
        expert_affinities_scaling_mode=args.expert_affinities_scaling_mode,
        block_sharding_strategy=args.block_sharding_strategy,
        compute_dtype = torch_to_nki_dtype(args.dtype),
    )

    return args.output


def _call_shard_hidden_kernel(args: BlockwiseMatmulArgs):
    """Call the shard hidden kernel for blockwise matmul."""
    _blockwise_mm_baseline_shard_hidden_nki_call[VNC(2)](
        hidden_states=args.hidden_states,
        expert_affinities_masked=args.expert_affinities_masked,
        gate_up_proj_weight=args.gate_up_proj_weight,
        down_proj_weight=args.down_proj_weight,
        block_size=args.block_size,
        token_position_to_id=args.token_position_to_id.to(dtype=torch.int32),
        block_to_expert=args.block_to_expert.to(dtype=torch.int32),
        output=args.output,
        gate_up_activations_T=args.gate_up_activations_T,
        down_activations=args.down_activations,
        skip_dma=args.skip_dma,
        is_tensor_update_accumulating=args.is_tensor_update_accumulating,
        expert_affinities_scaling_mode=args.expert_affinities_scaling_mode,
        compute_dtype=torch_to_nki_dtype(args.dtype),
    )

    return args.output, args.gate_up_activations_T, args.down_activations


def _call_default_kernel(args: BlockwiseMatmulArgs):
    """Call the default kernel for blockwise matmul."""

    gate_up_proj_scale, down_proj_scale = args.gate_up_proj_scale, args.down_proj_scale

    if gate_up_proj_scale is not None:
        # (1, 1, 2I) -> (2, I)
        assert gate_up_proj_scale.shape[0] == 1 and gate_up_proj_scale.shape[1] == 1
        gate_up_proj_scale = gate_up_proj_scale.view(2, -1)

    if down_proj_scale is not None:
        # (1, 1, H) -> (H)
        assert down_proj_scale.shape[0] == 1 and down_proj_scale.shape[1] == 1
        down_proj_scale = down_proj_scale.view(-1)

    _blockwise_mm_nki_call(
        hidden_states=args.hidden_states,
        expert_affinities_masked=args.expert_affinities_masked,
        gate_up_proj_weight=args.gate_up_proj_weight,
        down_proj_weight=args.down_proj_weight,
        block_size=args.block_size,
        token_position_to_id=args.token_position_to_id.to(dtype=torch.int32),
        block_to_expert=args.block_to_expert.to(dtype=torch.int32),
        output=args.output,
        skip_dma=args.skip_dma,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        is_tensor_update_accumulating=args.is_tensor_update_accumulating,
        compute_type=torch_to_nki_dtype(args.dtype),
    )

    return args.output

class TorchBlockwiseTraining(torch.autograd.Function):
    """
    PyTorch implementation of the blockwise matmul, with selective activation.

    In the forward and backward pass, use the custom dynamic slicing operation to avoid static slicing
    and use torch.Tensor.scatter_ to avoid dynamic update slicing. Both static slicing and dynamic
    update slicing introduce graph recompilation and significantly slows down performance. The original
    versions of the code is included as comments for clarity.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states,
        expert_affinities_masked,
        token_position_to_id,
        block_to_expert,
        gate_up_weight,
        down_weight,
    ):
        ctx.num_blocks = num_blocks = block_to_expert.shape[0]
        total_tokens, hidden_size = hidden_states.shape
        block_size = token_position_to_id.shape[0] // num_blocks
        # Add extra row for output of padding tokens (i.e. tokens which have -1 index)
        output = torch.zeros(
            total_tokens + 1,
            hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        # block_to_token_indices: (N, B)
        block_to_token_indices = token_position_to_id.view(
            num_blocks, block_size
        )
        gate_up_activations = torch.empty(
            num_blocks,
            block_size,
            gate_up_weight.shape[2],
            dtype=gate_up_weight.dtype,
            device=gate_up_weight.device,
        )
        down_activations = torch.empty(
            num_blocks,
            block_size,
            hidden_size,
            dtype=down_weight.dtype,
            device=down_weight.device,
        )
        for block_idx in range(num_blocks):
            xm.mark_step()
            # Casting to nn.Parameter so it's treated as an input by the compiler to avoid recompilation.
            block_idx = torch.nn.Parameter(
                torch.tensor(block_idx), requires_grad=False
            ).to(hidden_states.device)
            # Zero index to make taking the entire slice along a dimension more convenient.
            zero_idx = torch.nn.Parameter(torch.tensor(0), requires_grad=False).to(
                hidden_states.device
            )

            # Get token indices and expert id.
            # Original: block_token_indices = block_to_token_indices[block_idx]
            block_token_indices = dynamic_slice_2D(
                tensor=block_to_token_indices,
                start0=block_idx,
                start1=zero_idx,
                size0=1,
                size1=block_to_token_indices.shape[1],
            ).squeeze(0)
            # Original: block_expert_idx = block_to_expert[block_idx]
            block_expert_idx = dynamic_slice_1D(
                tensor=block_to_expert, start0=block_idx, size0=1
            )[0]

            # Gate up projection.
            block_hidden_states = hidden_states[block_token_indices]
            # original: gate_up_weight_slice = gate_up_weight[block_expert_idx]
            gate_up_weight_slice = dynamic_slice_3D(
                tensor=gate_up_weight,
                start0=block_expert_idx,
                start1=zero_idx,
                start2=zero_idx,
                size0=1,
                size1=gate_up_weight.shape[1],
                size2=gate_up_weight.shape[2],
            ).squeeze(0)
            gate_up_activation = block_hidden_states @ gate_up_weight_slice
            # original: gate_up_activations[block_idx] = gate_up_activation
            gate_up_activations.scatter_(
                dim=0,
                index=block_idx.unsqueeze(0).expand_as(gate_up_activation).unsqueeze(0),
                src=gate_up_activation.unsqueeze(0),
            )

            # Split and compute first dot activation.
            gate, up = torch.chunk(gate_up_activation, chunks=2, dim=-1)
            first_dot_activation = F.silu(gate) * up

            # Down projection.
            # original: down_weight_slice = down_weight[block_expert_idx]
            down_weight_slice = dynamic_slice_3D(
                tensor=down_weight,
                start0=block_expert_idx,
                start1=zero_idx,
                start2=zero_idx,
                size0=1,
                size1=down_weight.shape[1],
                size2=down_weight.shape[2],
            ).squeeze(0)
            down_activation = first_dot_activation @ down_weight_slice
            # original: down_activations[block_idx] = down_activation
            down_activations.scatter_(
                dim=0,
                index=block_idx.unsqueeze(0).expand_as(down_activation).unsqueeze(0),
                src=down_activation.unsqueeze(0),
            )

            # Scale by expert affinities.
            expert_affinities = expert_affinities_masked[block_token_indices]
            # original: block_expert_affinities = expert_affinities[:, block_expert_idx]
            block_expert_affinities = dynamic_slice_2D(
                tensor=expert_affinities,
                start0=zero_idx,
                start1=block_expert_idx,
                size0=expert_affinities.shape[0],
                size1=1,
            )
            block_output = down_activation * block_expert_affinities

            output[block_token_indices] += block_output
            xm.mark_step()
        # Drop the last row
        # [T, H]
        output = output[:total_tokens, :]
        ctx.save_for_backward(
            hidden_states,
            expert_affinities_masked,
            block_to_token_indices,
            block_to_expert,
            gate_up_weight,
            down_weight,
            gate_up_activations,
            down_activations,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        num_blocks = ctx.num_blocks
        (
            hidden_states,
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            gate_up_weight,
            down_weight,
            gate_up_activations,
            down_activations,
        ) = ctx.saved_tensors
        T, E = expert_affinities_masked.shape
        H = hidden_states.shape[-1]
        # add last row for -1 token index
        hidden_states_grad = torch.zeros(
            T + 1, H, device=hidden_states.device, dtype=hidden_states.dtype
        )
        affinities_grad = torch.zeros(
            T + 1,
            E,
            device=expert_affinities_masked.device,
            dtype=expert_affinities_masked.dtype,
        )
        down_weight_grad = torch.zeros_like(down_weight)
        gate_up_weight_grad = torch.zeros_like(gate_up_weight)
        # add last row for -1 token index to get 0 grad
        grad_output_padded = torch.concat(
            [
                grad_output,
                torch.zeros(1, H, device=grad_output.device, dtype=grad_output.dtype),
            ],
            dim=0,
        )
        for block_idx in range(num_blocks):
            xm.mark_step()
            # Casting to nn.Parameter so it's treated as an input by the compiler to avoid recompilation.
            block_idx = torch.nn.Parameter(
                torch.tensor(block_idx), requires_grad=False
            ).to(hidden_states.device)
            # Zero index to make taking the entire slice along a dimension more convenient.
            zero_idx = torch.nn.Parameter(torch.tensor(0), requires_grad=False).to(
                hidden_states.device
            )
            # Get token indices and expert id.
            # Original: block_token_indices = block_to_token_indices[block_idx]
            block_token_indices = dynamic_slice_2D(
                tensor=token_position_to_id,
                start0=block_idx,
                start1=zero_idx,
                size0=1,
                size1=token_position_to_id.shape[1],
            ).squeeze(0)
            # Original: block_expert_idx = block_to_expert[block_idx]
            block_expert_idx = dynamic_slice_1D(block_to_expert, block_idx, 1)[0]

            block_grad = grad_output_padded[block_token_indices]

            ## Gradient for inputs to the second dot product (expert affinities and down projection output)
            # original: down_activation = down_activations[block_idx]
            down_activation = dynamic_slice_3D(
                tensor=down_activations,
                start0=block_idx,
                start1=zero_idx,
                start2=zero_idx,
                size0=1,
                size1=down_activations.shape[1],
                size2=down_activations.shape[2],
            ).squeeze(0)
            # Gradient for expert affinities
            # original: affinities_grad[block_token_indices, block_expert_idx] = (block_grad * down_activation).sum(dim=1)
            block_affinities_expert_grad = (block_grad * down_activation).sum(
                dim=1, keepdim=True
            )
            block_affinities_grad = torch.zeros(
                block_grad.shape[0], E, device=block_grad.device
            )
            block_affinities_grad.scatter_(
                dim=1,
                index=block_expert_idx.unsqueeze(0).expand_as(
                    block_affinities_expert_grad
                ),
                src=block_affinities_expert_grad,
            )
            affinities_grad[block_token_indices] += block_affinities_grad
            # Gradient for down projection output
            expert_affinities = expert_affinities_masked[block_token_indices]
            # original: block_expert_affinities = expert_affinities[:, block_expert_idx]
            block_expert_affinities = dynamic_slice_2D(
                tensor=expert_affinities,
                start0=zero_idx,
                start1=block_expert_idx,
                size0=expert_affinities.shape[0],
                size1=1,
            )
            down_out_grad = block_grad * block_expert_affinities

            ## Gradient for inputs to the down projection (first dot activation and down projection weight)
            # Recompute activation
            # original: gate_up_activation = gate_up_activations[block_idx]
            gate_up_activation = dynamic_slice_3D(
                tensor=gate_up_activations,
                start0=block_idx,
                start1=zero_idx,
                start2=zero_idx,
                size0=1,
                size1=gate_up_activations.shape[1],
                size2=gate_up_activations.shape[2],
            ).squeeze(0)
            gate_activation, up_activation = torch.chunk(
                gate_up_activation, chunks=2, dim=-1
            )
            silu_activation = F.silu(gate_activation)
            first_dot_activation = silu_activation * up_activation
            # Gradient for down projection weight
            block_down_weight_grad = first_dot_activation.t() @ down_out_grad
            # original: down_weight_grad[block_expert_idx] += block_down_weight_grad
            down_weight_grad_slice = dynamic_slice_3D(
                tensor=down_weight_grad,
                start0=block_expert_idx,
                start1=zero_idx,
                start2=zero_idx,
                size0=1,
                size1=down_weight_grad.shape[1],
                size2=down_weight_grad.shape[2],
            ).squeeze(0)
            down_weight_grad_slice += block_down_weight_grad
            down_weight_grad.scatter_(
                dim=0,
                index=block_expert_idx.unsqueeze(0)
                .expand_as(down_weight_grad_slice)
                .unsqueeze(0),
                src=down_weight_grad_slice.unsqueeze(0),
            )
            # Gradient for first dot activation and silu activation
            # original: down_weight_slice = down_weight[block_expert_idx]
            down_weight_slice = dynamic_slice_3D(
                tensor=down_weight,
                start0=block_expert_idx,
                start1=zero_idx,
                start2=zero_idx,
                size0=1,
                size1=down_weight.shape[1],
                size2=down_weight.shape[2],
            ).squeeze(0)
            first_dot_grad = down_out_grad @ down_weight_slice.t()
            silu_grad = first_dot_grad * up_activation
            # Gradient for gate output and up output
            gate_output_grad = (
                silu_grad
                * torch.sigmoid(gate_activation)
                * (1 + gate_activation * (1 - torch.sigmoid(gate_activation)))
            )
            up_output_grad = first_dot_grad * silu_activation
            gate_up_out_grad = torch.cat([gate_output_grad, up_output_grad], dim=-1)

            ## Gradient for inputs to the gate up projection.
            # Gradient for gate up projection weight
            # original: gate_up_weight_grad[block_expert_idx] += block_gate_up_grad
            block_hidden_states = hidden_states[block_token_indices]
            block_gate_up_weight_grad = block_hidden_states.t() @ gate_up_out_grad
            gate_up_weight_grad_slice = dynamic_slice_3D(
                tensor=gate_up_weight_grad,
                start0=block_expert_idx,
                start1=zero_idx,
                start2=zero_idx,
                size0=1,
                size1=gate_up_weight_grad.shape[1],
                size2=gate_up_weight_grad.shape[2],
            ).squeeze(0)
            gate_up_weight_grad_slice += block_gate_up_weight_grad
            gate_up_weight_grad.scatter_(
                dim=0,
                index=block_expert_idx.unsqueeze(0)
                .expand_as(gate_up_weight_grad_slice)
                .unsqueeze(0),
                src=gate_up_weight_grad_slice.unsqueeze(0),
            )
            # Gradient for the hidden states
            # original: gate_up_weight_slice = gate_up_weight[block_expert_idx]
            gate_up_weight_slice = dynamic_slice_3D(
                tensor=gate_up_weight,
                start0=block_expert_idx,
                start1=zero_idx,
                start2=zero_idx,
                size0=1,
                size1=gate_up_weight.shape[1],
                size2=gate_up_weight.shape[2],
            ).squeeze(0)
            block_hidden_grad = gate_up_out_grad @ gate_up_weight_slice.t()

            # accumulate because one token can map to multiple experts
            hidden_states_grad[block_token_indices] += block_hidden_grad
            xm.mark_step()
        affinities_grad = affinities_grad[:T]
        hidden_states_grad = hidden_states_grad[:T]

        return (
            hidden_states_grad,
            affinities_grad,
            None,
            None,
            gate_up_weight_grad,
            down_weight_grad,
        )

@dataclass
class KernelConfig:
    logical_nc_config: LogicalNCConfig
    use_block_parallel: bool

class KernelAvailabilityError(Exception):
    pass

def check_kernel_availability(config: KernelConfig) -> None:
    """
    Check if required NKI kernels are available based on configuration.

    Args:
        config: KernelConfig object containing logical_nc_config and use_block_parallel

    Raises:
        KernelAvailabilityError: If the required kernel is not available
    """
    if config.logical_nc_config == LogicalNCConfig.LNC_2:
        if config.use_block_parallel:
            if _blockwise_mm_baseline_block_parallel_nki_call is None:
                raise KernelAvailabilityError("Block parallel NKI kernel not available")
        elif _blockwise_mm_baseline_shard_hidden_nki_call is None:
            raise KernelAvailabilityError("Shard hidden NKI kernel not available")
    elif config.logical_nc_config == LogicalNCConfig.LNC_1:
        if config.use_block_parallel:
            raise KernelAvailabilityError("Block parallel mode not supported with logical_nc_config=1")
        elif _blockwise_mm_nki_call is None:
            raise KernelAvailabilityError("Base NKI kernel not available")
    else:
        raise ValueError(f"Invalid logical_nc_config: {config.logical_nc_config}")

def can_use_blockwise_matmul_nki(
        hidden_size: int,
        intermediate_size_tp: int,
        block_size: int,
        glu_mlp: bool,
        use_torch_block_wise: bool,
        device: torch.device,
        logical_nc_config: int,
        use_block_parallel: bool = False,
) -> bool:
    """
    Determine if blockwise NKI kernel can be used based on configuration.

    Args:
        hidden_size: Size of the hidden layer
        intermediate_size_tp: Intermediate size with tensor parallelism
        block_size: Block size for matrix multiplication
        glu_mlp: Whether GLU MLP is enabled
        use_torch_block_wise: Whether to use torch implementation
        device: Target device
        logical_nc_config: LNC size (1 or 2)
        use_block_parallel: Whether to use block parallel mode

    Returns:
        bool: True if NKI kernel can be used, False otherwise
    """
    pre_validation_conditions = [
        (device.type == "cpu", "Cannot run blockwise NKI kernel on CPU"),
        (not glu_mlp, "Blockwise NKI kernel incompatible with glu_mlp=False"),
        (use_torch_block_wise, "use_torch_block_wise set, using torch implementation"),
    ]

    for condition, warning in pre_validation_conditions:
        if condition:
            warnings.warn(warning)
            return False

    try:
        kernel_config = KernelConfig(
            logical_nc_config=LogicalNCConfig(logical_nc_config),
            use_block_parallel=use_block_parallel
        )
        check_kernel_availability(kernel_config)
    except (KernelAvailabilityError, ValueError) as e:
        warnings.warn(f"Failed to load Blockwise NKI kernel. Error: {str(e)}")
        return False

    try:
        check_blockwise_mm_kernel_compatibility(
            hidden_size=hidden_size,
            block_size=block_size,
            intermediate_size_tp=intermediate_size_tp,
        )
    except AssertionError as e:
        warnings.warn(f"Blockwise kernel not compatible with model config. Reason: {str(e)}")
        return False

    return True


def augment_inputs_for_padded_blockwise_matmul(
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        token_position_to_id: torch.Tensor,
        expert_affinities_masked: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads the inputs to blockwise NKI matmul when skip_tokens or skip_dma is false. When skip_tokens is true,
    the kernel will handle -1s in token_position_to_id, and padding of an extra row of zeros is not needed.

    Args:
        output: Tensor of shape (T, H) where T is number of tokens and H is hidden size
        hidden_states: Tensor of shape (T, H)
        token_position_to_id: Tensor of shape (N*B,) containing token positions
        expert_affinities_masked: Tensor of shape (T, E) where E is number of experts

    Returns:
        Tuple of padded tensors in same order as inputs:
        - output padded to (T+1, H)
        - hidden_states padded to (T+1, H)
        - token_position_to_id with -1s replaced by T
        - expert_affinities_masked padded to (T+1, E)
    """
    total_tokens, hidden_size = hidden_states.shape
    num_experts = expert_affinities_masked.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Create padding tensors
    output_pad = torch.zeros(1, hidden_size, device=device, dtype=dtype)
    hidden_pad = output_pad.clone()  # Same shape/dtype/device
    expert_pad = torch.zeros(1, num_experts, device=device, dtype=dtype)

    # Pad tensors
    padded_output = torch.cat([output, output_pad])
    padded_hidden_states = torch.cat([hidden_states, hidden_pad])
    padded_expert_affinities = torch.cat([expert_affinities_masked, expert_pad])

    # Update token positions
    updated_token_positions = token_position_to_id.masked_fill(
        token_position_to_id == DEFAULT_PADDING_VALUE,
        total_tokens
    )

    return (
        padded_output,
        padded_hidden_states,
        updated_token_positions,
        padded_expert_affinities,
    )


class BlockwiseMatmulNKIFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        expert_affinities_masked,
        gate_up_proj_weight,
        down_proj_weight,
        token_position_to_id,
        block_to_expert,
        gate_up_proj_scale,
        down_proj_scale,
        is_training,
        blockwise_matmul_config:BlockwiseMatmulConfig,
        multi_expert_per_token=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    ):
        """
        Forward pass using the blockwise matmul NKI kernel.

        Common nomenclature:
            T = total tokens
            H = hidden size
            I = intermediate size
            E = number of experts
            N = number of blocks
            B = block size

        Arguments:
            hidden_states: (T, H)
            expert_affinities_masked: (T, E)
            gate_up_proj_weight: (E, H, 2I)
            down_proj_weight: (E, I, H)
            block_size: int
            token_position_to_id: (N*B,)
            block_to_expert: (N,)

        multi_expert_per_token: Indicates if a single token will be computed on multiple experts.
	                            If not (top_k = 1), we pass in an argument to the kernel for optimizations.

        expert_affinities_scaling_mode: Enable No, PRE or POST affinities scaling using this.
                                        Pass in int values or type <ExpertAffinityScaleMode>
                                        0 = ExpertAffinityScaleMode.NO_SCALE
                                        1 = ExpertAffinityScaleMode.POST_SCALE
                                        2 = ExpertAffinityScaleMode.PRE_SCALE


        """
        skip_dma = getattr(blockwise_matmul_config, 'skip_dma')
        block_size = blockwise_matmul_config.block_size
        logical_nc_config = blockwise_matmul_config.logical_nc_config
        use_block_parallel = blockwise_matmul_config.use_block_parallel
        always_augment_inputs_for_blockwise_matmul = blockwise_matmul_config.always_augment_inputs_for_blockwise_matmul
        block_sharding_strategy = blockwise_matmul_config.block_sharding_strategy
        if use_block_parallel:
            assert logical_nc_config == LogicalNCConfig.LNC_2, "use_block_parallel is currently only supported for LNC=2"
            assert not multi_expert_per_token, "use_block_parallel is currently not supported for multi_expert_per_token"

        total_tokens, hidden_size = hidden_states.shape
        ctx.total_tokens = total_tokens
        ctx.num_experts = expert_affinities_masked.shape[1]
        ctx.intermediate_size = down_proj_weight.shape[1]
        if is_training:
            num_block = block_to_expert.shape[0]
            gate_up_activations = torch.empty(num_block, 2, ctx.intermediate_size, block_size,
                                              dtype=gate_up_proj_weight.dtype, device=gate_up_proj_weight.device)
            down_activations = torch.empty(num_block, block_size, hidden_size, dtype=down_proj_weight.dtype,
                                           device=down_proj_weight.device)
        else:
            gate_up_activations = None
            down_activations = None
        # Add extra row to hidden_states and output for padding tokens (i.e. tokens which have -1 index)
        output = torch.zeros(total_tokens, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        if always_augment_inputs_for_blockwise_matmul or not skip_dma.skip_token:
            output, hidden_states, token_position_to_id, expert_affinities_masked = (
                augment_inputs_for_padded_blockwise_matmul(output, hidden_states, token_position_to_id,
                                                                     expert_affinities_masked))
        # Reshape gate_up_proj_weight to (E, H, 2, I) as expected by the kernel
        gate_up_proj_weight = gate_up_proj_weight.view(ctx.num_experts, hidden_size, 2, -1)
        # Flatten expert_affinities_masked: ((T+1)*E, 1)
        # TODO: cannot refactor to (T+1, E) as we currently don't support dynamic slice on both axis.
        expert_affinities_masked = expert_affinities_masked.view(-1, 1)
        
        args = BlockwiseMatmulArgs(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            block_size=block_size,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            output=output,
            gate_up_activations_T=gate_up_activations,
            down_activations=down_activations,
            skip_dma=skip_dma,
            is_tensor_update_accumulating=multi_expert_per_token,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            block_sharding_strategy=block_sharding_strategy,
            dtype=hidden_states.dtype,
        )

        # Select and call the appropriate kernel function
        if is_training:
            if logical_nc_config == 2:
                output, gate_up_activations, down_activations = _call_training_shard_hidden_kernel(args)
            else:
                output, gate_up_activations, down_activations = _call_training_kernel(args)
        elif logical_nc_config == 2:
            if down_proj_scale is not None:
                output = _call_block_parallel_allocated_kernel(args)
            elif use_block_parallel:
                output = _call_block_parallel_kernel(args)
            else:
                output, gate_up_activations, down_activations = _call_shard_hidden_kernel(args)
        else:
            output = _call_default_kernel(args)
        output = output[:total_tokens, :]
        ctx.save_for_backward(
            hidden_states,
            expert_affinities_masked.view(total_tokens + 1 if always_augment_inputs_for_blockwise_matmul
                                                            or not skip_dma.skip_token else total_tokens, ctx.num_experts),
            token_position_to_id,
            block_to_expert,
            gate_up_proj_weight,
            down_proj_weight,
            gate_up_activations,
            down_activations,
        )
        ctx.multi_expert_per_token = multi_expert_per_token
        ctx.block_size = block_size
        ctx.logical_nc_config = logical_nc_config
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert _blockwise_mm_nki_bwd_call is not None
        (
            hidden_states,
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            gate_up_proj_weight,
            down_proj_weight,
            gate_up_activations,
            down_activations,
        ) = ctx.saved_tensors
        _, E = expert_affinities_masked.shape
        T = ctx.total_tokens
        H = hidden_states.shape[-1]

        hidden_states_grad = torch.zeros(T + 1, H, device=hidden_states.device, dtype=hidden_states.dtype)
        affinities_grad = torch.zeros(
            T + 1, E, device=expert_affinities_masked.device, dtype=expert_affinities_masked.dtype
        )
        gate_up_proj_weight_grad = torch.zeros_like(gate_up_proj_weight, dtype=gate_up_proj_weight.dtype)
        down_weight_grad = torch.zeros_like(down_proj_weight, dtype=down_proj_weight.dtype)
        # add last row for -1 token index to get 0 grad
        grad_output_padded = torch.concat(
            [grad_output, torch.zeros(1, H, device=grad_output.device, dtype=grad_output.dtype)], dim=0
        )
        if ctx.logical_nc_config == 2:
            _blockwise_mm_bwd_baseline_shard_hidden_nki_call[VNC(2)](
                hidden_states,
                hidden_states_grad,
                expert_affinities_masked.reshape(-1, 1),
                affinities_grad.view(-1, 1),
                gate_up_proj_weight,
                gate_up_proj_weight_grad,
                gate_up_activations,
                down_proj_weight,
                down_weight_grad,
                down_activations,
                token_position_to_id.to(dtype=torch.int32),
                block_to_expert.to(dtype=torch.int32),
                grad_output_padded,
                block_size=ctx.block_size,
                is_tensor_update_accumulating=ctx.multi_expert_per_token,
            )
        else:
            _blockwise_mm_nki_bwd_call(
                hidden_states,
                hidden_states_grad,
                expert_affinities_masked.reshape(-1, 1),
                affinities_grad.view(-1, 1),
                gate_up_proj_weight,
                gate_up_proj_weight_grad,
                gate_up_activations,
                down_proj_weight,
                down_weight_grad,
                down_activations,
                token_position_to_id.to(dtype=torch.int32),
                block_to_expert.to(dtype=torch.int32),
                grad_output_padded,
                block_size=ctx.block_size,
                is_tensor_update_accumulating=ctx.multi_expert_per_token,
            )
        # FIXME: Compiler error without .clone()
        hidden_states_grad = hidden_states_grad[:T].clone()
        torch.distributed.all_reduce(
            hidden_states_grad,
            group=mappings.get_tensor_model_parallel_group(),
        )
        return (
            hidden_states_grad[:T],
            affinities_grad[:T],
            gate_up_proj_weight_grad.reshape(E, H, 2 * ctx.intermediate_size),
            down_weight_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
