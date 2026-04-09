import math
import warnings
import logging
import torch
import torch_xla.core.xla_model as xm
import torch.nn.functional as F

from neuronx_distributed.modules.moe.model_utils import GLUType, ACTFunc
from neuronx_distributed.modules.moe.model_utils import DEFAULT_PADDING_VALUE, DEFAULT_BLOCK_SIZE, DEFAULT_HIDDEN_ACT_SCALING_FACTOR
from neuronx_distributed.modules.moe.moe_configs import BlockwiseMatmulConfig
from neuronx_distributed.modules.moe.nki_import import NKIImport, import_nki_beta2, import_nki
from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.quantization.quantization_config import is_ocp_mx_quantized, QuantizationType
from neuronx_distributed.quantization.dequantize import blockwise_scale_dequantize
from neuronxcc.nki.compiler.backends.neuron.dimensions import VNC
from neuronx_distributed.utils.model_utils import LogicalNCConfig
from torch_neuronx.xla_impl.base import xla_call

from dataclasses import dataclass
from typing import Tuple, Any, Optional


logger = logging.getLogger("Neuron")

_TORCH_TO_NKI_DTYPE = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.int32: "int32",
}


def torch_to_nki_dtype(dtype: torch.dtype):
    """Convert a torch dtype to the corresponding nki.language dtype."""
    import nki.language as nl

    attr = _TORCH_TO_NKI_DTYPE.get(dtype)
    if attr is None:
        raise ValueError(f"Unsupported torch dtype for NKI conversion: {dtype}")
    return getattr(nl, attr)


def initialize_nki_components() -> dict:
    """
    Initialize all NKI components.

    Returns:
        dict: Mapping of component names to their imported values
    """
    imports = {
        "bwmm_shard_on_block": NKIImport("bwmm_shard_on_block", module_name="moe.moe_cte.bwmm_shard_on_block", is_kernel=True),
        "bwmm_shard_on_block_mx": NKIImport("bwmm_shard_on_block_mx", module_name="moe.moe_cte.bwmm_shard_on_block_mx", is_kernel=True),
        "blockwise_mm_baseline_shard_intermediate": NKIImport("blockwise_mm_baseline_shard_intermediate", module_name="moe.moe_cte.bwmm_shard_on_I", is_kernel=True),
        "blockwise_mm_baseline_shard_intermediate_hybrid": NKIImport("blockwise_mm_baseline_shard_intermediate_hybrid", module_name="moe.moe_cte.bwmm_shard_on_I", is_kernel=True),
        "blockwise_mm_shard_intermediate_dropping": NKIImport("blockwise_mm_shard_intermediate_dropping", module_name="moe.moe_cte.bwmm_shard_on_I", is_kernel=True),
        "moe_cte": NKIImport("moe_cte", module_name="moe.moe_cte.moe_cte", is_kernel=True),
        "block_shard_strategy": NKIImport("BlockShardStrategy", module_name="moe.moe_cte.moe_cte_utils"),
        "skip_mode": NKIImport("SkipMode", module_name="moe.moe_cte.moe_cte_utils"),
        "affinity_scale_mode": NKIImport("ExpertAffinityScaleMode", module_name="utils.common_types"),
        "act_fn_type": NKIImport("ActFnType", module_name="utils.common_types"),
    }

    components = {}
    for name, config in imports.items():
        component, error = import_nki_beta2(config)
        if error:
            warnings.warn(f"Warning: {error}")
        components[name] = component

    return components


# Initialize all components
nki_components = initialize_nki_components()

# Assign to module-level variables (nkilib beta2 kernels)
_bwmm_shard_on_block_nki_call = nki_components["bwmm_shard_on_block"]
_bwmm_shard_on_block_mx_nki_call = nki_components["bwmm_shard_on_block_mx"]
_blockwise_mm_baseline_shard_intermediate_nki_call = nki_components["blockwise_mm_baseline_shard_intermediate"]
_blockwise_mm_baseline_shard_intermediate_hybrid = nki_components["blockwise_mm_baseline_shard_intermediate_hybrid"]
_blockwise_mm_shard_intermediate_dropping_nki_call = nki_components["blockwise_mm_shard_intermediate_dropping"]
_moe_cte_nki_call = nki_components["moe_cte"]
BlockShardStrategy = nki_components["block_shard_strategy"]
SkipMode = nki_components["skip_mode"]
ExpertAffinityScaleMode = nki_components["affinity_scale_mode"]
ActFnType = nki_components["act_fn_type"]
ActivationFunction = nki_components["act_fn_type"]

# Training kernels from neuronxcc (not yet in nkilib)
def initialize_training_kernels() -> dict:
    """Initialize training-related NKI kernels from neuronxcc."""
    imports = {
        "blockwise_mm_training": NKIImport("blockwise_mm_baseline", module_name="blockwise_mm", nki_jit_type="use_nki_jit_decorator"),
        "blockwise_mm_baseline_shard_hidden": NKIImport("blockwise_mm_baseline_shard_hidden", module_name="blockwise_mm", nki_jit_type="use_nki_jit_decorator"),
        "blockwise_mm_bwd": NKIImport("blockwise_mm_bwd", module_name="blockwise_mm_bwd", nki_jit_type="use_nki_jit_decorator"),
        "blockwise_mm_bwd_baseline_shard_hidden": NKIImport("blockwise_mm_bwd_baseline_shard_hidden", module_name="blockwise_mm_bwd", nki_jit_type="use_nki_jit_decorator"),
    }
    components = {}
    for name, config in imports.items():
        component, error = import_nki(config)
        if error:
            warnings.warn(f"Warning: {error}")
        components[name] = component
    return components

_training_components = initialize_training_kernels()
_blockwise_mm_training_nki_call = _training_components["blockwise_mm_training"]
_blockwise_mm_baseline_shard_hidden_nki_call = _training_components["blockwise_mm_baseline_shard_hidden"]
_blockwise_mm_nki_bwd_call = _training_components["blockwise_mm_bwd"]
_blockwise_mm_bwd_baseline_shard_hidden_nki_call = _training_components["blockwise_mm_bwd_baseline_shard_hidden"]

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
    gate_clamp_upper_limit: Optional[float] = None
    gate_clamp_lower_limit: Optional[float] = None
    up_clamp_upper_limit: Optional[float] = None
    up_clamp_lower_limit: Optional[float] = None
    use_shard_on_block_dynamic_while: bool = False
    # Optional Input tensors
    conditions: Optional[torch.Tensor] = None
    num_static_blocks: Optional[int] = None
    gate_up_proj_bias: Optional[torch.tensor] = None
    down_proj_bias: Optional[torch.tensor] = None

    # This will be updated with actual default value ActivationFunction.SiLU once the ToT compiler flow into NxD
    kernel_act_fn: Any = None

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


def _call_bwmm_shard_on_block_kernel(args: BlockwiseMatmulArgs):
    """Call the shard on block kernel for blockwise matmul."""
    nki_grid = LogicalNCConfig.LNC_2
    E = args.gate_up_proj_weight.shape[0]
    # pass args not in the original interface as kwargs to ensure compatibility with different compiler versions
    optional_kwargs = {}
    if args.gate_clamp_upper_limit is not None:
        optional_kwargs["gate_clamp_upper_limit"] = args.gate_clamp_upper_limit
    if args.gate_clamp_lower_limit is not None:
        optional_kwargs["gate_clamp_lower_limit"] = args.gate_clamp_lower_limit
    if args.up_clamp_upper_limit is not None:
        optional_kwargs["up_clamp_upper_limit"] = args.up_clamp_upper_limit
    if args.up_clamp_lower_limit is not None:
        optional_kwargs["up_clamp_lower_limit"] = args.up_clamp_lower_limit

    output = _bwmm_shard_on_block_nki_call[nki_grid](
        hidden_states=get_data(args.hidden_states),
        expert_affinities_masked=get_data(args.expert_affinities_masked),
        gate_up_proj_weight=get_data(args.gate_up_proj_weight),
        down_proj_weight=get_data(args.down_proj_weight),
        block_size=args.block_size,
        token_position_to_id=get_data(args.token_position_to_id, lambda x: x.to(dtype=torch.int32)),
        block_to_expert=get_data(args.block_to_expert, lambda x: x.to(dtype=torch.int32)),
        gate_and_up_proj_bias=get_data(args.gate_up_proj_bias, lambda x: x.view(E, 2, -1)),
        down_proj_bias=get_data(args.down_proj_bias),
        gate_up_proj_scale=get_data(args.gate_up_proj_scale),
        down_proj_scale=get_data(args.down_proj_scale),
        skip_dma=args.skip_dma,
        is_tensor_update_accumulating=args.is_tensor_update_accumulating,
        expert_affinities_scaling_mode=args.expert_affinities_scaling_mode,
        block_sharding_strategy=args.block_sharding_strategy,
        compute_dtype=torch_to_nki_dtype(args.dtype),
        activation_function=args.kernel_act_fn,
        **optional_kwargs,
    )

    if args.is_tensor_update_accumulating:
        # The output from nkilib kernel is of shape (total_tokens + 1, 2, hidden_size), we need
        # to return the first value at second index from that result.
        return output[:, 0, ...]
    else:
        return output


def _call_shard_hidden_kernel(args: BlockwiseMatmulArgs):
    """Call the shard hidden kernel for blockwise matmul."""
    raise NotImplementedError("_call_shard_hidden_kernel is not available - kernel not imported from nkilib")

def get_data(tensor, transform=None):
    """Safely get .data from a tensor that might be None."""
    if tensor is None:
        return None
    data = tensor.data
    return transform(data) if transform else data

def _call_shard_on_intermediate_kernel(args: BlockwiseMatmulArgs):
    """Call the shard-on-intermediate kernel for blockwise matmul."""
    nki_grid = LogicalNCConfig.LNC_2
    conditions = args.conditions
    assert conditions is not None, "conditions must be passed in for shard-on-intermediate dynamic kernel"
    padded_conditions = torch.cat([conditions, torch.zeros(1, device=conditions.device)])
    E = args.gate_up_proj_weight.shape[0]
    # pass args not in the original interface as kwargs to ensure compatibility with different compiler versions
    optional_kwargs = {}
    if args.gate_clamp_upper_limit is not None:
        optional_kwargs["gate_clamp_upper_limit"] = args.gate_clamp_upper_limit
    if args.gate_clamp_lower_limit is not None:
        optional_kwargs["gate_clamp_lower_limit"] = args.gate_clamp_lower_limit
    if args.up_clamp_upper_limit is not None:
        optional_kwargs["up_clamp_upper_limit"] = args.up_clamp_upper_limit
    if args.up_clamp_lower_limit is not None:
        optional_kwargs["up_clamp_lower_limit"] = args.up_clamp_lower_limit
    return _blockwise_mm_baseline_shard_intermediate_hybrid[nki_grid](
        # Inputs
        conditions=get_data(padded_conditions),
        hidden_states=get_data(args.hidden_states),
        expert_affinities_masked=get_data(args.expert_affinities_masked),
        # MLP weights
        gate_up_proj_weight=get_data(args.gate_up_proj_weight),
        down_proj_weight=get_data(args.down_proj_weight),
        gate_and_up_proj_bias=get_data(args.gate_up_proj_bias, lambda x: x.view(E, 2, -1)),
        down_proj_bias=get_data(args.down_proj_bias),
        gate_up_proj_scale=get_data(args.gate_up_proj_scale),
        down_proj_scale=get_data(args.down_proj_scale),
        activation_function=args.kernel_act_fn,
        # Block related
        block_size=args.block_size,
        num_static_block=args.num_static_blocks,
        token_position_to_id=get_data(args.token_position_to_id, lambda x: x.to(dtype=torch.int32)),
        block_to_expert=get_data(args.block_to_expert, lambda x: x.to(dtype=torch.int32)),
        expert_affinities_scaling_mode=args.expert_affinities_scaling_mode,
        compute_dtype=torch_to_nki_dtype(args.dtype),
        # Output
        skip_dma=args.skip_dma,
        is_tensor_update_accumulating=args.is_tensor_update_accumulating,
        **optional_kwargs,
    )

def _call_shard_on_block_kernel(args: BlockwiseMatmulArgs):
    """Call the shard-on-block kernel for blockwise matmul."""
    raise NotImplementedError("_call_shard_on_block_kernel is not available - kernel not imported from nkilib")


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
    # TODO: Update kernel availability checks when more kernels are imported from nkilib
    if config.logical_nc_config == LogicalNCConfig.LNC_2:
        if config.use_block_parallel:
            raise KernelAvailabilityError("Block parallel NKI kernel not available in nkilib")
    elif config.logical_nc_config == LogicalNCConfig.LNC_1:
        if config.use_block_parallel:
            raise KernelAvailabilityError("Block parallel mode not supported with logical_nc_config=1")
        raise KernelAvailabilityError("LNC_1 kernels not available in nkilib")
    else:
        raise ValueError(f"Invalid logical_nc_config: {config.logical_nc_config}")

def check_blockwise_mm_kernel_compatibility(
    hidden_size,
    block_size,
    intermediate_size_tp,
    ):
    PSUM_SIZE = 512
    available_block_sizes = [128, 256, 512, 1024]
    assert block_size in available_block_sizes, f"Only support block_size in {available_block_sizes}, found {block_size}"
    
    assert 512 <= hidden_size <= 8192, f"Hidden dim must be between 512 and 8192, found {hidden_size}"
    assert hidden_size % PSUM_SIZE == 0, f"Hidden dim size must be multiples of {PSUM_SIZE}, found {hidden_size} "

def can_use_blockwise_matmul_nki(
        hidden_size: int,
        intermediate_size_tp: int,
        block_size: int,
        glu_mlp: bool,
        glu_type: GLUType,
        use_torch_block_wise: bool,
        device: torch.device,
        logical_nc_config: int,
        use_block_parallel: bool = False,
        use_shard_on_intermediate: bool = False,
        use_shard_on_block_dynamic_while: bool = False,
        use_bias: bool = False,
        scaling_factor: float = 1.0,
        act_fn: Any = "silu",
        gate_clamp_upper_limit: Optional[float] = None,
        gate_clamp_lower_limit: Optional[float] = None,
        up_clamp_upper_limit: Optional[float] = None,
        up_clamp_lower_limit: Optional[float] = None,
) -> bool:
    """
    Determine if blockwise NKI kernel can be used based on configuration.

    Args:
        hidden_size: Size of the hidden layer
        intermediate_size_tp: Intermediate size with tensor parallelism
        block_size: Block size for matrix multiplication
        glu_mlp: Whether GLU MLP is enabled
        glu_type: Type of GLU to use (GLU or SWIGLU)
        use_torch_block_wise: Whether to use torch implementation
        device: Target device
        logical_nc_config: LNC size (1 or 2)
        use_block_parallel: Whether to use block parallel mode

    Returns:
        bool: True if NKI kernel can be used, False otherwise
    """
    glu_type = GLUType.validate(glu_type)
    act_fn = ACTFunc.validate(act_fn)
    glu_supported = bias_supported = clamp_supported = (use_shard_on_intermediate or use_shard_on_block_dynamic_while)
    pre_validation_conditions = [
        (device.type == "cpu", "Cannot run blockwise NKI kernel on CPU"),
        (not glu_mlp, "Blockwise NKI kernel incompatible with glu_mlp=False"),
        (glu_type is not None and glu_type not in GLUType, "Blockwise NKI kernel only support glu_type=GLU or glu_type=SWIGLU"),
        (use_torch_block_wise, "use_torch_block_wise set, using torch implementation"),
        (glu_type == GLUType.SWIGLU and not glu_supported, "SWIGLU is not yet supported in the selected blockwise matmul NKI kernel"),
        (use_bias and not bias_supported, "gate_up_proj and down_proj bias is not yet supported in the selected blockwise matmul NKI kernel"),
        (scaling_factor != DEFAULT_HIDDEN_ACT_SCALING_FACTOR and glu_type == GLUType.SWIGLU, "scaling factor should be 1.702 for NKI kernel with SWIGLU"),
        (glu_type == GLUType.SWIGLU and act_fn != ACTFunc.SIGMOID, "SWIGLU is only supported with sigmoid activation function"),
        ((gate_clamp_upper_limit or gate_clamp_lower_limit or up_clamp_upper_limit or up_clamp_lower_limit) and not clamp_supported, "clamp limit is not yet supported in the selected blockwise matmul NKI kernel")
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

    # TODO: Re-import from nkilib when check_blockwise_mm_kernel_compatibility is available in nkilib
    try:
        check_blockwise_mm_kernel_compatibility(
            hidden_size=hidden_size,
            block_size=block_size,
            intermediate_size_tp=intermediate_size_tp,
        )
    except Exception as e:
        warnings.warn(f"Blockwise kernel not compatible with model config. Reason: {str(e)}")
        return False

    if intermediate_size_tp % 16 != 0:
        warnings.warn("Blockwise kernel not compatible with model config. Reason: intermediate size tp is not disivible by 16")
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
        conditions: Optional[torch.Tensor] = None,
        num_static_blocks: Optional[int] = None,
        gate_clamp_upper_limit: Optional[float] = None,
        gate_clamp_lower_limit: Optional[float] = None,
        up_clamp_upper_limit: Optional[float] = None,
        up_clamp_lower_limit: Optional[float] = None,
        gate_up_proj_bias: Optional[torch.tensor] = None,
        down_proj_bias: Optional[torch.tensor] = None,
        kernel_act_fn_id: int = 1,
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

        conditions: (N,) for shard-on-intermediate kernel
                    (ceil(N/2),) for shard-on-block kernel
                    Indicate whether blocks are padded or not in forward_blockwise with dynamic control flow
                    Example for shard-on-intermediate kernel: conditions = torch.tensor([1,1,1,1,1,0,0,0,0,0]) means the first 5 blocks are not fully padded block
                    and the last 5 blocks are fully padded block thus will be skipped
                    Example for shard-on-block kernel: conditions = torch.tensor([1,1,1,0,0]) means the first 3 * 2 blocks will be computed (block 0-4 is definitely not
                    fully padded, block 5 can be fully padded or not because shard-on-block kernel will only check block with even id 0,2,4,6...for skipping condition),
                    and the last 2 * 2 blocks will skipped.

        """
        skip_dma = getattr(blockwise_matmul_config, 'skip_dma')
        block_size = blockwise_matmul_config.block_size
        logical_nc_config = blockwise_matmul_config.logical_nc_config
        use_block_parallel = blockwise_matmul_config.use_block_parallel
        use_shard_on_intermediate_dynamic_while = blockwise_matmul_config.use_shard_on_intermediate_dynamic_while
        use_shard_on_block_dynamic_while = blockwise_matmul_config.use_shard_on_block_dynamic_while
        assert not (use_shard_on_intermediate_dynamic_while and use_shard_on_block_dynamic_while), \
            "shard_on_I_dynamic_while and use_shard_on_block_dynamic_while kernel cannot be enabled at the same time"

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
        #TODO: Currently blockwise scales are not supported in blockwise kernel. Deprecate this when supported.
        if gate_up_proj_scale is not None and len(gate_up_proj_scale.shape) > 2:
            assert len(gate_up_proj_scale.shape) == len(down_proj_scale.shape) == 3
            warnings.warn(
                f"""gate_up_proj_scale: {gate_up_proj_scale.shape} down_proj_scale: {down_proj_scale.shape} both are blockwise scales.
                Blockwise scaling is not supported in blockwise kernel for now and will be dequantized before the kernel"""
            )
            # mx swizzle currently only applies to gate_up and down proj weights
            mx_swizzle = is_ocp_mx_quantized(QuantizationType.BLOCKWISE_SYMMETRIC, gate_up_proj_weight.dtype, gate_up_proj_scale.dtype)
            gate_up_proj_weight = blockwise_scale_dequantize(gate_up_proj_weight, gate_up_proj_scale, upcast_dtype=hidden_states.dtype, mx_swizzle=mx_swizzle)
            mx_swizzle = is_ocp_mx_quantized(QuantizationType.BLOCKWISE_SYMMETRIC, down_proj_weight.dtype, down_proj_scale.dtype)
            down_proj_weight = blockwise_scale_dequantize(down_proj_weight, down_proj_scale, upcast_dtype=hidden_states.dtype, mx_swizzle=mx_swizzle)
            gate_up_proj_scale, down_proj_scale = None, None

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
            conditions=conditions,
            num_static_blocks=num_static_blocks,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            gate_up_proj_bias=gate_up_proj_bias,
            down_proj_bias=down_proj_bias,
            kernel_act_fn=ActivationFunction(kernel_act_fn_id),
        )

        # Select and call the appropriate kernel function
        if is_training:
            if logical_nc_config == 2:
                output, gate_up_activations, down_activations = _call_training_shard_hidden_kernel(args)
            else:
                output, gate_up_activations, down_activations = _call_training_kernel(args)
        elif logical_nc_config == 2:
            if use_shard_on_intermediate_dynamic_while:
                output = _call_shard_on_intermediate_kernel(args)
            elif use_shard_on_block_dynamic_while:
                output = _call_bwmm_shard_on_block_kernel(args)
            else:
                output, gate_up_activations, down_activations = _call_shard_hidden_kernel(args)
        else:
            raise NotImplementedError("LNC_1 kernels not available in nkilib")

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
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

def _call_bwmm_fp4_shard_on_block(args: BlockwiseMatmulArgs):
    #directly inject the conditions vector here
    # FIXME: move this code to expert_mlps_v2.py instead of injecting it directly here
    nki_grid = LogicalNCConfig.LNC_2
    # Reshape the token positions into blocks
    padded_conditions = None
    if args.use_shard_on_block_dynamic_while:
      num_blocks = args.block_to_expert.shape[0]
      blocks = args.token_position_to_id.view(num_blocks, args.block_size)
      # Check each block for non padded tokens (any position != -1)
      conditions = torch.any(blocks != -1, dim=1).to(torch.int32)
      padded_conditions = torch.cat([conditions, torch.zeros(2, device=conditions.device)])

    output = _bwmm_shard_on_block_mx_nki_call[nki_grid](
        hidden_states=get_data(args.hidden_states),
        expert_affinities_masked=get_data(args.expert_affinities_masked),
        gate_up_proj_weight=get_data(args.gate_up_proj_weight),
        down_proj_weight=get_data(args.down_proj_weight),
        token_position_to_id=get_data(args.token_position_to_id, lambda x: x.to(dtype=torch.int32)),
        block_to_expert=get_data(args.block_to_expert, lambda x: x.to(dtype=torch.int32)),
        conditions=get_data(padded_conditions),
        gate_and_up_proj_bias=get_data(args.gate_up_proj_bias),
        down_proj_bias=get_data(args.down_proj_bias),
        gate_up_proj_scale=get_data(args.gate_up_proj_scale),
        down_proj_scale=get_data(args.down_proj_scale),
        block_size=args.block_size,
        gate_up_activations_T=None,
        down_activations=None,
        # Meta parameters
        activation_function=args.kernel_act_fn,
        skip_dma=args.skip_dma,
        is_tensor_update_accumulating=args.is_tensor_update_accumulating,
        expert_affinities_scaling_mode=args.expert_affinities_scaling_mode,
        gate_clamp_upper_limit=args.gate_clamp_upper_limit,
        gate_clamp_lower_limit=args.gate_clamp_lower_limit,
        up_clamp_lower_limit=args.up_clamp_lower_limit,
        up_clamp_upper_limit=args.up_clamp_upper_limit,
    )

    if args.is_tensor_update_accumulating:
        # The output from kernel is of shape (2, total_tokens+1, hidden_size), we need
        # to return the first value at first index from that result.
        return output[0, ...]
    else:
        return output


class BlockwiseMatmulMXNKIFunc(torch.autograd.Function):
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
        blockwise_matmul_config: BlockwiseMatmulConfig,
        multi_expert_per_token=True,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        conditions: Optional[torch.Tensor] = None,
        num_static_blocks: Optional[int] = None,
        gate_clamp_upper_limit: Optional[float] = None,
        gate_clamp_lower_limit: Optional[float] = None,
        up_clamp_upper_limit: Optional[float] = None,
        up_clamp_lower_limit: Optional[float] = None,
        gate_up_proj_bias: Optional[torch.tensor] = None,
        down_proj_bias: Optional[torch.tensor] = None,
        kernel_act_fn_id: int = 1,
    ):
        assert not is_training, "Blockwise MX matmul is not supported for training yet"
        total_tokens, hidden_size = hidden_states.shape
        assert hidden_size % 512 == 0, "Hidden size must be divisible by 512"
        num_experts = expert_affinities_masked.shape[1]

        output = torch.zeros(total_tokens, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)

        # Pad T dim to T+1 when token DMA skipping is disabled
        skip_dma = getattr(blockwise_matmul_config, 'skip_dma')
        if not skip_dma.skip_token:
            output, hidden_states, token_position_to_id, expert_affinities_masked = (
                augment_inputs_for_padded_blockwise_matmul(output, hidden_states, token_position_to_id,
                                                                        expert_affinities_masked))

        I_TP = gate_up_proj_weight.shape[-1]
        num_I_TP_blocks = math.ceil(I_TP / 512)
        I_TP_block_size = I_TP // num_I_TP_blocks
        if gate_up_proj_bias is not None:
            gate_up_proj_bias = gate_up_proj_bias.reshape(num_experts, I_TP_block_size // 4, 2, num_I_TP_blocks, 4)

        expert_affinities_masked = expert_affinities_masked.view(-1, 1)

        args = BlockwiseMatmulArgs(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            block_size=blockwise_matmul_config.block_size,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            conditions=conditions,
            num_static_blocks=num_static_blocks,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            use_shard_on_block_dynamic_while=blockwise_matmul_config.use_shard_on_block_dynamic_while,
            gate_up_proj_bias=gate_up_proj_bias,
            down_proj_bias=down_proj_bias,
            kernel_act_fn=ActivationFunction(kernel_act_fn_id),
            is_tensor_update_accumulating=multi_expert_per_token,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            output=output,
            skip_dma=skip_dma,
        )

        output = _call_bwmm_fp4_shard_on_block(args)
        output = output[:total_tokens, :]

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Backward for blockwise MX matmul is not implemented yet.")
    