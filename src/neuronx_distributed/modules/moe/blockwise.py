import torch
import torch_xla.core.xla_model as xm
import torch.nn.functional as F
from neuronx_distributed.parallel_layers import mappings

try:
    from neuronxcc.nki._private_kernels.blockwise_mm import (
        blockwise_mm as blockwise_mm_nki,
        blockwise_mm_baseline as blockwise_mm_training_nki,
        check_blockwise_mm_kernel_compatibility,
    )
    from neuronxcc.nki._private_kernels.blockwise_mm_bwd import blockwise_mm_bwd as blockwise_mm_bwd_nki
    from torch_neuronx.xla_impl.ops import nki_jit
    from torch_neuronx.xla_impl.base import xla_call

    _blockwise_mm_nki_call = nki_jit()(blockwise_mm_nki)
    _blockwise_mm_training_nki_call = nki_jit()(blockwise_mm_training_nki)
    _blockwise_mm_nki_bwd_call = nki_jit()(blockwise_mm_bwd_nki)
except ImportError as e:
    import_error_msg = str(e)
    _blockwise_mm_nki_call = None
    _blockwise_mm_nki_bwd_call = None



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


def can_use_blockwise_matmul_nki(
    hidden_size,
    intermediate_size_tp,
    block_size,
    glu_mlp,
    device,
):
    if device.type == "cpu":
        print("Cannot run blockwise NKI kernel on cpu")
        return False

    if not glu_mlp:
        print("Blockwise NKI kernel incompatible with glu_mlp=False")
        return False

    if _blockwise_mm_nki_call is None:
        print(f"Failed to load Blockwise NKI kernel. Error: {str(import_error_msg)}")
        return False

    try:
        check_blockwise_mm_kernel_compatibility(
            hidden_size=hidden_size,
            block_size=block_size,
            intermediate_size_tp=intermediate_size_tp,
        )
    except AssertionError as e:
        print(f"Blockwise kernel not compatible with model config. Reason: {str(e)}")
        return False

    return True

class BlockwiseMatmulNKIFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states,
        expert_affinities_masked,
        gate_up_proj_weight,
        down_proj_weight,
        block_size,
        token_position_to_id,
        block_to_expert,
        gate_up_proj_scale,
        down_proj_scale,
        is_training,
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
        """
        assert _blockwise_mm_nki_call is not None

        total_tokens, hidden_size = hidden_states.shape
        ctx.total_tokens = total_tokens
        ctx.num_experts = expert_affinities_masked.shape[1]
        ctx.intermediate_size = down_proj_weight.shape[1]
        if is_training:
            num_block = block_to_expert.shape[0]
            gate_up_activations = torch.empty(num_block, 2, ctx.intermediate_size, block_size, dtype=gate_up_proj_weight.dtype, device=gate_up_proj_weight.device)
            down_activations = torch.empty(num_block, block_size, hidden_size, dtype=down_proj_weight.dtype, device=down_proj_weight.device)
        else:
            gate_up_activations = None
            down_activations = None

        # Add extra row to hidden_states and output for padding tokens (i.e. tokens which have -1 index)
        # TODO: Change this to (T, H) once the compiler issue with NaNs in skipped DMAs is fixed
        output = torch.zeros(total_tokens + 1, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        # hidden_states: (T, H) -> (T+1, H)
        hidden_states = torch.cat([
            hidden_states,
            torch.zeros(1, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        ])
        # expert_affinities_masked: (T+1, E)
        expert_affinities_masked = torch.cat([
            expert_affinities_masked,
            torch.zeros(1, ctx.num_experts, device=expert_affinities_masked.device, dtype=expert_affinities_masked.dtype)
        ])

        # TODO: Disable skipping DMAs until compiler issue is fixed
        token_position_to_id = token_position_to_id.masked_fill(torch.eq(token_position_to_id, -1), total_tokens)

        # Reshape gate_up_proj_weight to (E, H, 2, I) as expected by the kernel
        gate_up_proj_weight = gate_up_proj_weight.view(ctx.num_experts, hidden_size, 2, -1)
        # Flatten expert_affinities_masked: ((T+1)*E, 1)
        # TODO: cannot refactor to (T+1, E) as we currently don't support dynamic slice on both axis.
        expert_affinities_masked = expert_affinities_masked.view(-1, 1)

        if gate_up_proj_scale is not None:
            # (1, 1, 2I) -> (2, I)
            assert gate_up_proj_scale.shape[0] == 1 and gate_up_proj_scale.shape[1] == 1
            gate_up_proj_scale = gate_up_proj_scale.view(2, -1)
 
        if down_proj_scale is not None:
            # (1, 1, H) -> (H)
            assert down_proj_scale.shape[0] == 1 and down_proj_scale.shape[1] == 1
            down_proj_scale = down_proj_scale.view(-1)

        if is_training:
            _blockwise_mm_training_nki_call(
                # Inputs
                hidden_states=hidden_states,
                expert_affinities_masked=expert_affinities_masked,
                # MLP weights
                gate_up_proj_weight=gate_up_proj_weight,
                down_proj_weight=down_proj_weight,
                # Block related
                block_size=block_size,
                token_position_to_id=token_position_to_id.to(dtype=torch.int32),
                block_to_expert=block_to_expert.to(dtype=torch.int32),
                # Output
                output=output,
                gate_up_activations_T=gate_up_activations,
                down_activations=down_activations,
            )
        else:
            _blockwise_mm_nki_call(
                # Inputs
                hidden_states=hidden_states,
                expert_affinities_masked=expert_affinities_masked,
                # MLP weights
                gate_up_proj_weight=gate_up_proj_weight,
                down_proj_weight=down_proj_weight,
                # Block related
                block_size=block_size,
                token_position_to_id=token_position_to_id.to(dtype=torch.int32),
                block_to_expert=block_to_expert.to(dtype=torch.int32),
                # Output
                output=output,
                skip_dma=False,
                gate_up_proj_scale=gate_up_proj_scale,
                down_proj_scale=down_proj_scale,
            )

        # Drop the last row
        output = output[:total_tokens, :]
        ctx.save_for_backward(
            hidden_states,
            expert_affinities_masked.view(total_tokens + 1, ctx.num_experts),
            token_position_to_id,
            block_to_expert,
            gate_up_proj_weight,
            down_proj_weight,
            gate_up_activations,
            down_activations,
        )

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
            gate_up_proj_weight_grad.reshape(E, H, 2*ctx.intermediate_size),
            down_weight_grad,
            None,
            None,
            None,
            None,
            None,
            None,
        )

