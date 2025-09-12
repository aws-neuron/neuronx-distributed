import torch
from enum import IntEnum
import torch_xla.core.xla_model as xm
from neuronxcc.nki.compiler.backends.neuron.dimensions import VNC
from neuronxcc.nki._pre_prod_kernels.blockwise_mm import blockwise_mm_selective_cp, SkipMode
from neuronxcc.nki._private_kernels.blockwise_mm_bwd import blockwise_mm_bwd_selective_cp
import neuronxcc.nki.language as nl
import neuronxcc.nki as nki
 
device = xm.xla_device()
_blockwise_moe_shard_hidden_nki_call = nki.jit()(blockwise_mm_selective_cp)
_blockwise_moe_bwd_shard_hidden_dropping_nki_call = nki.jit()(blockwise_mm_bwd_selective_cp)


def map_skip_mode(skip_mode: int):
  """
  Creates a Skip Mode tuple defining the Weight Skip, Token Skip for improving performance
  Weight Skip - Avoids reloading expert weights from HBM for blocks that are assigned to the same expert.
  Token Skip - Skip computations on padded tokens in each block

  Args:
    skip_mode: Integer between 0 - 3
  Notes:
      Skip DMA is not applicable for perfectly balanced routers
  """
  if skip_mode == 0:
    return SkipMode(False, False)
  elif skip_mode == 1:
    return SkipMode(True, False)
  elif skip_mode == 2:
    return SkipMode(False, True)
  elif skip_mode == 3:
    return SkipMode(True, True)
  else:
    raise ValueError("Invalid skip_mode")

    
class ExpertAffinityScaleMode(IntEnum):
  """
  Defines the expert affinity scaling mode.
  NO_SCALE - Affinities are not applied.
  POST_SCALE - Affinities are applied after the down projection
  PRE_SCALE - Affinities are applied before the gate and up projection 
  """
  NO_SCALE = 0
  POST_SCALE = 1
  PRE_SCALE = 2


class BlockwiseMoeShardHiddenDropping(torch.autograd.Function):
  """
  Defines Torch Autograd Wrapper for the blockwise moe shard hidden dropping NKI kernel
  """
  @staticmethod
  def forward(ctx, hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight, token_position_to_id, block_to_expert, block_size, dtype, dma_skip_forward, ktype, expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE):
    """
    Forward Kernel Call

    Args:
        hidden_states: Tensor of input hidden states of size (T+1, H). The +1 is because the padding token position is set to index T.
        expert_affinities_masked: Tensor of expert affinities corresponding to each token of size ((T+1) * E, 1). 
        gate_up_proj_weight:  Tensor of concatenated gate and up projection weights (E, H, 2, I_TP)
        down_proj_weight: Tensor of down projection weights (E, I_TP, H)
        token_position_to_id: Tensor of block index of the corresponding tokens (N * B,) Note that we include tokens included for padding purposes and N * B >= T.
        block_to_expert: Tensor of expert indices of corresponding blocks (N, 1)
        block_size: Number of tokens per block
        dtype: Compute Dtype
        dma_skip_forward: Skip Mode for the forward kernel defining the Weight Skip, Token Skip. Weight Token Skip refers to the kernels ability to skip padding token computations.
        ktype: Kernel Type. 0 for the dropping kernel, 1 for non-dropping 
    
    Returns:
      output: Tensor of output hidden states of size (T+1, H).
    
    Notes:
      - All input/output tensors must have the same floating point dtype
      - token_position_to_id and block_to_expert must be np.int32 tensors
      - The following kernel works only for lnc2
      
    """
    assert expert_affinities_scaling_mode in (ExpertAffinityScaleMode.POST_SCALE, ExpertAffinityScaleMode.PRE_SCALE), "Only Pre-Scaling and Post scaling of expert affinities is supported for the forward kernel"

    nki_datatype = nl.float32 if dtype == torch.float32 else nl.bfloat16
    output, gate_up_activations_T, down_activations = _blockwise_moe_shard_hidden_nki_call[VNC(2)](
          hidden_states=hidden_states,
          expert_affinities_masked=expert_affinities_masked,
          gate_up_proj_weight=gate_up_proj_weight,
          down_proj_weight=down_proj_weight,
          block_size=block_size,
          token_position_to_id=token_position_to_id,
          block_to_expert=block_to_expert,
          compute_dtype=nki_datatype,
          expert_affinities_scaling_mode=expert_affinities_scaling_mode,
          skip_dma=dma_skip_forward
      )
    ctx.save_for_backward(
        hidden_states,
        expert_affinities_masked,
        gate_up_proj_weight,
        down_proj_weight,
        token_position_to_id,
        gate_up_activations_T,
        down_activations,
        block_to_expert,
    )
    ctx.block_size=block_size
    ctx.dtype=nki_datatype
    ctx.expert_affinities_scaling_mode = expert_affinities_scaling_mode
    ctx.ktype = ktype
    return output

  @staticmethod
  def backward(ctx, output_hidden_states_grad):
    """
    Calculates the gradients for Hidden States, Expert Affinities, Gate, Up, and Down Projection

    Args:
        output_hidden_states_grad: Gradients of the forward output.
    """
    hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight, token_position_to_id, gate_up_activations_T, down_activations, block_to_expert = ctx.saved_tensors
    assert ctx.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE, "Only Post scaling of expert affinities is supported for the backward kernel"

    hidden_states_grad, expert_affinities_masked_grad, gate_up_proj_weight_grad, down_proj_weight_grad = _blockwise_moe_bwd_shard_hidden_dropping_nki_call[VNC(2)](
                                                    hidden_states=hidden_states,
                                                    expert_affinities_masked=expert_affinities_masked, 
                                                    gate_up_proj_weight=gate_up_proj_weight, 
                                                    gate_up_proj_act_checkpoint_T=gate_up_activations_T,
                                                    down_proj_weight=down_proj_weight, 
                                                    down_proj_act_checkpoint=down_activations,
                                                    token_position_to_id=token_position_to_id, 
                                                    output_hidden_states_grad=output_hidden_states_grad,
                                                    compute_dtype=ctx.dtype,
                                                    block_size=ctx.block_size, 
                                                    ktype=ctx.ktype,
                                                    block_to_expert=block_to_expert
                                                    )

    return (hidden_states_grad, expert_affinities_masked_grad, gate_up_proj_weight_grad, down_proj_weight_grad, None, None, None, None, None, None, None)
  

class BlockwiseMoeShardHiddenDroppingBwd(torch.autograd.Function):
  """
  Defines Torch Autograd Wrapper for the blockwise moe shard hidden dropping Backward NKI kernel.
  This autograd expects the forward output to be passed so that the autograd graph can be built for the .backward() call
  """
  @staticmethod
  def forward(ctx, hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight, token_position_to_id, block_to_expert, block_size, dtype, ktype, reference_output, expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE):
    """
    Forward Kernel Call, only creates the compuation graph, does not call the forward kernel

    Args:
        hidden_states: Tensor of input hidden states of size (T+1, H). The +1 is because the padding token position is set to index T.
        expert_affinities_masked: Tensor of expert affinities corresponding to each token of size ((T+1) * E, 1). 
        gate_up_proj_weight:  Tensor of concatenated gate and up projection weights (E, H, 2, I_TP)
        down_proj_weight: Tensor of down projection weights (E, I_TP, H)
        token_position_to_id: Tensor of block index of the corresponding tokens (N * B,) Note that we include tokens included for padding purposes and N * B >= T.
        block_to_expert: Tensor of expert indices of corresponding blocks (N, 1)
        block_size: Number of tokens per block
        dtype: Compute Dtype
        ktype: Kernel Type. 0 for the dropping kernel, 1 for non-dropping 
    
    Returns:
      output: Tensor of output hidden states of size (T+1, H).
    
    Notes:
      - All input/output tensors must have the same floating point dtype
      - token_position_to_id and block_to_expert must be np.int32 tensors
      - The following kernel works only for lnc2
      
    """

    nki_datatype = nl.float32 if dtype == torch.float32 else nl.bfloat16
    output, gate_up_activations_T, down_activations = reference_output
    ctx.save_for_backward(
        hidden_states,
        expert_affinities_masked,
        gate_up_proj_weight,
        down_proj_weight,
        token_position_to_id,
        gate_up_activations_T,
        down_activations,
        block_to_expert,
    )
    ctx.block_size=block_size
    ctx.dtype=nki_datatype
    ctx.expert_affinities_scaling_mode = expert_affinities_scaling_mode
    ctx.ktype = ktype
    return output

  @staticmethod
  def backward(ctx, output_hidden_states_grad):
    """
    Calculates the gradients for Hidden States, Expert Affinities, Gate, Up, and Down Projection

    Args:
        output_hidden_states_grad: Gradients of the forward output.
    """
    hidden_states, expert_affinities_masked, gate_up_proj_weight, down_proj_weight, token_position_to_id, gate_up_activations_T, down_activations, block_to_expert = ctx.saved_tensors
    assert ctx.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE, "Only Post scaling of expert affinities is supported for the backward kernel"

    hidden_states_grad, expert_affinities_masked_grad, gate_up_proj_weight_grad, down_proj_weight_grad = _blockwise_moe_bwd_shard_hidden_dropping_nki_call[VNC(2)](
                                                    hidden_states=hidden_states,
                                                    expert_affinities_masked=expert_affinities_masked, 
                                                    gate_up_proj_weight=gate_up_proj_weight, 
                                                    gate_up_proj_act_checkpoint_T=gate_up_activations_T,
                                                    down_proj_weight=down_proj_weight, 
                                                    down_proj_act_checkpoint=down_activations,
                                                    token_position_to_id=token_position_to_id, 
                                                    output_hidden_states_grad=output_hidden_states_grad,
                                                    compute_dtype=ctx.dtype,
                                                    block_size=ctx.block_size, 
                                                    ktype=ctx.ktype,
                                                    block_to_expert=block_to_expert
                                                    )

    return (hidden_states_grad, expert_affinities_masked_grad, gate_up_proj_weight_grad, down_proj_weight_grad, None, None, None, None, None, None)