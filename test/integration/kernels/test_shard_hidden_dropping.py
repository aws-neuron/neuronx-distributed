import torch
from torch.nn import SiLU
import logging
import numpy as np
import hashlib
from blockwise_shard_hidden_dropping import BlockwiseMoeShardHiddenDropping, BlockwiseMoeShardHiddenDroppingBwd, ExpertAffinityScaleMode, map_skip_mode
from test_utils import verify_accuracy

import torch_xla.core.xla_model as xm
from neuronxcc.starfish.support.dtype import bfloat16
from neuronxcc.nki._pre_prod_kernels.blockwise_mm import SkipMode

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

device = xm.xla_device()

dtype_mapping = {
    torch.bfloat16: bfloat16,
    torch.float32: np.float32
}

def get_block_size_dropping(T, TOPK, cf, E, EP):
  # T includes batch size
  return T * TOPK * cf // (E*EP)

def generate_blockwise_fwd_golden(expert_affinities, down_proj_weights,
                                    token_position_to_id, block_to_expert, gate_and_up_proj_weights,
                                    hidden_states, T, H, B, N, E, I_TP, dtype, dma_skip:SkipMode,
                                    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                                    checkpoint_activation=False,
                                    conditions=None,
                                    ):
  """
  Generates the golden forward output for the blockwise Moe kernel using native Pytorch
  Arg:
    expert_affinities_masked: Tensor of expert affinities corresponding to each token of size ((T+1) * E, 1).
    down_proj_weights: Tensor of down projection weights (E, I_TP, H)
    token_position_to_id: Tensor of block index of the corresponding tokens (N * B,)
    block_to_expert: Tensor of expert indices of corresponding blocks (N, 1)
    gate_and_up_proj_weights:  Tensor of concatenated gate and up projection weights (E, H, 2, I_TP)
    hidden_states: Tensor of input hidden states of size (T+1, H). The +1 is because the padding token position is set to index T.
    T: Number of tokens, this includes the batch dimension colapsed to avoid padding
    H: Hidden Dimension Size
    B: Number of tokens per block
    N: Number of blocks
    E: Number of Experts
    I_TP: Intermediate size per TP group
    dtype: Compute Dtype
    dma_skip: DMA Skip Mode
    expert_affinities_scaling_mode: Scaling mode for expert afinitites
    checkpoint_activation: Whether to checkpoint the Gate, Up and down projection activations for the backward
    conditions: Flag for whether the block needs to be computed or skipped

  Returns:
    out_return: Output hidden states
    gate_up_activations_T: Gate and Up activations if checkpoint_activation = True
    down_activations: Down projection activations if checkpoint_activation = True
  """
  silu = SiLU()
  output_shape = [T+1, H]
  output = torch.zeros(output_shape).to(dtype)

  token_position_to_id = token_position_to_id.reshape(N, B)

  if checkpoint_activation:
      gate_up_activations_T = torch.zeros([N, 2, I_TP, B]).to(dtype)
      down_activations = torch.zeros([N, B, H]).to(dtype)

  if dma_skip.skip_weight:
    is_weight_same_as_prev = torch.zeros((N))
    is_weight_same_as_prev[1:] = block_to_expert[1:] == block_to_expert[:-1]
    is_weight_same_as_prev = is_weight_same_as_prev.to(torch.uint8)

  gate_up_weights = None
  down_weights = None

  for b in range(N):
    if conditions is not None and conditions[b] == 0:
      break

    local_token_position_to_id = token_position_to_id[b, :]
    # [Block Size, Hidden Size]
    if dma_skip.skip_token:
      zeros_hidden = torch.zeros((1, H)).to(dtype)  
      hidden_states = torch.cat([hidden_states, zeros_hidden], dim=0)
      zeros_exaf = torch.zeros((1, E)).to(dtype) 
      expert_affinities = torch.cat([expert_affinities, zeros_exaf], dim=0)

    local_hidden_states = hidden_states[local_token_position_to_id[:], :].to(dtype)
    expert_idx = block_to_expert[b]
    local_expert_affinities = expert_affinities[local_token_position_to_id, expert_idx].reshape(-1, 1).to(dtype)
    
    if expert_affinities_scaling_mode in [ExpertAffinityScaleMode.PRE_SCALE]:
      local_hidden_states = local_expert_affinities * local_hidden_states

    if dma_skip.skip_weight:
      expert_idx = E if is_weight_same_as_prev[b] else expert_idx

    # [H, 2, I]
    if expert_idx < E: # weight skip
      gate_up_weights = gate_and_up_proj_weights[expert_idx, :, :, :].reshape(H, 2*I_TP).to(dtype)

    # [B, 2, I]
    gate_up_activation = torch.matmul(local_hidden_states, gate_up_weights).reshape(B, 2, I_TP)
    if checkpoint_activation:
      gate_up_activations_T[b] = gate_up_activation.transpose(1, 2).transpose(0, 2)

    act_silu = silu(gate_up_activation[:, 0, :])
    # [B, I]
    multiply_1 = act_silu * gate_up_activation[:, 1, :]
    # [H, I]
    if expert_idx < E: # weight skip
      down_weights = down_proj_weights[expert_idx, :, :].to(dtype)

    down_activation = torch.matmul(multiply_1, down_weights)
    if checkpoint_activation:
      down_activations[b] = down_activation 
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
      scale = down_activation * local_expert_affinities
    else:
      scale = down_activation


    output[local_token_position_to_id[:], :] += scale.to(output.dtype)
    out_return = output[:T, :] if dma_skip.skip_token else output

  if checkpoint_activation:
    return out_return, gate_up_activations_T, down_activations
  return out_return

def generate_token_position_to_id_and_experts_dropping(T, BS, TOPK, E, B, N, rtype, cf, EP_DEG):
  """
  Generate token_experts and token_position_to_id
  Args:
    T: Number of tokens, this includes the batch dimension colapsed to avoid padding
    BS: Batch Size
    TOPK: Top K
    E: Number of Experts
    B: Number of tokens per block
    N: Number of blocks
    rtype: Flag for perfectly balanced routing
    cf: Capacity Factor
    EP_DEG: Expert Parallelism

  Returns:
    token_experts: Mask for expert selection
    token_position_to_id: Tensor of block index of the corresponding tokens (N * B,)
    block_to_expert: Tensor of expert indices of corresponding blocks (N, 1)
  """
  # Sequence length per batch
  S = T//BS 
  # E is E_local, the number of experts per EP rank
  token_experts = torch.zeros((T, E)) 
  # Routing type: 1=Perfect balancing
  if rtype==1:
    # For each local expert, sample tokens from each sequence
    for batch_idx in range(BS):
        seq_start = batch_idx*S
        for expert_idx in range(E):
            stride = (E * EP_DEG) // TOPK # first EP_DEG/TOPK will be sent 1, next 
            start_pos = expert_idx % stride # assumes each seq in a batch processes same set of tokens
            
            token_indices = torch.arange(start_pos, S, stride)
            global_token_indices = seq_start + token_indices
            token_experts[global_token_indices, expert_idx] = 1
  else:
    router = torch.zeros((T, TOPK))
    torch.manual_seed(0) # set random seed is required for while loop test stability
    for i in range(T):
        # generate random TOPK
        router[i] = torch.multinomial(torch.arange(E*EP_DEG, dtype=torch.float), (TOPK), replacement=False)

    one_hot = torch.arange(E)
    token_experts = torch.zeros((T, E))

    for i in range(TOPK):
      router_expanded = router[:, i].unsqueeze(1)  # Shape: (T, 1)
      one_hot_expanded = one_hot.unsqueeze(0)
      token_experts += router_expanded == one_hot_expanded
      
    expert_cap = int(torch.ceil(torch.tensor([T/E]) * cf))
    # Create mask for tokens that exceed capacity
    expert_counts = torch.zeros(E)
    dropped_tokens = torch.zeros_like(token_experts, dtype=torch.bool)
    
    # For each token, check if adding it would exceed capacity
    for t in range(T):
        for e in range(E):
            if token_experts[t, e]:
                if expert_counts[e] >= expert_cap:
                    dropped_tokens[t, e] = True
                else:
                    expert_counts[e] += 1
    
    # Remove dropped tokens using boolean indexing
    token_experts = torch.logical_and(token_experts, ~dropped_tokens)
    token_experts = token_experts.to(torch.float64)

  block_to_expert = torch.arange(E).to(torch.int32)
  
  token_position_to_id = torch.full((int(N * B),), T, dtype=torch.int32)
  
  for e in range(E):
    # Find all tokens assigned to this expert
    token_mask = token_experts[:, e] > 0
    if not torch.any(token_mask):
        continue  
    # Get the token indices for this expert
    expert_tokens = torch.where(token_mask)[0]
    # Calculate the starting position for this expert in the output array
    expert_start_idx = e * B
    num_tokens = min(len(expert_tokens), B)
    # Fill the positions in the output array
    token_position_to_id[expert_start_idx:expert_start_idx + num_tokens] = expert_tokens[:num_tokens]

  return token_experts, token_position_to_id, block_to_expert


def generate_blockwise_bwd_golden(grad_output,
                                        hidden_states,
                                        expert_affinities_masked,
                                        block_to_token_indices,
                                        block_to_expert,
                                        gate_up_weight,
                                        down_weight,
                                        gate_up_activations_T,
                                        down_activations,
                                        N,
                                        B,
                                        dma_skip,
                                        dtype):
  """
  Generates the backward gradients for the blockwise Moe kernel using native Pytorch
  Arg:
    grad_output: Gradient of the Block MOE output
    hidden_states: Tensor of input hidden states of size (T+1, H). The +1 is because the padding token position is set to index T.
    expert_affinities_masked: Tensor of expert affinities corresponding to each token of size ((T+1) * E, 1).
    block_to_token_indices: Tensor of block index of the corresponding tokens (N * B,)
    block_to_expert: Tensor of expert indices of corresponding blocks (N, 1)
    gate_up_weight:  Tensor of concatenated gate and up projection weights (E, H, 2, I_TP)
    down_weight: Tensor of down projection weights (E, I_TP, H)
    gate_up_activations_T: Checkpointed activations after the Gate and Up projections
    down_activations: Checkpointed activations after the down projection
    N: Number of blocks
    B: Number of tokens per block
    dma_skip: DMA Skip Mode
    dtype: Compute Dtype

  Returns:
    hidden_states_grad: Gradients of the hidden states
    affinities_grad: Gradients of the expert affinities
    gate_up_weight_grad: Gradients of the gate and up projection
    down_weight_grad: Gradients of the down projection
  """
  silu = SiLU()
  E, I_TP, H = down_weight.shape
  T, H = hidden_states.shape
    
  if dma_skip.skip_token:
    zeros_hidden = torch.zeros((1, H)).to(hidden_states.dtype)  
    hidden_states = torch.concatenate([hidden_states, zeros_hidden], dim=0)
    zeros_exaf = torch.zeros((1, E)).to(expert_affinities_masked.dtype) 
    expert_affinities_masked = torch.concatenate([expert_affinities_masked, zeros_exaf], dim=0)
    zeros_gradout = torch.zeros((1,H)).to(grad_output.dtype)
    grad_output = torch.concatenate([grad_output, zeros_gradout], dim=0)
  
  hidden_states_grad = torch.zeros_like(hidden_states)
  affinities_grad = torch.zeros_like(expert_affinities_masked)
  down_weight_grad = torch.zeros_like(down_weight)
  gate_up_weight_grad = torch.zeros_like(gate_up_weight)
  block_to_token_indices = block_to_token_indices.reshape(N, B)
  
  for block_idx in range(N):
    token_position_to_id = block_to_token_indices[block_idx]
    block_expert_idx = block_to_expert[block_idx]
    block_hidden_states = hidden_states[token_position_to_id]
    block_grad = grad_output[token_position_to_id]
    ## second dot
    down_activation = down_activations[block_idx]
    # assign because of unique mapping
    # upcast to fp32 to match nki kernel
    mul = block_grad.to(torch.float32) * down_activation.to(torch.float32)
    affinities_grad[token_position_to_id, block_expert_idx] = torch.sum(mul, dim=1, dtype=dtype)
    down_out_grad = block_grad * expert_affinities_masked[token_position_to_id, block_expert_idx].unsqueeze(-1)

    ## down proj
    gate_up_activation_T = gate_up_activations_T[block_idx]
    gate_activation_T, up_activation_T = torch.split(gate_up_activation_T, 1, dim=0)
    gate_activation, up_activation = gate_activation_T.squeeze(0).T, up_activation_T.squeeze(0).T
    silu_activation = silu(gate_activation)
    first_dot_activation = silu_activation * up_activation

    block_down_weight_grad = first_dot_activation.T @ down_out_grad
    down_weight_grad[block_expert_idx] += block_down_weight_grad

    first_dot_grad = down_out_grad @ down_weight[block_expert_idx].T
    silu_grad = first_dot_grad * up_activation

    gate_output_grad = silu_grad * torch.special.expit(gate_activation) * (1 + gate_activation * (1 - torch.special.expit(gate_activation)))
    up_output_grad = first_dot_grad * silu_activation

    gate_up_out_grad = torch.concatenate([gate_output_grad, up_output_grad], dim=-1)
    ## gate up proj
    block_gate_up_grad = block_hidden_states.T @ gate_up_out_grad
    gate_up_weight_grad[block_expert_idx] += block_gate_up_grad.reshape(H, 2, I_TP)

    block_hidden_grad = gate_up_out_grad @ gate_up_weight[block_expert_idx].reshape(H, 2*I_TP).T
    # accumulate because one token can map to multiple experts
    hidden_states_grad[token_position_to_id] += block_hidden_grad

  if dma_skip.skip_token:
    return hidden_states_grad[:T,:], affinities_grad[:T,:], gate_up_weight_grad, down_weight_grad
  else:
    return hidden_states_grad, affinities_grad, gate_up_weight_grad, down_weight_grad


def generate_inputs_and_goldens_for_bwd_dropping(T, TOPK, B, E, I_TP, H, BS, dtype, rtype, EP, cf=1, dma_skip=0):
  """
  Generates the Random inputs and reference golden forward and backward output for the given inputs
  arg:
    T:	Number of tokens, this includes the batch dimension colapsed to avoid padding
    TOPK: TopK
    B:	Block Size
    E:	Number of Experts
    I_TP:	Intermediate size per TP group
    H:	Hidden Dimension Size
    BS: Batch Size
    dtype: Compute Dtype
    rtype: Flag to choose between dropping and perfectly balanced routing
    EP: Expert Parallelism
    cf: Capacity Factor
  """
  # Each parameter combination has a unique seed to ensure independence
  param_string = f"{T}_{TOPK}_{B}_{E}_{I_TP}_{H}_{dma_skip}"
  seed = int(hashlib.sha256(param_string.encode()).hexdigest(), 16) % (2**32)
  torch.manual_seed(seed)

  dma_skip = map_skip_mode(dma_skip)
  N = E
  expert_masks, token_position_to_id, block_to_expert = generate_token_position_to_id_and_experts_dropping(T, BS, TOPK, E, B, N, rtype, cf, EP)

  down_proj_weights = torch.zeros(size=(E, I_TP, H), dtype=dtype).uniform_(-1.0, 1.0)
  gate_and_up_proj_weights = torch.zeros(size=[E, H, 2, I_TP]).uniform_(-0.1, 0.1).to(dtype)

  if dma_skip.skip_token:
    expert_affinities_masked = torch.rand([T, E]).to(dtype)
    expert_affinities_masked = (expert_affinities_masked * expert_masks).to(dtype)
    hidden_states = torch.rand([T, H]).to(dtype)
    grad_output = torch.zeros(size=[T, H]).to(dtype).uniform_(-1.0, 1.0)
  else:
    expert_affinities_masked = torch.rand([T+1, E]).to(dtype)
    expert_affinities_masked[:T] = expert_affinities_masked[:T] * expert_masks
    # zero the padded expert to avoid accuracy issues
    expert_affinities_masked[T] = 0
    hidden_states = torch.rand([T+1, H]).to(dtype) 
    grad_output = torch.zeros(size=[T+1, H]).to(dtype).uniform_(-1.0, 1.0)

    expert_masks = torch.vstack([expert_masks, torch.zeros([1, E])])
    hidden_states[T, ...] = 0
    # add last row for -1 token index to get 0 grad
    grad_output[T, ...] = 0

  hidden_states_output, gate_up_proj_act_checkpoint_T, down_proj_act_checkpoint = generate_blockwise_fwd_golden(expert_affinities_masked, down_proj_weights,
                                              token_position_to_id, block_to_expert, gate_and_up_proj_weights,
                                              hidden_states, T, H, B, N, E, I_TP, dtype, dma_skip, checkpoint_activation=True)

  inp_hw = {}
  inp_hw['expert_affinities'] = expert_affinities_masked.reshape(-1, 1) # TODO: need to make this shape as cannot dynamic index both T and E
  inp_hw['down_proj_weights'] = down_proj_weights
  inp_hw['token_position_to_id'] = token_position_to_id

  inp_hw['block_to_expert'] = block_to_expert

  inp_hw['gate_and_up_proj_weights'] = gate_and_up_proj_weights
  inp_hw['hidden_states'] = hidden_states
  inp_hw['gate_up_proj_act_checkpoint_T'] = gate_up_proj_act_checkpoint_T
  inp_hw['down_proj_act_checkpoint'] = down_proj_act_checkpoint
  inp_hw['grad_output'] = grad_output
  inp_hw['hidden_states_output'] = hidden_states_output

  hidden_states_grad, affinities_grad, gate_up_weight_grad, down_weight_grad = \
    generate_blockwise_bwd_golden(grad_output,
                                        hidden_states,
                                        expert_affinities_masked,
                                        token_position_to_id,
                                        block_to_expert,
                                        gate_and_up_proj_weights,
                                        down_proj_weights,
                                        gate_up_proj_act_checkpoint_T,
                                        down_proj_act_checkpoint,
                                        N, B, dma_skip, dtype)
  
  
  gold_hw = {}
  gold_hw['hidden_states_grad'] = hidden_states_grad
  gold_hw['expert_affinities_masked_grad'] = affinities_grad.reshape(-1, 1)
  gold_hw['gate_up_proj_weight_grad'] = gate_up_weight_grad
  gold_hw['down_proj_weight_grad'] = down_weight_grad

  return inp_hw, gold_hw

def moe_blockwise_fwd(H, T, E, TOPK, I_TP, BS, dtype, dma_skip_forward, rtype, cf, EP, ktype):
    """
    Performs accuracy check between the block wise moe shard hidden dropping kernel and a golden pytorch implementation for both forward and backward
    - Generates random inputs
    - Generates forward outputs using a reference forward implementation
    - Generetes the forward output using the NKI forward kernel
    - Compares the above two forward output for accuracy
    arg:
      H:	Hidden Dimension Size
      T:	Number of tokens, this includes the batch dimension colapsed to avoid padding
      E:	Number of Experts
      TOPK: TopK
      I_TP:	Intermediate size per TP group
      BS: Batch Size
      dtype: Compute Dtype
      dma_skip_forward: Skip Mode for the forward kernel defining the Weight Skip, Token Skip. Weight Token Skip refers to the kernels ability to skip padding token computations.
      rtype: Flag to choose between dropping and perfectly balanced routing
      cf: Capacity Factor
      EP: Expert Parallelism
      ktype: Kernel Type. 0 for the dropping kernel, 1 for non dropping 
    """
    logger.info("Generating Inputs and Goldens for Training")
    kpi_payload = dict()
    B = get_block_size_dropping(T, TOPK, cf, E, EP)
    np_dtype = dtype_mapping[dtype]

    data, _ = generate_inputs_and_goldens_for_bwd_dropping(T=T,TOPK=TOPK,B=B,E=E,I_TP=I_TP,H=H,dma_skip=dma_skip_forward,dtype=dtype,BS=BS,EP=EP,rtype=rtype)
    
    for key in data.keys():
        data[key] = data[key].to(device)
        if key in ['expert_affinities', 'gate_and_up_proj_weights', 'down_proj_weights', 'hidden_states']:
          data[key].requires_grad_()
        
    data['block_size'] = B
    data['dtype'] = dtype


    forward_tensor_mappings = {('hidden_states_output', 'HiddenStatesOutput'): data['hidden_states_output']}
    try:
      nki_function = BlockwiseMoeShardHiddenDropping.apply
      output_autograd = nki_function(data['hidden_states'], data['expert_affinities'], data['gate_and_up_proj_weights'], data['down_proj_weights'], data['token_position_to_id'], data['block_to_expert'], data['block_size'], dtype, map_skip_mode(dma_skip_forward), ktype)
      
      for kernel_output_name, golden in forward_tensor_mappings.items():
        largest_abs_diff, max_abs_diff_element_rel_diff = verify_accuracy(output_autograd, golden, np_dtype, kernel_output_name[0], "forward")
        kpi_payload["Forward" + kernel_output_name[1] + "LargestAbsDiff"] = largest_abs_diff
        kpi_payload["Forward" + kernel_output_name[1] + "MaxElementRelDiff"] = max_abs_diff_element_rel_diff
    except Exception as e:
       raise e

    return kpi_payload

def moe_blockwise_bwd(H, T, E, TOPK, I_TP, BS, dtype, dma_skip, rtype, cf, EP, ktype):
    """
    Performs accuracy check between the block wise moe shard hidden dropping kernel and a golden pytorch implementation for both forward and backward
    - Generates random inputs
    - Generates forward outputs using a reference forward implementation
    - Generates the gradients using a reference backward implementation
    - Generates the gradients using the NKI backward kernel
    - Compares the generated gradients for accuracy
    arg:
      H:	Hidden Dimension Size
      T:	Number of tokens, this includes the batch dimension colapsed to avoid padding
      E:	Number of Experts
      TOPK: TopK
      I_TP:	Intermediate size per TP group
      BS: Batch Size
      dtype: Compute Dtype
      dma_skip: Skip Mode for the defining the Weight Skip, Token Skip. Weight Token Skip refers to the kernels ability to skip padding token computations.
      rtype: Flag to choose between dropping and perfectly balanced routing
      cf: Capacity Factor
      EP: Expert Parallelism
      ktype: Kernel Type. 0 for the dropping kernel, 1 for non dropping 
    """
    logger.info("Generating Inputs and Goldens for Training")
    kpi_payload = dict()
    B = get_block_size_dropping(T, TOPK, cf, E, EP)
    np_dtype = dtype_mapping[dtype]

    data, golden_output = generate_inputs_and_goldens_for_bwd_dropping(T=T,TOPK=TOPK,B=B,E=E,I_TP=I_TP,H=H,dma_skip=dma_skip,dtype=dtype,BS=BS,EP=EP,rtype=rtype)
    
    for key in data.keys():
        data[key] = data[key].to(device)
        if key in ['expert_affinities', 'gate_and_up_proj_weights', 'down_proj_weights', 'hidden_states', 'hidden_states_output']:
          data[key].requires_grad_()
        
    data['block_size'] = B
    data['dtype'] = dtype

    reference_output = [data['hidden_states_output'], data['gate_up_proj_act_checkpoint_T'], data['down_proj_act_checkpoint']]

    nki_function = BlockwiseMoeShardHiddenDroppingBwd.apply
    output_autograd = nki_function(data['hidden_states'], data['expert_affinities'], data['gate_and_up_proj_weights'], data['down_proj_weights'], data['token_position_to_id'], data['block_to_expert'], data['block_size'], dtype, ktype, reference_output)
      
    # Backward
    grad_output = data["grad_output"].to(device)
    hidden_states_grad_np, affinities_grad_np, gate_up_weight_grad_np, down_weight_grad_np = golden_output["hidden_states_grad"], golden_output["expert_affinities_masked_grad"], golden_output["gate_up_proj_weight_grad"], golden_output["down_proj_weight_grad"] 
    
    backward_tensor_mappings = {('hidden_states', 'HiddenStatesGrad'):hidden_states_grad_np, ('expert_affinities', 'ExpertAffinitiesGrad'): affinities_grad_np, ('gate_and_up_proj_weights', 'GateAndUpProjWeightsGrad'): gate_up_weight_grad_np, ('down_proj_weights', 'DownProjWeightsGrad'): down_weight_grad_np}
    try:
      output_autograd.backward(gradient=grad_output)
    except Exception as e:
      raise e

    for kernel_output_name, golden in backward_tensor_mappings.items():
        try:
          largest_abs_diff, max_abs_diff_element_rel_diff = verify_accuracy(data[kernel_output_name[0]].grad, golden, np_dtype, kernel_output_name[0], "backward")
          kpi_payload["Backward"+kernel_output_name[1]+"LargestAbsDiff"] = largest_abs_diff
          kpi_payload["Backward"+kernel_output_name[1]+"MaxAbsDiffElementRelDiff"] = max_abs_diff_element_rel_diff
        except Exception as e:
          if len(e.args) > 1: # All close passes true if accuracy mismatch
            raise e
          
          logger.info(f"{H}, {T}, {E}, {TOPK}, {I_TP}, {BS}, {dtype}, {dma_skip}, {rtype}, {cf}, {EP}, {ktype} Compiler error for {kernel_output_name[1]} skipping all close")
          kpi_payload["Backward"+kernel_output_name[1]+"LargestAbsDiff"] = -1
          kpi_payload["Backward"+kernel_output_name[1]+"MaxAbsDiffElementRelDiff"] = -1
    
  
    return kpi_payload
      