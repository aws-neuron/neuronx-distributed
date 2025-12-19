"""
Model Execution Module
---------------------
This module contains functions for executing models on CPU and XLA devices.
It handles forward and backward passes, collecting outputs and gradients.
"""

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

from typing import Dict, Tuple, Any, List

from utils_testing import ExptCfg
from neuronx_distributed.modules.moe.moe_fused_tkg import MoEFusedTKG
from neuronx_distributed.parallel_layers import mappings

from utils import (
    CPU_DEVICE, XLA_DEVICE, TRAINING_MODE, INFERENCE_MODE,
    BATCH_DIM_TRAINING, BATCH_DIM_INFERENCE,
    reduce_loss, split_inputs_into_chunks, shard_batch
)
from utils_testing import token_shuffle_single_core
import utils_testing as ut


def get_router_logits_flag(model) -> bool:
    """
    Determine if the model should return router logits.
    
    Args:
        model: The model to check
        
    Returns:
        Boolean indicating whether the model should return router logits
    """
    if isinstance(model, MoEFusedTKG):
        return model.moe.return_router_logits
    else:
        return model.return_router_logits


def handle_token_shuffling(
    inputs: torch.Tensor, 
    cfg: ExptCfg, 
    dp_size: int, 
    is_cpu: bool
) -> Tuple[torch.Tensor, Any]:
    """
    Handle token shuffling for inputs if needed.
    
    Args:
        inputs: Input tensor
        cfg: Test configuration
        dp_size: Data parallel size
        is_cpu: Whether running on CPU
        
    Returns:
        Tuple of (shuffled_inputs, permutation_index)
    """
    permutation_index = None
    if is_cpu and getattr(cfg, 'token_shuffle_group_size', 1) > 1:
        assert inputs.device == torch.device(CPU_DEVICE)
        # in cpu mode, always tp = 1, not consider sequence_parallel
        inputs, permutation_index = token_shuffle_single_core(inputs, cfg, dp_size)
    
    return inputs, permutation_index


def process_model_outputs(
    outputs: List[torch.Tensor], 
    cfg: ExptCfg, 
    dp_size: int, 
    is_cpu: bool,
    ep_degree:int,
    permutation_index: Any = None
) -> torch.Tensor:
    """
    Process model outputs, including concatenation and token shuffling.
    
    Args:
        outputs: List of output tensors
        cfg: Test configuration
        dp_size: Data parallel size
        is_cpu: Whether running on CPU
        permutation_index: Permutation index for token shuffling
        
    Returns:
        Processed output tensor
    """
    batch_dim = BATCH_DIM_TRAINING if cfg.test_mode == TRAINING_MODE else BATCH_DIM_INFERENCE
    output = torch.cat(outputs, dim=batch_dim)
    
    if not is_cpu and ep_degree>1:	
        from neuronx_distributed.parallel_layers import mappings	
        output = mappings.gather_from_tensor_model_parallel_region(output)
    
    if is_cpu and getattr(cfg, 'token_shuffle_group_size', 1) > 1:
        assert output.device == torch.device(CPU_DEVICE)
        output = token_shuffle_single_core(output, cfg, dp_size, permutation_index=permutation_index)
    
    return output


def compute_loss_and_backward(
    output: torch.Tensor,
    target: torch.Tensor,
    model,
    optimizer,
    sequence_parallel_enabled: bool,
    is_cpu: bool,
    cfg: ExptCfg
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute loss, perform backward pass, and collect gradients.
    
    Args:
        output: Model output tensor
        target: Target tensor
        model: Model to compute gradients for
        optimizer: Optimizer to step
        sequence_parallel_enabled: Whether sequence parallelism is enabled
        is_cpu: Whether running on CPU
        cfg: Test configuration
        
    Returns:
        Tuple of (loss, gradient_dictionary)
    """
    if sequence_parallel_enabled:
        output_full = mappings.gather_from_sequence_parallel_region(
            output, to_model_parallel=False, sequence_dimension=model.sequence_dimension
        )
    else:
        output_full = output
    
    output_full = output_full.view(-1, cfg.hidden_size)
    loss = F.nll_loss(torch.nn.LogSoftmax(dim=1)(output_full), target)
    del output_full
    
    loss.backward()

    # prevents runtime errors when running back-to-back unit tests with cross-node ep
    xm.mark_step()

    optimizer.step()

    if not is_cpu:
        loss = reduce_loss(loss)

    if not cfg.zero1:
        grad_dict = ut.get_model_grads_dict(model)
    else:
        # in zero1, the gradients of trn are not the final gradients. They are before reduction.
        grad_dict = None

    return loss, grad_dict


def get_model_outputs(
    cfg: ExptCfg,
    model,
    optimizer,
    inputs,
    target,
    sequence_parallel_enabled,
    dp_size,
    dp_rank,
    exp_dp_rank,
    token_shuffle_group_size,
    tp_degree,
    ep_degree,
    is_cpu=False
):
    """
    Run model forward and backward passes and collect outputs.
    
    In CPU mode, we sequentially run each data-parallel shard to simulate a distributed backend.
    
    Args:
        cfg: Test configuration
        model: Model to run
        optimizer: Optimizer for training
        inputs: Input tensor
        target: Target tensor
        sequence_parallel_enabled: Whether sequence parallelism is enabled
        dp_size: Data parallel size
        dp_rank: Data parallel rank
        exp_dp_rank: Expert data parallel rank
        token_shuffle_group_size: Token shuffle group size
        is_cpu: Whether running on CPU
        
    Returns:
        Tuple of (router_logits, outputs, loss, grad_norm, grad_dict)
    """
    # ===== 1. DETERMINE MODEL CONFIGURATION =====
    return_router_logits = get_router_logits_flag(model)

    # ===== 2. HANDLE TOKEN SHUFFLING =====
    inputs, permutation_index = handle_token_shuffling(inputs, cfg, dp_size, is_cpu)

    # ===== 3. SPLIT INPUTS INTO CHUNKS =====
    input_chunks = split_inputs_into_chunks(inputs, dp_size, is_cpu, cfg.test_mode)
    if inputs.device != torch.device(CPU_DEVICE):
        assert len(input_chunks) == 1

    # ===== 4. PROCESS EACH CHUNK =====
    outputs = []
    router_logits = None
    
    for current_rank, chunk_input in enumerate(input_chunks):
        # Determine if we should capture router logits for this chunk
        should_capture_logits = (
            return_router_logits and
            (
                not hasattr(cfg, 'ep_degree') or
                ((cfg.ep_degree == 1 and (not is_cpu or current_rank == dp_rank)) or 
                 (cfg.ep_degree > 1 and (not is_cpu or current_rank == exp_dp_rank)))
            )
        )
        
        # Forward pass
        if should_capture_logits:
            output, router_logits = model(chunk_input)
            router_logits = router_logits.detach()
        else:
            output = model(chunk_input)[0]
            
        outputs.append(output)
    
    # Ensure router logits were captured if expected
    if return_router_logits:
        assert router_logits is not None, "Router logits were expected but not returned by the model. Check that the model is configured to return router logits."

    # ===== 5. PROCESS OUTPUTS =====
    output = process_model_outputs(outputs, cfg, dp_size, is_cpu, ep_degree, permutation_index)

    # ===== 6. HANDLE TRAINING MODE =====
    if cfg.test_mode == TRAINING_MODE:
        loss, grad_dict = compute_loss_and_backward(
            output, target, model, optimizer, sequence_parallel_enabled, is_cpu, cfg
        )
        return router_logits, output, loss, optimizer.grad_norm, grad_dict
    else:
        # ===== 7. HANDLE INFERENCE MODE =====
        assert cfg.test_mode == INFERENCE_MODE
        return router_logits, output, torch.Tensor([0]), torch.Tensor([0]), {}


def execute_model_on_cpu(
    cfg: ExptCfg, 
    env_config: Dict[str, Any], 
    models_data: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Execute model on CPU and collect outputs.
    
    Args:
        cfg: Test configuration
        env_config: Environment configuration
        models_data: Models and data
        
    Returns:
        Tuple of (router_logits, outputs, loss, grad_norm, grad_dict)
    """
    # Extract data
    model_cpu = models_data["model_cpu"]
    optimizer_cpu = models_data["optimizer_cpu"]
    inputs_cpu = models_data["inputs_cpu"]
    targets_cpu = models_data["targets_cpu"]
    
    # Extract environment config
    dp_size = env_config["dp_size"]
    dp_rank = env_config["dp_rank"]
    exp_dp_rank = env_config["exp_dp_rank"]
    sequence_parallel_enabled = env_config["sequence_parallel_enabled"]
    token_shuffle_group_size = env_config["token_shuffle_group_size"]
    
    # Get outputs and gradients from CPU
    return get_model_outputs(
        cfg,
        model_cpu,
        optimizer_cpu,
        inputs_cpu,
        targets_cpu,
        sequence_parallel_enabled,
        dp_size,
        dp_rank,
        exp_dp_rank,
        token_shuffle_group_size,
        tp_degree=1,
        ep_degree=1,
        is_cpu=True
    )

def execute_model_on_xla(
    cfg: ExptCfg, 
    env_config: Dict[str, Any], 
    models_data: Dict[str, Any],
    tp_degree:int,	
    ep_degree:int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Execute model on XLA and collect outputs.
    
    Args:
        cfg: Test configuration
        env_config: Environment configuration
        models_data: Models and data
        
    Returns:
        Tuple of (router_logits, outputs, loss, grad_norm, grad_dict)
    """
    # Extract data
    model_xla = models_data["model_xla"]
    optimizer_xla = models_data["optimizer_xla"]
    inputs_xla = models_data["inputs_xla"]
    targets_xla = models_data["targets_xla"]
    
    # Extract environment config
    dp_size = env_config["dp_size"]
    dp_rank = env_config["dp_rank"]
    exp_dp_rank = env_config["exp_dp_rank"]
    sequence_parallel_enabled = env_config["sequence_parallel_enabled"]
    token_shuffle_group_size = env_config["token_shuffle_group_size"]
    
    # Get outputs and gradients from XLA
    return get_model_outputs(
        cfg,
        model_xla,
        optimizer_xla,
        inputs_xla,
        targets_xla,
        sequence_parallel_enabled,
        dp_size,
        dp_rank,
        exp_dp_rank,
        token_shuffle_group_size,
        tp_degree,
        ep_degree,
        is_cpu=False
    )
