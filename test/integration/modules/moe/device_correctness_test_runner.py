"""
Device Correctness Test Runner
------------------------------
This module tests the correctness of MoE implementations on XLA devices.

Overall flow:
1. Set up test environment (distributed settings, parallelism)
2. For each iteration:
   a. Initialize models and data
   b. Handle precision differences between CPU and XLA
   c. Execute models on CPU and XLA
   d. Compare outputs and gradients
3. Clean up resources

Key components:
- Environment setup: Configure distributed environment
- Model execution: Run forward/backward passes
- Result comparison: Verify outputs match within tolerance
"""

import gc
import torch
import torch_xla.core.xla_model as xm

from typing import Dict, Any, List, Tuple
from utils_testing import ExptCfg

from utils import (
    print_rank0, get_appropriate_grad_context,
    TRAINING_MODE, XLA_DEVICE, BATCH_DIM_TRAINING, TOKEN_DIM
)
from environment_setup import (
    setup_test_environment, initialize_models_and_data, handle_expert_assignment_differences
)
from model_execution import execute_model_on_cpu, execute_model_on_xla
from tensor_ops import _slice_and_compare_tensors
import utils_testing as ut
from neuronx_distributed.parallel_layers import mappings
from utils import shard_batch


def prepare_xla_inputs(
    inputs_cpu: torch.Tensor, 
    targets_cpu: torch.Tensor,
    cfg: ExptCfg, 
    dp_size: int, 
    dp_rank: int,
    tp_degree: int, 	
    tp_rank: int,	
    ep_degree: int,
    sequence_parallel_enabled: bool,
    sequence_dimension: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare XLA inputs from CPU inputs.
    
    This function handles the conversion of CPU tensors to XLA tensors, including
    proper sharding for data parallelism and sequence parallelism.
    
    Common use case:
    In MoE model testing, this function is used to prepare inputs for the XLA device
    before model execution. For sequence parallel cases, it handles the scattering of
    inputs across the sequence dimension and creates a manually split version (inputs_xla_manual_split)
    for validation purposes. The batch dimension handling differs between training mode
    (BATCH_DIM_TRAINING=1) and inference mode (BATCH_DIM_INFERENCE=0).
    
    Args:
        inputs_cpu: CPU input tensor
        targets_cpu: CPU target tensor
        cfg: Test configuration
        dp_size: Data parallel size
        dp_rank: Data parallel rank
        sequence_parallel_enabled: Whether sequence parallelism is enabled
        sequence_dimension: Sequence dimension for parallelism
        
    Returns:
        Tuple of:
        - inputs_xla: Input tensor prepared for XLA device
        - targets_xla: Target tensor prepared for XLA device
        - inputs_xla_manual_split: Version of inputs_xla created through manual tensor splitting
          rather than collective operations. Used for validation to ensure that inputs created through
          collectives (scatter_to_sequence_parallel_region) match those created through manual CPU
          tensor splitting.
    """
    # Create and shard inputs_xla_full from inputs_cpu
    inputs_xla_full = inputs_cpu.detach().to(XLA_DEVICE)
    inputs_xla_full = shard_batch(inputs_xla_full, cfg, dp_size, dp_rank, cfg.test_mode)
    
    # Get sharded input for rank (for sequence parallel)
    if sequence_parallel_enabled:
        inputs_xla = mappings.scatter_to_sequence_parallel_region(inputs_xla_full, sequence_dimension=sequence_dimension)
    else:
        inputs_xla = inputs_xla_full
            
    # Prepare XLA target
    targets_xla = targets_cpu.clone().detach().to(XLA_DEVICE)
    
    # Create a manually split version of the input for validation
    # This version is created through direct tensor splitting rather than using collective operations
    # It's used to verify that inputs created through collectives match those created through manual splitting
    inputs_xla_manual_split = None
    if sequence_parallel_enabled:
        inputs_xla_manual_split = torch.tensor_split(inputs_xla_full.cpu(), tp_degree, dim=sequence_dimension)[tp_rank].clone().detach().to(XLA_DEVICE)
        
    # Data-parallel sharding
    targets_xla = shard_batch(targets_xla, cfg, dp_size, dp_rank, cfg.test_mode)
    
    if sequence_parallel_enabled and ep_degree > 1:	
        targets_xla = torch.tensor_split(targets_xla.cpu(), ep_degree, dim=sequence_dimension)[torch.distributed.get_rank() % ep_degree].clone().detach().to(XLA_DEVICE)
    
    return inputs_xla, targets_xla, inputs_xla_manual_split


def compare_model_outputs(
    router_logits_cpu: torch.Tensor,
    router_logits_xla: torch.Tensor,
    model_output_cpu: torch.Tensor,
    model_output_xla: torch.Tensor,
    tp_degree: int,
    ep_degree:int,
    dp_rank: int,
    batch_size: int,
    sequence_dimension: int,
    sequence_parallel_enabled: bool,
    test_mode: str,
    output_tols: Dict[str, float]
) -> None:
    """
    Compare outputs between CPU and XLA models.
    
    This function handles the comparison of model outputs between CPU and XLA implementations,
    accounting for differences in tensor shapes due to parallelism strategies. It ensures that
    the correct portions of tensors are compared based on the test mode and parallelism configuration.
        
    Args:
        router_logits_cpu: Router logits from CPU model
        router_logits_xla: Router logits from XLA model
        model_output_cpu: Output tensor from CPU model
        model_output_xla: Output tensor from XLA model
        sequence_parallel_enabled: Whether sequence parallelism is enabled
        tp_degree: Tensor parallel degree
        sequence_dimension: Sequence dimension for parallelism
        dp_rank: Data parallel rank
        batch_size: Batch size
        test_mode: Test mode (training or inference)
        output_tols: Tolerance parameters for comparison
    """
    # Compare output
    if sequence_parallel_enabled:
        # Compare with only output shard belonging to the TP rank
        tp_rank = torch.distributed.get_rank() % tp_degree
        model_output_cpu = torch.tensor_split(model_output_cpu, tp_degree, dim=sequence_dimension)[tp_rank]
        if ep_degree > 1:
            model_output_cpu = torch.tensor_split(model_output_cpu, ep_degree, dim=sequence_dimension)[torch.distributed.get_rank() % ep_degree]
        if test_mode != TRAINING_MODE:
            # Gather Router logits across sequence parallel region
            router_logits_xla = mappings.gather_from_sequence_parallel_region(router_logits_xla, sequence_dimension=TOKEN_DIM)
    if test_mode == TRAINING_MODE:
        model_output_cpu = model_output_cpu.narrow(BATCH_DIM_TRAINING, dp_rank * batch_size, batch_size)
    
    # Compare outputs
    ut.check_tensors(model_output_cpu.detach(), model_output_xla.detach(), **output_tols)
    ut.check_tensors(router_logits_cpu, router_logits_xla, **output_tols)


def compare_training_outputs(
    loss_cpu: torch.Tensor,
    loss_xla: torch.Tensor,
    grad_norm_cpu: torch.Tensor,
    grad_norm_xla: torch.Tensor,
    grad_dict_cpu: Dict[str, torch.Tensor],
    grad_dict_xla: Dict[str, torch.Tensor],
    models_data: Dict[str, Any],
    env_config: Dict[str, Any],
    sharding_info: Tuple[int, int, int, int],
    iteration: int,
    output_tols: Dict[str, float],
    grad_tols: Dict[str, float]
) -> None:
    """
    Compare training-specific outputs between CPU and XLA models.
    
    Args:
        loss_cpu: Loss from CPU model
        loss_xla: Loss from XLA model
        grad_norm_cpu: Gradient norm from CPU model
        grad_norm_xla: Gradient norm from XLA model
        grad_dict_cpu: Gradient dictionary from CPU model
        grad_dict_xla: Gradient dictionary from XLA model
        models_data: Dictionary containing models and data
        env_config: Environment configuration
        sharding_info: Tuple of (tp_rank, tp_size, ep_rank, ep_size)
        iteration: Current iteration number
        output_tols: Tolerance parameters for output comparison
        grad_tols: Tolerance parameters for gradient comparison
    """
    # Compare loss
    ut.check_tensors(loss_cpu.detach(), loss_xla.detach(), **output_tols)
    
    print_rank0(f"grad_norm_cpu={grad_norm_cpu}")
    print_rank0(f"grad_norm_xla={grad_norm_xla}")
    # TODO: verify after V1492568678
    # ut.check_tensors(grad_norm_cpu, grad_norm_xla, **output_tols)
    
    cfg_xla = env_config["cfg_xla"]
    if not cfg_xla.zero1:
        # Check gradients on each rank
        _slice_and_compare_tensors(grad_dict_cpu, grad_dict_xla, sharding_info, iteration, **grad_tols)
    else:
        # If zero1 is enabled then directly compare updated parameters, not the gradients
        # The true gradients used is private in zero1 optimizer
        xla_parameters = {n: p for n, p in models_data["model_xla"].named_parameters()}
        cpu_parameters = {n: p for n, p in models_data["model_cpu"].named_parameters()}
        param_tols = {k: cfg_xla.lr * v for k, v in grad_tols.items()}
        _slice_and_compare_tensors(cpu_parameters, xla_parameters, sharding_info, iteration, **param_tols)
        
        optimizer_cpu = models_data["optimizer_cpu"]
        optimizer_xla = models_data["optimizer_xla"]
        optimizer_cpu.zero_grad(set_to_none=True)
        optimizer_xla.zero_grad(set_to_none=True)


def run_device_correctness_test(cfg: ExptCfg, output_tols: Dict[str, float], grad_tols: Dict[str, float]) -> None:
    """
    Run device correctness test comparing CPU and XLA implementations.
    
    This function tests the correctness of the MoE implementation on XLA devices
    by comparing outputs and gradients with a CPU reference implementation.
    
    Args:
        cfg: Test configuration with model parameters
        output_tols: Tolerance parameters for output comparison
        grad_tols: Tolerance parameters for gradient comparison
    """
    # ===== 1. ENVIRONMENT SETUP =====
    env_config = setup_test_environment(cfg)
    
    # Get appropriate gradient context
    grad_ctx_mgr = get_appropriate_grad_context(cfg)
    
    with grad_ctx_mgr():
        for iteration in range(cfg.num_iters):
            print_rank0(f"iteration {iteration}")
            
            # ===== 2. MODEL INITIALIZATION =====
            models_data = initialize_models_and_data(cfg, env_config, iteration)
            
            # Handle expert assignment differences for bfloat16
            models_data = handle_expert_assignment_differences(cfg, env_config, models_data, iteration, output_tols)
            
            # Extract sharding info for tensor comparison
            tp_rank = env_config["tp_rank"]
            tp_size = env_config["tp_size"]
            ep_rank = env_config["ep_rank"]
            ep_size = env_config["ep_size"]
            tp_degree = env_config["tp_degree"]	
            ep_degree=env_config["ep_degree"]
            sharding_info = (tp_rank, tp_size, ep_rank, ep_size)
            
            # ===== 3. MODEL EXECUTION - CPU =====
            router_logits_cpu, model_output_cpu, loss_cpu, grad_norm_cpu, grad_dict_cpu = execute_model_on_cpu(
                cfg, env_config, models_data
            )
            
            # Re-init NxD with actual TP degree
            ut.nxd_init(
                tp_degree=env_config["tp_degree"],
                ep_degree=env_config["ep_degree"],
                token_shuffle_group_size=env_config["token_shuffle_group_size"],
                seed=iteration,
            )
            
            # ===== 4. PREPARE XLA INPUTS =====
            sequence_parallel_enabled = env_config["sequence_parallel_enabled"]
            sequence_dimension = models_data["sequence_dimension"]
            targets_cpu = models_data["targets_cpu"]
            dp_size = env_config["dp_size"]
            dp_rank = env_config["dp_rank"]
            tp_degree = env_config["tp_degree"]	
            tp_rank = env_config["tp_rank"]	
            ep_degree=env_config["ep_degree"]
            
            inputs_xla, targets_xla, inputs_xla_manual_split = prepare_xla_inputs(
                models_data["inputs_cpu"], targets_cpu, cfg, dp_size, dp_rank, tp_degree, tp_rank, ep_degree,
                sequence_parallel_enabled, sequence_dimension
            )
            
            # Check inputs if needed for sequence parallel
            if sequence_parallel_enabled and inputs_xla_manual_split is not None:
                # The inputs are the same from collectives and cpu split when checking on cpu explicitly,
                # however it would result in different router logits and output
                # adding this check_tensors trigger a different compilation which somehow resolves the issue
                ut.check_tensors(inputs_xla, inputs_xla_manual_split, 0, 0)
            
            # Update models_data with XLA inputs
            models_data["inputs_xla"] = inputs_xla
            models_data["targets_xla"] = targets_xla
            
            # ===== 5. MODEL EXECUTION - XLA =====
            router_logits_xla, model_output_xla, loss_xla, grad_norm_xla, grad_dict_xla = execute_model_on_xla(
                cfg, env_config, models_data, tp_degree, ep_degree
            )
            xm.mark_step()  # TRN enablement
            
            # ===== 6. CLEAN UP INPUT TENSORS =====
            del models_data["inputs_cpu"], models_data["inputs_xla"]
            del models_data["targets_cpu"], models_data["targets_xla"]
            
            # ===== 7. COMPARE MODEL OUTPUTS =====
            compare_model_outputs(
                router_logits_cpu, router_logits_xla,
                model_output_cpu, model_output_xla,
                env_config["tp_degree"], env_config["ep_degree"], 
                dp_rank, cfg.batch_size, 
                sequence_dimension, sequence_parallel_enabled, 
                cfg.test_mode, output_tols  
            )
            
            # Clean up output tensors
            del model_output_cpu, model_output_xla, router_logits_cpu, router_logits_xla
            
            # ===== 8. COMPARE TRAINING-SPECIFIC OUTPUTS =====
            if cfg.test_mode == TRAINING_MODE:
                compare_training_outputs(
                    loss_cpu, loss_xla, grad_norm_cpu, grad_norm_xla,
                    grad_dict_cpu, grad_dict_xla, models_data, env_config,
                    sharding_info, iteration, output_tols, grad_tols
                )
                
                # Clean up training tensors
                del loss_cpu, loss_xla, grad_dict_cpu, grad_dict_xla
                
            # ===== 9. CLEANUP FOR THIS ITERATION =====
            del models_data["model_cpu"], models_data["model_xla"]
            if "optimizer_cpu" in models_data:
                del models_data["optimizer_cpu"]
            if "optimizer_xla" in models_data:
                del models_data["optimizer_xla"]
            gc.collect()
            xm.mark_step()
