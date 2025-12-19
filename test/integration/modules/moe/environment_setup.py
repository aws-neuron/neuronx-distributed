"""
Environment Setup Module
-----------------------
This module contains functions for setting up the test environment,
initializing models and data, and handling precision differences.
"""

import dataclasses
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from typing import Dict, Any
import utils_testing as ut
from utils_testing import ExptCfg
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.modules.moe import token_shuffling

from utils import (
    XLA_DEVICE, CPU_DEVICE, PRECISION_SENSITIVE_DTYPE, TRAINING_MODE, INFERENCE_MODE,
    GRAD_CLIPPING_ENABLED, should_transpose_shared_experts_weights, shard_batch, logger,
    TOKEN_DIM
)


def setup_test_environment(cfg: ExptCfg) -> Dict[str, Any]:
    """
    Set up the test environment with the specified configuration parameters.
    
    Args:
        cfg: Test configuration
        
    Returns:
        Dict containing environment configuration
    """
    # ===== 1. VALIDATE CONFIGURATION =====
    # Validate test mode
    assert cfg.test_mode in {TRAINING_MODE, INFERENCE_MODE}, f"Unknown test_mode: {cfg.test_mode}"
    
    # Get parallelism parameters
    tp_degree = getattr(cfg, "tp_degree", 1)
    ep_degree = getattr(cfg, "ep_degree", 1)
    token_shuffle_group_size = getattr(cfg, "token_shuffle_group_size", 1)
    sequence_parallel_enabled = cfg.sequence_parallel_enabled
    
    # Validate configuration
    if not sequence_parallel_enabled:
        # Training without SP has BSH layout, which the test code does not currently account for
        assert tp_degree == 1 or cfg.test_mode != TRAINING_MODE, "Integration tests for training are only supported with SP"
    
    # ===== 2. INITIALIZE DISTRIBUTED ENVIRONMENT =====
    # This first nxd_init call is required to initialize the distributed environment
    # before retrieving parallel sizes and ranks (dp_size, dp_rank, tp_size, etc.).
    # These values are needed to create the environment configuration that will be used
    # for tensor sharding, input creation, and other distributed operations.
    # Later in initialize_models_and_data(), we'll reinitialize for CPU and XLA models separately.
    ut.nxd_init(tp_degree=tp_degree, ep_degree=ep_degree, token_shuffle_group_size=token_shuffle_group_size, seed=0)
    
    # ===== 3. GET PARALLEL SIZES AND RANKS =====
    dp_size = parallel_state.get_expert_data_parallel_size()
    dp_rank = parallel_state.get_expert_data_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    ep_size = parallel_state.get_expert_model_parallel_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    exp_dp_rank = parallel_state.get_expert_data_parallel_rank()
    
    # ===== 4. CREATE XLA CONFIG =====
    cfg_xla = dataclasses.replace(cfg, device=XLA_DEVICE)
    
    # ===== 5. SET LEARNING RATE FOR ZERO-1 =====
    # For zero-1, we need a non-zero learning rate because we can't compare gradients directly.
    # Instead, we make actual weight updates and compare the updated weights.
    # For non-zero-1, we use lr=0.0 since we directly compare gradients without updating weights.
    lr = cfg_xla.lr if cfg_xla.zero1 else 0.0
    
    # ===== 6. RETURN ENVIRONMENT CONFIG =====
    return {
        "tp_degree": tp_degree,
        "ep_degree": ep_degree,
        "token_shuffle_group_size": token_shuffle_group_size,
        "sequence_parallel_enabled": sequence_parallel_enabled,
        "dp_size": dp_size,
        "dp_rank": dp_rank,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "ep_size": ep_size,
        "ep_rank": ep_rank,
        "exp_dp_rank": exp_dp_rank,
        "cfg_xla": cfg_xla,
        "lr": lr,
        "grad_clipping": GRAD_CLIPPING_ENABLED,
    }


def initialize_models_and_data(cfg: ExptCfg, env_config: Dict[str, Any], iteration: int) -> Dict[str, Any]:
    """
    Initialize models, optimizers, and input data for testing.
    
    Args:
        cfg: Test configuration
        env_config: Environment configuration
        iteration: Current iteration number
        
    Returns:
        Dict containing models, optimizers, and input data
    """
    logger.info(f"Initializing models and data for iteration {iteration}")
    
    # ===== 1. EXTRACT ENVIRONMENT CONFIG =====
    tp_degree = env_config["tp_degree"]
    ep_degree = env_config["ep_degree"]
    token_shuffle_group_size = env_config["token_shuffle_group_size"]
    dp_size = env_config["dp_size"]
    dp_rank = env_config["dp_rank"]
    cfg_xla = env_config["cfg_xla"]
    lr = env_config["lr"]
    grad_clipping = env_config["grad_clipping"]
    
    # ===== 2. INITIALIZE CPU MODEL =====
    # Initialize with tp_degree=1, ep_degree=1 for CPU model
    ut.nxd_init(tp_degree=1, ep_degree=1, token_shuffle_group_size=1, seed=iteration)
    
    # enable_spmd_rank is not supported for CPU flow
    cfg.enable_spmd_rank = False
    model_cpu = ut.initialize_neuron_model(cfg)
    
    # ===== 3. INITIALIZE XLA MODEL =====
    # Initialize with actual parallelism parameters
    ut.nxd_init(
        tp_degree=tp_degree, ep_degree=ep_degree, token_shuffle_group_size=token_shuffle_group_size, seed=iteration
    )
    
    # Determine if shared experts weights should be transposed
    transpose_shared_experts_weights = should_transpose_shared_experts_weights(cfg_xla)
    
    # Initialize XLA model
    model_xla = ut.initialize_neuron_model(cfg_xla, transpose_shared_experts_weights=transpose_shared_experts_weights)
    
    # ===== 4. MATCH WEIGHTS BETWEEN MODELS =====
    ut.match_expert_weights(model_xla, model_cpu, cfg.glu_mlp, transpose_shared_experts_weights, 
                           cfg.shared_experts_sequence_parallel_enabled)
    
    sequence_dimension = model_xla.sequence_dimension
    
    # ===== 5. INITIALIZE OPTIMIZERS =====
    optimizer_cpu = None
    optimizer_xla = None
    
    if cfg.test_mode == TRAINING_MODE:
        # Set models to training mode
        model_cpu.train()
        model_xla.train()
        
        # Set sinkhorn_iterations=0, because small precision errors can cause differences in routing decisions
        model_cpu.router.sinkhorn_iterations = 0
        model_xla.router.sinkhorn_iterations = 0
        
        # Initialize optimizers
        optimizer_cpu = ut.initialize_neuron_optimizer(
            model_cpu, grad_clipping=grad_clipping, override_grad_reduction=True, zero1=False, lr=lr
        )
        optimizer_xla = ut.initialize_neuron_optimizer(
            model_xla, grad_clipping=grad_clipping, zero1=cfg_xla.zero1, lr=lr
        )
    else:
        # Set models to evaluation mode
        model_cpu.eval()
        model_xla.eval()
    
    # ===== 6. INITIALIZE INPUT TENSORS =====
    if cfg.test_mode == TRAINING_MODE:
        # Input is SBH in training when SP is enabled
        inputs_cpu = torch.randn(cfg.seq_len, cfg.batch_size * dp_size, cfg.hidden_size, dtype=cfg.dtype).detach()
        targets_cpu = torch.randint(
            0, cfg.hidden_size - 1, (cfg.seq_len * cfg.batch_size * dp_size,), dtype=torch.long
        ).detach()
    else:
        # Input is BSH in inference
        inputs_cpu = torch.randn(cfg.batch_size, cfg.seq_len, cfg.hidden_size, dtype=cfg.dtype).detach()
        targets_cpu = torch.randint(
            0, cfg.hidden_size - 1, (cfg.seq_len * cfg.batch_size,), dtype=torch.long
        ).detach()
    
    # ===== 7. PREPARE XLA INPUTS =====
    inputs_xla_full = inputs_cpu.detach().to(XLA_DEVICE)
    inputs_xla_full = shard_batch(inputs_xla_full, cfg, dp_size, dp_rank, cfg.test_mode)
    
    # ===== 8. RETURN MODELS AND DATA =====
    return {
        "model_cpu": model_cpu,
        "model_xla": model_xla,
        "optimizer_cpu": optimizer_cpu,
        "optimizer_xla": optimizer_xla,
        "inputs_cpu": inputs_cpu,
        "inputs_xla_full": inputs_xla_full,
        "targets_cpu": targets_cpu,
        "sequence_dimension": sequence_dimension,
    }


def handle_expert_assignment_differences(
    cfg: ExptCfg, 
    env_config: Dict[str, Any], 
    models_data: Dict[str, Any], 
    iteration: int,
    output_tols: Dict[str, float]
) -> Dict[str, Any]:
    """
    Handle differences in expert assignment between CPU and XLA for bfloat16 tests.
    
    torch.topk behavior is different on CPU and device in the case of ties.
    This causes mismatches in expert assignment for the TopK tests in bf16.
    
    Args:
        cfg: Test configuration
        env_config: Environment configuration
        models_data: Models and data
        iteration: Current iteration number
        output_tols: Tolerance parameters for output comparison
        
    Returns:
        Updated models_data with potentially modified inputs
    """
    # Skip if not using bfloat16 or if MoEFusedTKG is enabled
    if cfg.dtype != PRECISION_SENSITIVE_DTYPE or cfg.moe_fused_tkg_enabled:
        return models_data
        
    import torch_neuronx
    from neuronx_distributed.parallel_layers import mappings
    
    # ===== 1. EXTRACT DATA =====
    model_cpu = models_data["model_cpu"]
    model_xla = models_data["model_xla"]
    inputs_cpu = models_data["inputs_cpu"]
    sequence_dimension = models_data["sequence_dimension"]
    
    # ===== 2. EXTRACT ENVIRONMENT CONFIG =====
    tp_degree = env_config["tp_degree"]
    ep_degree = env_config["ep_degree"]
    token_shuffle_group_size = env_config["token_shuffle_group_size"]
    dp_size = env_config["dp_size"]
    dp_rank = env_config["dp_rank"]
    sequence_parallel_enabled = env_config["sequence_parallel_enabled"]
    
    # Create inputs_xla_full from inputs_cpu
    inputs_xla_full = inputs_cpu.detach().to(XLA_DEVICE)
    inputs_xla_full = shard_batch(inputs_xla_full, cfg, dp_size, dp_rank, cfg.test_mode)
    
    # ===== 3. INITIALIZE CPU ENVIRONMENT =====
    ut.nxd_init(tp_degree=1, ep_degree=1, token_shuffle_group_size=1, seed=iteration)
    
    # ===== 4. ENABLE EXPERT INDEX RETURN =====
    model_cpu.return_expert_index = True
    model_xla.return_expert_index = True

    # ===== 5. GET EXPERT INDICES FROM CPU MODEL =====
    with torch.no_grad():
        # Handle token shuffling if needed
        permutation_index = None
        if token_shuffle_group_size > 1:
            inputs_cpu, permutation_index = ut.token_shuffle_single_core(inputs_cpu, cfg, dp_size)
        
        # Get expert indices from CPU model
        router_logits_cpu, expert_index_cpu = model_cpu(inputs_cpu)[-2:]
        
        # ===== 6. HANDLE TOKEN SHUFFLING FOR ROUTER LOGITS AND EXPERT INDICES =====
        if token_shuffle_group_size > 1:
            inputs_cpu = ut.token_shuffle_single_core(inputs_cpu, cfg, dp_size, permutation_index=permutation_index)
            
            # Reshape and shuffle router logits
            router_logits_cpu = router_logits_cpu.reshape(
                cfg.seq_len, dp_size * cfg.batch_size, cfg.num_experts
            )
            router_logits_cpu = ut.token_shuffle_single_core(
                router_logits_cpu, cfg, dp_size, permutation_index=permutation_index
            )
            router_logits_cpu = router_logits_cpu.reshape(
                cfg.seq_len * dp_size * cfg.batch_size, cfg.num_experts
            )

            # Reshape and shuffle expert indices
            expert_index_cpu = expert_index_cpu.reshape(cfg.seq_len, dp_size * cfg.batch_size, cfg.top_k)
            expert_index_cpu = ut.token_shuffle_single_core(
                expert_index_cpu, cfg, dp_size, permutation_index=permutation_index
            )
            expert_index_cpu = expert_index_cpu.reshape(cfg.seq_len * dp_size * cfg.batch_size, cfg.top_k)

        # ===== 7. INITIALIZE XLA ENVIRONMENT =====
        ut.nxd_init(
            tp_degree=tp_degree,
            ep_degree=ep_degree,
            token_shuffle_group_size=token_shuffle_group_size,
            seed=iteration,
        )
        
        # ===== 8. PREPARE XLA INPUTS =====
        if sequence_parallel_enabled:
            inputs_xla = mappings.scatter_to_sequence_parallel_region(inputs_xla_full, sequence_dimension=sequence_dimension)
        else:
            inputs_xla = inputs_xla_full
        
        # ===== 9. GET EXPERT INDICES FROM XLA MODEL =====
        router_logits_xla, expert_index_xla = model_xla(inputs_xla)[-2:]
        
        # ===== 10. HANDLE TOKEN SHUFFLING FOR XLA EXPERT INDICES =====
        if token_shuffle_group_size > 1:
            if sequence_parallel_enabled:
                local_expert_index_xla = mappings.scatter_to_sequence_parallel_region(expert_index_xla)
            else:
                local_expert_index_xla = expert_index_xla
                
            # Unpermute the expert_index_xla to the original dp rank
            local_expert_index_xla = local_expert_index_xla.reshape(-1, cfg.batch_size, cfg.top_k)
            local_expert_index_xla = token_shuffling.token_unshuffle(
                local_expert_index_xla, model_xla.shuffle_permutation
            )
            local_expert_index_xla = local_expert_index_xla.reshape(-1, cfg.top_k)
            expert_index_xla = mappings.gather_from_sequence_parallel_region(local_expert_index_xla, sequence_dimension=TOKEN_DIM)

        # ===== 11. GATHER TENSORS FOR SEQUENCE PARALLEL =====
        if sequence_parallel_enabled and cfg.test_mode != TRAINING_MODE:
            expert_index_xla = mappings.gather_from_sequence_parallel_region(expert_index_xla, sequence_dimension=TOKEN_DIM)
            router_logits_xla = mappings.gather_from_sequence_parallel_region(router_logits_xla, sequence_dimension=TOKEN_DIM)

        # ===== 12. REINITIALIZE CPU ENVIRONMENT =====
        ut.nxd_init(tp_degree=1, ep_degree=1, token_shuffle_group_size=1, seed=iteration)
        
        # ===== 13. COMPARE EXPERT INDICES AND HANDLE MISMATCHES =====
        local_expert_index_cpu = shard_batch(expert_index_cpu, cfg, dp_size, dp_rank, cfg.test_mode)
        expert_mismatch_indices = set(
            torch.where(local_expert_index_cpu != expert_index_xla.cpu())[0].tolist()
        )
        
        if len(expert_mismatch_indices) > 0:
            # ===== 14. CHECK IF MISMATCHED EXPERT'S ROUTER LOGITS ARE CLOSE =====
            for mismatch_idx in expert_mismatch_indices:
                router_logits_xla_idx = router_logits_xla.cpu()[mismatch_idx]
                
                # Check each mismatched expert
                mismatch_idx_logits = \
                torch.where(local_expert_index_cpu[mismatch_idx] != expert_index_xla.cpu()[mismatch_idx])[
                    0].tolist()
                cpu_mismatched_indices = local_expert_index_cpu[mismatch_idx, mismatch_idx_logits]
                xla_mismatched_indices = expert_index_xla.cpu()[mismatch_idx, mismatch_idx_logits]
                
                # Check if mismatched expert's corresponding router logits are actually close due to bf16
                torch_neuronx.testing.assert_close(router_logits_xla_idx[cpu_mismatched_indices], router_logits_xla_idx[xla_mismatched_indices])
            
            # ===== 15. UPDATE INPUT TENSOR TO MASK TOKENS WITH EXPERT ASSIGNMENT MISMATCH =====
            local_inputs_cpu = shard_batch(inputs_cpu, cfg, dp_size, dp_rank, cfg.test_mode)
            local_inputs_cpu = ut.drop_tokens_in_tensor(local_inputs_cpu, expert_mismatch_indices)
            inputs_xla_full = inputs_cpu.detach().to(XLA_DEVICE)
            inputs_xla_full = shard_batch(inputs_xla_full, cfg, dp_size, dp_rank, cfg.test_mode)
        
        # ===== 16. DOUBLE CHECK INPUT IS STILL THE SAME =====
        local_inputs_cpu = shard_batch(inputs_cpu, cfg, dp_size, dp_rank, cfg.test_mode)
        ut.check_tensors(
            local_inputs_cpu.detach(), inputs_xla_full.detach(), **output_tols, additional_msg=f"Iteration {iteration}"
        )
    
    # ===== 17. RESET RETURN_EXPERT_INDEX =====
    model_cpu.return_expert_index = False
    model_xla.return_expert_index = False
    
    # ===== 18. UPDATE MODELS_DATA WITH POTENTIALLY MODIFIED INPUTS =====
    models_data["inputs_cpu"] = inputs_cpu
    models_data["inputs_xla_full"] = inputs_xla_full
    
    return models_data
