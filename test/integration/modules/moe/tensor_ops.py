"""
Tensor Operations Module
-----------------------
This module contains functions for tensor manipulation and comparison operations.
It handles slicing, reshaping, and comparing tensors between CPU and XLA implementations.
"""

import torch
import torch_neuronx
import utils_testing as ut

from typing import Dict, Tuple, Any, List


def _get_slice_for_rank(tensor: torch.Tensor, sharding_info: Tuple[int, int, int, int], split_dims: List[int]) -> torch.Tensor:
    """
    Get a slice of a tensor for a specific rank based on sharding information.
    
    Args:
        tensor: Input tensor to slice
        sharding_info: Tuple of (tp_rank, tp_size, ep_rank, ep_size)
        split_dims: Dimensions to split along
        
    Returns:
        Sliced tensor
    """
    tp_rank, tp_size, ep_rank, ep_size = sharding_info
    
    for dim in split_dims:
        # Determine which rank and size to use based on dimension
        rank, size = (tp_rank, tp_size) if dim > 0 else (ep_rank, ep_size)
        # Split tensor along dimension and select appropriate slice
        tensor = torch.tensor_split(tensor, size, dim=dim)[rank]
        
    return tensor


def _slice_gate_up_proj_tensor(
    tensor: torch.Tensor, 
    sharding_info: Tuple[int, int, int, int]
) -> torch.Tensor:
    """
    Slice a gate_up_proj tensor based on sharding information.
    
    Args:
        tensor: Input gate_up_proj tensor
        sharding_info: Tuple of (tp_rank, tp_size, ep_rank, ep_size)
        
    Returns:
        Sliced tensor
    """
    # Split into gate_proj and up_proj tensors
    gate_proj_tensor, up_proj_tensor = torch.tensor_split(tensor, 2, dim=2)
    
    # Slice each tensor separately
    gate_proj_tensor_for_rank = _get_slice_for_rank(gate_proj_tensor, sharding_info, split_dims=(0, 2))
    up_proj_tensor_for_rank = _get_slice_for_rank(up_proj_tensor, sharding_info, split_dims=(0, 2))
    
    # Concatenate sliced tensors
    return torch.cat([gate_proj_tensor_for_rank, up_proj_tensor_for_rank], dim=2)


def _get_tensor_for_rank(
    key: str, 
    cpu_tensor: torch.Tensor, 
    xla_tensor: torch.Tensor, 
    sharding_info: Tuple[int, int, int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get appropriate tensor slices for comparison based on key and sharding info.
    
    Args:
        key: Tensor key/name
        cpu_tensor: CPU tensor
        xla_tensor: XLA tensor
        sharding_info: Tuple of (tp_rank, tp_size, ep_rank, ep_size)
        
    Returns:
        Tuple of (cpu_tensor_for_rank, xla_tensor)
    """
    # If shapes match, no slicing needed
    if cpu_tensor.shape == xla_tensor.shape:
        return cpu_tensor, xla_tensor
    
    # Handle different tensor types based on key
    if "gate_up_proj" in key:
        return _slice_gate_up_proj_tensor(cpu_tensor, sharding_info), xla_tensor
    elif "up_proj" in key:
        return _get_slice_for_rank(cpu_tensor, sharding_info, split_dims=(0, 2)), xla_tensor
    elif "down_proj" in key:
        return _get_slice_for_rank(cpu_tensor, sharding_info, split_dims=(0, 1)), xla_tensor
    else:
        raise Exception(f"Unexpected shapes for key: {key}, {cpu_tensor.shape}, {xla_tensor.shape}")


def assert_cpu_xla_keys_match(cpu_keys: set, xla_keys: set) -> None:
    """
    Assert that XLA dict contains all CPU keys and validate key matching.
    
    Args:
        cpu_keys: Set of CPU tensor keys
        xla_keys: Set of XLA tensor keys
        
    Raises:
        AssertionError: If XLA keys don't contain all CPU keys
    """
    # Skip spmd_rank keys as they are XLA-specific constructs for SPMD indexing in inference
    xla_keys_filtered = {key for key in xla_keys if "spmd_rank.rank" not in key}
    
    missing_keys = cpu_keys - xla_keys_filtered
    assert not missing_keys, f"XLA dict missing CPU keys: {missing_keys}"


def _slice_and_compare_tensors(
    cpu_dict: Dict[str, torch.Tensor], 
    xla_dict: Dict[str, torch.Tensor], 
    sharding_info: Tuple[int, int, int, int], 
    iteration: int, 
    **tols
) -> None:
    """
    Slice and compare tensors between CPU and XLA models.
    
    This function handles the complexity of comparing tensors that may have different shapes
    due to tensor parallelism or expert parallelism.
    
    Args:
        cpu_dict: Dictionary of CPU tensors
        xla_dict: Dictionary of XLA tensors
        sharding_info: Tuple of (tp_rank, tp_size, ep_rank, ep_size)
        iteration: Current iteration number
        **tols: Tolerance parameters for comparison
    """
    cpu_keys = set(cpu_dict.keys())
    xla_keys = set(xla_dict.keys())
    
    # Validate that XLA contains all CPU keys
    assert_cpu_xla_keys_match(cpu_keys, xla_keys)
    
    # Process each tensor pair
    for key in sorted(cpu_keys):
        # Skip shared experts tensors (TODO: add per rank checking logic)
        if "shared_experts" in key:
            continue
            
        # Detach CPU tensor
        cpu_dict[key] = cpu_dict[key].detach()
        
        # Get appropriate tensor slices for comparison
        key_tensor_for_rank, xla_tensor = _get_tensor_for_rank(
            key, cpu_dict[key], xla_dict[key].detach(), sharding_info
        )

        # Add context information for error messages
        additional_msg = f"Iteration {iteration} \nKey: {key}"
        
        # Compare tensors with provided tolerances
        ut.check_tensors(key_tensor_for_rank, xla_tensor, **tols, additional_msg=additional_msg)
