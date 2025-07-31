import os
import torch

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace.mock_torchdist import mock_distributed

class NxDParallelState:
    """
    A consolidated context manager for NeuronX Distributed (NxD) that handles parallel state initialization
    and cleanup for distributed training scenarios.
    
    This context manager provides a unified interface to set up and tear down the distributed environment
    required for model parallelism in NeuronX Distributed.
    
    The class automatically manages:
    - Mock distributed environment setup when torch.distributed is not initialized
    - Process group initialization with XLA backend
    - Model parallel state configuration with specified parallelism
    - AOT (Ahead-of-Time) mode activation for tracing
    - Proper cleanup of all initialized components on exit
    
    Args:
        world_size (int, optional): Total number of processes in the distributed setup. Defaults to 1.
        rank (int, optional): Rank of the current process. Defaults to 0.
        tensor_model_parallel_size (int, optional): Size of tensor model parallel group. Defaults to 1.
        pipeline_model_parallel_size (int, optional): Size of pipeline model parallel group. Defaults to 1.
        context_parallel_size (int, optional): Size of context parallel group for sequence parallelism. Defaults to 1.
        expert_model_parallel_size (int, optional): Size of expert model parallel group for MoE models. Defaults to 1.
        lnc_size (int, optional): Logical neuron core size. Defaults to 1.
    
    Usage:
        ```python
        with NxDParallelState(
            world_size=32,
            tensor_model_parallel_size=32,
        ):
            # Your model tracing/compilation code here
            ...
        ```
    """
    def __init__(
        self,
        world_size=1,
        rank=0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        lnc_size=1
    ):
        self.world_size = world_size
        self.rank = rank
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.context_parallel_size = context_parallel_size
        self.expert_model_parallel_size = expert_model_parallel_size
        self.lnc_size = lnc_size
        self.mock_dist = None

    def __enter__(self):
        if not torch.distributed.is_initialized():
            # Setup mock distributed environment
            self.mock_dist = mock_distributed(world_size=self.world_size)
            self.mock_dist.__enter__()

            # Initialize process group
            torch.distributed.init_process_group(
                backend="xla",
                rank=self.rank,
                world_size=self.world_size
            )

        # Initialize model parallel state
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=self.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            context_parallel_size=self.context_parallel_size,
            expert_model_parallel_size=self.expert_model_parallel_size,
            skip_collective_init=True,
            lnc_size=self.lnc_size
        )

        # Set AOT mode
        parallel_state.set_aot_mode(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        parallel_state.set_aot_mode(False)
        parallel_state.destroy_model_parallel()
        
        if self.mock_dist is not None:
            torch.distributed.destroy_process_group()
            self.mock_dist.__exit__(exc_type, exc_val, exc_tb)