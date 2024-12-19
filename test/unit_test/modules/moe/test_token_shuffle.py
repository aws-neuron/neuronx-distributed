# Standard Library
import os
import unittest

# Third Party
import torch

from neuronx_distributed import parallel_layers
from neuronx_distributed.modules.moe import token_shuffling

from . import utils_testing as ut

if not torch.distributed.is_initialized():
    # Simulate torchrun (required because MoE uses parallel layers for TP)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group(backend="xla", init_method="env://")
parallel_layers.parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)
parallel_layers.parallel_state.initialize_token_shuffle_group(token_shuffle_group_size=1)
parallel_layers.random.model_parallel_xla_manual_seed(0)


class TestTokenShuffle(unittest.TestCase):
    def test_token_shuffle(self):
        hidden_states = torch.arange(2048).reshape(1, -1, 1)
        # print(f"{hidden_states=}")
        permuted_states, permutation = token_shuffling.token_shuffle(hidden_states)
        # print(f"{permuted_states=}")
        unpermuted_states = token_shuffling.token_unshuffle(permuted_states, permutation)
        # print(f"{unpermuted_states=}")
        ut.check_tensors(hidden_states, unpermuted_states, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=3, failfast=False)
