# Standard Library
import os
import unittest

# Third Party
import torch

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPs

from . import utils_testing as ut

if not torch.distributed.is_initialized():
    # Simulate torchrun (required because MoE uses parallel layers for TP)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group(backend="xla", init_method="env://")
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
)

class TestGetBlockwiseExpertAndTokenMapping(unittest.TestCase):
    def test_get_blockwise_expert_and_token_mapping(self):
        expert_mask = torch.tensor([
            [1, 0, 0],  # token 0 goes to expert 0
            [0, 1, 0],  # token 1 goes to expert 1
            [1, 0, 0],  # token 2 goes to expert 0
            [0, 0, 1],  # token 3 goes to expert 2
            [1, 0, 0],  # token 4 goes to expert 0
            [0, 1, 0],  # token 5 goes to expert 1
        ])
        expert_index = torch.tensor([
            [0], [1], [0], [2], [0], [1]
        ])
        # expert_affinities = torch.rand(seq_len, num_experts)
        # _, expert_index = torch.topk(expert_affinities, top_k)
        # expert_mask = ExpertMLPsV2.get_expert_mask(expert_index, num_experts)

        block_to_expert, token_position_to_id = ExpertMLPs.get_blockwise_expert_and_token_mapping(
            total_tokens=expert_mask.shape[0],
            num_blocks=4,
            expert_mask=expert_mask,
            block_size=2,
            device=expert_mask.device,
            enable_spmd_rank=False,
            spmd_rank=None,
            tensor_parallel_group=parallel_state.get_tensor_model_parallel_group(),
            expert_index=expert_index,
        )
        expected_block_to_expert = torch.tensor([0, 0, 1, 2])
        expected_token_position_to_id = torch.tensor([0, 2, 4, -1, 1, 5, 3, -1])
        ut.check_tensors(block_to_expert, expected_block_to_expert, atol=0, rtol=0)
        ut.check_tensors(token_position_to_id, expected_token_position_to_id, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=3, failfast=False)
