import os
import unittest

import torch

from unittest.mock import patch
from parameterized import parameterized
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPs

from . import utils_testing as ut

def validate_out_expert_mask(out_expert_mask, expert_start_ids, expert_end_ids, local_expert_indices, rank, num_experts):
    """
    Helper to validate the output expert mask to handle redundancy.
    """
    T, E = out_expert_mask.shape
    for t in range(T): # tokens
        explored = [False] * num_experts
        for e in range(E): # experts
            curr_mask = out_expert_mask[t][e]
            curr_expert_id = local_expert_indices[0][e]
            if explored[curr_expert_id]:
                assert not bool(curr_mask), f"Incorrect mask for token= {t} expert={e}"
                continue
            curr_expert_start_id = expert_start_ids[rank][curr_expert_id]
            curr_expert_end_id = expert_end_ids[rank][curr_expert_id]
            expected_mask = curr_expert_start_id<= t and t<=curr_expert_end_id
            assert bool(curr_mask) == expected_mask, f"Incorrect mask for token= {t} expert={e}"
            explored[curr_expert_id] = True


class TestMoeRedundancy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        @staticmethod
        def generate_mask_with_no_local_redundancy_cpu(mask):
            """CPU-compatible version using torch.cumsum"""
            mask_cumsum = torch.cumsum(mask, dim=1)
            mask_first_occurance = mask_cumsum == 1
            return torch.logical_and(mask_first_occurance, mask)
        
        # Monkey patch the method
        ExpertMLPs.generate_mask_with_no_local_redundancy = \
            generate_mask_with_no_local_redundancy_cpu

    def setUp(self):
        with patch.object(ExpertMLPs, '__init__', lambda self: None):
            self.generator = ExpertMLPs()

    @parameterized.expand([
        # No redundancy, EP=2, 4 experts
        [[[0, 0, 1, 1], [1, 1, 0, 0]], 64, [[0, 0, 0, 0], [0, 0, 64, 64]], [[-1, -1, 63, 63],[63, 63, 63, 63]]],
        [[[0, 0, 1, 1], [1, 1, 0, 0]], 63, [[0, 0, 0, 0], [0, 0, 63, 63]], [[-1, -1, 62, 62],[62, 62, 62, 62]]],
        # With redundancy, EP=2, 4 experts, E0 and E3 are redundant.
        [[[1, 1, 1, 0], [1, 0, 0, 2]], 64, [[0, 0, 0, 0], [32, 64, 64, 0]], [[31, 63, 63, -1], [63, 63, 63, 63]]],
        [[[1, 1, 1, 0], [1, 0, 0, 2]], 63, [[0, 0, 0, 0], [31, 63, 63, 0]], [[30, 62, 62, -1], [62, 62, 62, 62]]],
    ])
    def test_allocate_token_blocks(
        self,
        local_redudancy_degree,
        tokens,
        expected_start_indices,
        expected_end_indices
    ):
        redudancy_degree = torch.tensor(
            local_redudancy_degree,
            dtype=torch.int32)

        start_indices, end_indices = self.generator.allocate_token_blocks(
            redudancy_degree, tokens
        )
        expected_start_indices = torch.tensor(expected_start_indices, dtype=torch.int32)
        expected_end_indices = torch.tensor(expected_end_indices, dtype=torch.int32)

        ut.check_tensors(start_indices, expected_start_indices, atol=0, rtol=0)
        ut.check_tensors(end_indices, expected_end_indices, atol=0, rtol=0)

    @parameterized.expand([
        # No redundancy, EP=2, 4 experts
        [[[2,3]], [[0, 0, 0, 0], [0, 0, 64, 64]], [[-1, -1, 63, 63],[63, 63, 63, 63]], 4, [0]],
        [[[0,1]], [[0, 0, 0, 0], [0, 0, 63, 63]], [[-1, -1, 62, 62],[62, 62, 62, 62]], 4, [0]],
        # With redundancy, EP=2, 4 experts, E0 and E3 are redundant.
        [[[3, 0, 3]], [[0, 0, 0, 0], [32, 64, 64, 0]], [[31, 63, 63, -1], [63, 63, 63, 63]], 4, [1]],
        [[[3, 0, 3]], [[0, 0, 0, 0], [31, 63, 63, 0]], [[30, 62, 62, -1], [62, 62, 62, 62]], 4, [1]],
    ])
    def test_generate_local_expert_mask_with_redundancy(
        self,
        local_expert_indices,
        expert_start_ids,
        expert_end_ids,
        num_experts,
        rank,
    ):
        local_expert_indices_t = torch.tensor(local_expert_indices)
        expert_start_ids = torch.tensor(expert_start_ids)
        expert_end_ids = torch.tensor(expert_end_ids)
        num_experts = torch.tensor(num_experts)
        rank_t = torch.tensor(rank)
        T = 128
        local_expert_mask = torch.ones((T, local_expert_indices_t.shape[-1]))

        out_expert_mask = self.generator.generate_local_expert_mask_with_redundancy(
            local_expert_mask,
            local_expert_indices_t,
            expert_start_ids,
            expert_end_ids,
            num_experts,
            rank_t,
        )
        print(out_expert_mask)
        validate_out_expert_mask(out_expert_mask, expert_start_ids, expert_end_ids, local_expert_indices, rank[0], num_experts)


if __name__ == "__main__":
    unittest.main(verbosity=3, failfast=False)

