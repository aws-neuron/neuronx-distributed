import torch
from modules.flashdecode.util import mask_util


def verify(res, expected):
    active, prior = res
    expected_active, expected_prior = expected
    assert torch.equal(expected_active, active)
    assert torch.equal(expected_prior, prior)


def test_decoding(pos_ids, rank_id, num_cores_per_group, num_pos_per_core, expected):

    verify(mask_util(pos_ids=pos_ids, rank_id=rank_id,
                     num_cores_per_group=num_cores_per_group, num_pos_per_core=num_pos_per_core), expected)


if __name__ == '__main__':

    #################### BATCH_SZ = 1 ####################

    expected_active_mask = torch.tensor([[0]], dtype=torch.int32)
    expected_prior_mask = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
    cache_ids = torch.tensor([[6]], dtype=torch.int32)
    core_id = torch.tensor([19], dtype=torch.int32)
    test_decoding(pos_ids=cache_ids, rank_id=core_id, num_cores_per_group=4, num_pos_per_core=8,
                  expected=(expected_active_mask, expected_prior_mask))


    #################### BATCH_SZ = 3 ####################

    expected_active_mask = torch.tensor([[0], [1], [1]], dtype=torch.int32)
    expected_prior_mask = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.int32)
    cache_ids = torch.tensor([[5], [4], [10]], dtype=torch.int32)
    core_id = torch.tensor([0], dtype=torch.int32)
    test_decoding(pos_ids=cache_ids, rank_id=core_id, num_cores_per_group=2, num_pos_per_core=8,
                  expected=(expected_active_mask, expected_prior_mask))


    expected_active_mask = torch.tensor([[1], [0], [0]], dtype=torch.int32)
    expected_prior_mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0],
                                        [1, 1, 0, 0, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.int32)

    core_id = torch.tensor([1], dtype=torch.int32)
    test_decoding(pos_ids=cache_ids, rank_id=core_id, num_cores_per_group=2, num_pos_per_core=8,
                  expected=(expected_active_mask, expected_prior_mask))