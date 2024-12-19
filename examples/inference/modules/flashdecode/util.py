import torch
from typing import Tuple


def turn_2d_mask_to_4d(attention_mask, n_positions, batch_size):
    return attention_mask[:, None, None, :].expand(batch_size, 1, 1, n_positions).to(torch.bool)


def mask_util(pos_ids: torch.Tensor, rank_id: torch.Tensor, num_cores_per_group: int, cache_size: int) -> (
        Tuple)[torch.Tensor, torch.Tensor]:
    """
    @:param pos_ids: 1d tensor represents position ids for all sequences in a batch
    @:param rank_id: current rank of the device
    @:return num_cores_per_group: number of cores per kv group
    @:param cache_size: size of the cache per core
    """

    batch_sz = pos_ids.shape[0]
    """ 
    Core layout: 32 cores on 8 kv group (col) and 4 cores in each group
    
        0, 1, 2, 3, 4, 5, 6, 7
        -----------------------
    0 | 0, 4, 8, 12, 16, 20, 24, 28
    1 | 1, 5, 9, 13, 17, 21, 25, 29
    2 | 2, 6, 10, 14, 18, 22, 26, 30
    3 | 3, 7, 11, 15, 19, 23, 27, 31
    
    for rank id == 19:
    the rank_id_in_kv_group (row index) is 3, derived by 19 % 4
    
    """
    rank_id = torch.remainder(rank_id, num_cores_per_group)

    # active masks: select only one core to update active KV
    selected_core_idx = torch.remainder(pos_ids, num_cores_per_group)
    active_masks = torch.where(selected_core_idx == rank_id, 1, 0).to(dtype=pos_ids.dtype)

    # prior masks: infer and update it
    """ 
    Cache layout within 1 kv group: 4 cores (row) and each has 8 positions (col), that is cache_size=8
    Note num of positions = bucket_sz//num_cores_per_kv_group
    
        0, 1, 2, 3, 4, 5, 6, 7
        -----------------------
    0 | 0, 4, 8, 12, 16, 20, 24, 28
    1 | 1, 5, 9, 13, 17, 21, 25, 29
    2 | 2, 6, 10, 14, 18, 22, 26, 30
    3 | 3, 7, 11, 15, 19, 23, 27, 31
    
    for pos_id = 19:
    the selected_pos for prior masks to be updated (col index) is 4, derived by 19 // 4
    
    """
    selected_pos = torch.div(pos_ids, num_cores_per_group, rounding_mode='floor')

    # init prior mask: set True from the start to the selected_pos, and the rest False
    position_ids_to_compare = selected_pos.expand(selected_pos.shape[0], cache_size)
    mask = torch.arange(cache_size, device=pos_ids.device).view(1, -1).expand(batch_sz, cache_size)
    prior_masks = torch.where(position_ids_to_compare >= mask, 1, 0).to(dtype=pos_ids.dtype)

    # adjust the selected_pos value
    selected_pos_val = torch.add(rank_id, num_cores_per_group * selected_pos)
    is_future_val = torch.where(selected_pos_val >= pos_ids, 1, 0)
    to_exclude = torch.logical_or(active_masks, is_future_val).to(dtype=pos_ids.dtype)
    adjusted_mask = torch.sub(1, to_exclude)
    prior_masks = prior_masks.scatter(dim=1, index=selected_pos.to(torch.int64), src=adjusted_mask)

    return active_masks, prior_masks
