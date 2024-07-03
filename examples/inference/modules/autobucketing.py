from typing import List

import torch


def slice_rhs(tensor, bucket: int, max_idx: int, dim: int):
    tensor = torch.ops.aten.slice(tensor, dim, max_idx - bucket, max_idx, 1)
    return tensor


def slice_lhs(tensor, bucket: int, dim: int):
    tensor = torch.ops.aten.slice(tensor, dim, 0, bucket, 1)
    return tensor


@torch.jit.script
def token_generation_bk(tensors: List[torch.Tensor], buckets: torch.Tensor, padding_side: str):
    attention_mask = tensors[1]
    position_ids = tensors[2]
    seq_ids = tensors[3]

    # DO NOT USE argmax since we monkeypatch it, causing issues with torch.jit.script
    bucket_idx = torch.argmin(((buckets - position_ids) <= 0).to(torch.int))
    bucket = buckets[bucket_idx]

    if padding_side == "right":
        tensors[1] = slice_lhs(attention_mask, bucket, 1)
    else:
        tensors[1] = slice_rhs(attention_mask, bucket, buckets[-1], 1)

    return tensors, torch.tensor(bucket_idx).to(torch.int)


def get_token_generation_bk():
    return token_generation_bk


@torch.jit.script
def context_encoder_bk(tensors: List[torch.Tensor], buckets, padding_side: str):
    input_ids = tensors[0]

    position_idx = (input_ids > 0).sum(dim=1)[0]
    bucket_idx = torch.argmin(((buckets - position_idx) < 0).to(torch.int))
    bucket = buckets[bucket_idx]

    new_tensors = []
    if padding_side == "right":
        for i, tens in enumerate(tensors):
            if tens.shape[-1] == 1:
                new_tensors.append(tens)
            else:
                new_tensors.append(slice_lhs(tens, bucket, 1))
    else:
        max_idx = buckets[-1]
        for i, tens in enumerate(tensors):
            if i == len(tensors) - 1:
                new_tensors.append(tens)
            else:
                new_tensors.append(slice_rhs(tens, bucket, max_idx, 1))

    return new_tensors, torch.tensor(bucket_idx)


def get_context_encoder_bk():
    return context_encoder_bk


@torch.jit.script
def state_preprocessor(
    shapes_collection: List[List[List[int]]],
    states: List[torch.Tensor],
    bucket_idx_tensor: torch.Tensor,
    padding_side: str,
) -> List[torch.Tensor]:
    bucket_idx = torch.ops.aten.Int(bucket_idx_tensor)
    shapes = shapes_collection[bucket_idx]
    sliced_state_tensors = []
    for i in range(len(shapes)):
        expected_shape = shapes[i]
        state_tensor = states[i]
        state_tensor_shape = state_tensor.shape
        for j, npos in enumerate(expected_shape):
            state_tensor_dim_length = state_tensor_shape[j]
            if padding_side == "right":
                state_tensor = slice_lhs(state_tensor, npos, j)
            else:
                state_tensor = slice_rhs(state_tensor, npos, state_tensor_dim_length, j)
        sliced_state_tensors.append(state_tensor)
    return sliced_state_tensors


def get_state_preprocessor(layout="bsh"):
    return state_preprocessor
