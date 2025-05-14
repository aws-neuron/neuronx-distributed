import torch
from neuronx_distributed.parallel_layers import parallel_state

def shift_labels(batch):
    """
    Shift the labels by 1 from the front and pad the end with a padding token
    """
    label_pad_token_id: int = -100
    shifted_batch = {}
    for k, val in batch.items():
        if k=="labels": 
            val = val[:, 1:] # dimensions: [mbs, batch_size]. Shift 1 from the front of batch_size
            padding_value = torch.full((val.shape[0], 1), label_pad_token_id, device=val.device)
            val = torch.cat((val, padding_value), dim=1)
        shifted_batch[k] = val
    return shifted_batch

    
def get_batch_on_this_context_parallel_rank(batch):
    """
    Slice batch along sequence dimension for context parallelism
    """
    cp_size = parallel_state.get_context_model_parallel_size()
    cp_rank = parallel_state.get_context_model_parallel_rank()
    if not cp_size > 1:
        return batch  # Return the original batch if cp_size is not greater than 1
    batch = shift_labels(batch)
    seq_dim = 1
    cp_batch = {}
    for k, val in batch.items():
        if cp_size > 1 and val is not None and val.shape[seq_dim]>1:
            seq_len = val.shape[seq_dim]
            assert seq_len%cp_size==0, f"seq_len {seq_len} is not divisible by CP size {cp_size}"
            val = val.view(
                *val.shape[0:seq_dim],
                cp_size,
                seq_len // cp_size,
                *val.shape[(seq_dim + 1):],
            )
            index = torch.tensor([cp_rank], device=val.device)
            val = val.index_select(seq_dim, index)
            val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :]) 
        cp_batch[k] = val
    return cp_batch