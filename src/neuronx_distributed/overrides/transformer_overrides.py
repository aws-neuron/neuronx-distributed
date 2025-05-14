import torch

# overriding 
def rotate_half(x, flash_attn, transpose):
    """
    Rotates half the elements along a 
    
    Returns:
    torch.Tensor: Tensor with rotated elements
    """
    if flash_attn and transpose:
        x1 = x[:, :, : x.shape[-2] // 2, :]
        x2 = x[:, :, x.shape[-2] // 2 :, :]
        return torch.cat((-x2, x1), dim=-2)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Overriding function from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, flash_attn, transpose_nki_inputs=True):
    if position_ids is not None: # used for backward compatibility
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin.unsqueeze(1)
    if flash_attn and transpose_nki_inputs:
        cos = cos.permute(0, 1, 3, 2)  # [bs, 1, dim, seq_len]
        sin = sin.permute(0, 1, 3, 2)  # [bs, 1, dim, seq_len]
    q_embed = (q * cos) + (rotate_half(q, flash_attn, transpose_nki_inputs) * sin)
    k_embed = (k * cos) + (rotate_half(k, flash_attn, transpose_nki_inputs) * sin)
    return q_embed, k_embed