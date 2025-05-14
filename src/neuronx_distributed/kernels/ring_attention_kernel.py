import torch
from neuronx_distributed.kernels.kernel_utils import get_seed, move_seed, permute, cast

# FIXME: not capturing runtimeerror
def _ring_attn_placeholder(*args, **kwargs):
    raise RuntimeError(
        "Ring Attention initialization failed!\n"
        "Please check and upgrade neuronx-cc and torch_neuronx.\n"
        "python3 -m pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com neuronx-cc torch_neuronx"
    )


try:
    from neuronxcc.nki._pre_prod_kernels.ring_attention import (
        ring_attention_bwd,
        ring_attention_fwd,
    )
    from neuronxcc.nki.kernels.attention import FlashConfig

    _ring_fwd_nki_call = ring_attention_fwd
    _ring_bwd_nki_call = ring_attention_bwd

except Exception:
    try:
        from neuronxcc.nki._private_kernels.attention import (
            ring_attention_bwd,
            ring_attention_fwd,
        )
        _ring_fwd_nki_call = ring_attention_fwd
        _ring_bwd_nki_call = ring_attention_bwd
    except Exception:
        _ring_fwd_nki_call = _ring_attn_placeholder
        _ring_bwd_nki_call = _ring_attn_placeholder

    

def get_seq_tile_size(seqlen: int) -> int:
    if seqlen >= 4096:
        seq_tile_size = 2048
    else:
        seq_tile_size = seqlen // 2
    seq_tile_size = max(seq_tile_size, 1024)
    return seq_tile_size

class NkiRingAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        rank_id,
        src_tgt_pairs,
        softmax_scale,
        causal: bool,
        mixed_precision: bool,
        seed,
        dropout_p: float,
    ):
        bs, num_heads, head_dim, seqlen = q.shape
        if softmax_scale is None:
            softmax_scale = q.shape[-2] ** (-0.5)

        if seed is None:
            seed = get_seed(dropout_p, q.device)

        seq_tile_size = get_seq_tile_size(seqlen)
        flash_config = FlashConfig(should_transpose_v=True, seq_tile_size=seq_tile_size)
        
        attn_output, lse = _ring_fwd_nki_call[bs, num_heads](
            q,
            k,
            v,
            seed,
            rank_id=rank_id,
            src_tgt_pairs=src_tgt_pairs,
            use_causal_mask=causal,
            mixed_precision=mixed_precision,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            config=flash_config,
        )

        ctx.save_for_backward(q, k, v, attn_output, lse, seed)
        ctx.rank_id = rank_id
        ctx.src_tgt_pairs = src_tgt_pairs
        ctx.causal = causal
        ctx.mixed_precision = mixed_precision
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale

        move_seed(dropout_p)
        return attn_output

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, attn_output, lse, seed = ctx.saved_tensors
        dout = dout.permute(0, 1, 3, 2)
        attn_output = attn_output.permute(0, 1, 3, 2)
        bs, num_heads, _, _ = q.shape
        dq, dk, dv = _ring_bwd_nki_call[bs, num_heads](
            q,
            k,
            v,
            attn_output,
            dout,
            lse,
            seed,
            rank_id=ctx.rank_id,
            src_tgt_pairs=ctx.src_tgt_pairs,
            use_causal_mask=ctx.causal,
            mixed_precision=ctx.mixed_precision,
            dropout_p=ctx.dropout_p,
            softmax_scale=ctx.softmax_scale,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def nki_ring_attn_func(
    q,
    k,
    v,
    rank_id,
    src_tgt_pairs,
    dropout_p=0.0,
    softmax_scale=None,
    causal=True,
    mixed_precision=True,
    seed=None,
):
    """
    Arguments:
        q: (batch_size, nheads, seqlen, headdim)
        k: (batch_size, nheads_k, seqlen, headdim)
        v: (batch_size, nheads_k, seqlen, headdim)
        rank_id: int. The current rank id
        src_tgt_pairs: List[]. The src to target pairs of the ring.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        mixed_precision: bool. Whether to enable the higher precisions on the softmax.
        seed: int32 torch.Tensor. The seed for the dropout.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    bs, num_heads, seqlen, head_dim = q.shape
    if seqlen % 1024 != 0:
        raise NotImplementedError(
            f"Only support sequence as multiples of 1K, got {seqlen}"
        )

    q, k, v = permute(q, k, v)
    q, k, v = cast(q, k, v)

    return NkiRingAttnFunc.apply(
        q,
        k,
        v,
        rank_id,
        src_tgt_pairs,
        softmax_scale,
        causal,
        mixed_precision,
        seed,
        dropout_p,
    )