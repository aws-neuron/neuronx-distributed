"""NKI implementation of the forward and backward SSM kernels for a Mamba2 model.

Paper: https://arxiv.org/abs/2405.21060
"""

import torch
import neuronxcc.nki.language as nl
from torch_neuronx import nki_jit
import neuronxcc.nki.isa as nisa


def compute_a_factors(logA, ones_triu, chunk_size, d_state, d_head=None):
    dtype = logA.dtype
    i_p = nl.arange(chunk_size)[:, None]
    if d_head is None:
        d_head = chunk_size
    i_p_head = nl.arange(d_head)[:, None]
    i_f = nl.arange(chunk_size)[None, :]

    # we reuse this bcast later in the code, ensure it is large enough for all uses
    bcast_size = max(chunk_size, d_state)

    logA_bcast = logA.broadcast_to((chunk_size, bcast_size))
    l_bcast = nl.matmul(ones_triu, logA_bcast, transpose_x=True)
    l = l_bcast[:chunk_size, 0]
    l_t = nl.transpose(l_bcast, dtype=dtype)

    # === compute the _transpose_ of the 128x128 lower triangular matrix L ===
    partial_sums = l_t[:chunk_size, :chunk_size] - l
    L_full_t = nl.exp(partial_sums)
    L_t = nisa.affine_select(i_f >= i_p, L_full_t, 0)

    a_right = L_t[i_p, chunk_size - 1]
    a_left = nl.exp(l_t[i_p_head, i_f])

    if d_head != chunk_size:
        a_center_t = a_left[:, chunk_size - 1].broadcast_to((d_head, bcast_size))
        a_center = nl.transpose(a_center_t, dtype=dtype)
    else:
        a_center = a_left[:, chunk_size - 1]

    return L_t, a_left, a_center, a_right


def compute_chunk_output(BC_t, L_t, X, C_t, a_left, S,
                         transpose_gate=False,
                         transpose_broadcast=False):
    """Utility function for computing the output, shared between forward and backward"""
    # Diagonal term computation
    M_diag_t = L_t * BC_t
    Y_diag = nl.matmul(M_diag_t, X, transpose_x=not transpose_gate)
    # Compute the off-diagonal contribution using the state
    barC_t = C_t * a_left
    # utility function for
    Y_off = nl.matmul(barC_t, S, transpose_x=not transpose_broadcast)
    return Y_diag + Y_off


@nki_jit
def mamba_kernel_(dt_tensor, logA_tensor, X_tensor, B_tensor, C_tensor, out_Y_tensor, D_tensor=None):
    """
    dt_tensor: (batch_size, n_heads, seq_len)
    logA_tensor: (n_heads)
    X_tensor: (batch_size, seq_len, n_heads, d_head)
    B_tensor: (batch_size, seq_len, n_groups, d_state)
    C_tensor: (batch_size, seq_len, n_groups, d_state)
    D_tensor: (n_heads)
    """
    # Since this kernel requires high-precision, we run all internal computations in fp32.
    # Note: the speedup by using bf16 everywhere would be ~15%
    dtype = nl.float32
    block_size = 128  # we will split seq_len in chunks of size `block_size`

    batch_size, seq_len, n_heads, d_head = X_tensor.shape
    _, _, n_groups, d_state = B_tensor.shape
    assert seq_len % block_size == 0
    n_chunks = seq_len // block_size
    n_heads_per_group = n_heads // n_groups

    batch_id = nl.program_id(0)

    i_p = nl.arange(block_size)[:, None]
    i_f = nl.arange(block_size)[None, :]
    i_f_state = nl.arange(d_state)[None, :]
    i_f_head = nl.arange(d_head)[None, :]

    # creates a constant upper triangular matrix of ones
    ones_triu = nisa.affine_select(i_p <= i_f, nl.ones((block_size, block_size), dtype=dtype), 0)

    for group_id in nl.affine_range(n_groups):  # parallel for loop over each group (they are completely independent)
        # === Preload/compute logA, B, C_t and B @ C_t ====
        # (they are shared between multiple heads in the same group)
        B_cache = nl.ndarray((block_size, n_chunks, d_state), dtype=dtype)
        # todo: storing in this format may be a bad idea if d_state != 128?
        C_t_cache = nl.ndarray((d_state, n_chunks, block_size), dtype=dtype)
        BC_t_cache = nl.ndarray((block_size, n_chunks, block_size), dtype=dtype)
        for chunk_id in nl.affine_range(n_chunks):
            i_p_in = i_p + chunk_id * block_size
            B = nl.load(B_tensor[batch_id, i_p_in, group_id, i_f_state], dtype=dtype)
            C = nl.load(C_tensor[batch_id, i_p_in, group_id, i_f_state], dtype=dtype)
            C_t = nisa.nc_transpose(C)
            B_cache[:, chunk_id, :] = B
            C_t_cache[:, chunk_id, :] = C_t
            BC_t_cache[:, chunk_id, :] = nl.copy(nl.matmul(B, C_t), dtype=dtype)

        logA_cache = nl.load(logA_tensor.reshape((1, n_heads)), dtype=dtype).broadcast_to((block_size, n_heads))
        if D_tensor is not None:
            D = nl.load(D_tensor.reshape((1, n_heads)), dtype=dtype).broadcast_to((block_size, n_heads))
        else:
            D = None

        # == Actual code ===
        for head_id_in_group in nl.affine_range(n_heads_per_group):  # parallel for loop over the n_heads
            # get the global head_id given current group and current head in group
            head_id = group_id * n_heads_per_group + head_id_in_group
            # We iterate over the diagonal blocks and compute each Y_diag
            # At the same time, we update our running sum S and use it to compute Y_off.
            # We store Y = Y_diag + Y_off, and we move to the next block
            S = nl.zeros((d_state, d_head), dtype=dtype)
            for chunk_id in nl.sequential_range(n_chunks):
                i_p_in = i_p + chunk_id * block_size

                # broadcast dt and logA together
                dt = nl.load(dt_tensor[batch_id, i_p_in, head_id], dtype=dtype)
                logA = logA_cache[:, head_id] * dt

                # load from cache the relevant blocks
                B = B_cache[:, chunk_id, :]
                C_t = C_t_cache[:, chunk_id, :]
                BC_t = BC_t_cache[:, chunk_id, :]

                # broadcast X and dt
                X0 = nl.load(X_tensor[batch_id, i_p_in, head_id, i_f_head], dtype=dtype)
                X = dt * X0

                # Compute all logA related factors for this chunk
                L_t, a_left, a_center, a_right = compute_a_factors(logA, ones_triu, block_size, d_state)

                Y = compute_chunk_output(BC_t, L_t, X, C_t, a_left, S)

                # Update running sum S (will be used in the next iteration)
                barB = B * a_right
                barBX = nl.matmul(barB, X, transpose_x=True)
                S[...] = a_center * S + barBX

                if D is not None:
                    Y[...] = Y + D[:, head_id] * X0

                nl.store(out_Y_tensor[batch_id, i_p_in, head_id, i_f_head], Y)


@nki_jit
def mamba_kernel_bwd_(dt_tensor, logA_tensor, X_tensor, B_tensor, C_tensor,
                      d_out_tensor,
                      ddt_tensor,
                      dlogA_tensor,
                      dX_tensor,
                      dB_tensor,
                      dC_tensor,
                      D_tensor=None,
                      dD_tensor=None
                      ):
    """
    dt_tensor: (batch_size, seq_len, n_heads)
    logA_tensor: (n_heads)
    X_tensor: (batch_size, seq_len, n_heads, d_head)
    B_tensor: (batch_size, seq_len, n_groups, d_state)
    C_tensor: (batch_size, seq_len, n_groups, d_state)
    D_tensor: (n_heads)
    d_out_tensor: (batch_size, seq_len, n_heads, d_head)
    All other derivative tensors (d_*) have the same shape as their corresponding input counterparts.
    """

    # Note: since saving the intermediate results of the forward pass would use too much memory, this kernel also
    # recomputes the forward pass while computing the gradients.

    # Since this kernel requires high-precision, we run all internal computations in fp32.
    # Note: the speedup by using bf16 everywhere would be ~15%
    dtype = nl.float32
    block_size = 128  # we will split seq_len in chunks of size `block_size`
    batch_size, seq_len, n_heads, d_head = X_tensor.shape
    _, _, n_groups, d_state = B_tensor.shape
    assert seq_len % block_size == 0
    n_chunks = seq_len // block_size
    n_heads_per_group = n_heads // n_groups

    assert d_state == 128
    assert d_head <= 128
    assert block_size <= 128

    batch_id = nl.program_id(0)

    i_p = nl.arange(block_size)[:, None]
    i_f = nl.arange(block_size)[None, :]
    i_f_state = nl.arange(d_state)[None, :]
    i_f_head = nl.arange(d_head)[None, :]

    # upper triangular matrix of ones
    ones_triu = nisa.affine_select(i_p <= i_f, nl.ones((block_size, block_size), dtype=dtype), 0)
    ones_tril = nl.copy(nl.transpose(ones_triu), dtype=dtype)
    ones_sum_right = nl.ones([d_state, 1], dtype=dtype)
    ones_sum_left = nl.ones([1, d_state], dtype=dtype)
    ones_sum_right_head = nl.ones([d_head, 1], dtype=dtype)

    for group_id in nl.affine_range(n_groups):  # iterate in parallel over all channel groups (they are independent)
        # Preload/compute logA, B, C_t and B @ C_t (which are shared between multiple heads in the same group)
        B_cache = nl.ndarray((block_size, n_chunks, d_state), dtype=dtype)
        C_t_cache = nl.ndarray((d_state, n_chunks, block_size), dtype=dtype)
        BC_t_cache = nl.ndarray((block_size, n_chunks, block_size), dtype=dtype)
        for chunk_id in nl.affine_range(n_chunks):
            i_p_in = i_p + chunk_id * block_size
            B = nl.load(B_tensor[batch_id, i_p_in, group_id, i_f_state], dtype=dtype)
            C = nl.load(C_tensor[batch_id, i_p_in, group_id, i_f_state], dtype=dtype)
            C_t = nisa.nc_transpose(C)
            B_cache[:, chunk_id, :] = B
            C_t_cache[:, chunk_id, :] = C_t
            BC_t_cache[:, chunk_id, :] = nl.copy(nl.matmul(B, C_t), dtype=dtype)

        logA_cache = nl.load(logA_tensor.reshape((1, n_heads)), dtype=dtype).broadcast_to((block_size, n_heads))
        if D_tensor is not None:
            D = nl.load(D_tensor.reshape((1, n_heads)), dtype=dtype).broadcast_to((block_size, n_heads))
        else:
            D = None

        dC_accumulation = nl.zeros((block_size, n_chunks, d_state), dtype=dtype)
        dB_accumulation = nl.zeros((block_size, n_chunks, d_state), dtype=dtype)
        dA_final = nl.zeros((1, n_heads), dtype=dtype)
        if D is not None:
            dD_final = nl.zeros((1, n_heads), dtype=dtype)

        for head_id_in_group in nl.affine_range(n_heads_per_group):  # the n_heads are completely independent
            # get the global head_id given current group and current head in group
            head_id = group_id * n_heads_per_group + head_id_in_group
            dA_accumulation = nl.zeros((block_size, n_chunks, d_state), dtype=dtype)
            S = nl.zeros((d_state, d_head), dtype=dtype)
            for chunk_id in nl.sequential_range(n_chunks):
                # <forward pass>
                i_p_in = i_p + chunk_id * block_size
                # broadcast dt and logA together
                dt = nl.load(dt_tensor[batch_id, i_p_in, head_id])
                logA = logA_cache[:, head_id] * dt
                # Compute all logA related factors for this chunk
                L_t, a_left, a_center, a_right = compute_a_factors(logA, ones_triu, block_size, d_state, d_head=d_head)
                # load from cache the relevant blocks
                B = B_cache[:, chunk_id, :]
                C = nl.load(C_tensor[batch_id, i_p_in, group_id, i_f_state])
                # broadcast X and dt
                X0 = nl.load(X_tensor[batch_id, i_p_in, head_id, i_f_head])
                X = dt * X0
                # </forward pass>

                # compute dC gradient
                dO = nl.load(d_out_tensor[batch_id, i_p_in, head_id, i_f_head])
                dO_t = nisa.nc_transpose(dO)
                UdO_t = nl.matmul(X, dO_t)  # (B, L, nheads, hdim)
                S_t = nisa.nc_transpose(S)
                dC = compute_chunk_output(UdO_t, L_t, B, dO_t, a_left, S_t)

                # <forward pass>
                # Update the state: running sum S (will be used in the next iteration)
                barB = B * a_right
                barBX = nl.matmul(barB, X, transpose_x=True)
                S[...] = a_center * S + barBX
                # </forward pass>
                dC_accumulation[:, chunk_id, :] += dC
                dA_accumulation[:, chunk_id, :] = dA_accumulation[:, chunk_id, :] + C * dC

            dS = nl.zeros((d_state, d_head), dtype=dtype)
            cumsum_dA = nl.zeros((1, d_state), dtype=dtype)
            for chunk_id in nl.sequential_range(n_chunks):
                chunk_id = n_chunks - 1 - chunk_id  # To reverse time
                i_p_in = i_p + chunk_id * block_size

                # === Recompute forward pass ===
                # broadcast dt and logA together
                dt = nl.load(dt_tensor[batch_id, i_p_in, head_id])
                logA = logA_cache[:, head_id] * dt
                # Compute all logA related factors for this chunk
                L_t, a_left, a_center, a_right = compute_a_factors(logA, ones_triu, block_size, d_state, d_head=d_head)
                # load from cache the relevant blocks
                B = B_cache[:, chunk_id, :]
                C = nl.load(C_tensor[batch_id, i_p_in, group_id, i_f_state])
                C_t = nisa.nc_transpose(C)
                BC_t = BC_t_cache[:, chunk_id, :]
                # broadcast X and dt
                X0 = nl.load(X_tensor[batch_id, i_p_in, head_id, i_f_head])
                X = dt * X0

                # === Compute dX gradient ===
                dO = nl.load(d_out_tensor[batch_id, i_p_in, head_id, i_f_head])
                dU = compute_chunk_output(BC_t, L_t, dO, B, a_right, dS, transpose_gate=True, transpose_broadcast=True)

                # === Compute dB gradient ===
                X_t = nisa.nc_transpose(X)
                dO_Xt = nl.matmul(dO, X_t)
                L_t_ = nisa.nc_transpose(L_t)
                dS_t = nisa.nc_transpose(dS)

                C = nl.load(C_tensor[batch_id, i_p_in, group_id, i_f_state], dtype=dtype)

                # === Compute dB gradient ===
                dB = compute_chunk_output(dO_Xt, L_t_, C, X, a_right, dS_t, transpose_broadcast=True)

                # === Update reverse time state dState ===
                # Update the state: running sum dS (will be used in the next iteration)
                barC = C_t * a_left[:1, :].broadcast_to((block_size, block_size))

                barC_tX = nl.matmul(barC, dO, transpose_x=False)
                dS[...] = a_center * dS + barC_tX

                dB_accumulation[:, chunk_id, :] += dB
                dA_accumulation[:, chunk_id, :] -= B * dB

                # === Reverse cumulative sum for dA ===
                cumsum_chunk = nl.matmul(ones_tril, dA_accumulation[:, chunk_id, :], transpose_x=True)
                cumsum_chunk[...] = cumsum_chunk + nl.copy(cumsum_dA, dtype=dtype).broadcast_to((block_size, d_state))
                cumsum_dA[0, i_f_state] = cumsum_chunk[0, i_f_state]

                ddt = nl.matmul(cumsum_chunk * logA_cache[:, head_id], ones_sum_right) + nl.matmul(dU * X0,
                                                                                                   ones_sum_right_head)

                dA_chunk = nl.matmul(cumsum_chunk * dt, ones_sum_right)
                dA_final[:, head_id] += nl.matmul(ones_sum_left, dA_chunk)

                dX = dU * dt

                if D is not None:
                    dD_chunk = nl.matmul(dO * X0, ones_sum_right_head)
                    dD_final[:, head_id] += nl.copy(nl.matmul(ones_sum_left, dD_chunk), dtype=dtype)
                    dX[...] = dX + dO * D[:, head_id]

                nl.store(dX_tensor[batch_id, i_p_in, head_id, i_f_head], dX)
                nl.store(ddt_tensor[batch_id, i_p_in, head_id], ddt)

            nl.store(dlogA_tensor[batch_id, head_id], dA_final[0, head_id])
            if D is not None:
                nl.store(dD_tensor[batch_id, head_id], dD_final[0, head_id])

        for chunk_id in nl.sequential_range(n_chunks):
            i_p_in = i_p + chunk_id * block_size
            nl.store(dC_tensor[batch_id, i_p_in, group_id, i_f_state], dC_accumulation[:, chunk_id, :])
            nl.store(dB_tensor[batch_id, i_p_in, group_id, i_f_state], dB_accumulation[:, chunk_id, :])


class Mamba2Kernel(torch.autograd.Function):
    """Define the autograd function wih forward and backward kernel."""

    @staticmethod
    def forward(ctx, dt, A, X, B, C, D):
        batch_size, seq_len, n_heads, d_head = X.shape
        ctx.save_for_backward(dt, A, X, B, C, D)
        out_Y = torch.empty_like(X)
        mamba_kernel_[batch_size](dt, A, X, B, C, out_Y, D)
        return out_Y

    @staticmethod
    def backward(ctx, d_output):
        dt, A, X, B, C, D = ctx.saved_tensors
        batch_size, seq_len, n_heads, d_head = X.shape

        ddt = torch.empty_like(dt)
        dA = torch.empty_like(A.unsqueeze(0).repeat(batch_size, 1))
        dX = torch.empty_like(X)
        dB = torch.empty_like(B)
        dC = torch.empty_like(C)
        dD = torch.empty_like(D.unsqueeze(0).repeat(batch_size, 1))

        mamba_kernel_bwd_[batch_size](dt, A, X, B, C, d_output,
                                      ddt, dA, dX, dB, dC, D, dD)
        dA, dD = dA.sum(0), dD.sum(0)
        return ddt, dA, dX, dB, dC, dD


def mamba2_kernel(dt, A, X, B, C, D):
    return Mamba2Kernel.apply(dt, A, X, B, C, D)
