"""NKI implementation of a causal channel-wise 1d convolution.

This is a drop-in replacement of torch.nn.Conv1d when groups == in_channels == out_channels. It automatically
applies the right amount of left zero-padding to ensure the length of the output is the same as the input.
"""

from typing import Union

import torch

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from torch_neuronx import nki_jit
import neuronx_distributed.parallel_layers.parallel_state as ps

import numpy as np
import torch.nn as nn


def diag(w):
    """
    w: (128, 1)
    """
    m = w.shape[0]
    i_p = nl.arange(m)[:, None]
    i_f = nl.arange(m)[None, :]
    w_diag = nisa.affine_select(i_p == i_f, w.broadcast_to((m, m)), 0)
    return w_diag


def apply_activation(x, bias, activation: str):
    if activation is None:
        return x + bias if bias is not None else x
    elif activation == 'relu':
        return nisa.activation(nl.relu, x, bias=bias)
    elif activation == 'd_relu':
        assert bias is None
        return x >= 0
    elif activation == 'silu':
        z = x + bias if bias is not None else x
        return z * nisa.activation(nl.sigmoid, z)
    elif activation == 'd_silu':
        z = x + bias if bias is not None else x
        return nl.sigmoid(z) * (1 + z * (1 - nl.sigmoid(z)))
    elif activation == 'd_identity':
        return 1
    else:
        raise ValueError(f'Invalid activation {activation}')


def _conv_tile_tensor_e(data_tile, weight_tile, b_tile, dtype, activation=None):
    """Computes and returns the convolution of a data tile given weights and bias.

    Isolated from the rest of the code since we use it both for forward and backward.
    """
    p_size, n = data_tile.shape
    kernel_size = weight_tile.shape[1]

    conv = nl.zeros(shape=(p_size, n), dtype=dtype)

    chunk_size = n // kernel_size

    i_p = nl.arange(p_size)[:, None]

    for j in nl.affine_range(kernel_size):
        i_f_plus_j = j + nl.arange(chunk_size)[None, :] * kernel_size
        res = nl.zeros((p_size, chunk_size), dtype=nl.float32, buffer=nl.psum)
        for i in nl.affine_range(kernel_size):
            w_diag = diag(weight_tile[i_p, i])
            res += nisa.nc_matmul(w_diag, data_tile[i_p, i_f_plus_j + i])
        conv[i_p, i_f_plus_j] = apply_activation(res, bias=b_tile, activation=activation)

    return conv


def _conv_tile_scalar_e(data_tile, weight_tile, b_tile, dtype, activation=None):
    """Computes and returns the convolution of a data tile given weights and bias.

    Isolated from the rest of the code since we use it both for forward and backward.
    """
    p_size, n = data_tile.shape
    kernel_size = weight_tile.shape[1]

    conv = nl.ndarray(shape=(p_size, n), dtype=dtype)

    chunk_size = n // kernel_size

    i_p = nl.arange(p_size)[:, None]

    for j in nl.affine_range(kernel_size):
        i_f_plus_j = j + nl.arange(chunk_size)[None, :] * kernel_size
        res = nki.isa.tensor_scalar(data_tile[i_p, i_f_plus_j], op0=np.multiply,
                                    operand0=weight_tile[i_p, 0], dtype=dtype)
        for i in nl.static_range(1, kernel_size):
            res = res + nki.isa.tensor_scalar(data_tile[i_p, i_f_plus_j + i], op0=np.multiply,
                                              operand0=weight_tile[i_p, i],
                                              dtype=dtype)
        conv[i_p, i_f_plus_j] = apply_activation(res, bias=b_tile, activation=activation)
    return conv


# _conv_tile = _conv_tile_scalar_e
_conv_tile = _conv_tile_tensor_e


@nki_jit
def conv1d_grouped_kernel(input_data, w, b, output, activation=None):
    """NKI kernel to compute grouped 1d causal convolution, equivalent to:

        D, L = x.shape

        conv = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            bias=True,
            kernel_size=kernel_size,
            groups=D,
            padding=kernel_size - 1,
        )
        y = conv(x)[:, :L]

    Args:
      input_data: input tensor of shape [D,L]
      w: conv weights of shape [D, kernel_size]
      b: conv bias of shape [D]
      output: output tensor of shape [D, L]
    """

    batch_size, p, n = input_data.shape
    ch, _, ks = w.shape
    dtype = input_data.dtype

    # fixme: make the code work for any size
    assert p % 128 == 0 and ch == p and p == ch
    assert n % ks == 0 and n > ks  # check n is a multiple of kernel size
    assert ks == 4  # fixme: don't think this constrain is needed
    assert n <= 2048, "conv1d does not yet support sequence lengths larger than 2048"

    i_w = nl.arange(ks)[None, :]
    i_p = nl.arange(128)[:, None]
    i_y = nl.arange(n)[None, :]

    # Iterate over channel dimension then over batch dimension (so we load the weights only once for all samples)
    for k in nl.affine_range(input_data.shape[1] // 128):
        i_p_input = i_p + k * 128
        # weights and biases for current tile
        w_tile = nl.load(w.reshape((ch, ks))[i_p_input, i_w])
        b_tile = nl.load(b[i_p_input])

        for i_b in nl.affine_range(input_data.shape[0]):
            # Load with padding
            x = nl.zeros(shape=(128, n + ks - 1), dtype=dtype)
            x[i_p, ks - 1 + i_y] = nl.load(input_data[i_b, i_p_input, i_y])
            # run the convolution
            conv = _conv_tile(x, w_tile, b_tile, dtype, activation=activation)
            # The first positions contain the result of a zero padded window
            nl.store(output[i_b, i_p_input, i_y], value=conv[i_p, i_y])


@nki_jit
def conv1d_grouped_kernel_longseq(input_data, w, b, output, activation=None):
    """NKI kernel to compute grouped 1d causal convolution for sequences of all lengths by processing them in sub-sequences

    equivalent to:

        D, L = x.shape

        conv = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            bias=True,
            kernel_size=kernel_size,
            groups=D,
            padding=kernel_size - 1,
        )
        y = conv(x)[:, :L]

    Args:
      input_data: input tensor of shape [D,L]
      w: conv weights of shape [D, kernel_size]
      b: conv bias of shape [D]
      output: output tensor of shape [D, L]
    """

    _, channels, seq_len = input_data.shape
    ch, _, ks = w.shape
    dtype = input_data.dtype

    # fixme: make the code work for any size
    assert channels % 128 == 0 and ch == channels
    assert seq_len % ks == 0 and seq_len > ks  # check seq_len is a multiple of kernel size
    assert ks == 4  # fixme: don't think this constrain is needed

    i_w = nl.arange(ks)[None, :]
    i_p = nl.arange(128)[:, None]

    sub_seq_len = min(2048, seq_len)
    num_sub_seqs = (seq_len + sub_seq_len - 1) // sub_seq_len

    padded_len = sub_seq_len + ks - 1
    i_f_subseq = nl.arange(sub_seq_len)[None, :]
    i_f_subseq_padded = nl.arange(padded_len)[None, :]

    # Iterate over channel dimension then over batch dimension (so we load the weights only once for all samples)
    for k in nl.affine_range(input_data.shape[1] // 128):
        i_p_input = i_p + k * 128
        # weights and biases for current tile
        w_tile = nl.load(w.reshape((ch, ks))[i_p_input, i_w])
        b_tile = nl.load(b[i_p_input])

        for batch_id in nl.affine_range(input_data.shape[0]):

            for subseq_id in nl.affine_range(num_sub_seqs):
                i_f_subseq_padded_in = i_f_subseq_padded + subseq_id * sub_seq_len
                x = nl.zeros(shape=(128, padded_len), dtype=dtype)
                mask = (i_f_subseq_padded_in - (ks - 1) >= 0) & (i_f_subseq_padded_in - (ks - 1) < seq_len)
                x[i_p, i_f_subseq_padded] = nl.load(input_data[batch_id, i_p_input, i_f_subseq_padded_in - (ks - 1)],
                                                    mask=mask)
                # run the convolution
                conv = _conv_tile(x, w_tile, b_tile, dtype, activation=activation)
                mask_out = i_f_subseq + subseq_id * sub_seq_len < seq_len
                # store the result
                nl.store(output[batch_id, i_p_input, i_f_subseq + subseq_id * sub_seq_len], value=conv[i_p, i_f_subseq],
                         mask=mask_out)


@nki_jit
def conv1d_grouped_kernel_grad(input_data, w, d_output, d_input, d_w, d_b, activation=None):
    batch_size, p, n = input_data.shape
    ch, _, ks = w.shape
    dtype = input_data.dtype

    assert p % 128 == 0 and ch == p and p == ch
    assert n % ks == 0 and n > ks  # check n is a multiple of kernel size
    assert ks == 4

    i_p = nl.arange(128)[:, None]
    i_f_n = nl.arange(n)[None, :]
    i_f_w = nl.arange(ks)[None, :]
    seq_len = n + ks - 1
    i_f_seq_len = nl.arange(seq_len)[None, :]

    if activation is not None:
        d_activation = 'd_' + activation
    else:
        d_activation = 'd_identity'

    for chunk_id in nl.affine_range(input_data.shape[1] // 128):
        i_p_input = chunk_id * 128 + nl.arange(128)[:, None]
        w_tile = nl.load(w[i_p_input, 0, i_f_w])
        # we don't need the bias to compute gradients
        b_tile = None

        db_accumulation = nl.zeros([128, 1], dtype=dtype)
        dw_accumulation = nl.zeros([128, ks], dtype=dtype)

        for batch_id in nl.affine_range(input_data.shape[0]):
            # fixme: probably don't need to pad this
            x = nl.zeros(shape=(128, n + ks - 1), dtype=dtype)
            x[i_p, ks - 1 + i_f_n] = nl.load(input_data[batch_id, i_p_input, i_f_n])

            if activation is not None:
                preact_grad = _conv_tile(x, w_tile, b_tile, dtype, activation=d_activation)[i_p, i_f_n]
            else:
                preact_grad = 1
            dout_tile = nl.zeros(shape=(128, n + ks - 1), dtype=dtype)
            dout_tile[i_p, i_f_n] = preact_grad * nl.load(d_output[batch_id, i_p_input, i_f_n])

            # Compute db
            db_accumulation += nl.sum(dout_tile[i_p, i_f_n], axis=[1])

            # Compute d_input
            dout_reverse = nl.ndarray((128, seq_len), dtype=dtype)
            # fixme: we should simply index the tile with flipped indexes, no need for the copy
            #   but it will break down later as double indexing tile[i_p, i_f][i_p1, i_f1] is not supported
            dout_reverse[i_p, i_f_seq_len] = dout_tile[i_p, seq_len - 1 - i_f_seq_len]
            # dout_reverse = dout_tile[i_p, seq_len - 1 - i_f_seq_len]

            conv = _conv_tile(dout_reverse, w_tile, b_tile=None, dtype=dtype, activation=None)

            # We flip the result while storing
            nl.store(d_input[batch_id, i_p_input, i_f_n], conv[i_p, seq_len - ks - i_f_n])

            dw_batch = nl.ndarray((128, 4), dtype=dtype)
            # Compute dw
            for i in nl.static_range(ks):
                # todo: the vector engine should be able to execute both element-wise product and sum in one instruction
                dw_batch[i_p, i] = nl.sum(x[i_p, i + i_f_n] * dout_tile[i_p, i_f_n], axis=[1])
            dw_accumulation += dw_batch

        nl.store(d_b[i_p_input], db_accumulation[i_p, 0])
        nl.store(d_w[i_p_input, 0, i_f_w], dw_accumulation[i_p, i_f_w])


class GroupedConv1dNKI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, activation=None):
        # fixme: if output is too large we might avoid to store it and recomputed it during backprop
        output = torch.empty_like(input)
        # if input.shape[2] <= 2048:
        #     output = conv1d_grouped_kernel(input, weight, bias, output, activation=activation)
        # else:
        output = conv1d_grouped_kernel_longseq(input, weight, bias, output, activation=activation)
        ctx.save_for_backward(input, weight, bias)
        ctx.activation = activation
        return output

    @staticmethod
    def backward(ctx, d_output):
        input, weight, bias = ctx.saved_tensors
        dinput = torch.empty_like(input)
        dweight = torch.empty_like(weight)
        dbias = torch.empty_like(bias)
        if input.shape[2] > 2048:
            raise NotImplementedError('Gradient not implemented for conv1d with seq_len>2048')
        # dinput, dweight, dbias = conv1d_grouped_kernel_bwd(input, weight, bias, d_output)
        conv1d_grouped_kernel_grad(input, weight, d_output, dinput, dweight, dbias, activation=ctx.activation)
        return dinput, dweight, dbias, None


def nki_conv1d(input, weight, bias=None, activation=None):
    return GroupedConv1dNKI.apply(input, weight, bias, activation)


class ConvNKI(nn.Conv1d):
    """
    Custom layer implemented in NKI to compute efficiently a grouped convolution,
    equivalent to nn.Conv1d with groups == in_channels == out_channels.

    Parameters:
        input: (B_tensor, C_tensor, L)
        weight: (C_tensor, 1, kernel_size)
        bias: (C_tensor)
    Return:
        output: (B_tensor, C_tensor, L) Each input channel sequence input[b, c, :] is convolved with its own conv weight[c, 0, :].
                          The results are then stacked together.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: Union[str, int] = 0, dilation: int = 1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', device=None, dtype=None, activation=None) -> None:
        # We only support a very specific use case, check we are in it
        assert groups == in_channels, "NKI grouped conv kernel only supports groups == in_channels"
        assert padding == kernel_size - 1
        assert padding_mode == 'zeros'
        assert dilation == 1
        assert stride == 1
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
                         device, dtype)
        self.activation = activation
        self.parallel_split()

    def parallel_split(self):
        tp_rank = ps.get_tensor_model_parallel_rank()
        tp_size = ps.get_tensor_model_parallel_size()

        chunk = slice(self.out_channels // tp_size * tp_rank, self.out_channels // tp_size * (tp_rank + 1))
        self.weight.data = self.weight.data[chunk].detach().clone()
        self.bias.data = self.bias.data[chunk].detach().clone()
        self.in_channels = self.out_channels // tp_size
        self.out_channels = self.out_channels // tp_size

    def forward(self, input):
        return GroupedConv1dNKI.apply(input, self.weight, self.bias, self.activation)
