import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
import numpy as np
from neuronxcc.nki._private.private_api import sendrecv

@nki.jit
def indexed_flatten(input_tensor: nt.tensor,
                    f_len: int,
                    output_len: int,
                    row_offsets: nt.tensor,
                    row_offsets_start: nt.tensor = None,
                    padding_val: int=-1):
    '''
    This kernel performs the "indexed flatten" operation for the indices going into the blockwise MoE matmul kernel.
    For an `input_tensor` of shape [E, T], and a set of row_offsets.
    By reshaping the input tensor to [E, T//f_len, f_len], and the output tensor to [output_len//f_len, f_len]
    the kernel does the following logic:

    for e in E:
        write this row's [T//f_len, f_len] into the output tensor, at row_offset_local[e]

    Args:
        input_tensor: HBM tensor of shape [E,T].
        row_offsets: HBM tensor of shape [N,].
        row_offsets_start: HBM tensor containing 1 integer: the index at which to read `row_offsets`.
            i.e. the row_offsets for the `E` rows are row_offset_local=row_offsets[row_offsets_start:row_offsets_start+E].
        f_len: number of elements in each DMA copy in the free dimension.
        output_len: Length of the output array to write to.
            Note it's the kernel user's responsibility to make sure `output_len` is big enough to not result in OOB.

    Returns:
        flattened_array: flattened array of shape [output_len,]
    '''
    # 0. Input validation & setup.
    index_dtype = input_tensor.dtype
    n_partitions = nl.tile_size.pmax
    E, T = input_tensor.shape
    N, = row_offsets.shape

    assert output_len % n_partitions == 0, f"{output_len=} must be divisible by {n_partitions=}"
    assert output_len % f_len == 0, f"{output_len=} must be divisible by {f_len=}"
    assert T % f_len == 0, f"{T=} must be divisible by {f_len=}"
    assert (T // f_len) % 16 == 0, f"T // f_len must be divisible by 16 for DMAs to work. Got {T=}, {f_len=}"
    if row_offsets_start is None:
        assert N == E, f"When row_offsets_start is not specified, row_offsets size ({N=}) must match dim0 size of input_tensor ({E=})"
        row_offsets_start = 0
    else:
        assert N >= E, f"row_offsets size ({N=}) must be bigger than dim0 size of input_tensor ({E=})"
        row_offsets_start = nl.load(row_offsets_start).reshape((1,1))

    num_shards = nl.num_programs(0)
    shard_id = nl.program_id(0)
    assert num_shards == 2, "Expect kernel to run with LNC2"
    E_per_shard = E // num_shards
    E_offset = E_per_shard * shard_id

    # 1. Initialize input_tensor on HBM and fill it with `padding_val`.
    flattened_array_partial = nl.ndarray((output_len,), dtype=index_dtype, buffer=nl.hbm)
    sbuf_init = nl.full((n_partitions, output_len // n_partitions), padding_val, dtype=index_dtype, buffer=nl.sbuf)
    reshaped_partial_init = flattened_array_partial.reshape(sbuf_init.shape)
    nl.store(reshaped_partial_init, sbuf_init)
    
    # 2. Performs indexed flatten for the rows assigned to this NC.
    # Write using all lanes. Each write on the free dimension is `f_len`
    # The block size must be divisible by this number).
    n_partitions_per_row = T // f_len
    n_p_tiles = (n_partitions_per_row + n_partitions - 1) // n_partitions
    input_tensor_reshape = input_tensor.reshape((E, n_partitions_per_row, f_len))

    # Load offsets for each DMA copy.
    row_offsets_local = nl.ndarray((1, E_per_shard), dtype=nl.int32, buffer=nl.sbuf)
    i_0 = nl.arange(1)[:, None]
    i_e = nl.arange(E_per_shard)[None, :]
    off = row_offsets_start + E_offset
    row_offsets = row_offsets.reshape((1, N))
    row_offsets_local = nl.load(row_offsets[i_0, i_e + off])

    # DMA copies.
    reshaped_partial_dma = flattened_array_partial.reshape((output_len // f_len, f_len))
    for e in nl.affine_range(E_per_shard):
        for p in nl.affine_range(n_p_tiles):
            i_0 = nl.arange(n_partitions)[:, None]
            i_1 = nl.arange(f_len)[None, :]

            nisa.dma_copy(
                dst=reshaped_partial_dma[i_0 + p * n_partitions + row_offsets_local[:, nl.ds(e,1)], i_1],
                src=input_tensor_reshape[e + E_offset, i_0 + p * n_partitions, i_1],
                mask=i_0 < (n_partitions_per_row - p * n_partitions),
            )
    
    if E % num_shards != 0 and shard_id == 0:
        row_offset = nl.load(row_offsets[:, nl.ds(E-1, 1)])
        for p in nl.affine_range(n_p_tiles):
            i_0 = nl.arange(n_partitions)[:, None]
            i_1 = nl.arange(f_len)[None, :]

            nisa.dma_copy(
                dst=reshaped_partial_dma[i_0 + p * n_partitions + row_offset, i_1],
                src=input_tensor_reshape[E - 1, i_0 + p * n_partitions, i_1],
                mask=i_0 < (n_partitions_per_row - p * n_partitions),
            )

    # 3. Aggregate results from the two NCs.
    # All-reduce max the `flattened_array_partial`` between the two NCs.
    reshaped_reload = flattened_array_partial.reshape((n_partitions, output_len // n_partitions))
    reshaped_reload_local = nl.load(reshaped_reload)
    reshaped_reload_remote = nl.ndarray(reshaped_reload_local.shape, dtype=reshaped_reload_local.dtype)
    sendrecv(
        src=reshaped_reload_local,
        dst=reshaped_reload_remote,
        send_to_rank=(1-shard_id),
        recv_from_rank=(1-shard_id),
        pipe_id=0,
    )
    reshaped_reload = nisa.tensor_tensor(reshaped_reload_local, reshaped_reload_remote, op=np.maximum)
    
    # Write result to output
    flattened_array = nl.ndarray((output_len,), dtype=index_dtype, buffer=nl.shared_hbm)
    flattened_array_reshape = flattened_array.reshape((n_partitions, output_len // n_partitions))
    if shard_id == 0:
        nl.store(flattened_array_reshape, reshaped_reload)

    return flattened_array

