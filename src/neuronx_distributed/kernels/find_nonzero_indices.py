import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki._private.private_api import inline_asm_bytes


def nki_asm_nonzero_with_count(input_local, index_offset, padding_val=-1):
    '''
    Wrapper for the NonzeroWithCount ISA through inline asm.

    Args:
        input_local: SBUF tensor of shape (128, 1, T)
            Only data in partitions 0, 16, 32, ..., 112 are processed. 
        index_offset: offset at which the index starts counting
        padding_val: padding vaue to fill up spaces that do not have indices found.
    Returns
        output_local: SBUF tensor of shape (128, 1, T+1)
            Valid data also resides only in partitions 0, 16, 32, ..., 112, within a partition, assuming there are `N` nonzeros:
                The first `N` values in the output array contains the input array indices of the non-zero values, plus index_offset
                The next `T-N` values in the output array contain the same value of padding_value.
                The last value in the output array is a count of non-zero input values, exactly `N`.
    '''
    # INT32 = 0x8, FP32 = 0xa
    # Input can be either INT32 or FP32
    input_tpb_dtype = '8' if input_local.dtype==nl.int32 else 'a'
    # Output must always be INT32
    output_tpb_dtype = '8'
    # Convert `index_offset` and `padding_val` to little endian hex string
    index_offset_le_hex_str = '_'.join([f'{b:02x}' for b in index_offset.to_bytes(4, byteorder='big')[::-1]])
    padding_val_le_hex_str = '_'.join([f'{b:02x}' for b in (padding_val & 0xFFFFFFFF).to_bytes(4, byteorder='big')[::-1]])
    inline_bytes = (
            f"f2_10_00_00_{{events:NEURON_ISA_TBP_EVENTS:DATAPATH}}_80_0{input_tpb_dtype}_0{output_tpb_dtype}_"
            "00_{srcs[0]:NEURON_ISA_TPB_TENSOR3D}_"
            f"{index_offset_le_hex_str}_00_00_00_00_"
            f"{padding_val_le_hex_str}_"
            "{dsts[0]:NEURON_ISA_TPB_TENSOR3D}_00_00_00_00"
            )
    n_partitions, _, T = input_local.shape
    output_local = nl.ndarray((n_partitions, 1, T+1), dtype=nl.int32)
    inline_asm_bytes(dsts=[output_local], srcs=[input_local], engine=nisa.gpsimd_engine, asm_bytes=inline_bytes)

    return output_local


@nki.jit
def find_nonzero_indices(input_tensor: nt.tensor,
                 row_start_id: nt.tensor = None,
                 n_rows=None,
                 chunk_size=None,
                 index_dtype=nl.int32,):
    '''
    Index calculation kernel computes the indices of nonzero elements.
    For an `input_tensor` of shape [T,E]. It finds the indices along the T dimension.
    This is an LNC2 kernel that shards the E dimension.
    
    Args:
        input_tensor: HBM tensor of shape [T,E].
        row_start_id: HBM tensor containing 1 integer: the row index to start processing
            If unspecified, the entirety of `input_tensor` will be processed.
            If specified, `n_rows` must be specified too.
        n_rows: int, number of rows to process. This together with row_start_id enables processing of a chunk.
            If `row_start_id` is unspecified, this field is ignored.
            If `row_start_id` is specified, this field must be specified.
        chunk_size: size of the chunk if we process the T dimension in chunks.
            If unspecified, the chunk size is set to `T`.
            For longer sequence lengths, chunk_size will need to be set to avoid OOM on the free dimension.
        index_dtype: dtype of the output indices.
    
    Returns:
        indices: HBM tensor of shape [E,T], dtype `index_dtype`. For any given `e`, assume there are `N` nonzero values:
            The first `N` values in the output row, contains the input array indices of the nonzero values along dim T.
            The rest `T-N` values in the output array are -1s.
        nonzero_counts: HBM tensor of shape [E,], dtype int32. This consists the number of nonzero values for each `e`.
    '''
    # 1. Setup
    T, E = input_tensor.shape

    if row_start_id is None:
        row_start_id = 0
    else:
        row_start_id = nl.load(row_start_id).reshape((1,1))
        assert n_rows is not None, "Must specify n_rows when row_start_id is specified"
        E = n_rows

    num_shards = nl.num_programs(0)
    shard_id = nl.program_id(0)
    assert num_shards == 2, "Expect kernel to run with LNC2"
    assert E % num_shards == 0, f"Expect number of rows {E} is shardable by {num_shards}"
    
    if chunk_size is None:
        chunk_size = T
    else:
        assert T % chunk_size == 0, f"{T=} is not divisible by {chunk_size=}"

    # The 2 NCs shard the work along the expert dimension.
    E_per_shard = E // num_shards
    E_offset = E_per_shard * shard_id

    # Constants
    n_partitions = nl.tile_size.pmax
    quadrant=32
    n_quadrant = 4
    n_gpsimd_cores = 8
    n_gpsimd_cores_per_quadrant = 2
    n_partition_per_gpsimd_core = 16

    # Initialize (E,T) output on HBM to all -1s.
    indices = nl.ndarray((E, T), dtype=index_dtype, buffer=nl.shared_hbm)
    if chunk_size < T:
        sbuf_init = nl.full((n_partitions, E_per_shard*T // n_partitions), -1, dtype=index_dtype, buffer=nl.sbuf)
        reshaped_dst = indices.reshape((n_partitions * 2, E_per_shard*T // n_partitions))
        nl.store(reshaped_dst[nl.ds(n_partitions * shard_id, n_partitions), :], sbuf_init)

    # Keep counts in int32.
    nonzero_counts = nl.ndarray((E, ), dtype=nl.int32, buffer=nl.shared_hbm)
    nonzero_counts_local = nl.zeros((1, E_per_shard), dtype=nl.int32, buffer=nl.sbuf)

    # 2. Handle experts in groups of 8: 8 GPSIMD cores run in parallel.
    # Number of rounds to find index in groups of 8.
    n_rounds = (E_per_shard + n_gpsimd_cores - 1) // n_gpsimd_cores
    # Number of chunks to chunk along the T dim.
    n_T_chunks = T // chunk_size
    for r in nl.static_range(n_rounds):
        # Get number of experts to process this round.
        n_e = n_gpsimd_cores
        if r == n_rounds - 1:
            n_e = E_per_shard - n_gpsimd_cores * r

        # Counts of nonzeros so far. This is the offset at which the next chunk write should begin.
        # Keep offsets in int32.
        offsets = nl.zeros((1, n_gpsimd_cores), dtype=nl.int32, buffer=nl.sbuf)

        # Handle sequences in chunks with chunk_size
        for c in nl.static_range(n_T_chunks):
            # Load input: (T,E) layout on HBM
            # GPSIMD kernel needs: (128, chunk_size) with partition 0, 16, 32, ... each filled with a different expert.
            input_local_gpsimd_core_aligned = nl.ndarray((n_partitions, 1, chunk_size), dtype=input_tensor.dtype, buffer=nl.sbuf)
            n_tiles = chunk_size // n_partitions
            for i in nl.affine_range(n_tiles):
                # Read (128, n_e) chunk
                i_packed_t = nl.arange(n_partitions)[:, None]
                i_packed_e = nl.arange(n_e)[None, :]
                offset = r * n_gpsimd_cores + row_start_id + E_offset
                input_local_te = nl.load(input_tensor[i_packed_t + c * chunk_size + i * n_partitions, i_packed_e + offset])

                # Copy to (128, 128) chunk, putting the n_e columns at column 0, 16, 32, ..., 112
                input_local_aligned_te = nl.ndarray((n_partitions, n_partitions), dtype=input_tensor.dtype, buffer=nl.sbuf)
                for q in nl.affine_range(n_e):
                    input_local_aligned_te[:, nl.ds(q * n_partition_per_gpsimd_core, 1)] = input_local_te[:, nl.ds(q, 1)]

                # Transpose so we have expert data at row 0, 16, 32, ..., 112
                input_local_gpsimd_core_aligned[:, 0, nl.ds(i*n_partitions, n_partitions)] = nisa.nc_transpose(input_local_aligned_te)

            # Run GPSIMD kernel for ISA NonzeroWithCount.
            output_local = nki_asm_nonzero_with_count(input_local_gpsimd_core_aligned, c*chunk_size)

            # Write out rows 0, 32, 64, 96
            i_0 = nl.arange(1)[:, None]
            i_1 = nl.arange(1)[None, :]
            i_indices = nl.arange(chunk_size)[None, :]
            for q in nl.affine_range(n_quadrant):
                nl.store(
                    indices[i_0 + E_offset + r * n_gpsimd_cores + q * n_gpsimd_cores_per_quadrant, i_indices + offsets[i_0, i_1 + q * n_gpsimd_cores_per_quadrant]],
                    value=output_local[nl.ds(q*quadrant, 1), 0, nl.ds(0, chunk_size)],
                    mask=i_0 + q * n_gpsimd_cores_per_quadrant < n_e
                )
                offsets[i_0, i_1 + q * n_gpsimd_cores_per_quadrant] = nl.add(
                    offsets[i_0, i_1 + q * n_gpsimd_cores_per_quadrant],
                    output_local[nl.ds(q*quadrant, 1), 0, nl.ds(chunk_size, 1)],
                    mask=i_0 + q * n_gpsimd_cores_per_quadrant < n_e
                )

            # Stream shuffle to move rows 16, 48, 80, 112 to rows 0, 32, 64, 96, and then write them out
            quad_mask = [255] * quadrant
            quad_mask[0] = n_partition_per_gpsimd_core
            nisa.nc_stream_shuffle(src=output_local, dst=output_local, shuffle_mask = quad_mask)
            for q in nl.affine_range(n_quadrant):
                nl.store(
                    indices[i_0 + E_offset + r * n_gpsimd_cores + q * n_gpsimd_cores_per_quadrant + 1, i_indices + offsets[i_0, i_1 + q * n_gpsimd_cores_per_quadrant + 1]],
                    value=output_local[nl.ds(q*quadrant, 1), 0, nl.ds(0, chunk_size)],
                    mask=i_0 + q * n_gpsimd_cores_per_quadrant + 1 < n_e
                )
                offsets[i_0, i_1 + q * n_gpsimd_cores_per_quadrant + 1] = nl.add(
                    offsets[i_0, i_1 + q * n_gpsimd_cores_per_quadrant + 1],
                    output_local[nl.ds(q*quadrant, 1), 0, nl.ds(chunk_size, 1)],
                    mask=i_0 + q * n_gpsimd_cores_per_quadrant + 1 < n_e
                )

        # Final offsets are the nonzero counts per expert.
        i_n_e = nl.arange(n_e)[None, :]
        nonzero_counts_local[i_0, i_n_e + r * n_gpsimd_cores] = offsets[i_0, i_n_e]

    nonzero_counts_reshape = nonzero_counts.reshape((1, E))
    nl.store(nonzero_counts_reshape[:, nl.ds(E_offset, E_per_shard)], nonzero_counts_local)
    return indices, nonzero_counts
