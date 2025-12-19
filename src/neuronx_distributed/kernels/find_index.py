import warnings
import numpy as np

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt


@nki.jit
def count_nonzeros_shard_rows(
    mask: nt.tensor,
    row_start_id: nt.tensor=None,
    n_rows=None,
    boolean_mask: bool=True,
):
    '''
    Counts the number of nonzero elements along dim 1 of the 2D input mask.
    This is an LNC2 kernel that shards rows of the mask across the 2 NCs.

    Args:
        mask: HBM tensor of shape (E, T).
            This should contain 0s and 1s if boolean_mask=True.
            If boolean_mask=False, the kernel converts mask to 0/1.
        row_start_id: HBM tensor containing row index to start processing.
        n_rows: int, number of rows to process. This together with row_start_id enables processing of a chunk.
        boolean_mask: whether the input is already boolean. If False, the kernel converts the mask to 0/1.

    Returns:
        counts: int32 HBM tensor of shape (E,) or (n_rows,) if n_rows is specified.
    '''
    global_E, T = mask.shape

    # Update number of rows to process / row_start_id
    E = global_E
    if row_start_id is not None:
        assert n_rows is not None
        E = n_rows
        row_start_id = nl.load(row_start_id)[0,0]
    else:
        row_start_id = 0

    num_shards = nl.num_programs(0)
    shard_id = nl.program_id(0)
    print(num_shards, shard_id)
    assert E % num_shards == 0, f"Expect number of rows {E} is shardable by {num_shards}"

    E_per_shard = E // num_shards
    E_offset = E_per_shard * shard_id
    i_E_load = nl.arange(E_per_shard)[:, None] + row_start_id
    i_T_load = nl.arange(T)[None, :]

    mask_local = nl.ndarray((E_per_shard, T), dtype=mask.dtype, buffer=nl.sbuf)
    nisa.dma_copy(src=mask[i_E_load + E_offset, i_T_load], dst=mask_local, dge_mode=nisa.dge_mode.swdge)

    if not boolean_mask:
        # If input mask is not boolean, convert to 0/1
        mask_local = nisa.tensor_scalar(mask_local, nl.not_equal, 0)

    counts_local = nisa.tensor_reduce(nl.add, mask_local, axis=[1])

    counts = nl.ndarray((E,), dtype=nl.int32, buffer=nl.shared_hbm)

    reshaped_dst = counts.reshape((E, 1))
    i_E_write = nl.arange(E_per_shard)[:, None] + E_offset
    i_1_write = nl.arange(1)[None, :]
    nisa.dma_copy(src=counts_local, dst=reshaped_dst[i_E_write, i_1_write])

    return counts


@nki.jit
def find_index_shard_rows(
    mask: nt.tensor,
    n_values_max: int,
    n_values_min: int,
    row_start_id: nt.tensor=None,
    n_rows=None,
    boolean_mask: bool=True,
):
    '''
    Index calculation kernel computes the indices of nonzero elements along dim 1 of the 2D input mask.
    This is an LNC2 kernel that:
        - Across 2 NCs, shards rows of the mask.
        - Within each NC, performs index calculation for 1 row at a time:
            - Across 4 quadrants, shards the length of the row.
            - Across partitions within a quadrant, shards the values to find.

    Args:
        mask: HBM tensor of shape (E, T)
            This should contain 0s and 1s if boolean_mask=True.
            If boolean_mask=False, the kernel converts mask to 0/1.
        n_values_max: maximum number of nonzero elements to find per row in the mask.
            For dropping MoEs, set n_values_max = expert capacity.
            For dropless MoEs, set n_values_max = T.
        n_values_min: minimum number of nonzero elements to find per row in the mask (currently unused).
        row_start_id: HBM tensor containing row index to start processing.
        n_rows: int, number of rows to process. This together with row_start_id enables processing of a chunk.
        boolean_mask: whether the input is already boolean. If False, the kernel converts the mask to 0/1.

    Returns:
        true_indices: int32 HBM tensor of shape (E, n_values_max) or (n_rows, n_values_max) if n_rows is specified.
            The element at index (e, i) indicates the index of the i-th nonzero element in the e-th row.
            If a row has fewer than n_values_max 1s, the corresponding positions are filled with -1.
        true_counts: int32 tensor of shape (E,) or (n_rows,) if n_rows is specified.
            Actual true counts per row.
    '''
    global_E, T = mask.shape

    # Check conditions on T
    n_partitions = nl.tile_size.pmax
    quadrant = 32
    n_quadrants = n_partitions // quadrant
    assert T % n_quadrants == 0, f"Expect seqlen {T} is shardable by {n_quadrants}"
    assert T >= n_values_max, f"Invalid call with n_values_max {n_values_max} > T {T}"
    assert T <= 65536, f"Supports only T up to 64k, got T={T}"

    # Update number of rows to process / row_start_id
    E = global_E
    if row_start_id is not None:
        assert n_rows is not None
        E = n_rows
        row_start_id = nl.load(row_start_id)[0,0]
    else:
        row_start_id = 0
    
    # Check conditions on E
    num_shards = nl.num_programs(0)
    shard_id = nl.program_id(0)
    assert E % num_shards == 0, f"Expect number of rows {E} is shardable by {num_shards}"

    # Set up index load for E dim and T dim
    E_per_shard = E // num_shards
    E_offset = E_per_shard * shard_id
    i_E_load = nl.arange(1)[:, None] + E_offset + row_start_id
    T_per_quadrant = T // n_quadrants
    i_T_load = nl.arange(T_per_quadrant)[None, :]

    # Sequence lengths offset for each quadrant
    T_offset_np = np.repeat(np.arange(n_quadrants)*T_per_quadrant, quadrant).reshape(n_partitions, 1)
    T_offset = nl.shared_constant(T_offset_np, dtype=mask.dtype)
    T_offset_local = nl.load(T_offset)

    # Zeros as placeholder for cumsum with tensor_tensor_scan.
    unused_zeros = nl.zeros((n_partitions, T_per_quadrant), dtype=nl.float32, buffer=nl.sbuf)

    # Set up find8
    const8 = 8
    assert n_values_max % quadrant == 0, f"Expect n_values_max {n_values_max} is shardable by {quadrant}"
    values_per_partition = n_values_max // quadrant
    n_passes = int(np.ceil(values_per_partition/const8))

    # Output tensors in HBM
    true_indices = nl.ndarray((E, n_values_max), dtype=nl.int32, buffer=nl.shared_hbm)
    reshaped_dst = true_indices.reshape((E*quadrant, values_per_partition))
    true_counts = nl.ndarray((E,), dtype=nl.int32, buffer=nl.shared_hbm)

    # Process 1 row at a time, shard the sequence dimension on 4 quadrants.
    for e in nl.affine_range(E_per_shard):
        # Row has length T, load T/4 into each of the 4 quadrants.
        mask_local = nl.zeros((n_partitions, T_per_quadrant), dtype=mask.dtype, buffer=nl.sbuf)
        for q in nl.affine_range(n_quadrants):
            mask_local[nl.ds(q*quadrant, 1), :] = nl.load(mask[i_E_load+e, i_T_load + T_per_quadrant*q])

        # For all quadrants, stream shuffle to broadcast the mask from 1 partition to 32 partitions.
        quad_mask = [0] * quadrant
        # quad_mask[0] = 255
        nisa.nc_stream_shuffle(
            src=mask_local,
            dst=mask_local,
            shuffle_mask = quad_mask
        )

        if not boolean_mask:
            # If input mask is not boolean, convert to 0/1
            mask_local = nisa.tensor_scalar(mask_local, nl.not_equal, 0)

        # Cumsum
        # TODO: this cost is 2N cycles for N elements on free dim. Use the ISA whose cost is N cycles.
        # TODO: the 32 quadrants now performs the exact same computation. Maybe this can be optimized.
        cumulative_mask_local = nisa.tensor_tensor_scan(mask_local, unused_zeros, initial=0, op0=nl.add, op1=nl.add)

        # Make cumsum have the proper global results, by adding the previous quadrant's final sum to the later quadrants.
        subseq_cumsum = nl.ndarray((n_partitions, 1), dtype=mask.dtype, buffer=nl.sbuf)
        subseq_cumsum[nl.ds(0, quadrant)] = 0
        for q in nl.affine_range(n_quadrants - 1):
            subseq_cumsum[nl.ds((q+1)*quadrant, quadrant), :] = subseq_cumsum[nl.ds(q*quadrant, quadrant), :] + cumulative_mask_local[nl.ds(q*quadrant,quadrant), nl.ds(T_per_quadrant-1,1)]
        cumulative_mask_local[...] = nisa.tensor_scalar(cumulative_mask_local, nl.add, subseq_cumsum)

        final_sum = cumulative_mask_local[nl.ds(n_partitions-1, 1), nl.ds(T_per_quadrant-1,1)]
        nl.store(true_counts[nl.ds(E_offset + e, 1)], final_sum)

        # Find values
        # nc_find_index8 finds 8 values at a time.
        # Each partition within a quadrant finds for different values, because they have the same data.
        quadrant_find_pattern = (np.arange(1, const8+1))[None, :] + (np.arange(quadrant) * values_per_partition)[:, None]
        # Partitions with the same offset in the 4 quadrants find for the same values, because they have different data.
        full_find_pattern = nl.shared_constant(np.tile(quadrant_find_pattern, (n_quadrants,1)), dtype=cumulative_mask_local.dtype)
        full_find_pattern_local = nl.load(full_find_pattern)

        out_local = nl.ndarray((n_partitions, n_passes*const8), dtype=nl.uint32, buffer=nl.sbuf)
        # Issue with compiler where when n_passes=1, the loop doesn't work.
        if n_passes == 1:
            out_local[:, :]= nisa.nc_find_index8(data=cumulative_mask_local, vals = full_find_pattern_local)
        else:
            for i in nl.affine_range(n_passes):
                vals_to_find = full_find_pattern_local + i * const8
                out_local[:, nl.ds(const8 * i, const8)] = nisa.nc_find_index8(data=cumulative_mask_local, vals = vals_to_find)

        # Add offset to the local indices found in T/4 to be the global indices found in T.
        out_local = nisa.tensor_scalar(out_local[:, nl.ds(0, values_per_partition)], nl.add, T_offset_local, dtype=nl.uint32)

        # Reduce min on the indices found to aggregate the results to quadrant 0.
        # First reduce quadrant 0&1 with quadrant 2&3 and store to quadrant 0&1.
        # This is b/c nc_find_index8 returns UINT32_MAX for values not found, reduce min would pool the valid values.
        out_local[nl.ds(0, quadrant*2), :] = nisa.tensor_tensor(out_local[nl.ds(0, quadrant*2), :], out_local[nl.ds(quadrant*2, quadrant*2), :], op=nl.minimum)
        # Then reduce quadrant 0 with quadrant 1 and store to quadrant 0.
        out_local[nl.ds(0, quadrant), :] = nisa.tensor_tensor(out_local[nl.ds(0, quadrant), :], out_local[nl.ds(quadrant, quadrant), :], op=nl.minimum)

        # Store output from quadrant 0.
        nl.store(reshaped_dst[nl.ds((E_offset+e)*quadrant, quadrant), :], out_local[nl.ds(0, quadrant), :])

    return true_indices, true_counts
