import neuronxcc.nki.language as nl
from neuronx_distributed.kernels.expert_mlps_mx.constants import (
   ALLOWED_P_DIM_MX, ALLOWED_P_SCALE_IDX_OFFSETS
)


def _get_lnc_config():
    """
    Helper function to get LNC config + validate LNC correctness.
    """
    grid_ndim = nl.program_ndim()
    assert grid_ndim == 0 or grid_ndim == 1, "kernel only supports no specialization or specialization along one axis"
    n_prgs, prg_id = (nl.num_programs(axes=0), nl.program_id(axis=0)) if grid_ndim != 0 else (1, 0)
    assert n_prgs <= 2, f'kernel supports unsharded or LNC-2 sharded; but got a spmd grid size of {n_prgs}'
    return n_prgs, prg_id


def get_scale_idx(P, F, P_offset=0, F_offset=0):
    """
    Helper function to get scale indices for a given P, F.

    Can set P_offset to get indices to pack multiple scales into the same SBUF buffer.
    """

    assert P in ALLOWED_P_DIM_MX, f"Cannot calculate scale striding for P={P}, expected P in {ALLOWED_P_DIM_MX}"
    assert P_offset in ALLOWED_P_SCALE_IDX_OFFSETS, f"Cannot offset P by {P_offset}, expected P_offset in {ALLOWED_P_SCALE_IDX_OFFSETS}"

    QUADRANT_SIZE = 32
    NUM_QUADRANTS = P // QUADRANT_SIZE

    # Simple case: one quadrant
    # Throws '<class 'neuronxcc.pelican.ir.APIndex'> doesn't appear in params or loopnest' when unifying single and multi-quadrant mgrid.
    if P == 32:
        s_p, s_f = nl.mgrid[P_offset:P_offset + 4, 0:F]

    # Strided case (unpacked P of 256 or 512): split scale partitions to respect quadrant boundaries
    # Each quadrant gets 4 partitions worth of scales
    # Physical mapping: spread across quadrants with consistent offsets
    # s_p_0 * 32 + s_p_1 creates the required stride pattern
    else:
        s_p, s_f = nl.mgrid[P_offset:P_offset + P, 0:F]
        s_p_0, _, s_p_1 = s_p.split(NUM_QUADRANTS, 8, 4)  # Extract first four elements of each quadrant after the offset
        s_p = s_p_0 * 32 + s_p_1

    # Add F dim offset to s_f
    s_f += F_offset

    return s_p, s_f