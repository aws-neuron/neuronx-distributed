import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Type
import numpy as np

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
from neuronxcc import nki
from neuronxcc.nki.language import tile_size
from scipy.linalg import circulant # type: ignore[import-untyped]

# Global constant use for rolling topK

TOPK_PER_STAGE = 8 # Trn1 & Trn2 have 8 ALUs that can provide max8 per pass of data

"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                     Rolling TopK Algorithm Steps                             ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║ 1. Input Validation and Initialization:                                      ║
    ║    - Validate input tensor, k value, and dimensions                          ║
    ║    - Calculate staging parameters and dimension sizes                        ║
    ║    - Initialize rotation matrix for circular buffer operations               ║
    ║                                                                              ║
    ║ 2. Data Loading and Preparation:                                             ║
    ║    - Load input data using predicated_folded_load                            ║
    ║    - Initialize values and indices arrays                                    ║
    ║    - Set up initial indices for vocabulary items                             ║
    ║                                                                              ║
    ║ 3. Rolling TopK Process:                                                     ║
    ║    - For each stage (n_stages total):                                        ║
    ║      a. Find top k elements in the current chunk                             ║
    ║      b. Mask (-inf) these elements from further consideration                ║
    ║      c. Rotate the found top k elements and their indices                    ║
    ║      d. Insert rotated elements into the next stage                          ║
    ║                                                                              ║
    ║ 4. Final Processing:                                                         ║
    ║    - If sorted output is required:                                           ║
    ║      a. Reshape and sort the final top k elements                            ║
    ║      b. Perform final sort across all stages                                 ║
    ║    - Else:                                                                   ║
    ║      a. Use the last stage results directly                                  ║
    ║                                                                              ║
    ║ 5. Output Preparation:                                                       ║
    ║    - Store results in HBM (High Bandwidth Memory)                            ║
    ║    - Handle data distribution across multiple programs/cores                 ║
    ║                                                                              ║
    ║ 6. Return Results:                                                           ║
    ║    - Output top k values and their corresponding indices                     ║
    ║                                                                              ║
    ║ Key Features:                                                                ║
    ║ - Uses a rolling buffer technique to handle large vocabularies               ║
    ║ - Employs predicated loading for efficient memory access                     ║
    ║ - Utilizes rotation for maintaining global order across stages               ║
    ║ - Supports both sorted and unsorted output                                   ║
    ║ - Optimized for distributed processing across multiple cores                 ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
"""


class TopKConstants:

    @dataclass
    class DimensionSizes:
        """
        Actual dimension sizes (not tile counts) for the TopK computation

        Attributes
            B (int): batch dimension size for topk.
            S (int): sequence (or speculation dimension) size.
            V (int):  vocabulary dimension size.
            K (int):  The number of top elements to retrieve.
            BxS (int): Combined batch and sequence dimensions per lnc
            BxS0 (int): Base tile size for BxS per lnc
            n_stages (int): Number of stages in topk rolling
            input_ndim (int): Number of dimensions of input (2 or 3)
            chunk_size (int): Size of each chunk processed
            topk_per_stage (int): Number of top elements to extract per stage
        """

        # Input dimensions
        B: int
        S: int
        V: int
        K: int
        BxS: int
        BxS0: int
        n_stages: int
        chunk_size: int
        input_ndim: int 
        topk_per_stage: int = field(
            default=TOPK_PER_STAGE
        )
        V_padded: Optional[int] = None
        output_shape: Optional[Tuple[int]] = None


        def __post_init__(self):
            self.output_shape = (self.B, self.S, self.K) if self.input_ndim > 2 else (self.B, self.K)
            self.V_padded = self.chunk_size * self.n_stages

    @dataclass
    class TileCounts:
        """
        Attributes
          b_tiles (int): number of tiles to cover input BxS (`DimensionSizes.BxS`/ `DimensionSizes.BxS0`)
        """

        b_tiles: int

    @staticmethod
    def validate_and_get_inputs_sizes(
        input: nt.tensor,
        K: int,
        dim: Optional[int] = None,
    ) -> Tuple:
        assert input.ndim in [
            2,
            3,
        ], f"only 2D or 3D tensor supported but got tensor with shape {input.shape}"
        dim = dim or len(input.shape) - 1
        axis_len = input.shape[dim]
        if K > axis_len:
            raise ValueError(
                f"Parameter 'K' ({K}) is larger than the size of the specified axis ({axis_len})."
            )
        if K < 0:
            raise ValueError(f"Parameter 'K' ({K}) must be non-negative.")

        assert K % 8 == 0, f"only top k divisible by 8 is supported by got {K}"

        B, S, V = input.shape if input.ndim > 2 else (input.shape[0], 1, input.shape[1])
        params = {"B": B, "S": S, "V": V, "K": K}
        for name, value in params.items():
            if not isinstance(value, int):
                raise TypeError(f"Parameter {name} must be an integer.")

        return B, S, V, dim

    @staticmethod
    def calculate_constants(
        B: int, S: int, V: int, K: int, n_programs: int, input_ndim: int
    ) -> Tuple[DimensionSizes, TileCounts]:
        n_stages = math.ceil(min(K, V) / TopKConstants.DimensionSizes.topk_per_stage)

        # Fused batch: total number of "batch lines"
        BxS_global = B * S
        # Shard the fused batch lines across programs
        bxs_per_program = (BxS_global + n_programs - 1) // n_programs  # ceil div
        BxS_local = bxs_per_program
        # Per-stage chunking of vocab
        chunk_size = math.ceil(V / n_stages)

        # Tile size based on hardware or kernel configuration
        batch_seq_per_tile = min(math.ceil(tile_size.pmax / n_stages), BxS_local)
        n_batch_seq_tiles = math.ceil(BxS_local / batch_seq_per_tile)

        dim_sizes = TopKConstants.DimensionSizes(
            B=B,
            S=S,
            V=V,
            K=K,
            input_ndim=input_ndim,
            BxS=BxS_local,
            BxS0=batch_seq_per_tile,
            n_stages=n_stages,
            chunk_size=chunk_size,
            V_padded=chunk_size * n_stages
        )
        n_tiles = TopKConstants.TileCounts(b_tiles=n_batch_seq_tiles)
        return dim_sizes, n_tiles

    @staticmethod
    def calc_max_sbuf_batch(dim_sizes: DimensionSizes):
        # single tile memory requirement per partition
        values_and_idx_mem = 4 * 2 * (dim_sizes.chunk_size + dim_sizes.K)
        per_iteration_mem = 4 * 2 * dim_sizes.K
        shared_const_mem = 4 * dim_sizes.n_stages * dim_sizes.BxS0
        per_partition_mem = values_and_idx_mem + per_iteration_mem + shared_const_mem
        max_sbuf_batch = 20 * 2**20 / (tile_size.pmax * per_partition_mem) # take 20MB sbuf space
        return min(max_sbuf_batch, dim_sizes.BxS)

    @staticmethod
    def cost_estimate(dim_size: DimensionSizes, method="vanilla"):
        """
            Estimate the number of DVE instructions required for different attention implementations
            based on input dimensions.

            This function calculates the estimated DVE instruction count for different attention kernel
            implementations in transformer models, helping compare efficiency of different approaches.

            Args:
                dim_size (DimensionSizes): A dataclass containing the dimension sizes.
                    Expected fields include:
                    - B (int): Batch size
                    - S (int): Sequence length
                    - V (int): Vocabulary size
                    - K (int): Top-k elements to retrieve

                method (str, optional): The attention implementation method. Defaults to "vanilla".
                    Supported methods:
                    - "vanilla": Basic DVE implementation
                    - "rolling": Rolling-based implementation using circular buffers
                    - "cascading": Cascaded reduction implementation

            Returns:
                int: Estimated number of DVE instructions required

            Raises:
                ValueError: If an unsupported method is specified

            Note:
                - This provides a static analysis of instruction count
                - Actual performance may vary based on memory access patterns and hardware utilization

            Example:
                >>> dims = DimensionSizes(B=2, S=1, V=32000, K=8)
                >>> instr_count = cost_estimate(dims, method="rolling")
                >>> print(f"Estimated DVE instructions: {instr_count}")
        """
        supported_methods = ["vanilla", "rolling", "cascading"]

        if method not in supported_methods:
            raise ValueError(
                f"Unsupported method '{method}'. "
                f"Supported methods are: {', '.join(supported_methods)}"
            )
        POOL_GATHER_LIMIT = 512
        N_QUADRANT = 4
        GATHER_COST = 3e2
        cost = {
            'rolling': (math.ceil(dim_size.BxS / tile_size.pmax)
                        * sum([2 * (dim_size.chunk_size + i * dim_size.topk_per_stage)
                               + math.ceil(dim_size.chunk_size + i * dim_size.topk_per_stage / POOL_GATHER_LIMIT) * GATHER_COST
                               for i in range(math.ceil(dim_size.K / dim_size.topk_per_stage))])),
            'vanilla': (math.ceil(dim_size.BxS / tile_size.pmax)
                        * math.ceil(dim_size.K / dim_size.topk_per_stage) * 2 * dim_size.V),
            'cascading': (math.ceil(dim_size.BxS / tile_size.pmax)
                          * math.ceil(dim_size.K / dim_size.topk_per_stage)
                          * (math.ceil(dim_size.V / N_QUADRANT) + dim_size.K * N_QUADRANT)) + (dim_size.K * N_QUADRANT / 1.2)
        }
        return cost[method]


def get_permutation_matrix(
    block_size: int = 32, num_blocks: int = 1, dtype: Type[np.dtype] = np.float32
):
    shift = 1  # number of positions to roll
    # permutation base vector for circulant block
    base_perm = np.zeros(block_size)
    base_perm[shift % block_size] = 1  # 1 at shifted position
    P_block = circulant(base_perm)  # 32 x 32 roll matrix
    # build block-diagonal matrix with Kronecker product
    I_blocks = np.eye(num_blocks)
    B = np.kron(I_blocks, P_block)
    return B.astype(dtype)


def predicated_folded_load(
    data_hbm: nt.tensor,
    fold_factor: int,
    program_id: int = 0,
    n_programs: int = 1,
    fill_value: float = -9948.0,
) -> nt.tensor:
    """
        Reshapes  and load a HBM tensor of shape [b, n] into shape [b * fold_factor, n_folded] in SBUF
        tensor, where n_folded = ceil(n / fold_factor). If n is not divisible by fold_factor, the tensor
        is padded along the last dimension with -inf to make it divisible.

        Args:
            tensor (torch.Tensor): A 2D tensor of shape [b, n].
            fold_factor (int): The factor by which to fold the second dimension.

        Returns:
            torch.Tensor: A tensor of shape [b * fold_factor, n_folded], with padding
                        of -inf if needed.

        Example:
            >>> x = nt.tensor([[1, 2, 3, 4, 5]])
            >>> fold_tensor(x, 2)
            tensor([[1., 2., 3.],
                    [4., 5., -inf]])

            >>> data = nt.tensor([
                [1., 2., 3., 4., 5.],
                [6., 7., 8., 9., 10.]])
            >>> fold_factor = 2
            >>> n_prgs = 2

            # Program 0 (prg_id = 0) processes first row:
            >>> folding_load(data, fold_factor=2, prg_id=0, n_prgs=2)
            tensor([[1., 2., 3.],
                    [4., 5., -inf]])

            # Program 1 (prg_id = 1) processes second row:
            >>> folding_load(data, fold_factor=2, prg_id=1, n_prgs=2)
            tensor([[ 6.,  7.,  8.],
                    [ 9., 10., -inf]])

        Notes:
            - Each core folds only its portion of rows (B is sharded).
            - Padding is done to make N divisible by fold_factor, per core.
            - Output per core is [local_B * fold_factor, n_folded].
    """

    assert data_hbm.ndim == 2, "Expected input tensor to have shape [B, N]"
    batch_size, n = data_hbm.shape

    # Calculate the number of rows each program gets (ceil-based for uneven division)
    batch_size_sharded = (batch_size + n_programs - 1) // n_programs
    batch_line_offset = program_id * batch_size_sharded

    # Validate resource limits
    assert batch_size_sharded * fold_factor <= tile_size.pmax, (
        f"fold_factor x max local batch size ({batch_size_sharded}) exceeds tile limit {tile_size.pmax}"
    )

    # Compute folded shape: N -> [fold_factor, n_folded]
    n_folded = math.ceil(n / fold_factor)

    # Allocate scratch buffer filled with fill_value
    data_sb = nisa.memset((batch_size_sharded * fold_factor, n_folded), fill_value, dtype=data_hbm.dtype)

    # Fast path: if N is divisible by fold_factor, we can use reshape-based load
    if n == fold_factor * n_folded:
        # Reshape entire tensor to [B * fold_factor, n_folded]
        src_hbm_reshape = data_hbm.reshape((batch_size * fold_factor, n_folded))

        # Define grid over local shard
        ix, iy = nl.mgrid[0:batch_size_sharded * fold_factor, 0:n_folded]

        # Calculate flat offset into reshaped source
        base_offset = batch_line_offset * fold_factor
        mask = base_offset + ix < batch_size * fold_factor
        data_sb[ix, iy] = nl.load(src_hbm_reshape[base_offset + ix, iy], mask=mask)
        return data_sb

    # General case: predicated load for non-divisible N
    # Flatten input to 1D for offset calculation
    src_hbm_flat = data_hbm.reshape((batch_size * n,))

    # Index ranges for folds
    ix, iy = nl.mgrid[0:fold_factor, 0:n_folded]

    for batch_line in nl.affine_range(batch_size_sharded):
        row_idx = batch_line + batch_line_offset
        base_idx = row_idx * n  # Start index in flattened HBM
        # Compute flat offsets into flattened source
        flat_offsets = base_idx + ix * n_folded + iy
        # Predicate mask: only load valid elements within n and batch_local
        # Affine-safe mask: lift all bounds into predicate
        valid_row = row_idx < batch_size  # becomes a bool constant inside unrolled loop
        valid_col = (ix * n_folded + iy) < n
        valid_flat = flat_offsets < (batch_size * n)
        mask = valid_row & valid_col & valid_flat  # all affine-safe
        # Load with predication
        data_sb[ix + batch_line * fold_factor, iy] = nl.load(src=src_hbm_flat[flat_offsets], mask=mask)

    return data_sb


def predicated_folded_store(
    sbuf: nt.tensor,
    data_hbm: nt.tensor,
    fold_factor: int,
    program_id: int = 0,
    n_programs: int = 1,
) -> None:
    """
    Stores a local sbuf tensor [B_local * fold_factor, n_folded] back into the
    corresponding shard of a global HBM tensor [B, N], reversing predicated_folded_load.

    Args:
        sbuf (nt.tensor): Local buffer of shape [B_local * fold_factor, n_folded].
        data_hbm (nt.tensor): Global HBM tensor of shape [B, N].
        fold_factor (int): Number of folds applied during load.
        program_id (int): ID of the current core/program.
        n_programs (int): Total number of programs (cores).
    """

    assert data_hbm.ndim == 2, "Expected HBM tensor shape [B, N]"
    batch_size, n = data_hbm.shape
    n_folded = sbuf.shape[1]

    # Compute batch sharding
    batch_size_sharded = (batch_size + n_programs - 1) // n_programs
    batch_line_offset = program_id * batch_size_sharded

    # Fast path: n divisible by fold_factor
    if n == fold_factor * n_folded:
        reshaped_dst = data_hbm.reshape((batch_size * fold_factor, n_folded))
        ix, iy = nl.mgrid[0:batch_size_sharded * fold_factor, 0:n_folded]
        dst_offset = batch_line_offset * fold_factor
        mask = dst_offset + ix < batch_size * fold_factor
        nl.store(reshaped_dst[dst_offset + ix, iy], sbuf[ix, iy], mask=mask)
        return

    # General path: predicated store (flattened HBM)
    data_hbm_flat = data_hbm.reshape((batch_size * n,))
    ix, iy = nl.mgrid[0:fold_factor, 0:n_folded]

    for b in nl.affine_range(batch_size_sharded):
        row_idx = batch_line_offset + b
        base_idx = row_idx * n
        flat_indices = base_idx + ix * n_folded + iy
        mask = (ix * n_folded + iy < n) & (flat_indices < batch_size * n)
        sbuf_ix = ix + b * fold_factor
        nl.store(data_hbm_flat[flat_indices], sbuf[sbuf_ix, iy], mask=mask)


@nki.jit(experimental_flags='skip-non-top-level-shared-hbm-check')
def topk_rotated(
    input: nt.tensor,
    k: int = 256,
    dim: Optional[int] = None,
    largest: Optional[bool] = True,
    sorted: Optional[bool] = False,
) -> Tuple[nt.tensor, nt.tensor]:
    """
    Returns the k largest (or smallest) elements of a NumPy array along a given axis.

    Note: This function provides similar functionality to PyTorch's `torch.topk`.

    Args:
        input (nt.tensor): The input array.
        k (int): The number of top elements to retrieve.
        dim (int, optional): The axis along which to sort. The last axis (-1) is used by default.
        largest (bool, optional): Controls whether to return the largest (`True`) or smallest (`False`)
            k elements. Default: `True`.
        sorted (bool, optional): Controls whether the returned elements are sorted along the specified
            axis in descending order (for `largest=True`) or ascending order (for `largest=False`).
            Default: `False`.

    Returns:
        Tuple[nt.tensor, nt.tensor]: A tuple containing:
            - values (nt.tensor): The k largest (or smallest) elements along the specified axis.
              Shape: `a.shape[:dim] + (k,) + a.shape[dim+1:]`.
            - indices (nt.tensor): The indices of the corresponding elements in the original array
              along the specified axis. Shape: `a.shape[:dim] + (k,) + a.shape[axis+1:]`.

    Raises:
        TypeError: If `a` is not a nt.tesnor  or `k` is not an integer.
        ValueError: If `k` is greater than the size of the specified axis.

    Examples:
        >>> a = np.array([1, 3, 2, 4, 5])
        >>> values, indices = topk_rotated(a, 3)
        >>> print(values)
        [5 4 3]
        >>> print(indices)
        [4 3 1]

        >>> a = np.array([[1, 4, 2], [5, 2, 8]])
        >>> values, indices = topk_rotated(a, 2, dim=1)
        >>> print(values)
        [[4 2]
         [8 5]]
        >>> print(indices)
        [[1 2]
         [2 0]]
    """
    if not isinstance(input, nt.tensor):
        raise TypeError(f"Input must be a nt.tensor got {type(input)}.")

    grid_ndim = nl.program_ndim()
    assert grid_ndim == 0 or grid_ndim == 1, "topk only supports no specialization or specialization along one axis"
    n_programs, program_id = (nl.num_programs(axes=0), nl.program_id(axis=0)) if grid_ndim != 0 else (1, 0)

    batch_size, seq, elements, dim = TopKConstants.validate_and_get_inputs_sizes(
        input=input, K=k, dim=dim
    )
    dim_sizes, n_tiles = TopKConstants.calculate_constants(
        B=batch_size, S=seq, V=elements, K=k, n_programs=n_programs, input_ndim=input.ndim
    )
    # reshape HBM inputs to 2D
    input = input.reshape((dim_sizes.B * dim_sizes.S, dim_sizes.V))

    assert (
        dim_sizes.BxS % dim_sizes.BxS0 == 0
    ), f" only batch x seq multiple of {dim_sizes.BxS0} are supported, got {dim_sizes.BxS}"

    assert largest , (
        f" only largest=True got largest={largest}"
    )

    data = get_permutation_matrix(dim_sizes.n_stages, dim_sizes.BxS)
    rotate_hbm = nl.shared_constant(data, dtype=input.dtype)

    values = nl.ndarray(
        (
            dim_sizes.n_stages * dim_sizes.BxS0,
            dim_sizes.chunk_size + (dim_sizes.n_stages * dim_sizes.topk_per_stage),
        ),
        dtype=input.dtype,
    )
    indices = nl.full(
        (
            dim_sizes.n_stages * dim_sizes.BxS0,
            dim_sizes.chunk_size + (dim_sizes.n_stages * dim_sizes.topk_per_stage),
        ),
        0,
        dtype=input.dtype,
    )

    def sort(data_sbuf, indices=None, *, topk: int = 1):
        """
        Helper function to perform sort Perform sort using topk.
        - data_sbuf: 2D SBUF array of shape [m, n] which in unsorted where n == topk
        - indices: indices of the current topk elements

        Returns:
         - sorted values: 2D SBUF array of shape [m, n]
         - indices: 2D SBUF array of shape [m, n], global indices corresponding to sorted elements
        """
        m, n = data_sbuf.shape
        const_8 = 8
        assert topk == n, f"shape mismatch expected: {m} x {topk}, got {m}x{n}"
        num_pass = math.ceil(topk / const_8)

        topk_val_buf = nl.full((nl.par_dim(m), topk), 0, dtype=np.float32, name="val_buf", buffer=nl.sbuf)
        topk_idx_buf = nl.full((nl.par_dim(m), topk), 0, dtype=np.uint32, name="id_buf", buffer=nl.sbuf)
        global_topk_idx_buf = nl.full((nl.par_dim(m), topk), 0, dtype=np.uint32, name="global_id_buf", buffer=nl.sbuf)

        ix, iy = nl.mgrid[0:m, 0:const_8]
        ix_data, iy_data = nl.mgrid[0:m, 0:n]

        for i in nl.sequential_range(num_pass):
            topk_val_buf[:, nl.ds(i * const_8, const_8)] = nisa.max8(src=data_sbuf, mask=None, dtype=np.float32)
            if nisa.get_nc_version() <= nisa.nc_version.gen2:
                topk_idx_buf[:, nl.ds(i * const_8, const_8)] = nisa.nc_find_index8(data=data_sbuf[...],
                                                                                   vals=topk_val_buf[:, nl.ds(i * const_8, const_8)])
                data_sbuf[...] = nisa.nc_match_replace8(data=data_sbuf[...], vals=topk_val_buf[:, nl.ds(i * const_8, const_8)],
                                                        imm=float('-inf'), mask=None, dtype=np.float32)
            else:
                data_sbuf[...] = nisa.nc_match_replace8(data=data_sbuf[...], vals=topk_val_buf[:, nl.ds(i * const_8, const_8)],
                                                        imm=-np.inf, mask=None, dtype=np.float32,
                                                        dst_idx=topk_idx_buf[:, nl.ds(i * const_8, const_8)])
            global_topk_idx_buf[ix, iy + i * const_8] = nl.gather_flattened(data=indices[ix_data, iy_data],
                                                                            indices=topk_idx_buf[ix, iy + i * const_8])
        return topk_val_buf, global_topk_idx_buf

    def store_reshape_load(data_sb, fold_factor, dtype=None):
        m, n = data_sb.shape
        data_hbm = nl.ndarray(data_sb.shape, dtype=data_sb.dtype, buffer=nl.private_hbm)
        nl.store(data_hbm, data_sb)
        data_hbm = data_hbm.reshape((m // fold_factor, n * fold_factor))
        data_sb = nl.load(data_hbm, dtype=dtype)
        return data_sb

    def topk(data):
        return nisa.max8(src=data)

    def rotate(tensor):
        return nisa.nc_matmul(rotation, tensor)

    def delete(tensor, value, index=None):
        if nisa.get_nc_version() <= nisa.nc_version.gen2:
            index[:, :] = nisa.nc_find_index8(data=tensor, vals=value)
            return nisa.nc_match_replace8(data=tensor, vals=value, imm=float('-inf'))
        return nisa.nc_match_replace8(data=tensor, vals=value, dst_idx=index, imm=float("-inf"))

    def insert(tensor, values, offset=0):
        tensor[:, offset : offset + dim_sizes.topk_per_stage] = nl.copy(values, dtype=values.dtype)

    # can also be done with iota
    indices[:, : dim_sizes.chunk_size] = nl.load(
        nl.shared_constant(
            np.tile(
                np.arange(dim_sizes.V_padded)
                .astype(input.dtype)
                .reshape((dim_sizes.n_stages, dim_sizes.chunk_size)),
                (dim_sizes.BxS0, 1),
            )
        )
    )

    i_p = nl.arange(dim_sizes.n_stages * dim_sizes.BxS0)[:, None]
    i_f = nl.arange(dim_sizes.chunk_size)[None, :]
    values[i_p, i_f] = predicated_folded_load(input, fold_factor=dim_sizes.n_stages,
                                              n_programs=n_programs, program_id=program_id)
    i_f = nl.arange(dim_sizes.n_stages * dim_sizes.BxS0)[None, :]
    rotation = nl.load(rotate_hbm[i_p, i_f])

    local_index = nl.full(
        (dim_sizes.n_stages * dim_sizes.BxS0, dim_sizes.topk_per_stage), 0, dtype=nl.uint32
    )

    for i in nl.static_range(dim_sizes.n_stages):
        offset = dim_sizes.chunk_size + (dim_sizes.topk_per_stage * i)
        value = topk(values[:, :offset])
        values[:, :offset] = delete(values[:, :offset], value, local_index)
        global_index = nl.gather_flattened(indices, local_index)
        rotated_index = rotate(global_index)
        rotated = rotate(value)
        insert(values, rotated, offset=offset)
        insert(indices, rotated_index, offset=offset)

    if sorted:
        flat_value = store_reshape_load(value, dim_sizes.n_stages, dtype=value.dtype)
        flat_index = store_reshape_load(global_index, dim_sizes.n_stages, dtype=nl.uint32)
        # [32*b, 256/32=8] -> [b, 256]
        sorted_val, sorted_global_idx = sort(flat_value, topk=k, indices=flat_index)
        output_shape = (dim_sizes.B * dim_sizes.S, dim_sizes.K)
        output_hbm = nl.ndarray(output_shape, dtype=value.dtype, buffer=nl.shared_hbm)
        out_hbm_idx = nl.ndarray(output_shape, dtype=nl.uint32, buffer=nl.shared_hbm)

        # Compute batch sharding
        b_shard = (dim_sizes.B * dim_sizes.S + n_programs - 1) // n_programs
        batch_offset = program_id * b_shard
        ix, iy = nl.mgrid[0:b_shard, 0:dim_sizes.K]
        mask = ix + batch_offset < dim_sizes.B * dim_sizes.S
        nl.store(out_hbm_idx[ix + batch_offset, iy], sorted_global_idx, mask=mask)
        nl.store(output_hbm[ix + batch_offset, iy], sorted_val, mask=mask)
        output_hbm = output_hbm.reshape(dim_sizes.output_shape)
        out_hbm_idx = out_hbm_idx.reshape(dim_sizes.output_shape)
        return output_hbm, out_hbm_idx

    output_shape = (dim_sizes.B * dim_sizes.S, dim_sizes.K)
    global_index_int = nl.copy(global_index, dtype=nl.uint32)

    output_hbm = nl.ndarray(output_shape, dtype=value.dtype, buffer=nl.shared_hbm)
    out_hbm_idx = nl.ndarray(output_shape, dtype=nl.uint32, buffer=nl.shared_hbm)
    predicated_folded_store(global_index_int, out_hbm_idx, fold_factor=dim_sizes.n_stages,
                            program_id=program_id, n_programs=n_programs)
    predicated_folded_store(value, output_hbm, fold_factor=dim_sizes.n_stages,
                            program_id=program_id, n_programs=n_programs)
    # reshape to match input ndims
    output_hbm = output_hbm.reshape(dim_sizes.output_shape)
    out_hbm_idx = out_hbm_idx.reshape(dim_sizes.output_shape)
    return output_hbm, out_hbm_idx
