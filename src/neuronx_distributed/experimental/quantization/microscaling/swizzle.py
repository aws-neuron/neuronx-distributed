# TODO[release]: move from experimental to testing or utils folder, as this code is not intended to be a prod API

"""
Handles "swizzling" or interleaving/packing tensor from [M, N] to [M/4, N*4].
"""

def swizzle_tensor(input_tensor):
    """
    Swizzle input tensor from shape [M, N] to [M/4, N*4].
    Takes contiguous blocks of shape [32, 1] and converts to contiguous blocks of [8, 4].
    """
    M, N = input_tensor.shape

    assert M % 4 == 0, f"M dimension ({M}) must be divisible by 4"

    reshaped = input_tensor.reshape(M // 4, 4, N)

    # Transpose [M/4, N, 4] + reshape [M/4, N*4]
    swizzled = reshaped.transpose(1, 2).reshape(M // 4, N * 4)

    return swizzled.contiguous()


def swizzle_tiled_tensor(input_tensor):
    """
    Swizzle input tensor from shape [TILE_M, NUM_TILES_IN_M, N] to [TILE_M, NUM_TILES_IN_M/4, N*4].
    Groups every 4 consecutive elements in the NUM_TILES_IN_M dimension.
    """
    TILE_M, NUM_TILES_IN_M, N = input_tensor.shape
    
    # [TILE_M, NUM_TILES_IN_M, N] -> [TILE_M//4, 4, NUM_TILES_IN_M//4, 4, N]
    grouped = input_tensor.reshape(TILE_M//4, 4, NUM_TILES_IN_M//4, 4, N)
    
    # [TILE_M//4, 4_M, NUM_TILES_IN_M//4, 4_TILE, N] -> [4_TILE, TILE_M//4, NUM_TILES_IN_M//4, N, 4_M] -> [TILE_M, NUM_TILES_IN_M//4, N*4]
    swizzled = grouped.permute(3, 0, 2, 4, 1).reshape(TILE_M, NUM_TILES_IN_M//4, N*4)

    return swizzled.contiguous()



def unswizzle_tensor(swizzled_tensor, original_M, original_N):
    """
    Undo swizzling.
    """
    M_div_4, N_times_4 = swizzled_tensor.shape

    assert original_M == M_div_4 * 4, "Dimension mismatch"
    assert original_N * 4 == N_times_4, "Dimension mismatch"

    # Reshape to [M/4, N, 4]
    reshaped = swizzled_tensor.reshape(M_div_4, original_N, 4)

    # Transpose [M/4, 4, N] + reshape [M, N]
    unswizzled = reshaped.transpose(1, 2).reshape(original_M, original_N)

    return unswizzled.contiguous()
