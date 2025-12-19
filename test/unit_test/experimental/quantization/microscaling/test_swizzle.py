import pytest
import torch

from neuronx_distributed.experimental.quantization.microscaling.swizzle import swizzle_tensor, swizzle_tiled_tensor, unswizzle_tensor

def test_swizzle_tensor():
    X = torch.arange(128 * 512).reshape(128, 512).T
    print(f"Original: {X[:32, 0]=}")

    swizzled = swizzle_tensor(X)
    print(f"nSwizzled: {swizzled[:8, :4]=}")

    unswizzled = unswizzle_tensor(swizzled, X.shape[0], X.shape[1])
    print(f"Unswizzled: {unswizzled[:32, 0]=}")

    assert torch.equal(X, unswizzled)
    print("Tensors match!")

def test_swizzle_tiled_tensor():
    input_untiled = torch.arange(128 * 24 * 512).reshape(512, 3072).T
    input_tiled = input_untiled.reshape(24, 128, 512).permute(1, 0, 2)

    print("[32, 1] blocks before swizzle:")
    print(input_untiled[:32, :1])
    print(input_tiled[:32, :1, :1].squeeze(1))

    assert torch.equal(input_untiled[:32, :1], input_tiled[:32, :1, :1].squeeze(1))
    assert torch.equal(input_untiled, input_tiled.permute(1, 0, 2).reshape(3072, 512))

    swizzled_untiled = swizzle_tensor(input_untiled)
    swizzled_tiled = swizzle_tiled_tensor(input_tiled)

    print("[8, 4] blocks after swizzle:")
    print(f"{swizzled_untiled.shape=}")
    print(f"{swizzled_tiled.shape=}")
    print(swizzled_untiled[:8, :4])
    print(swizzled_tiled[:8, :1, :4].squeeze(1))
    
    assert torch.equal(swizzled_untiled[:8, :4], swizzled_tiled[:8, :1, :4].squeeze(1))
    assert torch.equal(swizzled_untiled, swizzled_tiled.permute(1, 0, 2).reshape(3072//4, 512*4))
    print("Tensors match!")

if __name__ == "__main__":
    pytest.main([__file__, '-v'])