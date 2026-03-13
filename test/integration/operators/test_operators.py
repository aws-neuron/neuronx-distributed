import functools
import importlib

from neuronx_distributed.operators.argmax import argmax as nxd_argmax, _can_use_nki_max
from neuronx_distributed.operators.topk import topk as nxd_topk
import pytest
import torch
import torch_neuronx
from unittest.mock import patch

from neuronx_distributed.parallel_layers import ColumnParallelLinear
from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance


class TestModelArgmax(torch.nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        is_nxd=True,
        dim=1,
        keepdim=False,
        gather_dim=1,
        disable_argmax_kernel=False,
    ):
        super().__init__()
        self.is_nxd = is_nxd
        self.dim = dim
        self.keepdim = keepdim
        self.gather_dim = gather_dim
        self.disable_argmax_kernel = disable_argmax_kernel
        self.lin = (
            ColumnParallelLinear(
                in_features, out_features, bias=False, gather_output=False
            )
            if self.is_nxd
            else torch.nn.Linear(in_features, out_features, False)
        )

    def forward(self, tensor):
        lin_out = self.lin(tensor)
        if self.is_nxd:
            return nxd_argmax(
                lin_out,
                dim=self.dim,
                gather_dim=self.gather_dim,
                keepdim=self.keepdim,
                disable_argmax_kernel=self.disable_argmax_kernel,
            )

        return torch.argmax(lin_out, dim=self.dim, keepdim=self.keepdim)


def validate_argmax(argmax_shape, dim, keepdim, tp_degree, disable_argmax_kernel=False):

    rank = len(argmax_shape)
    in_features = 1  # Use tiny dim since we don't care about the matmult

    # Prepare inputs
    input_shape = (*argmax_shape[:-1], in_features)
    tensor = torch.rand(input_shape)
    out_features = argmax_shape[-1] * tp_degree
    gather_dim = rank - 1

    mb = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=default_loader,
    )
    mb.add(
        "test",
        BaseModelInstance(
            functools.partial(
                TestModelArgmax,
                in_features,
                out_features,
                True,
                dim,
                keepdim,
                gather_dim,
                disable_argmax_kernel,
            ),
            {},
        ),
        [(tensor,)],
    )

    neuron_mod = mb.trace(initialize_model_weights=False)
    test_mod = TestModelArgmax(
        in_features, out_features, False, dim, keepdim, disable_argmax_kernel
    )
    weights_sharded = [
        {"lin.weight": torch.rand(out_features // tp_degree, in_features)}
        for _ in range(tp_degree)
    ]
    full_weight = torch.cat([d["lin.weight"] for d in weights_sharded], dim=0)

    start_rank_tensor = torch.tensor([0], dtype=torch.int32, device="cpu")
    neuron_mod.nxd_model.initialize(weights_sharded, start_rank_tensor)
    test_mod.load_state_dict({"lin.weight": full_weight})

    expected = test_mod(tensor)
    actual = neuron_mod(tensor)

    torch_neuronx.testing.assert_close(expected, actual)


def default_loader():
    # Not necessary for this test, we do it ourselves
    return {}


@pytest.mark.parametrize(
    "argmax_shape",
    [
        # Simple 2D
        (1, 64),
        (2, 64),
        (1, 128),
        (4, 128),
        (4, 256),
        # Simple 3D
        (1, 2, 64),
        (4, 4, 64),
        (4, 4, 256),
        # GPT-OSS use cases
        (1, 1, 25136),
        (8, 1, 25136),
        (128, 1, 1571),
    ],
    ids=lambda x: f"shape{x}",
)
@pytest.mark.parametrize("dim", [0, 1, 2], ids=lambda x: f"dim{x}")
@pytest.mark.parametrize("keepdim", [True, False], ids=lambda x: f"keepdim{x}")
@pytest.mark.parametrize("tp_degree", [2], ids=lambda x: f"tp{x}")
def test_nxd_argmax(argmax_shape, dim, keepdim, tp_degree):
    """
    Validate the accuracy of the distributed argmax implementation.

    `argmax_shape` is the shape that will get passed to the `argmax` function:
        (B, S, H // TP)
    """
    rank = len(argmax_shape)
    if dim >= rank:
        pytest.skip(f"Argmax dim={dim} invalid on rank {rank} tensor")
    validate_argmax(argmax_shape, dim, keepdim, tp_degree, disable_argmax_kernel=False)


@pytest.mark.parametrize(
    "hw_type,expected",
    [
        ("TRN1", False),
        ("TRN2", True),
        ("TRN3", True),
    ],
)
def test_can_use_nki_max_hw(hw_type, expected):
    """Test hardware type validation for _can_use_nki_max"""
    # Get the actual module, not the function
    # This is necessary due to overloaded argmax function and module naming
    argmax_module = importlib.import_module("neuronx_distributed.operators.argmax")
    with patch.object(argmax_module, "hardware") as mock_hw:
        # Mock the function call to return the hardware type
        mock_hw.return_value = hw_type

        # Mock the attributes used in comparison
        mock_hw.TRN2 = "TRN2"
        mock_hw.TRN3 = "TRN3"

        # Use a tensor that passes all other checks
        tensor = torch.rand(10, 128)
        result = argmax_module._can_use_nki_max(tensor, dim=1)

        assert result == expected


@pytest.mark.parametrize(
    "shape,dim,expected",
    [
        # Valid cases
        ((10, 128), 1, True),  # 2D, last dim, size >= 128
        ((10, 129), 1, True),  # 2D, last dim, size >= 128
        ((1, 10, 128), 2, True),  # 3D with shape[0]=1
        ((1, 10, 256), 2, True),  # 3D with shape[0]=1
        # Invalid: dim is not last dimension
        ((10, 128), 0, False),
        # Invalid: size < 128
        ((10, 127), 1, False),
        # Invalid: wrong number of dimensions
        ((128,), 0, False),  # 1D
        ((2, 10, 128), 2, False),  # 3D with shape[0] != 1
        ((1, 1, 10, 128), 3, False),  # 4D
    ],
)
def test_can_use_nki_max_inputs(shape, dim, expected):
    """Test input validation for _can_use_nki_max"""
    tensor = torch.rand(shape)
    assert _can_use_nki_max(tensor, dim) == expected


def test_disable_argmax_kernel():
    """
    Test that disable_argmax_kernel=True prevents NKI argmax kernel usage

    We first confirm _can_use_nki_max will return False, and then we run the
    argmax function to confirm outputs are correct.
    """
    argmax_shape = (10, 128)
    tensor = torch.rand(argmax_shape)  # Input that can use the kernel
    dim = 1  # Last dimension

    # Confirm that the _can_use_nki_max routing works as expected: Basecase
    result = _can_use_nki_max(tensor, dim, disable_argmax_kernel=False)
    assert result is True, "_can_use_nki_max should return True for valid kernel inputs"
    # Disable kernel
    result = _can_use_nki_max(tensor, dim, disable_argmax_kernel=True)
    assert (
        result is False
    ), "_can_use_nki_max should return False when kernel is disabled"

    # Confirm that the Neuron outputs (without the kernel) are accurate
    validate_argmax(
        argmax_shape, dim, keepdim=True, tp_degree=2, disable_argmax_kernel=True
    )


IN_FEATURES = 4
OUT_FEATURES = 64

class TestModelTopk(torch.nn.Module):

    def __init__(self, is_nxd=True, k=50, dim=1, gather_dim=1):
        super().__init__()
        self.is_nxd = is_nxd
        self.k = k
        self.dim = dim
        self.gather_dim = gather_dim
        self.lin = ColumnParallelLinear(
            IN_FEATURES,
            OUT_FEATURES,
            bias=False,
            gather_output=False
        ) if self.is_nxd else torch.nn.Linear(IN_FEATURES, OUT_FEATURES, False)

    def forward(self, tensor):
        lin_out = self.lin(tensor)
        if self.is_nxd:
            return nxd_topk(lin_out, k=self.k, dim=self.dim, gather_dim=self.gather_dim)

        return torch.topk(lin_out, k=self.k, dim=self.dim)


@pytest.mark.parametrize(
    ["input_is_3d","dim", "k"],
    [
        (False, 1, 10),
        # (False, 1, 50), # Failing - ticket: V2001877566
        # (True, 2, 2), # Failing - ticket: V2001877566
        # (True, 2, 50), # Failing - ticket: V2001877566
    ]
)
def test_nxd_topk(input_is_3d, dim, k):
    tp_degree = 2

    if input_is_3d:
        inp = torch.rand(1,2,IN_FEATURES)
        gather_dim = 2
    else:
        inp = torch.rand(2,IN_FEATURES)
        gather_dim = 1

    mb = ModelBuilder(router=None, tp_degree=tp_degree, checkpoint_loader=default_loader)
    mb.add(
        "test",
        BaseModelInstance(
            functools.partial(TestModelTopk, True, k, dim, gather_dim),
            {}
        ),
        [(inp,)]
    )

    neuron_mod = mb.trace(initialize_model_weights=False)
    test_mod = TestModelTopk(False, k, dim)

    weights_sharded = [{'lin.weight': torch.rand(OUT_FEATURES // tp_degree, IN_FEATURES)} for _ in range(tp_degree)]
    full_weight = torch.cat([d['lin.weight'] for d in weights_sharded], dim=0)

    start_rank_tensor = torch.tensor([0], dtype=torch.int32, device="cpu")
    neuron_mod.nxd_model.initialize(weights_sharded, start_rank_tensor)
    test_mod.load_state_dict({'lin.weight': full_weight})

    expected = test_mod(inp)
    expected_values, expected_indices = expected.values, expected.indices
    actual_values, actual_indices = neuron_mod(inp)

    assert expected_indices.shape == actual_indices.shape, "Shape Mismatch: expected {expected_indices.shape}, but got {actual_indices.shape}"
    assert torch.allclose(expected_indices, actual_indices)

    assert expected_values.shape == actual_values.shape, "Shape Mismatch: expected {expected_values.shape}, but got {actual_values.shape}"
    assert torch.allclose(expected_values, actual_values)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
