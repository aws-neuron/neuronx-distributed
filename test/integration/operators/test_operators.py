import functools
from neuronx_distributed.operators.argmax import argmax as nxd_argmax
from neuronx_distributed.operators.topk import topk as nxd_topk
import pytest
import torch
import torch_neuronx

from neuronx_distributed.parallel_layers import ColumnParallelLinear
from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance


IN_FEATURES = 4
OUT_FEATURES = 64

class TestModelArgmax(torch.nn.Module):

    def __init__(self, is_nxd=True, dim=1, keepdim=False, gather_dim=1):
        super().__init__()
        self.is_nxd = is_nxd
        self.dim = dim
        self.keepdim = keepdim
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
            return nxd_argmax(lin_out, dim=self.dim, gather_dim=self.gather_dim, keepdim=self.keepdim)

        return torch.argmax(lin_out,dim=self.dim, keepdim=self.keepdim)

def default_loader():
    # not necessary for this test, we do it ourselves
    return {}

@pytest.mark.parametrize(
    ["input_is_3d","dim", "keepdim"],
    [
        (False, 0, True),
        (False, 0, False),
        (False, 1, True),
        (False, 1, False),
        (True, 0, True),
        (True, 0, False),
        (True, 1, True),
        (True, 1, False),
        (True, 2, True),
        (True, 2, False),
    ]
)
def test_nxd_argmax(input_is_3d, dim, keepdim):
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
            functools.partial(TestModelArgmax, True, dim, keepdim, gather_dim),
            {}
        ),
        [(inp,)]
    )

    neuron_mod = mb.trace(initialize_model_weights=False)
    test_mod = TestModelArgmax(False, dim, keepdim)

    weights_sharded = [{'lin.weight': torch.rand(OUT_FEATURES // tp_degree, IN_FEATURES)} for _ in range(tp_degree)]
    full_weight = torch.cat([d['lin.weight'] for d in weights_sharded], dim=0)

    start_rank_tensor = torch.tensor([0], dtype=torch.int32, device="cpu")
    neuron_mod.nxd_model.initialize(weights_sharded, start_rank_tensor)
    test_mod.load_state_dict({'lin.weight': full_weight})

    expected = test_mod(inp)
    actual = neuron_mod(inp)

    assert expected.shape == actual.shape, "Shape Mismatch: expected {expected.shape}, but got {actual.shape}"
    assert torch.allclose(expected, actual)


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
        (False, 1, 50),
        (True, 2, 2),
        (True, 2, 50),
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
