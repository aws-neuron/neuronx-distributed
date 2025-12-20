import inspect
import types
import pytest
import torch
import torch.nn as nn

from neuronx_distributed.utils.tensor_replacement.model_modification import (
    modify_model_for_tensor_replacement,
    patch_forward_with_additional_args,
)
from neuronx_distributed.utils.tensor_replacement.registry import (
    RuntimeRegister,
)


# -----------------------------
# Tiny toy modules for testing
# -----------------------------
class Leaf(nn.Module):
    """Simple pass-through layer so hooks see a clean tensor."""
    def forward(self, x):
        return x


class TwoStage(nn.Module):
    """
    x -> child1 -> child2
    """
    def __init__(self):
        super().__init__()
        self.child1 = Leaf()
        self.child2 = Leaf()

    def forward(self, x):
        y = self.child1(x)
        z = self.child2(y)
        return z


class TwoStageScaled(TwoStage):
    """Like TwoStage, but has an extra optional arg in forward to test signatures."""
    def forward(self, x, scale: int = 1):
        out = super().forward(x)
        return out * scale


# -----------------------------
# RuntimeRegister patching
# -----------------------------
@pytest.fixture(autouse=True)
def reset_runtime_register(monkeypatch):
    """
    Ensure RuntimeRegister has a predictable contract for each test:
      - module_superset == ["child1", "child2"]
      - _tr_runtime_list / _tr_mask_list hold per-module tensors set by register_runtime_args
      - clear_runtime_args wipes those dicts
    """
    monkeypatch.setattr(RuntimeRegister, "module_superset", ["child1", "child2"], raising=False)
    monkeypatch.setattr(RuntimeRegister, "_tr_runtime_list", {}, raising=False)
    monkeypatch.setattr(RuntimeRegister, "_tr_mask_list", {}, raising=False)

    def register_runtime_args(*, tr_args, mask_args):
        # Store by module name in the canonical order
        RuntimeRegister._tr_runtime_list = {}
        RuntimeRegister._tr_mask_list = {}
        for name, t, m in zip(RuntimeRegister.module_superset, tr_args, mask_args):
            RuntimeRegister._tr_runtime_list[name] = t
            RuntimeRegister._tr_mask_list[name] = m

    def clear_runtime_args():
        RuntimeRegister._tr_runtime_list = {}
        RuntimeRegister._tr_mask_list = {}

    monkeypatch.setattr(
        RuntimeRegister,
        "register_runtime_args",
        staticmethod(register_runtime_args),
        raising=False,
    )
    monkeypatch.setattr(
        RuntimeRegister,
        "clear_runtime_args",
        staticmethod(clear_runtime_args),
        raising=False,
    )

    yield

    # best-effort cleanup
    RuntimeRegister._tr_runtime_list = {}
    RuntimeRegister._tr_mask_list = {}


# -----------------------------
# Helpers
# -----------------------------
def make_masks_and_tensors(shape, mask1=True, mask2=True):
    """
    Convenience to generate (tr_list, mask_list) in register order.
    """
    t1 = torch.full(shape, 111.0)
    t2 = torch.full(shape, 222.0)
    m1 = torch.ones_like(t1, dtype=torch.bool) if mask1 else torch.zeros_like(t1, dtype=torch.bool)
    m2 = torch.ones_like(t2, dtype=torch.bool) if mask2 else torch.zeros_like(t2, dtype=torch.bool)
    return [t1, t2], [m1, m2]


# -----------------------------
# Tests: patch_forward_with_additional_args
# -----------------------------
def test_patch_enforces_trailing_arg_count():
    model = TwoStage()
    # Provide module_superset explicitly
    model = patch_forward_with_additional_args(model, RuntimeRegister.module_superset)

    x = torch.randn(3, 4)
    t_list, _ = make_masks_and_tensors(x.shape, True, True)

    # Need 2*k trailing args, k=2 -> 4 tensors. Provide fewer to trigger error.
    with pytest.raises(ValueError, match=r"expected 4 trailing replacement args"):
        _ = model(x, *t_list)  # only 2 provided


def test_patch_calls_original_forward_and_clears_runtime():
    model = TwoStage()
    model = patch_forward_with_additional_args(model, RuntimeRegister.module_superset)

    x = torch.zeros(2, 3)
    t_list, m_list = make_masks_and_tensors(x.shape, True, True)

    # Ensure clean pre-state
    assert RuntimeRegister._tr_runtime_list == {}
    assert RuntimeRegister._tr_mask_list == {}

    out = model(x, *t_list, *m_list)

    # No hooks installed here, so output equals baseline forward(x).
    baseline = TwoStage()(x)
    assert torch.equal(out, baseline)

    # RuntimeRegister should be cleared after forward
    assert RuntimeRegister._tr_runtime_list == {}
    assert RuntimeRegister._tr_mask_list == {}


def test_patch_preserves_wrapper_metadata_and_binding():
    model = TwoStage()
    original_forward = model.forward
    model = patch_forward_with_additional_args(model, RuntimeRegister.module_superset)

    # wrapper preserves metadata via @wraps
    assert model.forward.__name__ == original_forward.__name__
    assert inspect.signature(model.forward) != inspect.signature(original_forward)  # now variadic
    # and it's still a bound method
    assert isinstance(model.forward, types.MethodType)


# -----------------------------
# Tests: modify_model_for_tensor_replacement (hooks)
# -----------------------------
def test_modify_model_installs_hooks_and_replaces_per_module():
    model = TwoStage()
    model, hooks = modify_model_for_tensor_replacement(model)

    # Hooks should be installed for the modules in the superset
    assert set(hooks.keys()) == {"child1", "child2"}

    x = torch.randn(5, 6)

    # A: mask1 True, mask2 False -> output after child2 == child2(child1(tf1)) == tf1
    t_list, m_list = make_masks_and_tensors(x.shape, mask1=True, mask2=False)
    out = model(x, *t_list, *m_list)
    assert torch.all(out == 111.0)

    # B: mask1 False, mask2 True -> final should be tf2
    t_list, m_list = make_masks_and_tensors(x.shape, mask1=False, mask2=True)
    out = model(x, *t_list, *m_list)
    assert torch.all(out == 222.0)

    # C: mask1 False, mask2 False -> baseline behavior
    t_list, m_list = make_masks_and_tensors(x.shape, mask1=False, mask2=False)
    baseline = TwoStage()(x)
    out = model(x, *t_list, *m_list)
    assert torch.equal(out, baseline)


def test_modify_model_with_multiarg_forward_signature():
    model = TwoStageScaled()
    model, _ = modify_model_for_tensor_replacement(model)

    x = torch.ones(2, 2)
    scale = 3
    t_list, m_list = make_masks_and_tensors(x.shape, mask1=False, mask2=True)
    out = model(x, scale, *t_list, *m_list)
    assert torch.all(out == 222.0 * scale)


def test_missing_runtime_tensor_or_mask_raises_from_arg_guard():
    """
    If the patched forward is called with insufficient trailing args,
    it should raise before hooks run.
    """
    model = TwoStage()
    model, _ = modify_model_for_tensor_replacement(model)

    x = torch.randn(2, 3)
    t1 = torch.full_like(x, 111.0)
    m1 = torch.ones_like(x, dtype=torch.bool)

    with pytest.raises(ValueError, match=r"expected 4 trailing replacement args"):
        _ = model(x, t1, m1)  # only 2 provided; need 4


def test_shape_mismatch_in_where_raises_runtime_error():
    model = TwoStage()
    model, _ = modify_model_for_tensor_replacement(model)

    x = torch.randn(2, 3)
    t1 = torch.full_like(x, 111.0)
    t2 = torch.full((2, 2), 222.0)  # not broadcastable to (2,3)
    m1 = torch.ones_like(x, dtype=torch.bool)
    m2 = torch.ones_like(x, dtype=torch.bool)

    with pytest.raises(RuntimeError):
        _ = model(x, t1, t2, m1, m2)


def test_masks_can_disable_all_replacements():
    model = TwoStage()
    model, _ = modify_model_for_tensor_replacement(model)

    x = torch.randn(4, 4)
    t_list, m_list = make_masks_and_tensors(x.shape, mask1=False, mask2=False)
    model(x, *t_list, *m_list)
