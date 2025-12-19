import pytest
import torch

from neuronx_distributed.utils.tensor_replacement.registry import RuntimeRegister


@pytest.fixture(autouse=True)
def reset_register():
    # Clean slate for each test
    RuntimeRegister.module_superset = []
    RuntimeRegister.clear_runtime_args()
    yield
    RuntimeRegister.module_superset = []
    RuntimeRegister.clear_runtime_args()


def test_register_valid_inputs_clones_and_isolation():
    RuntimeRegister.module_superset = ["m1", "m2"]

    x1 = torch.ones(2, 3)
    x2 = torch.full((2, 3), 2.0)
    m1 = torch.ones_like(x1, dtype=torch.bool)
    m2 = torch.zeros_like(x2, dtype=torch.bool)

    RuntimeRegister.register_runtime_args([x1, x2], [m1, m2])

    # Keys recorded in order
    assert set(RuntimeRegister._tr_runtime_list.keys()) == {"m1", "m2"}
    assert set(RuntimeRegister._tr_mask_list.keys()) == {"m1", "m2"}

    t1 = RuntimeRegister._tr_runtime_list["m1"]
    t2 = RuntimeRegister._tr_runtime_list["m2"]
    k1 = RuntimeRegister._tr_mask_list["m1"]
    k2 = RuntimeRegister._tr_mask_list["m2"]

    # distinct objects (cloned)
    assert t1 is not x1
    assert t2 is not x2
    assert k1 is not m1
    assert k2 is not m2

    # modifying the originals does not mutate registry values
    x1.add_(5.0)
    m2.logical_not_()

    assert torch.allclose(RuntimeRegister._tr_runtime_list["m1"], torch.ones(2, 3))
    assert torch.equal(RuntimeRegister._tr_mask_list["m2"], torch.zeros_like(k2))



def test_register_length_mismatch_raises():
    RuntimeRegister.module_superset = ["m1", "m2", "m3"]
    x = torch.zeros(1)
    m = torch.ones(1, dtype=torch.bool)
    with pytest.raises(ValueError, match=r"Expected 3 tf tensors .* got 2 and 3|Expected 3"):
        RuntimeRegister.register_runtime_args([x, x], [m, m, m])


def test_register_non_tensor_runtime_raises_typeerror():
    RuntimeRegister.module_superset = ["m1"]
    m = torch.ones(1, dtype=torch.bool)
    with pytest.raises(TypeError, match=r"must be a torch\.Tensor"):
        RuntimeRegister.register_runtime_args([123], [m])


def test_register_non_tensor_mask_raises():
    """
    Code doesn't explicitly type-check masks, but it calls .clone(), so a non-tensor
    mask will still fail. Assert that an exception is raised.
    """
    RuntimeRegister.module_superset = ["m1"]
    x = torch.zeros(1)
    with pytest.raises(Exception):
        RuntimeRegister.register_runtime_args([x], ["not-a-tensor"])  # .clone() will fail


def test_clear_runtime_args_empties_state():
    RuntimeRegister.module_superset = ["m1"]
    x = torch.zeros(1)
    m = torch.ones(1, dtype=torch.bool)
    RuntimeRegister.register_runtime_args([x], [m])

    assert RuntimeRegister._tr_runtime_list and RuntimeRegister._tr_mask_list

    RuntimeRegister.clear_runtime_args()
    assert RuntimeRegister._tr_runtime_list == {}
    assert RuntimeRegister._tr_mask_list == {}


def test_empty_superset_allows_empty_lists():
    RuntimeRegister.module_superset = []
    # Should not raise; just clears and stays empty
    RuntimeRegister.register_runtime_args([], [])
    assert RuntimeRegister._tr_runtime_list == {}
    assert RuntimeRegister._tr_mask_list == {}
