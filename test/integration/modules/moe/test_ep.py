"""
torchrun --no_python --nproc_per_node=8 pytest -rA test/moe/test_ep.py
"""
import os

import neuronx_distributed as nxd
import pytest
import torch
import torch.distributed
from neuronx_distributed.parallel_layers.layers import divide
from neuronx_distributed.parallel_layers.mappings import (
    enter_expert_parallel_region,
    exit_expert_parallel_region,
)

# do distributed setup. test configuration for parallelism:
# - TP within node
# - EP across node
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="xla")
n_proc_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
n_nodes = divide(torch.distributed.get_world_size(), n_proc_per_node)
nxd.parallel_layers.initialize_model_parallel(
    tensor_model_parallel_size=n_proc_per_node,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=n_nodes,
)
nxd.parallel_layers.random.model_parallel_xla_manual_seed(0)


@pytest.mark.parametrize("n_experts", [4], ids=lambda n: f"e={n}")
@pytest.mark.parametrize("expert_capacity", [128], ids=lambda n: f"ec={n}")
@pytest.mark.parametrize("hidden_sz", [4, 64], ids=lambda n: f"h={n}")
@pytest.mark.parametrize("sp_input", [False, True], ids=lambda b: f"sp={int(b)}")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_ep_enter(
    n_experts: int,
    expert_capacity: int,
    hidden_sz: int,
    sp_input: bool,
    dtype: torch.dtype,
) -> None:
    trn = torch.device("xla")
    ep_group = nxd.parallel_layers.parallel_state.get_expert_model_parallel_group()
    tp_group = nxd.parallel_layers.parallel_state.get_tensor_model_parallel_group()

    n_experts_per_ep_rank = divide(n_experts, ep_group.size())
    assert n_experts_per_ep_rank == 1, "haven't implemented expert packing yet"

    # fully unpartitioned set of tokens. expert_capacity is for the full EP group.
    x_global = torch.rand(
        n_experts, expert_capacity, hidden_sz, dtype=dtype, device=trn
    )

    # tokens that are held by each DP_nonexp rank. (e, c/ep, h)
    if not sp_input:
        x = x_global.chunk(ep_group.size(), dim=1)[ep_group.rank()].contiguous()
    else:
        # fmt: off
        x = (
            x_global
            .chunk(ep_group.size(), dim=1)[ep_group.rank()]  # EP
            .chunk(tp_group.size(), dim=1)[tp_group.rank()]  # SP
        )
        # fmt: on

    x_ep = enter_expert_parallel_region(x, input_is_sequence_parallel=sp_input)

    # generate expected tensor after alltoall.
    expected = x_global.chunk(ep_group.size(), dim=0)[ep_group.rank()].view(
        n_experts_per_ep_rank,
        ep_group.size(),
        divide(expert_capacity, ep_group.size()),
        hidden_sz,
    )

    torch.testing.assert_close(x_ep, expected)


@pytest.mark.parametrize("n_experts", [4, 8], ids=lambda n: f"n={n}")
@pytest.mark.parametrize("expert_capacity", [16], ids=lambda n: f"c={n}")
@pytest.mark.parametrize("hidden_sz", [4], ids=lambda n: f"h: {n}")
@pytest.mark.parametrize("seq_parallel", [False, True], ids=lambda b: f"sp={int(b)}")
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
def test_ep_exit(
    n_experts: int,
    expert_capacity: int,
    hidden_sz: int,
    seq_parallel: bool,
    dtype: torch.dtype,
):
    trn = torch.device("xla")
    ep_group = nxd.parallel_layers.parallel_state.get_expert_model_parallel_group()
    tp_group = nxd.parallel_layers.parallel_state.get_tensor_model_parallel_group()

    capacity_per_ep_rank = divide(expert_capacity, ep_group.size())

    x_global = torch.rand(
        (n_experts, ep_group.size(), capacity_per_ep_rank, hidden_sz),
        dtype=dtype,
        device=trn,
    )

    # input: (e/ep, ep, c/sp, h)
    x = (
        x_global.detach()
        .chunk(ep_group.size(), dim=0)[ep_group.rank()]  # EP
        .chunk(tp_group.size(), dim=2)[tp_group.rank()]  # SP
        .contiguous()
    )

    out = exit_expert_parallel_region(x, output_in_sequence_parallel=seq_parallel)

    if seq_parallel:
        # expected output: (e, c/sp, h)
        expected = (
            x_global.detach()
            .chunk(ep_group.size(), dim=1)[ep_group.rank()]  # EP
            .chunk(tp_group.size(), dim=2)[tp_group.rank()]  # SP
            .view(n_experts, divide(capacity_per_ep_rank, tp_group.size()), hidden_sz)
        )
    else:
        # expected output: (e, c, h)
        expected = (
            x_global.detach()
            .chunk(ep_group.size(), dim=1)[ep_group.rank()]
            .view(n_experts, capacity_per_ep_rank, hidden_sz)
        )
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("expert_capacity", [256], ids=lambda n: f"ec={n}")
@pytest.mark.parametrize("hidden_sz", [128], ids=lambda n: f"h={n}")
@pytest.mark.parametrize("n_experts", [4, 8], ids=lambda n: f"e={n}")
@pytest.mark.parametrize("output_sp", [False, True], ids=lambda b: f"sp_out={int(b)}")
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
def test_ep_enter_then_exit_inversion(
    n_experts: int,
    expert_capacity: int,
    hidden_sz: int,
    output_sp: bool,
    dtype: torch.dtype,
) -> None:
    """we expect that exiting EP should be an inversion of entering EP, or in other
    words that entering and then immediate exiting should get us back to the
    original input.
    """
    trn = torch.device("xla")
    dp_group = nxd.parallel_layers.parallel_state.get_data_parallel_group()
    tp_group = nxd.parallel_layers.parallel_state.get_tensor_model_parallel_group()

    n_tokens_per_dp_rank = expert_capacity * n_experts

    # need to create these values based off DP rank. (e, c, h)
    x = (
        (dp_group.rank() * n_tokens_per_dp_rank + torch.arange(0, n_tokens_per_dp_rank))
        .view(n_experts, expert_capacity, 1)
        .repeat([1, 1, hidden_sz])
        .to(dtype=dtype, device=trn)
    )
    # (e, c, h) -> (e/ep, ep, c, h)
    x_ep = enter_expert_parallel_region(x, input_is_sequence_parallel=False)
    # here we mimic a dropping operation, normally there would be an MLP in
    # between the two operations which would do this via a reduce-scatter.
    # (e/ep, ep, c/sp, h)
    x_dropped = x_ep.chunk(dim=2, chunks=tp_group.size())[tp_group.rank()]
    # (e/ep, ep, c/sp, h) -> (e, c, h)
    x_out = exit_expert_parallel_region(
        x_dropped, output_in_sequence_parallel=output_sp
    )

    if output_sp:
        torch.testing.assert_close(
            x_out, x.chunk(tp_group.size(), dim=1)[tp_group.rank()]
        )
    else:
        torch.testing.assert_close(x_out, x)
