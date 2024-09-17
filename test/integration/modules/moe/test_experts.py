import os
from types import SimpleNamespace
from typing import Callable, Dict

import neuronx_distributed as nxd
import pytest
import torch
import torch.distributed
import torch_xla.core.xla_model as xm
from neuronx_distributed.modules.moe.experts import Experts
from neuronx_distributed.parallel_layers.layers import divide
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_tensor_model_parallel_rank,
)
from torch import Tensor
from torch.nn import Module, ModuleList
from transformers.models.llama.modeling_llama import LlamaMLP

torch.set_printoptions(precision=2, linewidth=320, sci_mode=False)


if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="xla")

# test configuration for parallelism:
# - TP within node
# - EP across node
# TODO. also need to test cases where there is DP_exp > 1
n_proc_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
n_nodes = divide(torch.distributed.get_world_size(), n_proc_per_node)
nxd.parallel_layers.initialize_model_parallel(
    tensor_model_parallel_size=n_proc_per_node,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=n_nodes,
)
nxd.parallel_layers.random.model_parallel_xla_manual_seed(0)

neuron_cc_flags = [
    "--auto-cast none",
    # "--internal-compiler-debug-mode=all"
]
os.environ["NEURON_CC_FLAGS"] = " ".join(neuron_cc_flags)


def assert_close(actual, expected, name):
    tp_rank = get_tensor_model_parallel_rank()
    ep_rank = get_expert_model_parallel_rank()
    dp_rank = get_data_parallel_rank()

    def rank_msg(m):
        return f"TP={tp_rank}, EP={ep_rank}, DP={dp_rank}, {m}"

    try:
        torch.testing.assert_close(actual, expected, atol=1e-1, rtol=5e-3, msg=rank_msg)
    except AssertionError as e:
        print(f"EP={ep_rank}, TP={tp_rank}, DP={dp_rank}")
        print(f"actual {name}")
        print(actual)
        print(f"expected {name}")
        print(expected)

        raise e


class ReferenceLlamaExperts(Module):
    """unpartitioned, naive implementation of expert MLPs"""

    def __init__(
        self, n_experts: int, hidden_size: int, intermediate_size: int
    ) -> None:
        super().__init__()
        self.__n_experts = n_experts
        self.__hidden_size = hidden_size
        cfg = SimpleNamespace(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            pretraining_tp=1,
            hidden_act="silu",
        )
        self.experts = ModuleList([LlamaMLP(cfg) for _ in range(n_experts)])

    def forward(self, x_routed: Tensor) -> Tensor:
        """works by iterating through experts and applying MLP to each

        Args:
            x_routed (n_experts, expert_capacity, hidden_sz)

        Returns:
            output (n_experts, expert_capacity, hidden_sz)
        """
        input_n_experts, _, input_hidden_sz = x_routed.shape
        assert input_n_experts == self.__n_experts
        assert input_hidden_sz == self.__hidden_size

        expert_outputs = [None] * self.__n_experts
        for expert_idx, expert in enumerate(self.experts):
            expert_outputs[expert_idx] = expert.forward(x_routed[expert_idx, :, :])
        output = torch.stack(expert_outputs)

        return output


def _tp_partition(x: Tensor, dim: int) -> Tensor:
    tp_group = nxd.parallel_layers.parallel_state.get_tensor_model_parallel_group()
    return x.chunk(tp_group.size(), dim=dim)[tp_group.rank()]


def _convert_state_dict_expert_fused(
    state_dict: Dict[str, Tensor], n_experts: int
) -> Dict[str, Tensor]:
    ep_group = nxd.parallel_layers.parallel_state.get_expert_model_parallel_group()

    experts_per_ep_group = divide(n_experts, ep_group.size())
    local_expert_indices = list(
        range(
            (ep_group.rank() + 0) * experts_per_ep_group,
            (ep_group.rank() + 1) * experts_per_ep_group,
        )
    )

    # copy the MLP parameters, accounting for TP.
    up_gate_proj = torch.stack(
        [
            torch.cat(
                [
                    _tp_partition(state_dict[f"experts.{e}.up_proj.weight"], dim=0),
                    _tp_partition(state_dict[f"experts.{e}.gate_proj.weight"], dim=0),
                ],
                dim=0,
            )
            for e in local_expert_indices
        ]
    )
    down_proj = torch.stack(
        [
            _tp_partition(state_dict[f"experts.{e}.down_proj.weight"], dim=1)
            for e in local_expert_indices
        ]
    )

    return {
        "up_gate_proj.weight": up_gate_proj,
        "down_proj.weight": down_proj,
    }


@pytest.mark.parametrize("expert_capacity", [32, 64], ids=lambda n: f"c={n}")
@pytest.mark.parametrize("hidden_sz", [4], ids=lambda n: f"h={n}")
@pytest.mark.parametrize("inter_sz", [32], ids=lambda n: f"i={n}")
@pytest.mark.parametrize("n_experts", [4, 8], ids=lambda n: f"e={n}")
@pytest.mark.parametrize("glu", [True], ids=lambda b: f"glu={int(b)}")
@pytest.mark.parametrize(
    "activation_fn", [torch.nn.functional.silu], ids=lambda f: f"act={f}"
)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
def test_forward_backward(
    expert_capacity: int,
    hidden_sz: int,
    inter_sz: int,
    n_experts: int,
    glu: bool,
    activation_fn: Callable[[Tensor], Tensor],
    dtype: torch.dtype,
):
    ######################################################################################
    # ARRANGE
    ######################################################################################
    tp_group = nxd.parallel_layers.parallel_state.get_tensor_model_parallel_group()
    ep_group = nxd.parallel_layers.parallel_state.get_expert_model_parallel_group()

    trn = torch.device("xla")
    cpu = torch.device("cpu")
    # removed support but leaving test code in case we need it later
    output_sp = False

    n_experts_per_ep_rank = divide(n_experts, ep_group.size())
    inter_sz_local = divide(inter_sz, tp_group.size())

    ref_experts = (
        ReferenceLlamaExperts(
            n_experts=n_experts, hidden_size=hidden_sz, intermediate_size=inter_sz
        )
        .to(dtype=dtype, device=cpu)
        .requires_grad_(True)
    )

    experts = Experts(
        n_experts=n_experts,
        hidden_size=hidden_sz,
        intermediate_size=inter_sz,
        # only testing EP path here, which isn't compatible with this being false
        reduce_output=True,
        glu=glu,
        activation_fn=activation_fn,
        dtype=dtype,
        device=trn,
    )
    experts.load_state_dict(
        _convert_state_dict_expert_fused(ref_experts.state_dict(), n_experts=n_experts),
        strict=True,
    )

    xm.mark_step()

    x_global_ref = torch.randn(
        # in this case the "expert_capacity" is for the full EP group.
        # so it will contain tokens from multiple DP_nonexp groups.
        (n_experts, expert_capacity, hidden_sz),
        requires_grad=True,
        dtype=dtype,
        device=cpu,
    )
    # input: (e, c/ep, h)
    x = (
        x_global_ref.detach()
        .chunk(ep_group.size(), dim=1)[ep_group.rank()]  # EP
        .to(dtype=dtype, device=trn)
        .requires_grad_(True)
    )
    xm.mark_step()

    ######################################################################################
    # ACT
    ######################################################################################
    # impl fwd
    output = experts.forward(x)
    output.sum().backward(retain_graph=True)

    xm.mark_step()

    # reference fwd
    output_global_ref = ref_experts.forward(x_global_ref)
    output_global_ref.sum().backward(retain_graph=True)

    ######################################################################################
    # ASSERT
    ######################################################################################
    # check output
    if not output_sp:
        expected_output = output_global_ref.chunk(ep_group.size(), dim=1)[
            ep_group.rank()
        ]
    else:
        # fmt: off
        expected_output = (
            output_global_ref
            .chunk(ep_group.size(), dim=1)[ep_group.rank()]
            .chunk(tp_group.size(), dim=1)[tp_group.rank()]
        )
        # fmt: on
    assert_close(output.cpu(), expected_output, "output")

    # check weight grads
    for local_expert_idx in range(n_experts_per_ep_rank):
        global_expert_idx = ep_group.rank() * n_experts_per_ep_rank + local_expert_idx
        ref_expert: LlamaMLP = ref_experts.experts[global_expert_idx]

        # down (row parallel projection)
        assert_close(
            experts.down_proj.weight.grad[local_expert_idx, :, :].cpu(),
            _tp_partition(ref_expert.down_proj.weight.grad, dim=1),
            "down_weight.grad",
        )

        # up (col parallel projection)
        assert_close(
            experts.up_gate_proj.weight.grad[
                local_expert_idx, :inter_sz_local, :
            ].cpu(),
            _tp_partition(ref_expert.up_proj.weight.grad, dim=0),
            "up_weight.grad",
        )

        # gate (col parallel projection)
        assert_close(
            experts.up_gate_proj.weight.grad[
                local_expert_idx, inter_sz_local:, :
            ].cpu(),
            _tp_partition(ref_expert.gate_proj.weight.grad, dim=0),
            "gate_weight.grad",
        )

    # check input grad
    assert_close(
        x.grad.cpu(),
        x_global_ref.grad.chunk(ep_group.size(), dim=1)[ep_group.rank()],
        "input.grad",
    )


if __name__ == "__main__":
    """pytest swallows some of the neuron errors so run as python script if you need
    to debug things like that."""
    test_forward_backward(
        expert_capacity=32, hidden_sz=4, inter_sz=32, n_experts=4, dtype=torch.bfloat16
    )
