from dataclasses import dataclass

import torch

FP32_TEST_TOLS = {
    "atol": 5e-6,
    "rtol": 1e-4,
}

BF16_TEST_TOLS = {
    "atol": 2e-2,
    "rtol": 2e-2,
}


@dataclass
class LossFnExptCfg:
    # encapsulate configs
    batch_size: int
    seq_len: int
    num_layers: int
    num_experts: int
    top_k: int
    dtype: torch.dtype
    device: torch.device = "cpu"
    num_iters: int = 10


# TODO: Call directly from HuggingFace transformers once it is more stable and released.
# For now replicating, because there are bug fixes actively being made here,
# which are not yet in any release branch
def hf_load_balancing_loss_func(gate_logits: torch.Tensor, num_experts: int = -1, top_k=2) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # Compute the percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * (num_experts / top_k)


def get_loss_fn_correctness_test_configs(dtypes):
    test_configs = []
    for dtype in dtypes:
        test_configs_for_dtype = [
            LossFnExptCfg(batch_size=1, seq_len=128, num_layers=32, num_experts=8, top_k=2, dtype=dtype),
            LossFnExptCfg(batch_size=1, seq_len=128, num_layers=128, num_experts=8, top_k=1, dtype=dtype),
            LossFnExptCfg(batch_size=1, seq_len=128, num_layers=128, num_experts=8, top_k=2, dtype=dtype),
            LossFnExptCfg(batch_size=1, seq_len=2048, num_layers=32, num_experts=8, top_k=2, dtype=dtype),
        ]
        test_configs.extend(test_configs_for_dtype)
    return test_configs
