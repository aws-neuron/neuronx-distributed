import torch
import torch.nn.functional as F


def load_balancing_loss_func(router_logits, num_experts, top_k):
    """Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper, which penalizes imbalanced expert assignments by the
    router.

    Args:
        router_logits: Router logits from all layers, concatenated into a single tensor.
        num_experts: Total number of experts.
        top_k: Number of experts activated per token. Should be less than or equal to num_experts.

    Returns:
        load_balancing_loss
    """

    # router_logits: (batch_size * seq_len * num_layers, num_experts)

    # Normalize router_logits to get expert_affinities
    # (perform in fp64 to prevent auto-downcasting of operation to bf16, for numerical accuracy)
    expert_affinities = F.softmax(router_logits, dim=-1, dtype=torch.float64)

    _, selected_experts = torch.topk(expert_affinities, top_k)

    # Implement efficient version of one-hot encoding for xla device
    expert_masks_list = []
    expert_num_idx_arr = torch.arange(num_experts, device=selected_experts.device, dtype=torch.float64)
    for e in range(top_k):
        # Append one-hot (BSL, E) to expert_masks_list
        expert_masks_list.append((selected_experts[:, e].unsqueeze(1) == expert_num_idx_arr).to(torch.float32))
    # expert_mask = (BSL, top_k, E)
    expert_mask = torch.stack(expert_masks_list, dim=1)

    # Find percentage of tokens routed to each expert
    tokens_per_expert = torch.mean(expert_mask, dim=0)

    # Compute the average probability of routing to these experts
    router_prob_per_expert = torch.mean(expert_affinities.to(torch.float32), dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))

    # Scale the router loss in a way that is agnostic to the model architecture
    # Reference: https://github.com/stanford-futuredata/megablocks/blob/f05609ce69c1e1a7dd008c49cf435ef74df84b69/megablocks/layers/moe.py#L84
    return overall_loss * (num_experts / top_k)
