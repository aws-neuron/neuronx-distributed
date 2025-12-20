from neuronx_distributed.utils.tensor_capture.api import get_captured_tensors_dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        B, T, H = x.size()
        head_dim = H // self.num_heads
        q = self.q_proj(x).view(B, T, self.num_heads, head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim**0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v).reshape(B, T, H)
        out = self.o_proj(attn_out)
        return out  # (B, T, H)


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size // 2)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        g = F.silu(self.gate_proj(x))                # (B, T, inter)
        u = self.up_proj(x)                          # (B, T, inter//2)
        # pad mismatch
        if g.size(-1) != u.size(-1):
            diff = g.size(-1) - u.size(-1)
            u = F.pad(u, (0, diff))
        return self.down_proj(g * u)                 # (B, T, H)


class SimpleMoE(nn.Module):
    def __init__(self, hidden_size, expert_size=64, num_experts=4, top_k=2):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([nn.Linear(hidden_size, expert_size) for _ in range(num_experts)])
        self.combine = nn.Linear(expert_size, hidden_size)
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x):  # x: (B,T,H)
        # 1) Router
        router_logits = self.router(x)                      # (B,T,E)
        scores = F.softmax(router_logits, dim=-1)           # (B,T,E)
        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)  # (B,T,K), (B,T,K)

        # 2) Expert outputs for all experts in one stack: (B,T,E,S)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=2)   # S = expert_size

        # 3) Gather only the selected experts per token: (B,T,K,S)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, expert_outs.size(-1))
        chosen = torch.gather(expert_outs, 2, gather_idx)               # (B,T,K,S)

        # 4) Weight by top-k scores and reduce over K: (B,T,S)
        weighted = (chosen * topk_scores.unsqueeze(-1)).sum(dim=2)      # (B,T,S)

        # 5) Project back to hidden: (B,T,H)
        return self.combine(weighted)                                   # (B,T,H)



class ToyLayer(nn.Module):
    def __init__(self, hidden_size=128, intermediate_size=256, num_heads=4, moe=True):
        super().__init__()
        self.attn = SimpleAttention(hidden_size, num_heads)
        self.mlp = SimpleMLP(hidden_size, intermediate_size)
        self.moe = SimpleMoE(hidden_size) if moe else None
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Attention
        attn_out = self.attn(x)
        x = x + attn_out

        # MLP
        mlp_out = self.mlp(x)
        x = x + mlp_out

        # MoE
        if self.moe:
            moe_out = self.moe(x)
            x = x + moe_out

        return self.norm(x)


class ToyDeepModel(nn.Module):
    def __init__(self, num_layers=5, hidden_size=2048, intermediate_size=8192, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            ToyLayer(hidden_size, intermediate_size*(i+1), num_heads, moe=True)
            for i in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

        out = self.final_ln(x)
        # Only retrieve after final layer
        tensor_dict = get_captured_tensors_dict()
        if tensor_dict:
            return out, tensor_dict
        return out

