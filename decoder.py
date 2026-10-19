

import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentionLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 主线性变换权重和偏置
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.bias = nn.Parameter(torch.randn(out_dim))


        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Softmax(dim=-1)
        )

        # 残差映射：将输入维度映射到输出维度，用于残差连接
        self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):

        attn_weights = self.attention(x)
        weighted_x = x * attn_weights
        linear_out = F.linear(weighted_x, self.weight, self.bias)


        residual = self.residual_proj(x)
        out = linear_out + 1*residual
        return out

class MultiEndBranchAttention(nn.Module):
    """
    针对每个端元一支分支：
      输入:  (n_end, hidden_dim)
      输出:  (n_end, hidden_dim)  —— 每个端元语义单独增强
    """
    def __init__(self, n_end, hidden_dim):
        super().__init__()
        self.n_end = n_end
        self.hidden_dim = hidden_dim

        # 为每个端元单独建一支 AttentionLinear: 200 -> 200
        self.branches = nn.ModuleList([
            AttentionLinear(hidden_dim, hidden_dim) for _ in range(n_end)
        ])

    def forward(self, x):
        """
        x: (n_end, hidden_dim)
        """
        assert x.dim() == 2, f"Expect x shape (n_end, hidden_dim), got {x.shape}"
        assert x.shape[0] == self.n_end, f"n_end mismatch: {x.shape[0]} vs {self.n_end}"

        outs = []
        for i in range(self.n_end):

            xi = x[i].unsqueeze(0)
            yi = self.branches[i](xi)
            outs.append(yi.squeeze(0))

        out = torch.stack(outs, dim=0)
        return out
