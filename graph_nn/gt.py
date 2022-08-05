from functools import partial
from typing import Optional

import torch
from torch import nn

from .types import _Activation
from .utils import sparse_aggregate, sparse_row_softmax


class GTLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: _Activation = partial(nn.ReLU, inplace=True),
        norm_first: bool = False,
    ):
        assert embed_dim % num_heads == 0
        if feedforward_dim is None:
            feedforward_dim = embed_dim * 4
        super().__init__()
        self.attn_linear = nn.Linear(embed_dim, embed_dim * 3)
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.scale = embed_dim ** (-0.5)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            activation(),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(p=dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            x = x + self._self_attention(self.norm1(x), edge_indices)
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(x + self._self_attention(x, edge_indices))
            x = self.norm2(x + self.ffn(x))
        return x

    def _self_attention(
        self, x: torch.Tensor, edge_indices: torch.Tensor
    ) -> torch.Tensor:
        row_idx, col_idx = edge_indices[0], edge_indices[1]
        qkv = self.attn_linear(x).reshape(-1, self.num_heads, self.head_dim * 3)
        q, k, v = qkv.chunk(3, dim=2)
        alpha = (q[row_idx] * k[col_idx]).sum(2) * self.scale
        weights = sparse_row_softmax(row_idx, alpha, x.shape[0])
        values = v[col_idx] * weights.unsqueeze(2)
        return sparse_aggregate(row_idx, values, x.shape[0]).flatten(1)

