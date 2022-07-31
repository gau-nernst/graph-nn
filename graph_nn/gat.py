from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from .types import _Activation
from .utils import init_glorot_uniform, sparse_aggregate, sparse_row_softmax

__all__ = ["GATLayer"]


class GATLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        activation: _Activation = partial(nn.LeakyReLU, negative_slope=0.2),
        dropout: float = 0.0,
        aggregate: str = "concat",
    ):
        assert aggregate in ("concat", "mean")
        if aggregate == "concat":
            assert output_dim % num_heads == 0
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads if aggregate == "concat" else output_dim
        self.aggregate = aggregate

        self.linear = nn.Linear(input_dim, self.head_dim * num_heads, bias=False)
        self.src_attn = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.dst_attn = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.act = activation()
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_glorot_uniform(self.linear.weight, self.linear.in_features, self.head_dim)
        init_glorot_uniform(self.src_attn, self.head_dim * 2, 1)
        init_glorot_uniform(self.dst_attn, self.head_dim * 2, 1)

    def forward(self, x: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        row_idx, col_idx = edge_indices[0], edge_indices[1]
        x = self.linear(x).reshape(-1, self.num_heads, self.head_dim)

        src_attn = F.embedding(row_idx, (x * self.src_attn).sum(dim=-1))
        dst_attn = F.embedding(col_idx, (x * self.dst_attn).sum(dim=-1))
        attn_coef = self.act(src_attn + dst_attn)

        values = sparse_row_softmax(row_idx, attn_coef, x.shape[0])
        values = self.dropout(values)  # DropEdge might be more efficient

        values = x[col_idx] * values.unsqueeze(2)
        x = sparse_aggregate(row_idx, values, x.shape[0])
        if self.aggregate == "concat":
            return x.flatten(1)
        if self.aggregate == "mean":
            return x.mean(1)
