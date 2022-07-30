from functools import partial
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .types import _Activation
from .utils import init_glorot_uniform


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
        x = self.linear(x).reshape(-1, self.num_heads, self.head_dim)
        
        src_attn = F.embedding(edge_indices[0], (x * self.src_attn).sum(dim=-1))
        dst_attn = F.embedding(edge_indices[1], (x * self.dst_attn).sum(dim=-1))
        attn_coef = self.act(src_attn + dst_attn)
        # attn_mat = torch.sparse_coo_tensor(edge_indices, attn_coef, requires_grad=True)
        # attn_weights = torch.sparse.softmax(attn_mat, dim=1).coalesce()

        # It might be more efficient to apply edge dropping to inputs
        # instead of applying dropout to softmax weights
        # indices = attn_weights.indices()
        # values = self.dropout(attn_weights.values())
        values = sparse_softmax(edge_indices, attn_coef, 1, [x.shape[0], x.shape[0], self.num_heads])
        values = self.dropout(values)

        # head_outputs = []
        # for i in range(self.num_heads):
        #     w_i = torch.sparse_coo_tensor(indices, values[:, i], requires_grad=True)
        #     head_outputs.append(torch.sparse.mm(w_i, x[:, i]))

        # if self.aggregate == "concat":
        #     return torch.cat(head_outputs, dim=1)
        # if self.aggregate == "mean":
        #     return sum(head_outputs) / self.num_heads

        values = x[edge_indices[1]] * values.unsqueeze(2)
        out = torch.zeros_like(x)
        out = out.scatter_add_(0, edge_indices[0, :, None, None].expand_as(values), values)
        if self.aggregate == "concat":
            return out.flatten(1)
        if self.aggregate == "mean":
            return out.mean(1)


def sparse_softmax(indices: torch.Tensor, values: torch.Tensor, dim: int, shape: List[int]) -> torch.Tensor:
    sparse_dim, dense_dim = indices.shape[0], values.dim() - 1
    assert dim < sparse_dim + dense_dim
    if dim >= sparse_dim:
        return values.softmax(dim - sparse_dim)

    index = indices[dim]
    reduce_shape = list(shape)
    reduce_shape.pop(dim)
    max_val = torch.zeros(reduce_shape, device=values.device)

    index_shape = list(index.shape) + [1] * dense_dim
    index_scatter = index.view(index_shape).expand_as(values)
    max_val = max_val.scatter_reduce(0, index_scatter, values, reduce="amax", include_self=False)
    values = values - max_val[index]
    
    values = values.exp()
    sum_val = torch.zeros(reduce_shape, device=values.device)
    sum_val = sum_val.scatter_add_(0, index_scatter, values)
    values = values / sum_val[index]

    return values
