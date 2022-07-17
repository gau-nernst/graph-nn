from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
# import torch_sparse

from .types import _Activation
from .utils import init_glorot_uniform


class GATLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        activation: _Activation = partial(nn.LeakyReLU, negative_slope=0.2),
        dropout: float = 0.0,
    ):
        assert output_dim % num_heads == 0
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.src_attn = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.dst_attn = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.act = activation()
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        init_glorot_uniform(self.linear.weight, self.linear.in_features, self.head_dim)
        init_glorot_uniform(self.src_attn, self.head_dim, 1, gain=0.5**0.5)
        init_glorot_uniform(self.dst_attn, self.head_dim, 1, gain=0.5**0.5)

    def forward(self, x: torch.Tensor, edge_indices: torch.Tensor):
        x = self.linear(x).reshape(-1, self.num_heads, self.head_dim)

        src_attn = F.embedding(edge_indices[0], (x * self.src_attn).sum(dim=-1))
        dst_attn = F.embedding(edge_indices[1], (x * self.dst_attn).sum(dim=-1))
        attn_coef = self.act(src_attn + dst_attn)
        attn_mat = torch.sparse_coo_tensor(edge_indices, attn_coef, requires_grad=True)
        attn_weights = torch.sparse.softmax(attn_mat, dim=1).coalesce()

        # It might be more efficient to apply edge dropping to inputs
        # instead of applying dropout to softmax weights
        indices = attn_weights.indices()
        values = self.dropout(attn_weights.values())

        head_outputs = []
        for i in range(self.num_heads):
            # Backward for sparse tensor can be buggy
            w_i = torch.sparse_coo_tensor(indices, values[:, i], requires_grad=True)
            head_outputs.append(torch.sparse.mm(w_i, x[:, i]))

            # Alternatively, use torch_sparse for sparse x dense matmul
            # head_outputs.append(torch_sparse.spmm(indices, values[i], x.shape[0], x.shape[0], x[:, i]))

        return torch.cat(head_outputs, dim=1)
