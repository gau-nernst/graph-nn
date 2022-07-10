from functools import partial
from typing import Optional

import torch
from torch import nn

from utils import _Activation, _Norm, add_self_loops


class GCNLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        norm: Optional[_Norm] = None,
        activation: _Activation = partial(nn.ReLU, inplace=True),
    ):
        super().__init__()
        self.norm = None if norm is None else norm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.act = activation()

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: Optional[torch.Tensor] = None,
        norm_adj_mat: Optional[torch.Tensor] = None,
    ):
        assert (edge_indices is not None) or (norm_adj_mat is not None)
        if norm_adj_mat is None:
            norm_adj_mat = self.create_norm_adj_mat(edge_indices, x.shape[0])
        if self.norm is not None:
            x = self.norm(x)
        return self.act(norm_adj_mat @ self.linear(x))

    @staticmethod
    def create_norm_adj_mat(edge_indices: torch.Tensor, num_nodes: int):
        edge_indices = add_self_loops(edge_indices=edge_indices, num_nodes=num_nodes)
        deg_rsqrt = torch.bincount(edge_indices[0], minlength=num_nodes).rsqrt()
        values = deg_rsqrt[edge_indices[0]] * deg_rsqrt[edge_indices[1]]
        return torch.sparse_coo_tensor(edge_indices, values).coalesce()
