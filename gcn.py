from functools import partial
from typing import Callable, Optional

import torch
from torch import nn

from utils import add_self_loops


class GCNLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        norm: Optional[Callable[[int], nn.Module]] = None,
        activation: Callable[[], nn.Module] = partial(nn.ReLU, inplace=True),
    ):
        super().__init__()
        self.norm = None if norm is None else norm(input_dim)
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        self.act = activation()

        nn.init.kaiming_normal_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_indices: Optional[torch.Tensor] = None,
        norm_adj_mat: Optional[torch.Tensor] = None,
    ):
        assert (edge_indices is not None) or (norm_adj_mat is not None)
        if norm_adj_mat is None:
            norm_adj_mat = self.create_norm_adj_mat(edge_indices)
        if self.norm is not None:
            x = self.norm(x)
        return self.act(norm_adj_mat @ x @ self.weight)

    @staticmethod
    def create_norm_adj_mat(edge_indices: torch.Tensor, num_nodes: int):
        edge_indices = add_self_loops(edge_indices=edge_indices, num_nodes=num_nodes)
        deg_rsqrt = torch.bincount(edge_indices[0], minlength=num_nodes).rsqrt()
        values = deg_rsqrt[edge_indices[0]] * deg_rsqrt[edge_indices[1]]
        return torch.sparse_coo_tensor(edge_indices, values).coalesce()
