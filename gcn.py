from functools import partial
from typing import Callable, Optional

import torch
from torch import nn

from utils import add_self_loops


class GCNLayer(nn.Module):
    def __init__(
        self,
        adjacency_matrix: torch.Tensor,
        input_dim: int,
        output_dim: int,
        is_normalized: bool = False,
        norm: Optional[Callable[[int], nn.Module]] = None,
        activation: Callable[[], nn.Module] = partial(nn.ReLU, inplace=True),
    ):
        super().__init__()
        if not is_normalized:
            adjacency_matrix = self.normalize_adjacency_matrix(adjacency_matrix)
        self.register_buffer("norm_adj_mat", adjacency_matrix)
        self.norm = None if norm is None else norm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.act = activation()

    def forward(self, x: torch.Tensor):
        if self.norm is not None:
            x = self.norm(x)
        return self.act(self.norm_adj_mat @ self.linear(x))

    @staticmethod
    def normalize_adjacency_matrix(adj_mat: torch.Tensor):
        assert adj_mat.is_sparse
        adj_mat = add_self_loops(adj_mat).float()
        degree_invsqrt = torch.sparse.sum(adj_mat, dim=0).to_dense().rsqrt()

        indices, values = adj_mat.indices(), adj_mat.values()
        values *= degree_invsqrt[indices[0]] * degree_invsqrt[indices[1]]
        return adj_mat
