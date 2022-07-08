from functools import partial
from typing import Callable, Optional

import torch
from torch import nn


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
    def normalize_adjacency_matrix(adjacency_matrix: torch.Tensor):
        assert adjacency_matrix.is_sparse
        adjacency_matrix = adjacency_matrix.cpu().float()
        n = adjacency_matrix.shape[0]
        diag, size = [list(range(n))] * 2, [n] * 2

        identity_matrix = torch.sparse_coo_tensor(diag, [1.0] * n, size=size)
        adjacency_matrix = (adjacency_matrix + identity_matrix).coalesce()
        degree_invsqrt = torch.sparse.sum(adjacency_matrix, dim=0).to_dense().rsqrt()

        indices = adjacency_matrix._indices()
        values = adjacency_matrix._values()
        for idx, (i, j) in enumerate(zip(indices[0], indices[1])):
            values[idx] *= degree_invsqrt[i] * degree_invsqrt[j]
        return torch.sparse_coo_tensor(indices, values, size=size)
