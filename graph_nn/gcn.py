import torch
from torch import nn

from .utils import append_identity_matrix


__all__ = ["GCNLayer"]


class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
        return adj_mat @ self.linear(x)

    @staticmethod
    def normalize_adjacency_matrix(
        edge_indices: torch.Tensor, add_self_loops: bool = True
    ) -> torch.Tensor:
        if add_self_loops:
            edge_indices = append_identity_matrix(edge_indices)
        deg_rsqrt = torch.bincount(edge_indices[0]).rsqrt()  # out degree
        values = deg_rsqrt[edge_indices[0]] * deg_rsqrt[edge_indices[1]]
        return torch.sparse_coo_tensor(edge_indices, values).coalesce()
