import torch
from torch import nn

from .utils import append_identity_matrix

__all__ = ["GCNLayer"]


class GCNLayer(nn.Module):
    """Graph Convolutional Network as proposed in https://arxiv.org/pdf/1609.02907.pdf (ICLR 2017)

    Original implementation: https://github.com/tkipf/gcn
    """

    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.empty(output_dim)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
        out = adj_mat @ self.linear(x)
        if self.bias is not None:
            out = out + self.bias
        return out

    @staticmethod
    def normalize_adjacency_matrix(
        edge_indices: torch.Tensor, add_self_loops: bool = True
    ) -> torch.Tensor:
        if add_self_loops:
            edge_indices = append_identity_matrix(edge_indices)
        deg_rsqrt = torch.bincount(edge_indices[0]).rsqrt()  # out degree
        values = deg_rsqrt[edge_indices[0]] * deg_rsqrt[edge_indices[1]]
        return torch.sparse_coo_tensor(edge_indices, values).coalesce()
