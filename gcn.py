import torch
from torch import nn

from utils import add_self_loops


class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
        return adj_mat @ self.linear(x)

    @staticmethod
    def normalize_adj_mat(edge_indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
        edge_indices = add_self_loops(edge_indices=edge_indices, num_nodes=num_nodes)
        deg_rsqrt = torch.bincount(edge_indices[0], minlength=num_nodes).rsqrt()
        values = deg_rsqrt[edge_indices[0]] * deg_rsqrt[edge_indices[1]]
        return torch.sparse_coo_tensor(edge_indices, values).coalesce()
