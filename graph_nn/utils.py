from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "append_identity_matrix",
    "init_glorot_uniform",
    "Dropout",
    "sparse_softmax",
    "sparse_aggregate",
]


def append_identity_matrix(
    edge_indices: torch.Tensor, num_nodes: Optional[int] = None
) -> torch.Tensor:
    num_nodes = num_nodes or edge_indices.max() + 1
    id_indices = torch.arange(num_nodes, device=edge_indices.device)
    return torch.cat([edge_indices, id_indices.unsqueeze(0).expand(2, -1)], dim=1)


@torch.no_grad()
def init_glorot_uniform(
    tensor: torch.Tensor, fan_in: int, fan_out: int, gain: float = 1.0
) -> torch.Tensor:
    a = gain * (6 / (fan_in + fan_out)) ** 0.5
    tensor.uniform_(-a, a)
    return tensor


class Dropout(nn.Dropout):
    """Dropout with support for sparse COO tensors."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            x = x.coalesce()
            values = super().forward(x.values())
            return torch.sparse_coo_tensor(x.indices(), values, x.size())
        return super().forward(x)


def sparse_softmax(
    idx: torch.Tensor, values: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Apply softmax for COO sparse tensor. Softmax is applied for each row if `idx` is row indices, or similarly for column indices.
    Values can represent N-dim weight coefficients, such as multi-head mechanism in GAT.

    If you pass in row indices, it is equivalent to `torch.sparse.softmax(torch.sparse_coo_tensor(indices, values), dim=1).coalesce().values()`.

    Args:
        idx: row or column indices. Shape (nnz,)
        values: weight coefficients. Shape (nnz, dim1, dim2, ...)
        num_nodes: number of nodes
    """

    reduced_shape = (num_nodes,) + values.shape[1:]
    expanded_row_idx = idx.view([-1] + [1] * (values.dim() - 1)).expand_as(values)

    # Tensor.scatter_reduce() is only available in PyTorch 1.12
    if torch.__version__ >= (1, 12, 0):
        max_val = torch.zeros(reduced_shape, device=values.device)
        max_val = max_val.scatter_reduce(
            0, expanded_row_idx, values, reduce="amax", include_self=False
        )
        values = values - F.embedding(idx, max_val)

    values = values.exp()
    sum_val = torch.zeros(reduced_shape, device=values.device)
    sum_val = sum_val.scatter_add_(0, expanded_row_idx, values)
    values = values / F.embedding(idx, sum_val)

    return values


def sparse_aggregate(
    idx: torch.Tensor, values: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Aggregate N-dim features from neighbors. This can be used for multi-head mechanism in GAT.

    This is equivalent to `torch.sparse.sum(torch.sparse_coo_tensor(indices, values), dim=1).to_dense()`.

    Args:
        idx: row indices. Shape (nnz,)
        values: features from neighbors. Shape (nnz, dim1, dim2, ...)
        num_nodes: number of nodes
    """

    idx = idx.view([-1] + [1] * (values.dim() - 1)).expand_as(values)
    out = torch.zeros((num_nodes,) + values.shape[1:], device=values.device)
    return out.scatter_add_(0, idx, values)
