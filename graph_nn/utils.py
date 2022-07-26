from typing import Optional

import torch
from torch import nn


__all__ = ["append_identity_matrix", "init_glorot_uniform", "Dropout"]


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
