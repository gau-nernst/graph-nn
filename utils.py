from typing import Callable

import torch
from torch import nn
from torch.types import _device, _dtype

_Activation = Callable[[], nn.Module]
_Norm = Callable[[int], nn.Module]


def eye_sparse(n: int, dtype: _dtype = None, device: _device = None) -> torch.Tensor:
    indices = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)
    values = torch.ones(n, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, size=[n, n])


def add_self_loops(
    *,
    adj_mat: torch.Tensor = None,
    edge_indices: torch.Tensor = None,
    num_nodes: int = None
) -> torch.Tensor:
    if adj_mat is not None:
        assert adj_mat.is_sparse
        n, dtype, device = adj_mat.shape[0], adj_mat.dtype, adj_mat.device
        id_mat = eye_sparse(n, dtype, device)
        return (adj_mat + id_mat).coalesce()

    if edge_indices is not None:
        if num_nodes is None:
            num_nodes = edge_indices.max() + 1
        id_indices = torch.arange(num_nodes, device=edge_indices.device)
        return torch.cat([edge_indices, id_indices.unsqueeze(0).expand(2, -1)], dim=1)

    raise ValueError


@torch.no_grad()
def init_glorot(
    tensor: torch.Tensor, fan_in: int, fan_out: int, gain: float = 1.0
) -> torch.Tensor:
    a = gain * (6 / (fan_in + fan_out)) ** 0.5
    tensor.uniform_(-a, a)
    return tensor
