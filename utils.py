import torch
from torch.types import _device, _dtype


def eye_sparse(n: int, dtype: _dtype = None, device: _device = None) -> torch.Tensor:
    indices = torch.arange(n, device=device).unsqueeze(0).repeat(2, 1)
    values = torch.ones(n, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values)


def add_self_loops(adj_mat: torch.Tensor) -> torch.Tensor:
    assert adj_mat.is_sparse
    id_mat = eye_sparse(adj_mat.shape[0], dtype=adj_mat.dtype, device=adj_mat.device)
    return (adj_mat + id_mat).coalesce()
