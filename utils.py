from typing import Callable

import torch
from torch import nn

_Activation = Callable[[], nn.Module]
_Norm = Callable[[int], nn.Module]


def add_self_loops(edge_indices: torch.Tensor, num_nodes: int = None) -> torch.Tensor:
    if num_nodes is None:
        num_nodes = edge_indices.max() + 1
    id_indices = torch.arange(num_nodes, device=edge_indices.device)
    return torch.cat([edge_indices, id_indices.unsqueeze(0).expand(2, -1)], dim=1)


@torch.no_grad()
def init_glorot(
    tensor: torch.Tensor, fan_in: int, fan_out: int, gain: float = 1.0
) -> torch.Tensor:
    a = gain * (6 / (fan_in + fan_out)) ** 0.5
    tensor.uniform_(-a, a)
    return tensor
