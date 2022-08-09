from functools import partial

import torch
from torch import nn

from .types import _Activation, _Norm

__all__ = ["GINLayer"]


class GINLayer(nn.Module):
    """Graph Attention Network as proposed in https://arxiv.org/pdf/1810.00826.pdf (ICLR 2019)

    Original implementation: https://github.com/weihua916/powerful-gnns
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: _Activation = partial(nn.ReLU, inplace=True),
        norm: _Norm = nn.BatchNorm1d,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            norm(output_dim),
            activation(),
            nn.Linear(output_dim, output_dim),
        )
        self.coef = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor) -> torch.Tensor:
        return self.mlp(x * self.coef + adj_mat @ x)
