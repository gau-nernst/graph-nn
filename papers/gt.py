import argparse
from functools import partial
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch_geometric.data import Data as PyGData

from graph_nn.gt import GTLayer
from graph_nn.types import _Activation
from graph_nn.utils import append_identity_matrix


class GTModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        # activation: _Activation = partial(nn.ELU, inplace=True),
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GTLayer(hidden_dim, num_heads, dropout=dropout))
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        for layer in self.layers:
            x = layer(x, edge_indices)
        return self.output(x)

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--dropout", type=float, default=0.2)

    @staticmethod
    def build_model(
        config: Dict[str, Any], data: PyGData
    ) -> Tuple[nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
        model = GTModel(**config)
        edge_index = append_identity_matrix(data.edge_index)
        return model, (data.x, edge_index)
