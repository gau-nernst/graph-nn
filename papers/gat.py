import argparse
from functools import partial
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch_geometric.data import Data as PyGData

from graph_nn import GATLayer, Dropout
from graph_nn.types import _Activation
from graph_nn.utils import append_identity_matrix


class GATModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int,
        dropout: float,
        activation: _Activation = partial(nn.ELU, inplace=True),
    ):
        super().__init__()
        self.layer1 = GATLayer(input_dim, hidden_dim, num_heads, dropout=dropout)
        self.layer2 = GATLayer(hidden_dim, num_classes, 1, dropout=dropout)
        self.act = activation()
        self.dropout = Dropout(p=dropout)

    def forward(self, x: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        x = self.act(self.layer1(self.dropout(x), edge_indices))
        return self.layer2(self.dropout(x), edge_indices)

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--num_heads", type=int, default=8)
        parser.add_argument("--dropout", type=float, default=0.6)

    @staticmethod
    def build_model(
        config: Dict[str, Any], data: PyGData
    ) -> Tuple[nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
        model = GATModel(**config)
        edge_index = append_identity_matrix(data.edge_index)
        return model, (data.x, edge_index)
