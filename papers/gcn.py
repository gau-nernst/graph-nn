import argparse
from functools import partial
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch_geometric.data import Data as PyGData

from graph_nn import Dropout, GCNLayer
from graph_nn.types import _Activation


class GCNModel(nn.Module):
    l2_prefix = "layer1"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        activation: _Activation = partial(nn.ReLU, inplace=True),
    ):
        super().__init__()
        self.layer1 = GCNLayer(input_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, num_classes)
        self.act = activation()
        self.dropout = Dropout(p=dropout)

    def forward(self, x: torch.Tensor, norm_adj_mat: torch.Tensor) -> torch.Tensor:
        x = self.act(self.layer1(self.dropout(x), norm_adj_mat))
        return self.layer2(self.dropout(x), norm_adj_mat)

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument("--hidden_dim", type=int, default=16)
        parser.add_argument("--dropout", type=float, default=0.5)

    @staticmethod
    def build_model(
        config: Dict[str, Any], data: PyGData
    ) -> Tuple[nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
        model = GCNModel(**config)
        adj_mat = GCNLayer.normalize_adjacency_matrix(data.edge_index)
        return model, (data.x, adj_mat)
