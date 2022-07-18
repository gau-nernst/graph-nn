from functools import partial
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch_geometric.data import Data as PyGData

from graph_nn.gat import GATLayer
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
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        x = self.act(self.layer1(self.dropout(x), edge_indices))
        return self.layer2(self.dropout(x), edge_indices)

    @staticmethod
    def get_config(dataset: str) -> Dict[str, Any]:
        base_config = {
            "num_epochs": 1000,
            "dropout": 0.6,
            "hidden_dim": 64,
            "num_heads": 8,
        }
        if dataset in ("CiteSeer", "Cora"):
            return {
                **base_config,
                "l2_regularization": 5e-4,
                "optimizer": partial(torch.optim.Adam, lr=5e-3),
            }
        if dataset == "PubMed":
            return {
                **base_config,
                "l2_regularization": 1e-3,
                "optimizer": partial(torch.optim.Adam, lr=1e-2),
            }
        raise ValueError(f"Dataset {dataset} is not supported")

    @staticmethod
    def build_model(
        input_dim: int, num_classes: int, config: dict, data: PyGData
    ) -> Tuple[nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
        model = GATModel(
            input_dim,
            config["hidden_dim"],
            num_classes,
            config["num_heads"],
            config["dropout"],
        )
        edge_index = append_identity_matrix(data.edge_index)
        return model, (data.x, edge_index)
