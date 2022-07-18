from functools import partial
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch_geometric.data import Data as PyGData

from graph_nn.gcn import GCNLayer
from graph_nn.types import _Activation


class GCNModel(nn.Module):
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
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, norm_adj_mat: torch.Tensor) -> torch.Tensor:
        x = self.act(self.layer1(self.dropout(x), norm_adj_mat))
        return self.layer2(self.dropout(x), norm_adj_mat)

    @staticmethod
    def get_config(dataset: str) -> Dict[str, Any]:
        base_config = {
            "num_epochs": 300,
            "optimizer": partial(torch.optim.Adam, lr=1e-2),
            "l2_prefix": "layer1",
        }
        if dataset in ("CiteSeer", "Cora", "PubMed"):
            return {
                **base_config,
                "dropout": 0.5,
                "l2_regularization": 5e-4,
                "hidden_dim": 16,
            }
        if dataset == "NELL":
            return {
                **base_config,
                "dropout": 0.1,
                "l2_regularization": 1e-5,
                "hidden_dim": 64,
            }
        raise ValueError(f"Dataset {dataset} is not supported")

    @staticmethod
    def build_model(
        input_dim: int, num_classes: int, config: dict, data: PyGData
    ) -> Tuple[nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
        model = GCNModel(
            input_dim, config["hidden_dim"], num_classes, config["dropout"]
        )
        adj_mat = GCNLayer.normalize_adjacency_matrix(data.edge_index)
        return model, (data.x, adj_mat)
