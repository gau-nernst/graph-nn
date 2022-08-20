import argparse
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch_geometric.data import Data as PyGData

from graph_nn import Dropout, SGCLayer


class SGCModel(nn.Module):
    l2_prefix = "layer1"

    def __init__(self, input_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.sgc = SGCLayer(input_dim, num_classes)
        self.dropout = Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sgc(self.dropout(x))

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument("--K", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0)

    @staticmethod
    def build_model(
        config: Dict[str, Any], data: PyGData
    ) -> Tuple[nn.Module, Tuple[torch.Tensor]]:
        config = config.copy()
        x = SGCLayer.preprocess_features(data.x, data.edge_index, config.pop("K"))
        model = SGCModel(**config)
        return model, (x,)
