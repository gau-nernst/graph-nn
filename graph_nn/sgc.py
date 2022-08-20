import torch
from torch import nn

from .gcn import GCNLayer


class SGCLayer(nn.Linear):
    """Simple Graph Convolution as proposed in https://arxiv.org/pdf/1902.07153.pdf (ICML 2019)

    Official implementation: https://github.com/Tiiiger/SGC
    """

    @staticmethod
    def preprocess_features(
        x: torch.Tensor,
        edge_indices: torch.Tensor,
        K: int,
        add_self_loops: bool = True,
    ) -> torch.Tensor:
        adj = GCNLayer.normalize_adjacency_matrix(
            edge_indices, add_self_loops=add_self_loops
        )
        for _ in range(K):
            x = adj @ x
        return x
