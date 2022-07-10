from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from utils import _Activation, add_self_loops


class GATLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        activation: _Activation = partial(nn.LeakyReLU, negative_slope=0.2),
        dropout: float = 0.6,
    ):
        assert output_dim % num_heads == 0
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.dropout = dropout

        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.src_attn = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.dst_attn = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.act = activation()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.src_attn, nonlinearity="linear")
        nn.init.kaiming_normal_(self.dst_attn, nonlinearity="linear")

    def forward(self, x: torch.Tensor, edge_indices: torch.Tensor):
        x = F.dropout(x, p=self.dropout, training=self.training)
        edge_indices = add_self_loops(edge_indices=edge_indices, num_nodes=x.shape[0])

        # dropout after linear is not mentioned in the paper
        x = self.linear(x).reshape(-1, self.num_heads, self.head_dim)
        x = F.dropout(x, p=self.dropout, training=self.training)

        src_attn = (x * self.src_attn).sum(dim=-1)[edge_indices[0]]
        dst_attn = (x * self.dst_attn).sum(dim=-1)[edge_indices[1]]
        attn = self.act(src_attn + dst_attn)

        head_outputs = []
        for i in range(self.num_heads):
            attn_coef_mat = torch.sparse_coo_tensor(edge_indices, attn[:, i])
            attn_weights_mat = torch.sparse.softmax(attn_coef_mat, dim=1)
            attn_weights_mat = sparse_dropout(
                attn_weights_mat, self.dropout, training=self.training
            )
            head_outputs.append(torch.sparse.mm(attn_weights_mat, x[:, i]))
        return torch.cat(head_outputs, dim=1)


def sparse_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training:
        return x
    x = x.coalesce()
    values = F.dropout(x.values(), p=p, training=True)
    return torch.sparse_coo_tensor(x.indices(), values, x.shape)
