from functools import partial

import torch
from torch import nn

from utils import _Activation, add_self_loops


class GATLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        activation: _Activation = partial(nn.LeakyReLU, negative_slope=0.2),
        dropout: float = 0.0,
    ):
        assert output_dim % num_heads == 0
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.src_attn = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.dst_attn = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        self.act = activation()
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.src_attn, nonlinearity="linear")
        nn.init.kaiming_normal_(self.dst_attn, nonlinearity="linear")

    def forward(self, x: torch.Tensor, edge_indices: torch.Tensor):
        edge_indices = add_self_loops(edge_indices=edge_indices, num_nodes=x.shape[0])

        # Input dropout is expensive when node emmbeddings have high dimensionality,
        # while not effective when the embeddings are sparse.
        # Dropout after linear is not mentioned in the paper.
        # x = self.dropout(x)
        x = self.linear(x).reshape(-1, self.num_heads, self.head_dim)
        x = self.dropout(x)

        src_attn = (x * self.src_attn).sum(dim=-1)[edge_indices[0]]
        dst_attn = (x * self.dst_attn).sum(dim=-1)[edge_indices[1]]
        attn_coef = torch.sparse_coo_tensor(edge_indices, self.act(src_attn + dst_attn))
        attn_weights = torch.sparse.softmax(attn_coef, dim=1).coalesce()

        indices, values = attn_weights.indices(), attn_weights.values()
        values = self.dropout(values)
        crow_idx = torch._convert_indices_from_coo_to_csr(indices[0], x.shape[0])

        head_outputs = []
        for i in range(self.num_heads):
            values_i = values[:, i].contiguous()
            weights_i = torch.sparse_csr_tensor(crow_idx, indices[1], values_i)
            head_outputs.append(torch.sparse.mm(weights_i, x[:, i]))

        return torch.cat(head_outputs, dim=1)
