from copy import deepcopy

import torch
from torch_geometric.nn import GINConv

from graph_nn import GINLayer


def test_gin_against_pyg(num_nodes: int, symmetric_adj_mat_idx: torch.Tensor):
    indices = symmetric_adj_mat_idx
    in_dim, out_dim = 50, 64

    gin = GINLayer(in_dim, out_dim)
    pyg_gin = GINConv(deepcopy(gin.mlp), train_eps=True, flow="target_to_source")
    pyg_gin.nn.load_state_dict(deepcopy(gin.mlp.state_dict()))

    x = torch.randn(num_nodes, in_dim)
    adj_mat = torch.sparse_coo_tensor(indices, torch.ones(indices.shape[1]))

    out1 = pyg_gin(x, indices)
    out2 = gin(x, adj_mat)
    assert torch.allclose(out1, out2)

    out1.abs().sum().backward()
    out2.abs().sum().backward()

    grad1 = pyg_gin.nn[0].weight.grad
    grad2 = gin.mlp[0].weight.grad
    assert torch.allclose(grad1, grad2)
