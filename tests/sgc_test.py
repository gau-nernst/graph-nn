import torch
from torch_geometric.nn import SGConv

from graph_nn import SGCLayer


def test_sgc_against_pyg(num_nodes: int, asymmetric_adj_mat_idx: torch.Tensor):
    indices = asymmetric_adj_mat_idx
    in_dim, out_dim, K = 50, 64, 2

    pyg_sgc = SGConv(in_dim, out_dim, K=K)
    sgc = SGCLayer(in_dim, out_dim)
    sgc.weight.data = pyg_sgc.lin.weight.data.clone()
    sgc.bias.data = pyg_sgc.lin.bias.data.clone()

    x = torch.randn(num_nodes, in_dim)
    preprocessed_x = SGCLayer.preprocess_features(x, indices, K)

    # by default PyG aggregates from source to target nodes
    out1 = pyg_sgc(x, indices[[1, 0]])
    out2 = sgc(preprocessed_x)
    assert torch.allclose(out1, out2, atol=1e-7)

    out1.abs().sum().backward()
    out2.abs().sum().backward()

    grad1 = pyg_sgc.lin.weight.grad
    grad2 = sgc.weight.grad
    assert torch.allclose(grad1, grad2, atol=1e-5)
