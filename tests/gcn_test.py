import pytest
import torch
from graph_nn import GCNLayer
from main import run
from torch_geometric.nn import GCNConv


def test_gcn_against_pyg(num_nodes: int, symmetric_adj_mat_idx: torch.Tensor):
    indices = symmetric_adj_mat_idx
    in_dim, out_dim = 50, 64

    pyg_gcn = GCNConv(in_dim, out_dim, bias=False)
    gcn = GCNLayer(in_dim, out_dim)
    gcn.linear.weight.data = pyg_gcn.lin.weight.data.clone()

    x = torch.randn(num_nodes, in_dim)
    adj_mat = GCNLayer.normalize_adjacency_matrix(indices)

    out1 = pyg_gcn(x, indices)
    out2 = gcn(x, adj_mat)
    assert torch.allclose(out1, out2, atol=1e-7)

    out1.abs().sum().backward()
    out2.abs().sum().backward()

    grad1 = pyg_gcn.lin.weight.grad
    grad2 = gcn.linear.weight.grad
    assert torch.allclose(grad1, grad2, atol=1e-5)


@pytest.mark.parametrize("dataset", ["Cora", "CiteSeer"])
def test_gcn_cora_citeseer(dataset: str):
    test_acc = run(
        model="gcn",
        dataset=dataset,
        num_epochs=200,
        optimizer="Adam",
        lr=1e-2,
        weight_decay=0,
        momentum=0,
        l2_regularization=5e-4,
        device="cpu",
        hidden_dim=16,
        dropout=0.5,
    )
    if dataset == "Cora":
        assert test_acc > 0.81
    elif dataset == "CiteSeer":
        assert test_acc > 0.70
