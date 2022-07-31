import torch
from graph_nn import GATLayer
from graph_nn.utils import append_identity_matrix
from torch_geometric.nn import GATConv


def test_gat_against_pyg(num_nodes: int, asymmetric_adj_mat_idx: torch.Tensor):
    indices = asymmetric_adj_mat_idx
    in_dim, out_dim, heads = 50, 64, 8

    pyg_gat = GATConv(
        in_dim, out_dim // heads, heads=heads, bias=False, flow="target_to_source"
    )
    gat = GATLayer(in_dim, out_dim, heads)
    gat.linear.weight.data = pyg_gat.lin_src.weight.data.clone()
    gat.src_attn.data = pyg_gat.att_src.data.clone()
    gat.dst_attn.data = pyg_gat.att_dst.data.clone()

    x = torch.randn(num_nodes, in_dim)
    new_indices = append_identity_matrix(indices, num_nodes=num_nodes)

    out1 = pyg_gat(x, indices)
    out2 = gat(x, new_indices)
    assert torch.allclose(out1, out2, atol=1e-7)

    out1.abs().sum().backward()
    out2.abs().sum().backward()

    grad1 = pyg_gat.lin_src.weight.grad
    grad2 = gat.linear.weight.grad
    assert torch.allclose(grad1, grad2, atol=1e-5)
