import pytest
from main import run


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


@pytest.mark.parametrize("dataset", ["Cora", "CiteSeer"])
def test_gat_cora_citeseer(dataset: str):
    test_acc = run(
        model="gat",
        dataset=dataset,
        num_epochs=1000,
        optimizer="Adam",
        lr=5e-3,
        weight_decay=0,
        momentum=0,
        l2_regularization=5e-4,
        device="cpu",
        hidden_dim=64,
        num_heads=8,
        output_heads=1,
        dropout=0.6,
    )
    if dataset == "Cora":
        assert test_acc > 0.825
    elif dataset == "CiteSeer":
        assert test_acc > 0.70
