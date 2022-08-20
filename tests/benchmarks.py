import pytest

from main import run


@pytest.mark.parametrize("dataset", ["Cora", "CiteSeer"])
def test_gcn_cora_citeseer(dataset: str):
    test_acc, test_std = run(
        model="gcn",
        dataset=dataset,
        num_epochs=200,
        optimizer="Adam",
        lr=1e-2,
        weight_decay=0,
        momentum=0,
        l2_regularization=5e-4,
        device="cpu",
        n_runs=10,
        hidden_dim=16,
        dropout=0.5,
    )
    if dataset == "Cora":
        assert test_acc + test_std > 0.81
    elif dataset == "CiteSeer":
        assert test_acc + test_std > 0.70


@pytest.mark.parametrize("dataset", ["Cora", "CiteSeer"])
def test_gat_cora_citeseer(dataset: str):
    test_acc, test_std = run(
        model="gat",
        dataset=dataset,
        num_epochs=1000,
        optimizer="Adam",
        lr=5e-3,
        weight_decay=0,
        momentum=0,
        l2_regularization=5e-4,
        device="cpu",
        n_runs=10,
        hidden_dim=64,
        num_heads=8,
        output_heads=1,
        dropout=0.6,
    )
    if dataset == "Cora":
        assert test_acc + test_std > 0.825
    elif dataset == "CiteSeer":
        assert test_acc + test_std > 0.70


@pytest.mark.parametrize("dataset", ["Cora", "CiteSeer", "PubMed"])
def test_sgc_cora_citeseer_pubmed(dataset: str):
    wd_dict = {
        "Cora": 1.3e-5,
        "CiteSeer": 2.35e-5,
        "PubMed": 7.4e-5,
    }
    test_acc, test_std = run(
        model="sgc",
        dataset=dataset,
        num_epochs=150 if dataset == "CiteSeer" else 100,
        optimizer="Adam",
        lr=0.2,
        weight_decay=wd_dict[dataset],
        momentum=0,
        l2_regularization=0,
        device="cpu",
        n_runs=10,
        K=2,
        dropout=0,
    )
    if dataset == "Cora":
        assert test_acc + test_std > 0.809
    elif dataset == "CiteSeer":
        assert test_acc + test_std > 0.717
    elif dataset == "PubMed":
        assert test_acc + test_std > 0.788
