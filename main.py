import argparse
import pprint
import time
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.datasets import NELL, Planetoid

from papers import GATModel, GCNModel

_model_mapper = {"gcn": GCNModel, "gat": GATModel}


def get_dataset(dataset_name: str) -> PyGDataset:
    if dataset_name in ("Cora", "CiteSeer", "PubMed"):
        return Planetoid("data", name=dataset_name)
    if dataset_name == "NELL":
        return NELL("data")
    raise ValueError(f"Dataset {dataset_name} is not supported")


def row_normalize(x: torch.Tensor) -> torch.Tensor:
    # A row may be all zeros e.g. PubMed dataset
    return x / x.sum(dim=1, keepdim=True).clip_(min=1e-8)


def train(
    model: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    y: torch.Tensor,
    mask: torch.Tensor,
    optim: torch.optim.Optimizer,
    l2_regularization: float,
    l2_prefix: Optional[str] = None,
) -> float:
    model.train()
    optim.zero_grad()
    out = model(*inputs)
    loss = F.cross_entropy(out[mask], y[mask])
    l2_model = model if l2_prefix is None else getattr(model, l2_prefix)
    loss = loss + l2_loss(l2_model) * l2_regularization

    loss.backward()
    optim.step()
    return loss.item()


@torch.inference_mode()
def eval(
    model: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    y: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    model.eval()
    out = model(*inputs)
    loss = F.cross_entropy(out[mask], y[mask]).item()

    preds = out[mask].argmax(dim=1)
    acc = (preds == y[mask]).sum().item() / mask.sum().item()

    return loss, acc


def l2_loss(model: nn.Module) -> torch.Tensor:
    return sum(p.square().sum() for p in model.parameters())


def run(model_name: str, dataset_name: str) -> None:
    print(f"{model_name = }, {dataset_name = }")
    model_name = model_name.lower()
    assert model_name in _model_mapper
    model_cls = _model_mapper[model_name]
    config = model_cls.get_config(dataset_name)
    pprint.pprint(config)
    print()

    ds = get_dataset(dataset_name)
    data = ds[0]
    print(data)
    for x in ["train", "val", "test"]:
        size = data[f"{x}_mask"].sum().item()
        print(f"{x}_mask: {size}")
    print()

    data.x = row_normalize(data.x)
    model, inputs = model_cls.build_model(
        data.num_features, ds.num_classes, config, data
    )
    optim = config["optimizer"](model.parameters())

    best_acc = best_epoch = 0
    best_loss = float("inf")
    best_state_dict = None
    time0 = time.time()
    for i in range(config["num_epochs"]):
        epoch = i + 1

        train_loss = train(
            model,
            inputs,
            data.y,
            data.train_mask,
            optim,
            config["l2_regularization"],
            config.get("l2_prefix"),
        )
        val_loss, val_acc = eval(model, inputs, data.y, data.val_mask)

        if val_loss < best_loss:
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            best_state_dict = deepcopy(model.state_dict())

        if epoch % 100 == 0:
            log_msg = (
                f"Epoch {epoch}, "
                f"train loss {train_loss:.4f}, "
                f"val loss {val_loss:.4f}, "
                f"val acc {val_acc * 100:.2f}"
            )
            print(log_msg)

    print(f"Time taken: {time.time() - time0:.2f}")
    msg = (
        f"Best Epoch: {best_epoch}, "
        f"Val Loss {best_loss:.4f}, "
        f"Val Acc: {best_acc * 100:.2f}"
    )
    print(msg)

    model.load_state_dict(best_state_dict)
    test_loss, test_acc = eval(model, inputs, data.y, data.test_mask)
    print(f"Test Acc: {test_acc * 100:.2f}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    run(args.model, args.dataset)
