import argparse
import pprint
import time
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.datasets import NELL, Planetoid

from papers import GATModel, GCNModel, SGCModel

_model_mapper = {"gcn": GCNModel, "gat": GATModel, "sgc": SGCModel}


def get_dataset(dataset_name: str) -> PyGDataset:
    if dataset_name in ("Cora", "CiteSeer", "PubMed"):
        ds = Planetoid("data", name=dataset_name)
        return ds[0], ds.num_classes

    if dataset_name == "NELL":
        ds = NELL("data")
        data = ds[0]
        data.x = data.x.to_torch_sparse_coo_tensor()
        return data, ds.num_classes

    raise ValueError(f"Dataset {dataset_name} is not supported")


def row_normalize(x: torch.Tensor) -> torch.Tensor:
    # A row may be all zeros e.g. PubMed dataset
    if x.is_sparse:
        x = x.coalesce()
        x_sum = torch.sparse.sum(x, dim=1).to_dense().clip_(min=1e-8)
        indices = x.indices()
        values = x.values() / x_sum[indices[0]]
        return torch.sparse_coo_tensor(indices, values, x.size())

    return x / x.sum(dim=1, keepdim=True).clip_(min=1e-8)


def train(
    model: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    y: torch.Tensor,
    mask: torch.Tensor,
    optim: torch.optim.Optimizer,
    l2_regularization: float,
) -> float:
    model.train()
    optim.zero_grad()
    out = model(*inputs)
    loss = F.cross_entropy(out[mask], y[mask])
    if l2_regularization > 0:
        if hasattr(model, "l2_prefix"):
            l2_model = getattr(model, model.l2_prefix)
        else:
            l2_model = model
        loss = loss + l2_loss(l2_model) * l2_regularization

    loss.backward()
    optim.step()
    return loss.cpu().item()


@torch.inference_mode()
def eval(
    model: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    y: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[float, float]:
    model.eval()
    out = model(*inputs)
    loss = F.cross_entropy(out[mask], y[mask]).cpu().item()

    preds = out[mask].argmax(dim=1)
    acc = ((preds == y[mask]).sum() / mask.sum()).cpu().item()

    return loss, acc


def l2_loss(model: nn.Module) -> torch.Tensor:
    return sum(p.square().sum() for p in model.parameters())


def run(
    model: str,
    dataset: str,
    num_epochs: int,
    optimizer: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    l2_regularization: float,
    device: str,
    n_runs: int,
    **model_params,
) -> None:
    data, num_classes = get_dataset(dataset)
    data.x = row_normalize(data.x)
    print(data)
    for x in ["train", "val", "test"]:
        size = data[f"{x}_mask"].sum().item()
        print(f"{x}_mask: {size}")
    print()

    model_params = {
        "input_dim": data.x.shape[-1],
        "num_classes": num_classes,
        **model_params,
    }
    model_cls = _model_mapper[model.lower()]
    all_test_acc = []
    for run_i in range(n_runs):
        print(f"Run {run_i + 1}/{n_runs}")

        model, inputs = model_cls.build_model(model_params, data)
        model.to(device)
        inputs = tuple(x.to(device) for x in inputs)
        data = data.to(device)

        optim_cls = getattr(torch.optim, optimizer)
        optim_params = {"lr": lr, "weight_decay": weight_decay}
        if optimizer == "SGD":
            optim_params["momentum"] = momentum
        optim = optim_cls(model.parameters(), **optim_params)

        best_acc = best_epoch = 0
        best_loss = float("inf")
        best_state_dict = None
        time0 = time.time()
        for i in range(num_epochs):
            epoch = i + 1

            train_loss = train(
                model, inputs, data.y, data.train_mask, optim, l2_regularization
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
        print(f"Test Acc: {test_acc* 100:.2f}")
        all_test_acc.append(test_acc)

        print()

    acc_mean = np.mean(all_test_acc)
    acc_std = np.std(all_test_acc, ddof=1)
    print(f"Test Acc: {acc_mean * 100:.2f} (±{acc_std * 100:.2f})")
    return acc_mean, acc_std


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_runs", type=int, default=1)

    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--l2_regularization", type=float, default=0.0)

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, extra = parser.parse_known_args()
    model_cls = _model_mapper[args.model.lower()]
    model_cls.add_arguments(parser)
    args = parser.parse_args()

    args = vars(args)
    pprint.pprint(args)
    print()

    run(**args)
