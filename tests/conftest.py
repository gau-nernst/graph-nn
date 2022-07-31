import random

import pytest
import torch


@pytest.fixture(scope="module")
def num_nodes():
    return 100


def _adj_mat_idx(num_nodes: int, density: float, symmetric: bool):
    nnz = int(num_nodes * num_nodes * density)

    indices = set()
    while len(indices) < nnz:
        i, j = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
        if i == j:
            continue
        indices.add((i, j))
        if symmetric:
            indices.add((j, i))

    indices = torch.tensor(sorted(indices)).t()
    return indices


@pytest.fixture(scope="module")
def symmetric_adj_mat_idx(num_nodes: int):
    density = 0.1
    return _adj_mat_idx(num_nodes, density, True)


@pytest.fixture(scope="module")
def asymmetric_adj_mat_idx(num_nodes: int):
    density = 0.1
    return _adj_mat_idx(num_nodes, density, False)
