# Graph Neural Networks

[![Test](https://github.com/gau-nernst/graph-nn/actions/workflows/test.yaml/badge.svg)](https://github.com/gau-nernst/graph-nn/actions/workflows/test.yaml)
![python>=3.7](https://img.shields.io/badge/python-%3E%3D%203.7-informational)
![torch>=1.10](https://img.shields.io/badge/torch-%3E%3D%201.10-informational)

This repo re-implements some popular Graph Neural Networks (GNNs) for learning purpose. The code depends only on PyTorch, but the datasets are taken from PyTorch Geometric.

Implemented models
- [Graph Convolutional Network](https://arxiv.org/abs/1609.02907) (GCN)
- [Graph Attention Network](https://arxiv.org/abs/1710.10903) (GAT)

## Set up environment

```bash
conda create -n gnn python=3.10
conda activate gnn
conda install pytorch -c pytorch

# if you need CUDA
# conda install pytorch cudatoolkit -c pytorch

# optional
conda install pyg -c pyg
```

Note that you don't need `torch-scatter` and `torch-sparse` to use the datasets from PyTorch Geometric, except for NELL (to be confirmed).

## Reproduce paper results

Unless otherwise stated, hyperparameters are taken directly from the respective papers. Each experiment is ran 100 times, and the results are averaged.

```bash
python main.py --model gcn --dataset Cora   # other options: CiteSeer, PubMed
```

Add `--device cuda` if you want to train on GPU.

### GCN

```bash
# For Cora, CiteSeer, and PubMed
python main.py --model gcn --dataset Cora --dropout 0.5 --l2_regularization 5e-4 --lr 1e-2 --num_epochs 200

# For NELL
python main.py --model gcn --dataset NELL --dropout 0.1 --l2_regularization 1e-5 --lr 1e-2 --hidden_dim 64 --num_epochs 200
```

Dataset   | Cora          | CiteSeer      | PubMed        | NELL
----------|---------------|---------------|---------------|---------------
Paper     | 81.5          | 70.3          | 79.0          | 66.0
This repo | 81.36 (±0.70) | 71.07 (±0.80) | 79.22 (±0.32) | 56.94 (±1.24)

Related issues to GCN on NELL (possibly due to data preprocessing):

- https://github.com/pyg-team/pytorch_geometric/issues/2392
- https://github.com/tkipf/gcn/issues/14

### GAT

```bash
# For Cora and CiteSeer
python main.py --model gat --dataset Cora --dropout 0.6 --l2_regularization 5e-4 --lr 5e-3 --num_epochs 1000

# For PubMed (different from paper)
python main.py --model gat --dataset PubMed --dropout 0.6 --l2_regularization 5e-4 --lr 5e-3 --num_epochs 1000 --output_heads 8
```

Dataset   | Cora          | CiteSeer      | PubMed
----------|---------------|---------------|---------------
Paper     | 83.0          | 72.5          | 79.0
This repo | 82.65 (±0.41) | 71.04 (±0.41) | 79.11 (±0.28)

Related issues to GAT on CiteSeer and PubMed:

- https://github.com/PetarV-/GAT/issues/14
- https://github.com/PetarV-/GAT/issues/12

For PubMed, using hyperparameters provided by the paper `--l2_regularization 1e-3 --lr 1e-2`, I can only achieve 77.36 (±0.52) accuracy.

## Unit tests

The implementations here are tested against PyTorch Geometric's implementations for correctness. You will need to install `torch-scatter` and `torch-sparse` together with PyTorch Geometric. See their [doc](https://github.com/pyg-team/pytorch_geometric) for installation instructions.

As of this writing, there are no pre-built `torch-scatter` and `torch-sparse` binaries for macOS ARM64 (Apple Silicon). You will need to install them from source

```bash
pip install torch-scatter torch-sparse
pip install torch-geometric
```

Once you have the pre-requisites, install `pytest` via conda or pip then run:

```bash
python -m pytest
```
