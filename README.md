# Graph Neural Networks

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

As of this writing:
- PyTorch Geometric doesn't have pre-built binaries for PyTorch 1.12. You might need to install PyTorch 1.11 instead.
  ```bash
  conda install pytorch=1.11 -c pytorch
  ```
- PyTorch Geometric doesn't have pre-built binaries for macOS ARM64 (Apple Silicon). You might need to install it from source
  ```bash
  pip install torch-sparse torch-scatter
  pip install torch-geometric
  ```

## Reproduce paper results

Unless otherwise stated, hyperparameters are taken directly from the respective papers.

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

Dataset   | Cora | CiteSeer | PubMed | NELL
----------|------|----------|--------|-----
Paper     | 81.5 | 70.3     | 79.0   | 66.0
This repo | 82.0 | 72.3     | 79.2   | 58.0

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

Dataset   | Cora | CiteSeer | PubMed
----------|------|----------|--------
Paper     | 83.0 | 72.5     | 79.0
This repo | 83.0 | 71.0     | 79.4

Related issues to GAT on CiteSeer and PubMed:

- https://github.com/PetarV-/GAT/issues/14
- https://github.com/PetarV-/GAT/issues/12

For PubMed, using hyperparameters provided by the paper `--l2_regularization 1e-3 --lr 1e-2`, I can only achieve 77.7 accuracy.
