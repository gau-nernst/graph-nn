# Graph Neural Networks

This repo re-implements some popular Graph Neural Networks (GNNs) for learning purpose. The code depends only on PyTorch, but the datasets are taken from PyTorch Geometric.

Implemented models
- [Graph Convolutional Network](https://arxiv.org/abs/1609.02907) (GCN) ✅
- [Graph Attention Network](https://arxiv.org/abs/1710.10903) (GAT)

✅ means paper results can be re-produced by this repo.

## Set up environment

```bash
conda create -n gnn python=3.10
conda activate gnn
conda install pytorch -c pytorch

# if you need CUDA, run this instead
# conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch

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
