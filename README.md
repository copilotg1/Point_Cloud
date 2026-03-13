# PointNet — PyTorch Implementation

A complete PyTorch implementation of **PointNet** (*Qi et al., CVPR 2017*), the pioneering deep learning architecture for directly processing 3D point clouds.

> **Paper**: [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)

## Overview

PointNet is the first deep learning model that directly consumes unordered point sets as input, without converting them to voxels or images. It achieves state-of-the-art performance on 3D classification and segmentation tasks through three key innovations:

1. **Permutation Invariance** — Uses max pooling as a symmetric function to achieve invariance to input point ordering.
2. **Spatial Transformer Networks (T-Net)** — Learns transformation matrices to align input points and features, providing robustness to geometric transformations.
3. **Global Feature Aggregation** — Extracts a compact 1024-dimensional global feature vector from arbitrary-sized point clouds.

## Architecture

```
Input (N×3) → Input T-Net → Shared MLP (64,64) → Feature T-Net
                                                       ↓
                                                 Shared MLP (64,128,1024)
                                                       ↓
                                                  Max Pooling → Global Feature (1024)
                                                       ↓
                                    ┌──────────────────┴──────────────────┐
                              Classification Head              Segmentation Head
                              FC(512,256,k)                  Concat local+global
                              → k classes                    MLP(512,256,128,m)
                                                             → N×m per-point labels
```

For a detailed architecture analysis, see [`docs/architecture.md`](docs/architecture.md).

## Project Structure

```
pointnet/
├── __init__.py                 # Package init & public API
├── model.py                    # PointNetEncoder, Classification, Segmentation
├── transform_nets.py           # TNet, InputTransformNet, FeatureTransformNet
├── dataset.py                  # ModelNet40Dataset, data augmentation utilities
├── utils.py                    # Loss functions (with regularization), metrics
├── train_classification.py     # Classification training script
└── train_segmentation.py       # Segmentation training script

tests/
├── test_model.py               # Tests for model architecture & training
└── test_transform.py           # Tests for T-Net & regularization

docs/
└── architecture.md             # In-depth architecture analysis (中文/English)
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**: PyTorch ≥ 1.9, NumPy ≥ 1.21, tqdm ≥ 4.62

## Quick Start

### Classification (ModelNet40)

```bash
python -m pointnet.train_classification \
    --data_root ./data/modelnet40 \
    --num_points 1024 \
    --batch_size 32 \
    --epochs 200 \
    --feature_transform
```

### Segmentation (ShapeNet Part)

```bash
python -m pointnet.train_segmentation \
    --data_root ./data/shapenet \
    --num_points 2048 \
    --batch_size 16 \
    --epochs 200 \
    --feature_transform
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## Usage Example

```python
import torch
from pointnet import PointNetClassification, PointNetSegmentation

# Classification: predict object category
cls_model = PointNetClassification(num_classes=40, feature_transform=True)
points = torch.randn(8, 1024, 3)  # batch of 8, 1024 points each
predictions, input_trans, feat_trans = cls_model(points)
# predictions: (8, 40) log-probabilities

# Segmentation: predict per-point part labels
seg_model = PointNetSegmentation(num_parts=50, feature_transform=True)
predictions, input_trans, feat_trans = seg_model(points)
# predictions: (8, 1024, 50) per-point log-probabilities
```

## Reference

```bibtex
@inproceedings{qi2017pointnet,
  title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```
