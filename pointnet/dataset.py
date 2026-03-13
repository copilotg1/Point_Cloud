"""
Dataset utilities for PointNet.

Provides dataset classes for loading and preprocessing point cloud data:
    - ModelNet40Dataset: The ModelNet40 benchmark for 3D shape classification.

Point cloud preprocessing includes:
    - Random sampling of a fixed number of points.
    - Normalization to unit sphere.
    - Optional data augmentation (random rotation, jittering).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Normalize a point cloud to fit within a unit sphere centered at origin.

    Args:
        points: Point cloud of shape (N, 3).

    Returns:
        Normalized point cloud of shape (N, 3).
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
    return points


def random_rotate_point_cloud(points: np.ndarray) -> np.ndarray:
    """Randomly rotate a point cloud around the Y-axis.

    This is a common augmentation for 3D shape classification since most
    objects have a canonical up direction (Y-axis).

    Args:
        points: Point cloud of shape (N, 3).

    Returns:
        Rotated point cloud of shape (N, 3).
    """
    angle = np.random.uniform(0, 2 * np.pi)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a],
    ])
    return points @ rotation_matrix.T


def jitter_point_cloud(points: np.ndarray, sigma: float = 0.01,
                        clip: float = 0.05) -> np.ndarray:
    """Add random Gaussian noise to each point.

    Args:
        points: Point cloud of shape (N, 3).
        sigma: Standard deviation of the Gaussian noise.
        clip: Maximum absolute value to clip the noise.

    Returns:
        Jittered point cloud of shape (N, 3).
    """
    noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
    return points + noise


class ModelNet40Dataset(Dataset):
    """ModelNet40 Dataset for 3D point cloud classification.

    Expects data in the following directory structure:
        root/
          airplane/
            train/
              airplane_0001.npy
              ...
            test/
              airplane_0627.npy
              ...
          bathtub/
            ...

    Each .npy file contains a point cloud of shape (N, 3) or (N, 6) where
    the first 3 columns are XYZ coordinates and the optional last 3 are normals.

    Args:
        root: Root directory of the dataset.
        num_points: Number of points to sample from each point cloud.
        split: Dataset split ('train' or 'test').
        augment: Whether to apply data augmentation (rotation + jittering).
    """

    def __init__(self, root: str, num_points: int = 1024,
                 split: str = "train", augment: bool = True):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.split = split
        self.augment = augment and (split == "train")

        # Discover classes from subdirectories
        self.classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect all file paths and labels
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root, cls, split)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith(".npy"):
                    self.samples.append((
                        os.path.join(cls_dir, fname),
                        self.class_to_idx[cls],
                    ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        filepath, label = self.samples[idx]

        # Load point cloud
        points = np.load(filepath).astype(np.float32)

        # Use only XYZ coordinates
        if points.shape[1] > 3:
            points = points[:, :3]

        # Random sampling
        if len(points) >= self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)
        points = points[choice, :]

        # Normalize
        points = normalize_point_cloud(points)

        # Data augmentation
        if self.augment:
            points = random_rotate_point_cloud(points)
            points = jitter_point_cloud(points)

        points = torch.from_numpy(points).float()
        label = torch.tensor(label, dtype=torch.long)

        return points, label
