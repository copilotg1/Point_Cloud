"""
Transform Networks (T-Net) for PointNet.

T-Net learns a transformation matrix to align input point clouds or feature
spaces. The architecture consists of shared MLPs, max pooling for global
feature aggregation, and fully connected layers to predict the transformation.

Two variants are provided:
    - InputTransformNet:   Learns a 3x3 matrix to spatially align input points.
    - FeatureTransformNet: Learns a KxK matrix to align learned features.

Reference:
    Section 3.1, "Symmetry Function for Unordered Input"
    Section 3.4, "Joint Alignment Network"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Transformation Network (T-Net).

    Predicts a k×k transformation matrix from an input point cloud.

    Architecture:
        Input (N, k) → shared MLP (64, 128, 1024) → max pool → FC (512, 256) → k*k

    Each shared MLP and FC layer uses batch normalization and ReLU activation.
    The output matrix is initialized as an identity matrix via a learned bias,
    ensuring the initial transformation is close to identity.

    Args:
        k: Dimensionality of the transformation (e.g. 3 for input, 64 for features).
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k

        # Shared MLP layers (applied per-point via Conv1d)
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to predict the transformation matrix.

        Args:
            x: Input tensor of shape (batch_size, k, num_points).

        Returns:
            Transformation matrix of shape (batch_size, k, k).
        """
        batch_size = x.size(0)

        # Shared MLP: per-point feature extraction
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))   # (B, 1024, N)

        # Symmetric function: max pooling over points
        x = torch.max(x, dim=2)[0]            # (B, 1024)

        # Fully connected layers
        x = F.relu(self.bn4(self.fc1(x)))     # (B, 512)
        x = F.relu(self.bn5(self.fc2(x)))     # (B, 256)
        x = self.fc3(x)                        # (B, k*k)

        # Add identity matrix to bias toward identity transformation
        identity = torch.eye(self.k, dtype=x.dtype, device=x.device)
        identity = identity.view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity

        # Reshape to transformation matrix
        x = x.view(batch_size, self.k, self.k)  # (B, k, k)
        return x


class InputTransformNet(TNet):
    """Input Transform Network.

    Specialization of T-Net for 3D input point clouds.
    Learns a 3×3 spatial transformation matrix to canonicalize the input
    by aligning it to a consistent reference frame.

    This makes the network invariant to rigid transformations (rotation,
    translation) of the input point cloud.
    """

    def __init__(self):
        super().__init__(k=3)


class FeatureTransformNet(TNet):
    """Feature Transform Network.

    Specialization of T-Net for feature space alignment.
    Learns a k×k transformation matrix in the feature space.

    The paper uses k=64 and adds a regularization loss to constrain the
    feature transformation matrix to be close to orthogonal:
        L_reg = ||I - A * A^T||^2_F
    This prevents the transformation from collapsing the feature space.

    Args:
        k: Feature dimensionality (default: 64).
    """

    def __init__(self, k: int = 64):
        super().__init__(k=k)


def feature_transform_regularization(transform: torch.Tensor) -> torch.Tensor:
    """Compute the regularization loss for the feature transformation matrix.

    Encourages the transformation matrix to be orthogonal:
        L_reg = ||I - A * A^T||^2_F

    This prevents the high-dimensional feature transform from degenerating.

    Args:
        transform: Feature transformation matrix of shape (batch_size, k, k).

    Returns:
        Scalar regularization loss.
    """
    batch_size, k, _ = transform.size()
    identity = torch.eye(k, dtype=transform.dtype, device=transform.device)
    identity = identity.unsqueeze(0).expand(batch_size, -1, -1)

    product = torch.bmm(transform, transform.transpose(1, 2))
    loss = torch.mean(torch.norm(identity - product, dim=(1, 2)))
    return loss
