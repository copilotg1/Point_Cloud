"""
PointNet Model for Classification and Segmentation.

This module implements the two main PointNet architectures:

1. PointNetClassification:
   - Takes raw point clouds (N×3) and outputs class predictions.
   - Uses T-Net for input and feature alignment.
   - Aggregates global features via max pooling.

2. PointNetSegmentation:
   - Takes raw point clouds and outputs per-point class predictions.
   - Concatenates local features with the global feature vector.
   - Produces per-point semantic labels.

Architecture Overview (Classification):
    Input (B, N, 3)
      → Input Transform (T-Net 3×3)
      → Shared MLP (64, 64)
      → Feature Transform (T-Net 64×64)
      → Shared MLP (64, 128, 1024)
      → Max Pooling → Global Feature (B, 1024)
      → FC (512, 256, num_classes)
      → Output (B, num_classes)

Architecture Overview (Segmentation):
    Input (B, N, 3)
      → [Same as classification up to global feature]
      → Concatenate local features (B, N, 64) + global feature (B, 1024)
      → Shared MLP (512, 256, 128)
      → Per-point FC → Output (B, N, num_parts)

Reference:
    Charles R. Qi et al., "PointNet: Deep Learning on Point Sets for
    3D Classification and Segmentation", CVPR 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet.transform_nets import (
    InputTransformNet,
    FeatureTransformNet,
)


class PointNetEncoder(nn.Module):
    """PointNet Feature Encoder (shared backbone).

    Extracts both local and global features from point clouds.
    This encoder is shared by both the classification and segmentation heads.

    Architecture:
        Input (B, N, 3)
          → Input T-Net → transform points
          → Conv1d(3, 64) → Conv1d(64, 64)
          → Feature T-Net → transform features
          → Conv1d(64, 64) → Conv1d(64, 128) → Conv1d(128, 1024)
          → Max Pool → Global Feature (B, 1024)

    Args:
        global_feat: If True, return only the global feature (for classification).
                     If False, return both local and global features (for segmentation).
        feature_transform: If True, apply feature T-Net alignment.
    """

    def __init__(self, global_feat: bool = True, feature_transform: bool = True):
        super().__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        # Input Transform Network (3x3)
        self.input_transform = InputTransformNet()

        # First shared MLP block: 3 → 64 → 64
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Feature Transform Network (64x64)
        if self.feature_transform:
            self.feat_transform = FeatureTransformNet(k=64)

        # Second shared MLP block: 64 → 128 → 1024
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor):
        """Forward pass through the PointNet encoder.

        Args:
            x: Input point cloud of shape (batch_size, num_points, 3).

        Returns:
            If global_feat=True:
                global_feature: (batch_size, 1024)
                input_transform: (batch_size, 3, 3)
                feature_transform: (batch_size, 64, 64) or None
            If global_feat=False:
                (point_features, global_feature): ((B, N, 1088), ...)
                input_transform: (batch_size, 3, 3)
                feature_transform: (batch_size, 64, 64) or None
        """
        batch_size, num_points, _ = x.size()

        # ---- Input Transform ----
        # Transpose for Conv1d: (B, N, 3) → (B, 3, N)
        x = x.transpose(1, 2)

        # Predict and apply input transformation
        input_trans = self.input_transform(x)  # (B, 3, 3)
        x = x.transpose(1, 2)                  # (B, N, 3)
        x = torch.bmm(x, input_trans)          # (B, N, 3)
        x = x.transpose(1, 2)                  # (B, 3, N)

        # ---- First Shared MLP ----
        x = F.relu(self.bn1(self.conv1(x)))    # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))    # (B, 64, N)

        # ---- Feature Transform ----
        feat_trans = None
        if self.feature_transform:
            feat_trans = self.feat_transform(x)  # (B, 64, 64)
            x = x.transpose(1, 2)                # (B, N, 64)
            x = torch.bmm(x, feat_trans)          # (B, N, 64)
            x = x.transpose(1, 2)                # (B, 64, N)

        # Save local (point-wise) features for segmentation
        local_features = x  # (B, 64, N)

        # ---- Second Shared MLP ----
        x = F.relu(self.bn3(self.conv3(x)))    # (B, 64, N)
        x = F.relu(self.bn4(self.conv4(x)))    # (B, 128, N)
        x = F.relu(self.bn5(self.conv5(x)))    # (B, 1024, N)

        # ---- Symmetric Function: Max Pooling ----
        global_feature = torch.max(x, dim=2)[0]  # (B, 1024)

        if self.global_feat:
            return global_feature, input_trans, feat_trans

        # For segmentation: concatenate local features with global feature
        # Expand global feature and concatenate with per-point local features
        global_feature_expanded = global_feature.unsqueeze(2).repeat(
            1, 1, num_points
        )  # (B, 1024, N)
        combined = torch.cat(
            [local_features, global_feature_expanded], dim=1
        )  # (B, 1088, N)
        combined = combined.transpose(1, 2)  # (B, N, 1088)

        return combined, global_feature, input_trans, feat_trans


class PointNetClassification(nn.Module):
    """PointNet Classification Network.

    Classifies an entire point cloud into one of `num_classes` categories.

    Architecture:
        PointNetEncoder(global_feat=True)
          → FC(1024, 512) + BN + ReLU + Dropout(0.3)
          → FC(512, 256) + BN + ReLU + Dropout(0.3)
          → FC(256, num_classes)
          → Log-Softmax

    Args:
        num_classes: Number of output classes (e.g., 40 for ModelNet40).
        feature_transform: Whether to use the feature T-Net.
    """

    def __init__(self, num_classes: int = 40, feature_transform: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.feature_transform = feature_transform

        # Shared encoder
        self.encoder = PointNetEncoder(
            global_feat=True, feature_transform=feature_transform
        )

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor):
        """Forward pass for point cloud classification.

        Args:
            x: Input point cloud of shape (batch_size, num_points, 3).

        Returns:
            output: Log-softmax scores of shape (batch_size, num_classes).
            input_transform: (batch_size, 3, 3) input transformation matrix.
            feature_transform: (batch_size, 64, 64) feature transformation matrix,
                               or None if feature_transform is disabled.
        """
        # Encode point cloud
        global_feature, input_trans, feat_trans = self.encoder(x)

        # Classification head
        x = F.relu(self.bn1(self.fc1(global_feature)))  # (B, 512)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))                # (B, 256)
        x = self.dropout(x)
        x = self.fc3(x)                                   # (B, num_classes)

        output = F.log_softmax(x, dim=1)
        return output, input_trans, feat_trans


class PointNetSegmentation(nn.Module):
    """PointNet Part Segmentation Network.

    Predicts per-point class labels for semantic/part segmentation.

    Architecture:
        PointNetEncoder(global_feat=False)
          → per-point features (B, N, 1088)
          → Conv1d(1088, 512) + BN + ReLU
          → Conv1d(512, 256) + BN + ReLU
          → Conv1d(256, 128) + BN + ReLU
          → Conv1d(128, num_parts)
          → Log-Softmax per point

    Args:
        num_parts: Number of part/semantic classes per point.
        feature_transform: Whether to use the feature T-Net.
    """

    def __init__(self, num_parts: int = 50, feature_transform: bool = True):
        super().__init__()
        self.num_parts = num_parts
        self.feature_transform = feature_transform

        # Shared encoder (returns local + global concatenated features)
        self.encoder = PointNetEncoder(
            global_feat=False, feature_transform=feature_transform
        )

        # Segmentation head (per-point MLP via Conv1d)
        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_parts, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor):
        """Forward pass for point cloud segmentation.

        Args:
            x: Input point cloud of shape (batch_size, num_points, 3).

        Returns:
            output: Log-softmax scores of shape (batch_size, num_points, num_parts).
            input_transform: (batch_size, 3, 3) input transformation matrix.
            feature_transform: (batch_size, 64, 64) feature transformation matrix,
                               or None if feature_transform is disabled.
        """
        batch_size, num_points, _ = x.size()

        # Encode: combined features (B, N, 1088)
        combined, global_feat, input_trans, feat_trans = self.encoder(x)

        # Segmentation head
        x = combined.transpose(1, 2)                      # (B, 1088, N)
        x = F.relu(self.bn1(self.conv1(x)))                # (B, 512, N)
        x = F.relu(self.bn2(self.conv2(x)))                # (B, 256, N)
        x = F.relu(self.bn3(self.conv3(x)))                # (B, 128, N)
        x = self.conv4(x)                                  # (B, num_parts, N)

        x = x.transpose(1, 2)                              # (B, N, num_parts)
        output = F.log_softmax(x, dim=2)

        return output, input_trans, feat_trans
