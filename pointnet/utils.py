"""
Utility functions for PointNet training and evaluation.

Includes:
    - PointNetLoss: Combined classification + regularization loss.
    - compute_accuracy: Accuracy computation for classification.
    - compute_mean_iou: Mean IoU computation for segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet.transform_nets import feature_transform_regularization


class PointNetClassificationLoss(nn.Module):
    """Combined loss for PointNet classification.

    Combines NLL loss with feature transform regularization:
        L = L_cls + alpha * L_reg

    where:
        L_cls = NLLLoss(predictions, labels)
        L_reg = ||I - A * A^T||^2_F  (orthogonality constraint on feature transform)

    Args:
        alpha: Weight for the regularization term (default: 0.001 as in the paper).
    """

    def __init__(self, alpha: float = 0.001):
        super().__init__()
        self.alpha = alpha
        self.nll_loss = nn.NLLLoss()

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor,
                feat_transform: torch.Tensor = None) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            predictions: Log-softmax output of shape (B, num_classes).
            labels: Ground truth labels of shape (B,).
            feat_transform: Feature transformation matrix of shape (B, 64, 64).

        Returns:
            Combined loss scalar.
        """
        cls_loss = self.nll_loss(predictions, labels)

        if feat_transform is not None:
            reg_loss = feature_transform_regularization(feat_transform)
            return cls_loss + self.alpha * reg_loss

        return cls_loss


class PointNetSegmentationLoss(nn.Module):
    """Combined loss for PointNet segmentation.

    Combines per-point NLL loss with feature transform regularization:
        L = L_seg + alpha * L_reg

    Args:
        alpha: Weight for the regularization term (default: 0.001).
    """

    def __init__(self, alpha: float = 0.001):
        super().__init__()
        self.alpha = alpha
        self.nll_loss = nn.NLLLoss()

    def forward(self, predictions: torch.Tensor, labels: torch.Tensor,
                feat_transform: torch.Tensor = None) -> torch.Tensor:
        """Compute the combined segmentation loss.

        Args:
            predictions: Log-softmax output of shape (B, N, num_parts).
            labels: Ground truth per-point labels of shape (B, N).
            feat_transform: Feature transformation matrix of shape (B, 64, 64).

        Returns:
            Combined loss scalar.
        """
        # Reshape for NLLLoss: (B*N, num_parts) and (B*N,)
        batch_size, num_points, num_parts = predictions.size()
        predictions = predictions.reshape(-1, num_parts)
        labels = labels.reshape(-1)

        seg_loss = self.nll_loss(predictions, labels)

        if feat_transform is not None:
            reg_loss = feature_transform_regularization(feat_transform)
            return seg_loss + self.alpha * reg_loss

        return seg_loss


def compute_accuracy(predictions: torch.Tensor,
                     labels: torch.Tensor) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Log-softmax output of shape (B, num_classes).
        labels: Ground truth labels of shape (B,).

    Returns:
        Accuracy as a float between 0 and 1.
    """
    pred_classes = predictions.argmax(dim=1)
    correct = (pred_classes == labels).sum().item()
    return correct / labels.size(0)


def compute_mean_iou(predictions: torch.Tensor, labels: torch.Tensor,
                     num_classes: int) -> float:
    """Compute mean Intersection-over-Union (mIoU) for segmentation.

    Args:
        predictions: Log-softmax output of shape (B, N, num_parts).
        labels: Ground truth per-point labels of shape (B, N).
        num_classes: Total number of part classes.

    Returns:
        Mean IoU across all classes present in the batch.
    """
    pred_classes = predictions.argmax(dim=2).reshape(-1)
    labels_flat = labels.reshape(-1)

    ious = []
    for cls in range(num_classes):
        pred_mask = (pred_classes == cls)
        label_mask = (labels_flat == cls)
        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0
    return sum(ious) / len(ious)
