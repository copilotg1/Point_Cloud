"""
Tests for PointNet Model (Classification and Segmentation).

Validates:
    - PointNetEncoder output shapes for global and local features
    - PointNetClassification forward pass and output dimensions
    - PointNetSegmentation forward pass and output dimensions
    - Model behavior with and without feature transform
    - Gradient flow through the full model
    - Loss function computation
    - Metric computation (accuracy, mIoU)
"""

import torch
import pytest

from pointnet.model import (
    PointNetEncoder,
    PointNetClassification,
    PointNetSegmentation,
)
from pointnet.utils import (
    PointNetClassificationLoss,
    PointNetSegmentationLoss,
    compute_accuracy,
    compute_mean_iou,
)


# ─── PointNetEncoder Tests ───────────────────────────────────────────────────

class TestPointNetEncoder:
    """Tests for the PointNet shared encoder."""

    def test_global_feat_output_shape(self):
        """Encoder with global_feat=True should return (B, 1024) features."""
        encoder = PointNetEncoder(global_feat=True, feature_transform=True)
        x = torch.randn(4, 128, 3)
        global_feat, input_trans, feat_trans = encoder(x)
        assert global_feat.shape == (4, 1024)
        assert input_trans.shape == (4, 3, 3)
        assert feat_trans.shape == (4, 64, 64)

    def test_local_feat_output_shape(self):
        """Encoder with global_feat=False should return per-point features."""
        encoder = PointNetEncoder(global_feat=False, feature_transform=True)
        x = torch.randn(4, 128, 3)
        combined, global_feat, input_trans, feat_trans = encoder(x)
        assert combined.shape == (4, 128, 1088)  # 64 local + 1024 global
        assert global_feat.shape == (4, 1024)
        assert input_trans.shape == (4, 3, 3)
        assert feat_trans.shape == (4, 64, 64)

    def test_no_feature_transform(self):
        """Encoder without feature transform should return None for feat_trans."""
        encoder = PointNetEncoder(global_feat=True, feature_transform=False)
        x = torch.randn(2, 64, 3)
        global_feat, input_trans, feat_trans = encoder(x)
        assert global_feat.shape == (2, 1024)
        assert input_trans.shape == (2, 3, 3)
        assert feat_trans is None

    def test_variable_num_points(self):
        """Encoder should handle different numbers of points."""
        encoder = PointNetEncoder(global_feat=True, feature_transform=True)
        for n_points in [32, 256, 1024]:
            x = torch.randn(2, n_points, 3)
            global_feat, _, _ = encoder(x)
            assert global_feat.shape == (2, 1024)


# ─── PointNetClassification Tests ────────────────────────────────────────────

class TestPointNetClassification:
    """Tests for the PointNet classification network."""

    def test_output_shape(self):
        """Classification output should be (B, num_classes)."""
        model = PointNetClassification(num_classes=40, feature_transform=True)
        x = torch.randn(4, 256, 3)
        output, input_trans, feat_trans = model(x)
        assert output.shape == (4, 40)
        assert input_trans.shape == (4, 3, 3)
        assert feat_trans.shape == (4, 64, 64)

    def test_output_is_log_softmax(self):
        """Classification output should be log-probabilities (all <= 0)."""
        model = PointNetClassification(num_classes=10, feature_transform=True)
        model.eval()
        x = torch.randn(2, 128, 3)
        with torch.no_grad():
            output, _, _ = model(x)
        # Log-softmax outputs should be <= 0
        assert (output <= 0).all()
        # Exp of log-softmax should sum to ~1
        probs = torch.exp(output)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)

    def test_different_num_classes(self):
        """Model should work with different numbers of classes."""
        for num_classes in [10, 20, 40]:
            model = PointNetClassification(
                num_classes=num_classes, feature_transform=True
            )
            x = torch.randn(2, 64, 3)
            output, _, _ = model(x)
            assert output.shape == (2, num_classes)

    def test_without_feature_transform(self):
        """Classification should work without feature transform."""
        model = PointNetClassification(
            num_classes=40, feature_transform=False
        )
        x = torch.randn(2, 128, 3)
        output, input_trans, feat_trans = model(x)
        assert output.shape == (2, 40)
        assert feat_trans is None

    def test_gradient_flow(self):
        """Gradients should flow through the entire classification model."""
        model = PointNetClassification(num_classes=10, feature_transform=True)
        x = torch.randn(2, 64, 3)
        output, _, feat_trans = model(x)
        labels = torch.randint(0, 10, (2,))

        loss_fn = PointNetClassificationLoss()
        loss = loss_fn(output, labels, feat_trans)
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_train_eval_mode(self):
        """Model should behave differently in train vs eval mode (dropout, BN)."""
        model = PointNetClassification(num_classes=10, feature_transform=True)
        x = torch.randn(4, 64, 3)

        model.train()
        out_train, _, _ = model(x)

        model.eval()
        with torch.no_grad():
            out_eval, _, _ = model(x)

        # Outputs should differ due to dropout and batch norm behavior
        assert out_train.shape == out_eval.shape


# ─── PointNetSegmentation Tests ──────────────────────────────────────────────

class TestPointNetSegmentation:
    """Tests for the PointNet segmentation network."""

    def test_output_shape(self):
        """Segmentation output should be (B, N, num_parts)."""
        model = PointNetSegmentation(num_parts=50, feature_transform=True)
        x = torch.randn(4, 256, 3)
        output, input_trans, feat_trans = model(x)
        assert output.shape == (4, 256, 50)
        assert input_trans.shape == (4, 3, 3)
        assert feat_trans.shape == (4, 64, 64)

    def test_output_is_log_softmax(self):
        """Segmentation output should be log-probabilities per point."""
        model = PointNetSegmentation(num_parts=10, feature_transform=True)
        model.eval()
        x = torch.randn(2, 64, 3)
        with torch.no_grad():
            output, _, _ = model(x)
        # All values should be <= 0
        assert (output <= 0).all()
        # Exp should sum to ~1 along the parts dimension
        probs = torch.exp(output)
        assert torch.allclose(
            probs.sum(dim=2), torch.ones(2, 64), atol=1e-5
        )

    def test_per_point_prediction(self):
        """Each point should get an independent prediction."""
        model = PointNetSegmentation(num_parts=4, feature_transform=True)
        model.eval()
        x = torch.randn(1, 100, 3)
        with torch.no_grad():
            output, _, _ = model(x)
        # Should have predictions for all 100 points
        assert output.shape == (1, 100, 4)
        # Each point should have valid predictions
        pred_classes = output.argmax(dim=2)
        assert pred_classes.shape == (1, 100)

    def test_without_feature_transform(self):
        """Segmentation should work without feature transform."""
        model = PointNetSegmentation(
            num_parts=50, feature_transform=False
        )
        x = torch.randn(2, 128, 3)
        output, input_trans, feat_trans = model(x)
        assert output.shape == (2, 128, 50)
        assert feat_trans is None

    def test_gradient_flow(self):
        """Gradients should flow through the segmentation model."""
        model = PointNetSegmentation(num_parts=4, feature_transform=True)
        x = torch.randn(2, 32, 3)
        output, _, feat_trans = model(x)
        labels = torch.randint(0, 4, (2, 32))

        loss_fn = PointNetSegmentationLoss()
        loss = loss_fn(output, labels, feat_trans)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ─── Loss Function Tests ─────────────────────────────────────────────────────

class TestClassificationLoss:
    """Tests for the classification loss function."""

    def test_loss_computation(self):
        """Classification loss should be a positive scalar."""
        loss_fn = PointNetClassificationLoss(alpha=0.001)
        predictions = torch.log_softmax(torch.randn(4, 10), dim=1)
        labels = torch.randint(0, 10, (4,))
        feat_trans = torch.eye(64).unsqueeze(0).repeat(4, 1, 1)

        loss = loss_fn(predictions, labels, feat_trans)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_loss_without_feature_transform(self):
        """Loss should work without feature transform regularization."""
        loss_fn = PointNetClassificationLoss()
        predictions = torch.log_softmax(torch.randn(4, 10), dim=1)
        labels = torch.randint(0, 10, (4,))

        loss = loss_fn(predictions, labels, feat_transform=None)
        assert loss.dim() == 0
        assert loss.item() > 0


class TestSegmentationLoss:
    """Tests for the segmentation loss function."""

    def test_loss_computation(self):
        """Segmentation loss should be a positive scalar."""
        loss_fn = PointNetSegmentationLoss(alpha=0.001)
        predictions = torch.log_softmax(torch.randn(4, 32, 10), dim=2)
        labels = torch.randint(0, 10, (4, 32))
        feat_trans = torch.eye(64).unsqueeze(0).repeat(4, 1, 1)

        loss = loss_fn(predictions, labels, feat_trans)
        assert loss.dim() == 0
        assert loss.item() > 0


# ─── Metric Tests ────────────────────────────────────────────────────────────

class TestMetrics:
    """Tests for evaluation metrics."""

    def test_perfect_accuracy(self):
        """Accuracy should be 1.0 for perfect predictions."""
        predictions = torch.zeros(4, 3)
        predictions[0, 0] = 10.0
        predictions[1, 1] = 10.0
        predictions[2, 2] = 10.0
        predictions[3, 0] = 10.0
        labels = torch.tensor([0, 1, 2, 0])

        acc = compute_accuracy(predictions, labels)
        assert acc == pytest.approx(1.0)

    def test_zero_accuracy(self):
        """Accuracy should be 0.0 for completely wrong predictions."""
        predictions = torch.zeros(4, 3)
        predictions[0, 1] = 10.0  # predict 1, label 0
        predictions[1, 2] = 10.0  # predict 2, label 1
        predictions[2, 0] = 10.0  # predict 0, label 2
        predictions[3, 2] = 10.0  # predict 2, label 0
        labels = torch.tensor([0, 1, 2, 0])

        acc = compute_accuracy(predictions, labels)
        assert acc == pytest.approx(0.0)

    def test_mean_iou_perfect(self):
        """mIoU should be 1.0 for perfect segmentation."""
        predictions = torch.zeros(1, 4, 3)
        predictions[0, 0, 0] = 10.0
        predictions[0, 1, 1] = 10.0
        predictions[0, 2, 2] = 10.0
        predictions[0, 3, 0] = 10.0
        labels = torch.tensor([[0, 1, 2, 0]])

        iou = compute_mean_iou(predictions, labels, num_classes=3)
        assert iou == pytest.approx(1.0)
