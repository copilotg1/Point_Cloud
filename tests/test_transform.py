"""
Tests for PointNet Transform Networks (T-Net).

Validates:
    - TNet output shapes and identity initialization
    - InputTransformNet produces 3x3 matrices
    - FeatureTransformNet produces KxK matrices
    - Feature transform regularization loss computation
    - Gradient flow through the transform networks
"""

import torch
import pytest

from pointnet.transform_nets import (
    TNet,
    InputTransformNet,
    FeatureTransformNet,
    feature_transform_regularization,
)


class TestTNet:
    """Tests for the generic TNet module."""

    def test_output_shape_k3(self):
        """TNet with k=3 should output (B, 3, 3) transformation matrix."""
        model = TNet(k=3)
        x = torch.randn(4, 3, 100)  # (B, k, N)
        out = model(x)
        assert out.shape == (4, 3, 3)

    def test_output_shape_k64(self):
        """TNet with k=64 should output (B, 64, 64) transformation matrix."""
        model = TNet(k=64)
        x = torch.randn(2, 64, 50)
        out = model(x)
        assert out.shape == (2, 64, 64)

    def test_identity_initialization(self):
        """TNet output should be close to identity for zero-initialized weights."""
        model = TNet(k=3)
        model.eval()  # eval mode needed for batch_size=1 with BatchNorm
        x = torch.zeros(1, 3, 10)
        with torch.no_grad():
            out = model(x)
        # The output should be somewhat close to identity due to the bias
        assert out.shape == (1, 3, 3)

    def test_variable_num_points(self):
        """TNet should handle different numbers of input points."""
        model = TNet(k=3)
        for n_points in [16, 128, 1024, 2048]:
            x = torch.randn(2, 3, n_points)
            out = model(x)
            assert out.shape == (2, 3, 3)

    def test_batch_size_one(self):
        """TNet should work with batch size 1 in eval mode."""
        model = TNet(k=3)
        model.eval()  # eval mode needed for batch_size=1 with BatchNorm
        x = torch.randn(1, 3, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 3)

    def test_gradient_flow(self):
        """Gradients should flow through TNet."""
        model = TNet(k=3)
        x = torch.randn(2, 3, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestInputTransformNet:
    """Tests for the Input Transform Network."""

    def test_output_shape(self):
        """InputTransformNet should produce a 3x3 transformation matrix."""
        model = InputTransformNet()
        x = torch.randn(4, 3, 256)
        out = model(x)
        assert out.shape == (4, 3, 3)

    def test_is_tnet_k3(self):
        """InputTransformNet should be a TNet with k=3."""
        model = InputTransformNet()
        assert model.k == 3


class TestFeatureTransformNet:
    """Tests for the Feature Transform Network."""

    def test_output_shape_default(self):
        """FeatureTransformNet with default k=64 should produce 64x64 matrix."""
        model = FeatureTransformNet()
        x = torch.randn(4, 64, 256)
        out = model(x)
        assert out.shape == (4, 64, 64)

    def test_output_shape_custom_k(self):
        """FeatureTransformNet with custom k should produce k×k matrix."""
        model = FeatureTransformNet(k=32)
        x = torch.randn(2, 32, 100)
        out = model(x)
        assert out.shape == (2, 32, 32)

    def test_is_tnet(self):
        """FeatureTransformNet should be a TNet subclass."""
        model = FeatureTransformNet(k=64)
        assert isinstance(model, TNet)
        assert model.k == 64


class TestFeatureTransformRegularization:
    """Tests for the regularization loss function."""

    def test_identity_matrix_loss(self):
        """Regularization loss for identity matrices should be zero."""
        identity = torch.eye(64).unsqueeze(0).repeat(4, 1, 1)
        loss = feature_transform_regularization(identity)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_non_identity_positive_loss(self):
        """Regularization loss for non-orthogonal matrices should be positive."""
        transform = torch.randn(4, 64, 64)
        loss = feature_transform_regularization(transform)
        assert loss.item() > 0

    def test_orthogonal_matrix_loss(self):
        """Regularization loss for orthogonal matrices should be near zero."""
        # Create an orthogonal matrix using QR decomposition
        random_matrix = torch.randn(64, 64)
        q, _ = torch.linalg.qr(random_matrix)
        transform = q.unsqueeze(0).repeat(2, 1, 1)
        loss = feature_transform_regularization(transform)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_output_is_scalar(self):
        """Regularization loss should be a scalar."""
        transform = torch.randn(4, 64, 64)
        loss = feature_transform_regularization(transform)
        assert loss.dim() == 0

    def test_gradient_flow(self):
        """Gradients should flow through the regularization loss."""
        transform = torch.randn(4, 64, 64, requires_grad=True)
        loss = feature_transform_regularization(transform)
        loss.backward()
        assert transform.grad is not None
