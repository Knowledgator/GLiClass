"""Tests for gliclass.loss_functions module."""

import pytest
import torch

from gliclass.loss_functions import sequence_contrastive_loss, focal_loss_with_logits


class TestSequenceContrastiveLoss:
    """Test suite for sequence_contrastive_loss function."""

    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        batch_size = 2
        seq_len = 4
        embed_dim = 8
        return torch.randn(batch_size, seq_len, embed_dim)

    @pytest.fixture
    def sample_mask(self):
        """Sample mask indicating valid positions."""
        return torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)

    def test_returns_scalar_loss(self, sample_embeddings, sample_mask):
        """Should return a scalar loss value."""
        loss = sequence_contrastive_loss(sample_embeddings, sample_mask)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar

    def test_loss_is_positive(self, sample_embeddings, sample_mask):
        """Should return positive loss value."""
        loss = sequence_contrastive_loss(sample_embeddings, sample_mask)

        assert loss >= 0

    def test_identical_sequences_low_loss(self):
        """Should give low loss for identical sequences."""
        embeddings = torch.ones(2, 4, 8)  # Identical embeddings
        mask = torch.ones(2, 4, dtype=torch.float32)

        loss = sequence_contrastive_loss(embeddings, mask)

        # Identical sequences should have low contrastive loss
        assert loss < 10.0  # Reasonable upper bound

    def test_handles_masked_positions(self):
        """Should ignore masked-out positions."""
        embeddings = torch.randn(2, 4, 8)
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float32)

        loss = sequence_contrastive_loss(embeddings, mask)

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flows_through_loss(self, sample_embeddings, sample_mask):
        """Should allow gradient to flow through."""
        sample_embeddings.requires_grad = True

        loss = sequence_contrastive_loss(sample_embeddings, sample_mask)
        loss.backward()

        assert sample_embeddings.grad is not None


class TestFocalLossWithLogits:
    """Test suite for focal_loss_with_logits function."""

    @pytest.fixture
    def sample_logits(self):
        """Sample logits for testing."""
        return torch.randn(2, 4)

    @pytest.fixture
    def sample_targets(self):
        """Sample binary targets."""
        return torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)

    def test_returns_tensor_with_reduction_none(self, sample_logits, sample_targets):
        """Should return per-element loss with reduction='none' (default)."""
        loss = focal_loss_with_logits(sample_logits, sample_targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == sample_logits.shape

    def test_returns_scalar_with_reduction_mean(self, sample_logits, sample_targets):
        """Should return scalar with reduction='mean'."""
        loss = focal_loss_with_logits(sample_logits, sample_targets, reduction="mean")

        assert loss.dim() == 0  # Scalar

    def test_loss_is_positive(self, sample_logits, sample_targets):
        """Should return non-negative loss."""
        loss = focal_loss_with_logits(sample_logits, sample_targets, reduction="mean")

        assert loss >= 0

    def test_perfect_predictions_low_loss(self):
        """Should give low loss for perfect predictions."""
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        loss = focal_loss_with_logits(logits, targets, reduction="mean")

        # Perfect predictions should have very low loss
        assert loss < 0.1

    def test_wrong_predictions_high_loss(self):
        """Should give higher loss for wrong predictions."""
        logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
        targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # Opposite

        loss = focal_loss_with_logits(logits, targets, reduction="mean")

        # Wrong predictions should have higher loss
        assert loss > 1.0

    def test_alpha_parameter_effect(self, sample_logits, sample_targets):
        """Should respect alpha weighting parameter."""
        loss_alpha_1 = focal_loss_with_logits(sample_logits, sample_targets, alpha=1.0, reduction="mean")
        loss_alpha_05 = focal_loss_with_logits(sample_logits, sample_targets, alpha=0.5, reduction="mean")

        # Different alpha should give different losses
        assert not torch.allclose(loss_alpha_1, loss_alpha_05)

    def test_gamma_parameter_effect(self, sample_logits, sample_targets):
        """Should respect gamma focusing parameter."""
        loss_gamma_0 = focal_loss_with_logits(sample_logits, sample_targets, gamma=0.0, reduction="mean")
        loss_gamma_2 = focal_loss_with_logits(sample_logits, sample_targets, gamma=2.0, reduction="mean")

        # Different gamma should give different losses
        # gamma=0 is equivalent to BCE loss
        assert not torch.allclose(loss_gamma_0, loss_gamma_2)

    def test_reduction_sum(self, sample_logits, sample_targets):
        """Should reduce loss by sum."""
        loss = focal_loss_with_logits(sample_logits, sample_targets, reduction="sum")

        assert loss.dim() == 0  # Scalar

    def test_reduction_none(self, sample_logits, sample_targets):
        """Should return per-element loss when reduction='none'."""
        loss = focal_loss_with_logits(sample_logits, sample_targets, reduction="none")

        assert loss.shape == sample_logits.shape

    def test_handles_extreme_logits(self):
        """Should handle very large positive and negative logits."""
        logits = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        loss = focal_loss_with_logits(logits, targets, reduction="mean")

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flows_through_loss(self, sample_logits, sample_targets):
        """Should allow gradient to flow through."""
        sample_logits.requires_grad = True

        loss = focal_loss_with_logits(sample_logits, sample_targets, reduction="mean")
        loss.backward()

        assert sample_logits.grad is not None

    def test_all_zeros_targets(self):
        """Should handle all-zero targets."""
        logits = torch.randn(2, 4)
        targets = torch.zeros(2, 4)

        loss = focal_loss_with_logits(logits, targets, reduction="mean")

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_all_ones_targets(self):
        """Should handle all-one targets."""
        logits = torch.randn(2, 4)
        targets = torch.ones(2, 4)

        loss = focal_loss_with_logits(logits, targets, reduction="mean")

        assert not torch.isnan(loss)
        assert loss >= 0
