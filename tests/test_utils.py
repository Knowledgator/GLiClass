"""Tests for gliclass.utils module."""

import pytest
import torch

from gliclass.utils import is_module_available, retrieval_augmented_text, default_f1_reward


class TestIsModuleAvailable:
    """Test suite for is_module_available function."""

    def test_detects_installed_module(self):
        """Should return True for installed modules."""
        assert is_module_available("torch") is True
        assert is_module_available("pytest") is True

    def test_detects_missing_module(self):
        """Should return False for non-existent modules."""
        assert is_module_available("nonexistent_module_12345") is False

    def test_handles_submodules(self):
        """Should work with submodule paths."""
        assert is_module_available("torch.nn") is True


class TestRetrievalAugmentedText:
    """Test suite for retrieval_augmented_text function."""

    def test_with_structured_examples(self):
        """Should concatenate input text with structured examples."""
        text = "This is a test."
        examples = [
            {"text": "Example 1", "true_labels": ["label1"], "all_labels": ["label1", "label2"]},
            {"text": "Example 2", "true_labels": ["label2"], "all_labels": ["label1", "label2"]},
        ]

        result = retrieval_augmented_text(text, examples)

        assert isinstance(result, str)
        assert text in result

    def test_empty_examples_returns_original_text(self):
        """Should return original text when no examples provided."""
        text = "This is a test."
        examples = []

        result = retrieval_augmented_text(text, examples)

        assert result == text

    def test_includes_true_label_markers(self):
        """Should include TRUE_LABEL markers for positive labels."""
        text = "Query text"
        examples = [{"text": "Example", "true_labels": ["tech"], "all_labels": ["tech", "sports"]}]

        result = retrieval_augmented_text(text, examples)

        assert "<<TRUE_LABEL>>" in result
        assert "tech" in result


class TestDefaultF1Reward:
    """Test suite for default_f1_reward function."""

    @pytest.fixture
    def sample_inputs(self):
        """Sample inputs matching the function signature."""
        batch_size = 2
        num_labels = 4
        return {
            "probs": torch.rand(batch_size, num_labels),
            "actions": torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.long),
            "original_targets": torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.long),
            "valid_mask": torch.ones(batch_size, num_labels),
        }

    def test_returns_tensor(self, sample_inputs):
        """Should return a torch tensor."""
        reward = default_f1_reward(**sample_inputs)

        assert isinstance(reward, torch.Tensor)

    def test_output_shape(self, sample_inputs):
        """Should return (N, 1) shaped tensor."""
        reward = default_f1_reward(**sample_inputs)

        assert reward.shape == (2, 1)

    def test_perfect_predictions(self):
        """Should give F1=1.0 for perfect predictions."""
        probs = torch.rand(1, 4)
        actions = torch.tensor([[1, 0, 1, 0]], dtype=torch.long)
        targets = torch.tensor([[1, 0, 1, 0]], dtype=torch.long)
        valid_mask = torch.ones(1, 4)

        reward = default_f1_reward(probs, actions, targets, valid_mask)

        assert torch.allclose(reward, torch.tensor([[1.0]]))

    def test_zero_f1_for_wrong_predictions(self):
        """Should give F1=0.0 when predictions and targets don't overlap."""
        probs = torch.rand(1, 4)
        actions = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        targets = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
        valid_mask = torch.ones(1, 4)

        reward = default_f1_reward(probs, actions, targets, valid_mask)

        assert torch.allclose(reward, torch.tensor([[0.0]]))

    def test_handles_valid_mask(self):
        """Should respect valid_mask to ignore certain positions."""
        probs = torch.rand(1, 4)
        actions = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        targets = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
        valid_mask = torch.tensor([[1, 1, 0, 0]])  # Mask out last two

        reward = default_f1_reward(probs, actions, targets, valid_mask)

        # Should get F1=1.0 since masked positions are ignored
        assert torch.allclose(reward, torch.tensor([[1.0]]))
