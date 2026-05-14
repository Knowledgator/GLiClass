"""Tests for gliclass.poolings module."""

import pytest
import torch

from gliclass.poolings import (
    GlobalMaxPooling1D,
    GlobalAvgPooling1D,
    GlobalSumPooling1D,
    GlobalRMSPooling1D,
    GlobalAbsMaxPooling1D,
    GlobalAbsAvgPooling1D,
    FirstTokenPooling1D,
    LastTokenPooling1D,
    PassPooling1D,
)


class TestGlobalMaxPooling1D:
    """Test suite for GlobalMaxPooling1D."""

    @pytest.fixture
    def pooling_layer(self):
        """Create pooling layer for testing."""
        return GlobalMaxPooling1D()

    @pytest.fixture
    def sample_input(self):
        """Sample input tensor (batch_size, seq_len, hidden_dim)."""
        return torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]])

    def test_returns_max_across_sequence(self, pooling_layer, sample_input):
        """Should return maximum values across sequence dimension."""
        output = pooling_layer(sample_input)

        expected = torch.tensor([[5.0, 6.0], [11.0, 12.0]])
        assert torch.allclose(output, expected)

    def test_output_shape(self, pooling_layer, sample_input):
        """Should reduce sequence dimension."""
        output = pooling_layer(sample_input)

        assert output.shape == (2, 2)  # (batch_size, hidden_dim)


class TestGlobalAvgPooling1D:
    """Test suite for GlobalAvgPooling1D."""

    @pytest.fixture
    def pooling_layer(self):
        """Create pooling layer for testing."""
        return GlobalAvgPooling1D()

    def test_returns_average_across_sequence(self, pooling_layer):
        """Should return average values across sequence dimension."""
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

        output = pooling_layer(inputs)

        expected = torch.tensor([[2.0, 3.0]])  # (1+3)/2, (2+4)/2
        assert torch.allclose(output, expected)

    def test_handles_attention_mask(self, pooling_layer):
        """Should average only over non-masked positions."""
        inputs = torch.tensor([[[2.0, 4.0], [4.0, 6.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]]).unsqueeze(-1)  # (batch, seq, 1)

        output = pooling_layer(inputs, mask)

        expected = torch.tensor([[3.0, 5.0]])  # (2+4)/2, (4+6)/2
        assert torch.allclose(output, expected)

    def test_output_shape(self, pooling_layer):
        """Should reduce sequence dimension."""
        inputs = torch.randn(2, 5, 10)  # (batch, seq, hidden)

        output = pooling_layer(inputs)

        assert output.shape == (2, 10)


class TestGlobalSumPooling1D:
    """Test suite for GlobalSumPooling1D."""

    @pytest.fixture
    def pooling_layer(self):
        """Create pooling layer for testing."""
        return GlobalSumPooling1D()

    def test_returns_sum_across_sequence(self, pooling_layer):
        """Should return sum of values across sequence dimension."""
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

        output = pooling_layer(inputs)

        expected = torch.tensor([[4.0, 6.0]])  # 1+3, 2+4
        assert torch.allclose(output, expected)

    def test_handles_attention_mask(self, pooling_layer):
        """Should sum only over non-masked positions."""
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]]).unsqueeze(-1)  # (batch, seq, 1)

        output = pooling_layer(inputs, mask)

        expected = torch.tensor([[4.0, 6.0]])  # 1+3, 2+4 (masked position becomes 0)
        assert torch.allclose(output, expected)


class TestFirstTokenPooling1D:
    """Test suite for FirstTokenPooling1D."""

    @pytest.fixture
    def pooling_layer(self):
        """Create pooling layer for testing."""
        return FirstTokenPooling1D()

    def test_returns_first_token(self, pooling_layer):
        """Should return the first token representation."""
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

        output = pooling_layer(inputs)

        expected = torch.tensor([[1.0, 2.0]])
        assert torch.allclose(output, expected)

    def test_works_with_batch(self, pooling_layer):
        """Should work with batched inputs."""
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

        output = pooling_layer(inputs)

        expected = torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        assert torch.allclose(output, expected)

    def test_output_shape(self, pooling_layer):
        """Should reduce sequence dimension."""
        inputs = torch.randn(4, 10, 16)

        output = pooling_layer(inputs)

        assert output.shape == (4, 16)


class TestLastTokenPooling1D:
    """Test suite for LastTokenPooling1D."""

    @pytest.fixture
    def pooling_layer(self):
        """Create pooling layer for testing."""
        return LastTokenPooling1D()

    def test_returns_last_token(self, pooling_layer):
        """Should return the last token representation."""
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

        output = pooling_layer(inputs)

        expected = torch.tensor([[5.0, 6.0]])
        assert torch.allclose(output, expected)

    def test_works_with_batch(self, pooling_layer):
        """Should work with batched inputs."""
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

        output = pooling_layer(inputs)

        expected = torch.tensor([[3.0, 4.0], [7.0, 8.0]])
        assert torch.allclose(output, expected)

    def test_output_shape(self, pooling_layer):
        """Should reduce sequence dimension."""
        inputs = torch.randn(4, 10, 16)

        output = pooling_layer(inputs)

        assert output.shape == (4, 16)


class TestGlobalRMSPooling1D:
    """Test suite for GlobalRMSPooling1D."""

    @pytest.fixture
    def pooling_layer(self):
        """Create pooling layer for testing."""
        return GlobalRMSPooling1D()

    def test_returns_rms_across_sequence(self, pooling_layer):
        """Should return RMS values across sequence dimension."""
        inputs = torch.tensor([[[3.0, 4.0], [0.0, 0.0]]])

        output = pooling_layer(inputs)

        expected = torch.sqrt(torch.tensor([[4.5, 8.0]]))
        assert torch.allclose(output, expected)

    def test_handles_attention_mask(self, pooling_layer):
        """Should compute RMS only over non-masked positions."""
        inputs = torch.tensor([[[3.0, 4.0], [3.0, 4.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]]).unsqueeze(-1)

        output = pooling_layer(inputs, mask)

        expected = torch.tensor([[3.0, 4.0]])
        assert torch.allclose(output, expected)

    def test_output_shape(self, pooling_layer):
        """Should reduce sequence dimension."""
        inputs = torch.randn(2, 5, 10)

        output = pooling_layer(inputs)

        assert output.shape == (2, 10)


class TestGlobalAbsMaxPooling1D:
    """Test suite for GlobalAbsMaxPooling1D."""

    @pytest.fixture
    def pooling_layer(self):
        """Create pooling layer for testing."""
        return GlobalAbsMaxPooling1D()

    def test_returns_abs_max_across_sequence(self, pooling_layer):
        """Should return maximum absolute values across sequence dimension."""
        inputs = torch.tensor([[[-5.0, 2.0], [3.0, -4.0], [1.0, 1.0]]])

        output = pooling_layer(inputs)

        expected = torch.tensor([[5.0, 4.0]])
        assert torch.allclose(output, expected)

    def test_handles_attention_mask(self, pooling_layer):
        """Should find abs max only over non-masked positions."""
        inputs = torch.tensor([[[-2.0, 3.0], [4.0, -1.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]]).unsqueeze(-1)

        output = pooling_layer(inputs, mask)

        expected = torch.tensor([[4.0, 3.0]])
        assert torch.allclose(output, expected)

    def test_output_shape(self, pooling_layer):
        """Should reduce sequence dimension."""
        inputs = torch.randn(2, 5, 10)

        output = pooling_layer(inputs)

        assert output.shape == (2, 10)


class TestGlobalAbsAvgPooling1D:
    """Test suite for GlobalAbsAvgPooling1D."""

    @pytest.fixture
    def pooling_layer(self):
        """Create pooling layer for testing."""
        return GlobalAbsAvgPooling1D()

    def test_returns_abs_avg_across_sequence(self, pooling_layer):
        """Should return average of absolute values across sequence dimension."""
        inputs = torch.tensor([[[-2.0, 4.0], [2.0, -4.0]]])

        output = pooling_layer(inputs)

        expected = torch.tensor([[2.0, 4.0]])
        assert torch.allclose(output, expected)

    def test_handles_attention_mask(self, pooling_layer):
        """Should average abs values only over non-masked positions."""
        inputs = torch.tensor([[[-2.0, 4.0], [4.0, -2.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]]).unsqueeze(-1)

        output = pooling_layer(inputs, mask)

        expected = torch.tensor([[3.0, 3.0]])
        assert torch.allclose(output, expected)

    def test_output_shape(self, pooling_layer):
        """Should reduce sequence dimension."""
        inputs = torch.randn(2, 5, 10)

        output = pooling_layer(inputs)

        assert output.shape == (2, 10)


class TestPassPooling1D:
    """Test suite for PassPooling1D."""

    @pytest.fixture
    def pooling_layer(self):
        """Create pooling layer for testing."""
        return PassPooling1D()

    def test_returns_input_unchanged(self, pooling_layer):
        """Should return input tensor without modification."""
        inputs = torch.randn(2, 5, 10)

        output = pooling_layer(inputs)

        assert torch.allclose(output, inputs)

    def test_ignores_attention_mask(self, pooling_layer):
        """Should ignore attention mask and return full input."""
        inputs = torch.randn(2, 5, 10)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])

        output = pooling_layer(inputs, mask)

        assert torch.allclose(output, inputs)

    def test_maintains_shape(self, pooling_layer):
        """Should maintain input shape exactly."""
        inputs = torch.randn(3, 7, 12)

        output = pooling_layer(inputs)

        assert output.shape == inputs.shape
