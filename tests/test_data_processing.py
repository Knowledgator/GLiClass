"""Tests for gliclass.data_processing module."""

import pytest
import torch

from gliclass.data_processing import pad_2d_tensor


class TestPad2DTensor:
    """Test suite for pad_2d_tensor function."""

    @pytest.fixture
    def sample_tensors(self):
        """Fixture providing sample 2D tensors of varying sizes."""
        return [
            torch.tensor([[1, 2], [3, 4]]),  # 2x2
            torch.tensor([[5, 6, 7]]),  # 1x3
            torch.tensor([[8], [9], [10]]),  # 3x1
        ]

    def test_pads_to_maximum_dimensions(self, sample_tensors):
        """Should pad all tensors to match the maximum rows and columns."""
        result = pad_2d_tensor(sample_tensors)

        # batch_size=3, max_rows=3, max_cols=3
        assert result.shape == (3, 3, 3)

    def test_preserves_original_values(self, sample_tensors):
        """Should preserve all original tensor values in padded output."""
        result = pad_2d_tensor(sample_tensors)

        # First tensor: check original values
        assert result[0, 0, 0] == 1
        assert result[0, 0, 1] == 2
        assert result[0, 1, 0] == 3
        assert result[0, 1, 1] == 4

        # Second tensor
        assert result[1, 0, 0] == 5
        assert result[1, 0, 1] == 6
        assert result[1, 0, 2] == 7

        # Third tensor
        assert result[2, 0, 0] == 8
        assert result[2, 1, 0] == 9
        assert result[2, 2, 0] == 10

    def test_pads_with_zeros(self, sample_tensors):
        """Should fill padding positions with zeros."""
        result = pad_2d_tensor(sample_tensors)

        # Check padded positions
        assert result[0, 2, 0] == 0  # Row padding
        assert result[0, 0, 2] == 0  # Column padding
        assert result[1, 1, 0] == 0  # Row padding in second tensor
        assert result[2, 0, 1] == 0  # Column padding in third tensor

    def test_single_tensor(self):
        """Should handle a single tensor correctly."""
        single_tensor = [torch.tensor([[1, 2], [3, 4]])]

        result = pad_2d_tensor(single_tensor)

        assert result.shape == (1, 2, 2)
        assert torch.allclose(result[0], single_tensor[0].long())

    def test_uniform_size_tensors(self):
        """Should handle tensors that are already the same size."""
        uniform_tensors = [
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[5, 6], [7, 8]]),
        ]

        result = pad_2d_tensor(uniform_tensors)

        assert result.shape == (2, 2, 2)
        # No padding needed, should match originals exactly
        assert torch.allclose(result[0], uniform_tensors[0].long())
        assert torch.allclose(result[1], uniform_tensors[1].long())

    def test_empty_tensor_handling(self):
        """Should handle tensors with zero dimensions."""
        tensors_with_empty = [
            torch.tensor([[1, 2]]),
            torch.tensor([[]]),  # Empty second dimension
        ]

        result = pad_2d_tensor(tensors_with_empty)

        # Should not crash and should return valid tensor
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 2  # batch size

    def test_preserves_dtype(self):
        """Should preserve data type of input tensors."""
        float_tensors = [
            torch.tensor([[1.5, 2.5]], dtype=torch.float32),
            torch.tensor([[3.5]], dtype=torch.float32),
        ]

        result = pad_2d_tensor(float_tensors)

        assert result.dtype == torch.float32

    def test_varying_row_counts(self):
        """Should handle tensors with different numbers of rows."""
        tensors = [
            torch.tensor([[1]]),  # 1 row
            torch.tensor([[2], [3], [4], [5]]),  # 4 rows
            torch.tensor([[6], [7]]),  # 2 rows
        ]

        result = pad_2d_tensor(tensors)

        # Max rows = 4
        assert result.shape == (3, 4, 1)

    def test_varying_column_counts(self):
        """Should handle tensors with different numbers of columns."""
        tensors = [
            torch.tensor([[1, 2, 3, 4, 5]]),  # 5 cols
            torch.tensor([[6, 7]]),  # 2 cols
            torch.tensor([[8]]),  # 1 col
        ]

        result = pad_2d_tensor(tensors)

        # Max cols = 5
        assert result.shape == (3, 1, 5)

    def test_batch_consistency(self):
        """Should maintain batch order and size."""
        tensors = [
            torch.tensor([[1]]),
            torch.tensor([[2]]),
            torch.tensor([[3]]),
        ]

        result = pad_2d_tensor(tensors)

        assert result.shape[0] == 3
        assert result[0, 0, 0] == 1
        assert result[1, 0, 0] == 2
        assert result[2, 0, 0] == 3
