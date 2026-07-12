"""Tests for gliclass.scorers module."""

from types import SimpleNamespace

import pytest
import torch

from gliclass.scorers import (
    ScorerWeightedDot,
    ScorerDot,
    MLPScorer,
    HopfieldScorer,
    CrossAttnScorer,
    DecoderKVScorer,
)


class TestScorerWeightedDot:
    @pytest.fixture
    def scorer(self):
        return ScorerWeightedDot(hidden_size=128)

    def test_forward_pass(self, scorer):
        text_rep = torch.randn(4, 128)
        label_rep = torch.randn(4, 10, 128)

        scores = scorer(text_rep, label_rep)

        assert scores.shape == (4, 10)
        assert not torch.isnan(scores).any()

    def test_gradient_flow(self, scorer):
        text_rep = torch.randn(4, 128, requires_grad=True)
        label_rep = torch.randn(4, 10, 128, requires_grad=True)

        scores = scorer(text_rep, label_rep)
        loss = scores.sum()
        loss.backward()

        assert text_rep.grad is not None
        assert label_rep.grad is not None


class TestScorerDot:
    @pytest.fixture
    def scorer(self):
        return ScorerDot()

    def test_forward_pass(self, scorer):
        text_rep = torch.randn(4, 128)
        label_rep = torch.randn(4, 10, 128)

        scores = scorer(text_rep, label_rep)

        assert scores.shape == (4, 10)
        assert not torch.isnan(scores).any()

    def test_gradient_flow(self, scorer):
        text_rep = torch.randn(4, 128, requires_grad=True)
        label_rep = torch.randn(4, 10, 128, requires_grad=True)

        scores = scorer(text_rep, label_rep)
        loss = scores.sum()
        loss.backward()

        assert text_rep.grad is not None
        assert label_rep.grad is not None


class TestMLPScorer:
    @pytest.fixture
    def scorer(self):
        return MLPScorer(hidden_size=128)

    def test_forward_pass(self, scorer):
        text_rep = torch.randn(4, 128)
        label_rep = torch.randn(4, 10, 128)

        scores = scorer(text_rep, label_rep)

        assert scores.shape == (4, 10)
        assert not torch.isnan(scores).any()

    def test_different_batch_sizes(self, scorer):
        for batch_size in [1, 2, 8]:
            text_rep = torch.randn(batch_size, 128)
            label_rep = torch.randn(batch_size, 10, 128)

            scores = scorer(text_rep, label_rep)

            assert scores.shape == (batch_size, 10)

    def test_gradient_flow(self, scorer):
        text_rep = torch.randn(4, 128, requires_grad=True)
        label_rep = torch.randn(4, 10, 128, requires_grad=True)

        scores = scorer(text_rep, label_rep)
        loss = scores.sum()
        loss.backward()

        assert text_rep.grad is not None
        assert label_rep.grad is not None


class TestHopfieldScorer:
    @pytest.fixture
    def scorer(self):
        return HopfieldScorer(hidden_size=128)

    def test_forward_pass(self, scorer):
        text_rep = torch.randn(4, 128)
        label_rep = torch.randn(4, 10, 128)

        scores = scorer(text_rep, label_rep)

        assert scores.shape == (4, 10)
        assert not torch.isnan(scores).any()

    def test_multiple_iterations(self):
        scorer = HopfieldScorer(hidden_size=128, num_iteration=3)
        text_rep = torch.randn(4, 128)
        label_rep = torch.randn(4, 10, 128)

        scores = scorer(text_rep, label_rep)

        assert scores.shape == (4, 10)

    def test_gradient_flow(self, scorer):
        text_rep = torch.randn(4, 128, requires_grad=True)
        label_rep = torch.randn(4, 10, 128, requires_grad=True)

        scores = scorer(text_rep, label_rep)
        loss = scores.sum()
        loss.backward()

        assert text_rep.grad is not None
        assert label_rep.grad is not None


class TestCrossAttnScorer:
    @pytest.fixture
    def scorer(self):
        return CrossAttnScorer(hidden_size=128, num_heads=8)

    def test_forward_pass_with_text_mask(self, scorer):
        text_rep = torch.randn(4, 20, 128)
        label_rep = torch.randn(4, 10, 128)
        text_mask = torch.ones(4, 20, dtype=torch.bool)
        text_mask[:, 15:] = 0

        scores = scorer(text_rep, label_rep, text_mask=text_mask)

        assert scores.shape == (4, 10)
        assert not torch.isnan(scores).any()

    def test_forward_pass_without_text_mask(self, scorer):
        text_rep = torch.randn(4, 20, 128)
        label_rep = torch.randn(4, 10, 128)

        scores = scorer(text_rep, label_rep)

        assert scores.shape == (4, 10)
        assert not torch.isnan(scores).any()

    def test_different_seq_lengths(self, scorer):
        for seq_len in [10, 20, 50]:
            text_rep = torch.randn(4, seq_len, 128)
            label_rep = torch.randn(4, 10, 128)

            scores = scorer(text_rep, label_rep)

            assert scores.shape == (4, 10)

    def test_gradient_flow(self, scorer):
        text_rep = torch.randn(4, 20, 128, requires_grad=True)
        label_rep = torch.randn(4, 10, 128, requires_grad=True)

        scores = scorer(text_rep, label_rep)
        loss = scores.sum()
        loss.backward()

        assert text_rep.grad is not None
        assert label_rep.grad is not None

    def test_eval_mode(self, scorer):
        scorer.eval()
        text_rep = torch.randn(4, 20, 128)
        label_rep = torch.randn(4, 10, 128)

        with torch.no_grad():
            scores = scorer(text_rep, label_rep)

        assert scores.shape == (4, 10)
        assert not torch.isnan(scores).any()


class TestDecoderKVScorerRepresentationExtraction:
    @pytest.fixture
    def scorer(self):
        scorer = DecoderKVScorer.__new__(DecoderKVScorer)
        torch.nn.Module.__init__(scorer)
        scorer.config = SimpleNamespace(sep_token_index=98, class_token_index=99)
        return scorer

    def test_extracts_last_sep_and_labels_with_padding(self, scorer):
        hidden_states = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
        input_ids = torch.tensor(
            [
                [98, 4, 99, 5, 99, 98, 0, 0],
                [98, 6, 99, 98, 0, 99, 98, 0],
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
            ]
        )

        text_repr, label_repr = scorer._extract_representations(hidden_states, input_ids, attention_mask)

        torch.testing.assert_close(text_repr, torch.stack([hidden_states[0, 5], hidden_states[1, 3]]))
        torch.testing.assert_close(label_repr[0], torch.stack([hidden_states[0, 2], hidden_states[0, 4]]))
        torch.testing.assert_close(label_repr[1, 0], hidden_states[1, 2])
        torch.testing.assert_close(label_repr[1, 1], torch.zeros(3))

    def test_missing_sep_returns_zero_representation(self, scorer):
        hidden_states = torch.randn(1, 4, 3)
        input_ids = torch.tensor([[1, 99, 2, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])

        text_repr, label_repr = scorer._extract_representations(hidden_states, input_ids, attention_mask)

        torch.testing.assert_close(text_repr, torch.zeros_like(text_repr))
        torch.testing.assert_close(label_repr[0, 0], hidden_states[0, 1])

    def test_extracted_representations_preserve_gradients(self, scorer):
        hidden_states = torch.randn(1, 5, 3, requires_grad=True)
        input_ids = torch.tensor([[98, 1, 99, 2, 98]])
        attention_mask = torch.ones_like(input_ids)

        text_repr, label_repr = scorer._extract_representations(hidden_states, input_ids, attention_mask)
        (text_repr.sum() + label_repr.sum()).backward()

        assert hidden_states.grad is not None
        torch.testing.assert_close(hidden_states.grad[0, 2], torch.ones(3))
        torch.testing.assert_close(hidden_states.grad[0, 4], torch.ones(3))
