"""Tests for gliclass.scorers module."""

import pytest
import torch

from gliclass.scorers import (
    ScorerWeightedDot,
    ScorerDot,
    MLPScorer,
    HopfieldScorer,
    CrossAttnScorer,
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
