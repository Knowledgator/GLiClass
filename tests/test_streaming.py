"""Tests for gliclass.streaming — strategies, cache management, BatchedKVHelper."""

import pytest
import torch
from transformers.cache_utils import DynamicCache

from gliclass.streaming.cache import (
    CacheState,
    BatchedKVHelper,
    create_empty_cache,
    truncate_cache,
)
from gliclass.streaming.strategies import (
    ClassificationStrategy,
    EveryChunkStrategy,
    EveryNTokensStrategy,
    OnDelimiterStrategy,
    NeverStrategy,
    SlidingWindowStrategy,
    ComposedStrategy,
)
from gliclass.streaming.types import SessionInput, SessionOutput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dynamic_cache(num_layers: int, num_heads: int, seq_len: int, head_dim: int, batch: int = 1) -> DynamicCache:
    cache = DynamicCache()
    for i in range(num_layers):
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)
        cache.update(k, v, i)
    return cache


def _make_cache_state(seq_len: int, num_layers: int = 2, num_heads: int = 2, head_dim: int = 8) -> CacheState:
    past_kv = _make_dynamic_cache(num_layers, num_heads, seq_len, head_dim)
    return CacheState(
        past_key_values=past_kv,
        input_ids=torch.ones(seq_len, dtype=torch.long),
        attention_mask=torch.ones(seq_len, dtype=torch.long),
        cached_length=seq_len,
    )


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestEveryChunkStrategy:
    def test_triggers_on_any_positive_tokens(self):
        s = EveryChunkStrategy()
        assert s.should_classify(1, 10, "text") is True
        assert s.should_classify(50, 200, "text") is True

    def test_does_not_trigger_on_zero_tokens(self):
        s = EveryChunkStrategy()
        assert s.should_classify(0, 10, "text") is False

    def test_get_window_returns_full_cache(self):
        s = EveryChunkStrategy()
        cache = _make_cache_state(20)
        result = s.get_window(cache)
        assert result is cache


class TestEveryNTokensStrategy:
    def test_triggers_after_n_tokens(self):
        s = EveryNTokensStrategy(n=10)
        assert s.should_classify(5, 5, "") is False
        assert s.should_classify(5, 10, "") is True

    def test_counter_resets_after_trigger(self):
        s = EveryNTokensStrategy(n=10)
        s.should_classify(10, 10, "")   # trigger, reset
        assert s._accumulated == 0
        assert s.should_classify(5, 15, "") is False

    def test_accumulates_across_calls(self):
        s = EveryNTokensStrategy(n=30)
        assert s.should_classify(10, 10, "") is False
        assert s.should_classify(10, 20, "") is False
        assert s.should_classify(10, 30, "") is True

    def test_exact_n(self):
        s = EveryNTokensStrategy(n=7)
        assert s.should_classify(7, 7, "") is True

    def test_overshoot_still_triggers(self):
        s = EveryNTokensStrategy(n=5)
        assert s.should_classify(20, 20, "") is True

    def test_independent_instances(self):
        s1 = EveryNTokensStrategy(n=10)
        s2 = EveryNTokensStrategy(n=10)
        s1.should_classify(8, 8, "")
        assert s2._accumulated == 0


class TestOnDelimiterStrategy:
    def test_triggers_when_delimiter_present(self):
        s = OnDelimiterStrategy(delimiter=".")
        assert s.should_classify(5, 5, "End of sentence.") is True

    def test_does_not_trigger_without_delimiter(self):
        s = OnDelimiterStrategy(delimiter=".")
        assert s.should_classify(5, 5, "No period here") is False

    def test_newline_delimiter(self):
        s = OnDelimiterStrategy(delimiter="\n\n")
        assert s.should_classify(3, 3, "para1\n\npara2") is True
        assert s.should_classify(3, 3, "para1\npara2") is False

    def test_empty_text(self):
        s = OnDelimiterStrategy(delimiter=".")
        assert s.should_classify(0, 0, "") is False

    def test_delimiter_in_middle(self):
        s = OnDelimiterStrategy(delimiter=";")
        assert s.should_classify(5, 5, "part1; part2") is True


class TestNeverStrategy:
    def test_never_triggers(self):
        s = NeverStrategy()
        assert s.should_classify(100, 1000, "text") is False
        assert s.should_classify(0, 0, "") is False

    def test_get_window_returns_full_cache(self):
        s = NeverStrategy()
        cache = _make_cache_state(10)
        assert s.get_window(cache) is cache


class TestSlidingWindowStrategy:
    def test_triggers_on_positive_tokens(self):
        s = SlidingWindowStrategy(window_size=50)
        assert s.should_classify(1, 10, "text") is True

    def test_does_not_trigger_on_zero_tokens(self):
        s = SlidingWindowStrategy(window_size=50)
        assert s.should_classify(0, 10, "text") is False

    def test_window_truncates_cache(self):
        s = SlidingWindowStrategy(window_size=10)
        cache = _make_cache_state(30)
        windowed = s.get_window(cache)
        assert windowed.cached_length == 10

    def test_window_larger_than_cache_returns_full(self):
        s = SlidingWindowStrategy(window_size=100)
        cache = _make_cache_state(20)
        windowed = s.get_window(cache)
        assert windowed.cached_length == 20

    def test_original_cache_unmodified(self):
        s = SlidingWindowStrategy(window_size=5)
        cache = _make_cache_state(20)
        s.get_window(cache)
        assert cache.cached_length == 20


class TestComposedStrategy:
    def test_trigger_delegated_to_trigger_strategy(self):
        trigger = EveryNTokensStrategy(n=10)
        window = SlidingWindowStrategy(window_size=50)
        s = ComposedStrategy(trigger=trigger, window=window)
        assert s.should_classify(5, 5, "") is False
        assert s.should_classify(5, 10, "") is True

    def test_window_delegated_to_window_strategy(self):
        trigger = EveryChunkStrategy()
        window = SlidingWindowStrategy(window_size=10)
        s = ComposedStrategy(trigger=trigger, window=window)
        cache = _make_cache_state(30)
        windowed = s.get_window(cache)
        assert windowed.cached_length == 10

    def test_never_trigger_with_every_chunk_window(self):
        s = ComposedStrategy(trigger=NeverStrategy(), window=SlidingWindowStrategy(20))
        assert s.should_classify(100, 100, "text") is False


class TestCustomStrategy:
    def test_custom_strategy_interface(self):
        class CountStrategy(ClassificationStrategy):
            def __init__(self, n):
                self.n = n
                self.calls = 0

            def should_classify(self, tokens_added, cached_length, text):
                self.calls += 1
                return self.calls >= self.n

        s = CountStrategy(n=3)
        assert s.should_classify(1, 1, "") is False
        assert s.should_classify(1, 2, "") is False
        assert s.should_classify(1, 3, "") is True


# ---------------------------------------------------------------------------
# CacheState tests
# ---------------------------------------------------------------------------

class TestCacheState:
    def test_create_empty_cache(self):
        cache = create_empty_cache(session_id="test")
        assert cache.cached_length == 0
        assert cache.past_key_values is None
        assert cache.session_id == "test"
        assert cache.input_ids.shape == (0,)

    def test_create_empty_cache_with_device(self):
        device = torch.device("cpu")
        cache = create_empty_cache(session_id="s", device=device)
        assert cache.input_ids.device.type == "cpu"

    def test_to_device(self):
        cache = _make_cache_state(10)
        moved = cache.to(torch.device("cpu"))
        assert moved.input_ids.device.type == "cpu"
        assert moved.cached_length == 10
        assert moved.session_id == cache.session_id

    def test_metadata_preserved_on_move(self):
        cache = _make_cache_state(5)
        cache.metadata = {"key": "value"}
        moved = cache.to(torch.device("cpu"))
        assert moved.metadata == {"key": "value"}


class TestTruncateCache:
    def test_truncate_reduces_length(self):
        cache = _make_cache_state(50)
        truncated = truncate_cache(cache, max_length=20)
        assert truncated.cached_length == 20

    def test_truncate_keeps_last_tokens(self):
        cache = _make_cache_state(10)
        cache.input_ids = torch.arange(10, dtype=torch.long)
        truncated = truncate_cache(cache, max_length=5)
        assert truncated.cached_length == 5
        assert truncated.input_ids.tolist() == [5, 6, 7, 8, 9]

    def test_truncate_attention_mask(self):
        cache = _make_cache_state(10)
        truncated = truncate_cache(cache, max_length=4)
        assert truncated.attention_mask.shape == (4,)

    def test_no_truncate_when_within_limit(self):
        cache = _make_cache_state(10)
        result = truncate_cache(cache, max_length=20)
        assert result is cache

    def test_no_truncate_on_empty_cache(self):
        cache = create_empty_cache()
        result = truncate_cache(cache, max_length=10)
        assert result is cache

    def test_truncate_kv_shape(self):
        num_layers, num_heads, seq_len, head_dim = 2, 2, 30, 8
        cache = _make_cache_state(seq_len, num_layers, num_heads, head_dim)
        truncated = truncate_cache(cache, max_length=10)
        for layer in truncated.past_key_values.layers:
            if layer.is_initialized:
                assert layer.keys.shape[2] == 10

    def test_session_id_preserved(self):
        cache = _make_cache_state(10)
        cache.session_id = "my_session"
        truncated = truncate_cache(cache, max_length=5)
        assert truncated.session_id == "my_session"

    def test_metadata_preserved(self):
        cache = _make_cache_state(10)
        cache.metadata = {"info": 42}
        truncated = truncate_cache(cache, max_length=5)
        assert truncated.metadata == {"info": 42}


# ---------------------------------------------------------------------------
# BatchedKVHelper tests
# ---------------------------------------------------------------------------

class TestBatchedKVHelperStack:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def _make_inputs(self, lengths, device):
        ids = [torch.ones(l, dtype=torch.long, device=device) for l in lengths]
        masks = [torch.ones(l, dtype=torch.long, device=device) for l in lengths]
        return ids, masks

    def test_stack_homogeneous_caches(self, device):
        caches = [_make_cache_state(10) for _ in range(3)]
        ids, masks = self._make_inputs([5, 5, 5], device)
        result = BatchedKVHelper.stack_for_update(caches, ids, masks, device)

        assert result["input_ids"].shape == (3, 5)
        assert result["max_cached_len"] == 10
        assert result["cached_lengths"] == [10, 10, 10]
        assert result["new_lengths"] == [5, 5, 5]

    def test_stack_heterogeneous_caches(self, device):
        caches = [_make_cache_state(5), _make_cache_state(20), _make_cache_state(10)]
        ids, masks = self._make_inputs([3, 3, 3], device)
        result = BatchedKVHelper.stack_for_update(caches, ids, masks, device)

        assert result["max_cached_len"] == 20
        assert result["input_ids"].shape == (3, 3)

    def test_stack_with_empty_cache(self, device):
        empty = create_empty_cache(device=device)
        filled = _make_cache_state(10)
        caches = [empty, filled]
        ids, masks = self._make_inputs([4, 4], device)
        result = BatchedKVHelper.stack_for_update(caches, ids, masks, device)

        assert result["cached_lengths"] == [0, 10]
        assert result["max_cached_len"] == 10

    def test_attention_mask_shape(self, device):
        caches = [_make_cache_state(8), _make_cache_state(12)]
        ids, masks = self._make_inputs([5, 3], device)
        result = BatchedKVHelper.stack_for_update(caches, ids, masks, device)

        # full mask = max_cached + max_new
        assert result["attention_mask"].shape == (2, 12 + 5)

    def test_position_ids_per_session(self, device):
        caches = [_make_cache_state(5), _make_cache_state(10)]
        ids, masks = self._make_inputs([3, 3], device)
        result = BatchedKVHelper.stack_for_update(caches, ids, masks, device)

        pos = result["position_ids"]
        # session 0: positions [5, 6, 7]
        assert pos[0, :3].tolist() == [5, 6, 7]
        # session 1: positions [10, 11, 12]
        assert pos[1, :3].tolist() == [10, 11, 12]

    def test_all_empty_caches_no_stacked_kv(self, device):
        caches = [create_empty_cache(device=device) for _ in range(2)]
        ids, masks = self._make_inputs([4, 4], device)
        result = BatchedKVHelper.stack_for_update(caches, ids, masks, device)

        assert result["past_key_values"] is None
        assert result["max_cached_len"] == 0

    def test_padded_new_tokens_are_zero(self, device):
        caches = [_make_cache_state(5), _make_cache_state(5)]
        ids = [torch.ones(6, dtype=torch.long, device=device),
               torch.ones(3, dtype=torch.long, device=device)]
        masks = [torch.ones(6, dtype=torch.long, device=device),
                 torch.ones(3, dtype=torch.long, device=device)]
        result = BatchedKVHelper.stack_for_update(caches, ids, masks, device)

        # session 1 has 3 tokens, padded to 6 — positions 3,4,5 should be 0
        assert result["input_ids"][1, 3:].sum().item() == 0


class TestBatchedKVHelperUnstack:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_unstack_restores_cache_lengths(self, device):
        caches = [_make_cache_state(5), _make_cache_state(10)]
        ids = [torch.ones(3, dtype=torch.long, device=device),
               torch.ones(3, dtype=torch.long, device=device)]
        masks = [torch.ones(3, dtype=torch.long, device=device),
                 torch.ones(3, dtype=torch.long, device=device)]

        stacked = BatchedKVHelper.stack_for_update(caches, ids, masks, device)

        # simulate decoder output with updated KV
        num_layers, num_heads, head_dim = 2, 2, 8
        max_total = stacked["max_cached_len"] + max(stacked["new_lengths"])
        new_kv = _make_dynamic_cache(num_layers, num_heads, max_total, head_dim, batch=2)

        updated = BatchedKVHelper.unstack_after_update(new_kv, stacked, caches, ids, masks)

        assert updated[0].cached_length == 5 + 3
        assert updated[1].cached_length == 10 + 3

    def test_unstack_preserves_session_ids(self, device):
        c1 = _make_cache_state(5)
        c1.session_id = "alice"
        c2 = _make_cache_state(8)
        c2.session_id = "bob"
        caches = [c1, c2]

        ids = [torch.ones(2, dtype=torch.long, device=device),
               torch.ones(2, dtype=torch.long, device=device)]
        masks = ids[:]

        stacked = BatchedKVHelper.stack_for_update(caches, ids, masks, device)
        max_total = stacked["max_cached_len"] + 2
        new_kv = _make_dynamic_cache(2, 2, max_total, 8, batch=2)
        updated = BatchedKVHelper.unstack_after_update(new_kv, stacked, caches, ids, masks)

        assert updated[0].session_id == "alice"
        assert updated[1].session_id == "bob"

    def test_unstack_input_ids_concatenated(self, device):
        cache = _make_cache_state(3)
        cache.input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        cache.attention_mask = torch.ones(3, dtype=torch.long)
        caches = [cache]

        new_ids = [torch.tensor([4, 5], dtype=torch.long, device=device)]
        new_masks = [torch.ones(2, dtype=torch.long, device=device)]

        stacked = BatchedKVHelper.stack_for_update(caches, new_ids, new_masks, device)
        new_kv = _make_dynamic_cache(2, 2, stacked["max_cached_len"] + 2, 8, batch=1)
        updated = BatchedKVHelper.unstack_after_update(new_kv, stacked, caches, new_ids, new_masks)

        assert updated[0].input_ids.tolist() == [1, 2, 3, 4, 5]


class TestBatchedKVHelperClassify:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_stack_for_classify_shapes(self, device):
        caches = [_make_cache_state(10), _make_cache_state(15)]
        label_ids = [torch.ones(8, dtype=torch.long, device=device),
                     torch.ones(6, dtype=torch.long, device=device)]
        label_masks = [torch.ones(8, dtype=torch.long, device=device),
                       torch.ones(6, dtype=torch.long, device=device)]

        result = BatchedKVHelper.stack_for_classify(caches, label_ids, label_masks, device)

        assert result["input_ids"].shape == (2, 8)   # max_label = 8
        assert result["attention_mask"].shape == (2, 15 + 8)   # max_cached + max_label
        assert result["label_lengths"] == [8, 6]
        assert result["max_label_len"] == 8

    def test_stack_for_classify_position_ids(self, device):
        caches = [_make_cache_state(5), _make_cache_state(12)]
        label_ids = [torch.ones(4, dtype=torch.long, device=device),
                     torch.ones(4, dtype=torch.long, device=device)]
        label_masks = label_ids[:]

        result = BatchedKVHelper.stack_for_classify(caches, label_ids, label_masks, device)

        pos = result["position_ids"]
        assert pos[0, :4].tolist() == [5, 6, 7, 8]
        assert pos[1, :4].tolist() == [12, 13, 14, 15]

    def test_label_mask_is_label_only(self, device):
        caches = [_make_cache_state(10)]
        label_ids = [torch.ones(5, dtype=torch.long, device=device)]
        label_masks = [torch.ones(5, dtype=torch.long, device=device)]

        result = BatchedKVHelper.stack_for_classify(caches, label_ids, label_masks, device)

        # label_mask covers only label tokens (not cache)
        assert result["label_mask"].shape == (1, 5)


# ---------------------------------------------------------------------------
# Data types tests
# ---------------------------------------------------------------------------

class TestDataTypes:
    def test_session_input_defaults(self):
        from gliclass.streaming.strategies import EveryChunkStrategy
        inp = SessionInput(
            session_id="s1",
            text="hello",
            labels=["a", "b"],
            strategy=EveryChunkStrategy(),
        )
        assert inp.classification_type == "multi-label"

    def test_session_output_not_triggered(self):
        out = SessionOutput(
            session_id="s1",
            triggered=False,
            predictions=None,
            cached_length=10,
            tokens_added=5,
        )
        assert out.predictions is None
        assert not out.triggered
        assert out.metadata == {}

    def test_session_output_triggered(self):
        preds = [{"label": "positive", "score": 0.9}]
        out = SessionOutput(
            session_id="s1",
            triggered=True,
            predictions=preds,
            cached_length=50,
            tokens_added=10,
        )
        assert out.triggered
        assert out.predictions[0]["label"] == "positive"
