"""Tests for streaming strategies, cache management, and classification."""

from types import SimpleNamespace

import torch
import pytest
from torch import nn
from transformers.cache_utils import DynamicCache

from gliclass.streaming.cache import (
    CacheState,
    BatchedKVHelper,
    truncate_cache,
    create_empty_cache,
)
from gliclass.streaming.pipeline import StreamingZeroShotClassificationPipeline
from gliclass.streaming.strategies import (
    NeverStrategy,
    ComposedStrategy,
    EveryChunkStrategy,
    OnDelimiterStrategy,
    EveryNTokensStrategy,
    SlidingWindowStrategy,
    ClassificationStrategy,
)

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
        next_position_id=seq_len,
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
        s.should_classify(10, 10, "")  # trigger, reset
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

    def test_next_position_preserved_on_move(self):
        cache = _make_cache_state(5)
        cache.next_position_id = 12
        assert cache.to(torch.device("cpu")).next_position == 12


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

    def test_truncate_preserves_absolute_next_position(self):
        cache = _make_cache_state(10)
        cache.next_position_id = 20
        truncated = truncate_cache(cache, max_length=4)
        assert truncated.cached_length == 4
        assert truncated.next_position == 20


# ---------------------------------------------------------------------------
# BatchedKVHelper tests
# ---------------------------------------------------------------------------


class TestBatchedKVHelperStack:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def _make_inputs(self, lengths, device):
        ids = [torch.ones(length, dtype=torch.long, device=device) for length in lengths]
        masks = [torch.ones(length, dtype=torch.long, device=device) for length in lengths]
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
        ids = [torch.ones(6, dtype=torch.long, device=device), torch.ones(3, dtype=torch.long, device=device)]
        masks = [torch.ones(6, dtype=torch.long, device=device), torch.ones(3, dtype=torch.long, device=device)]
        result = BatchedKVHelper.stack_for_update(caches, ids, masks, device)

        # session 1 has 3 tokens, padded to 6 — positions 3,4,5 should be 0
        assert result["input_ids"][1, 3:].sum().item() == 0


class TestBatchedKVHelperUnstack:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_unstack_restores_cache_lengths(self, device):
        caches = [_make_cache_state(5), _make_cache_state(10)]
        ids = [torch.ones(3, dtype=torch.long, device=device), torch.ones(3, dtype=torch.long, device=device)]
        masks = [torch.ones(3, dtype=torch.long, device=device), torch.ones(3, dtype=torch.long, device=device)]

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

        ids = [torch.ones(2, dtype=torch.long, device=device), torch.ones(2, dtype=torch.long, device=device)]
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

    def test_unstack_advances_absolute_position(self, device):
        cache = _make_cache_state(3)
        cache.next_position_id = 10
        ids = [torch.ones(2, dtype=torch.long, device=device)]
        masks = [torch.ones(2, dtype=torch.long, device=device)]
        stacked = BatchedKVHelper.stack_for_update([cache], ids, masks, device)
        new_kv = _make_dynamic_cache(2, 2, 5, 8, batch=1)
        updated = BatchedKVHelper.unstack_after_update(new_kv, stacked, [cache], ids, masks)
        assert updated[0].next_position == 12

    def test_sliced_cache_owns_its_storage(self):
        source = _make_dynamic_cache(1, 2, 8, 4, batch=2)
        sliced = BatchedKVHelper._slice_past_kv(source, 0, 2, 6)
        source.layers[0].keys.fill_(123)
        assert not torch.all(sliced.layers[0].keys == 123)


class TestBatchedKVHelperClassify:
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_stack_for_classify_shapes(self, device):
        caches = [_make_cache_state(10), _make_cache_state(15)]
        label_ids = [torch.ones(8, dtype=torch.long, device=device), torch.ones(6, dtype=torch.long, device=device)]
        label_masks = [torch.ones(8, dtype=torch.long, device=device), torch.ones(6, dtype=torch.long, device=device)]

        result = BatchedKVHelper.stack_for_classify(caches, label_ids, label_masks, device)

        assert result["input_ids"].shape == (2, 8)  # max_label = 8
        assert result["attention_mask"].shape == (2, 15 + 8)  # max_cached + max_label
        assert result["label_lengths"] == [8, 6]
        assert result["max_label_len"] == 8

    def test_stack_for_classify_position_ids(self, device):
        caches = [_make_cache_state(5), _make_cache_state(12)]
        label_ids = [torch.ones(4, dtype=torch.long, device=device), torch.ones(4, dtype=torch.long, device=device)]
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


class _Tokenizer:
    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        del return_tensors, add_special_tokens
        ids = []
        cursor = 0
        special = {"<<SEP>>": 98, "<<LABEL>>": 99, "<<EXAMPLE>>": 97}
        while cursor < len(text):
            match = next((token for token in special if text.startswith(token, cursor)), None)
            if match is not None:
                ids.append(special[match])
                cursor += len(match)
            else:
                ids.append((ord(text[cursor]) % 90) + 1)
                cursor += 1
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}


class _StreamingModel:
    def __init__(self):
        self.config = SimpleNamespace(architecture_type="decoder-kv", max_labels_alloc="dynamic")
        self.device = torch.device("cpu")
        self.model = SimpleNamespace(scorer=nn.Linear(1, 1))
        self.position_history = []

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def update_decoder_cache(self, input_ids, attention_mask, position_ids, past_key_values):
        del attention_mask
        self.position_history.append(position_ids.clone())
        batch, new_length = input_ids.shape
        old_length = 0 if past_key_values is None else past_key_values.layers[0].keys.shape[2]
        cache = _make_dynamic_cache(1, 1, old_length + new_length, 2, batch=batch)
        return SimpleNamespace(past_key_values=cache)

    def classify_from_decoder_cache(
        self,
        input_ids,
        attention_mask,
        label_mask,
        position_ids,
        past_key_values,
    ):
        del attention_mask, label_mask, position_ids, past_key_values
        counts = input_ids.eq(99).sum(dim=1)
        max_labels = int(counts.max().item())
        logits = torch.zeros(input_ids.shape[0], max_labels)
        if max_labels:
            logits[:, 0] = 2.0
        return SimpleNamespace(logits=logits)


def _make_pipeline(**kwargs):
    return StreamingZeroShotClassificationPipeline(
        _StreamingModel(),
        _Tokenizer(),
        device="cpu",
        **kwargs,
    )


class TestStreamingPipeline:
    def test_new_api_and_automatic_session_cleanup(self):
        pipeline = _make_pipeline()
        first = pipeline(["ab", "cd"], ["x", "y"], session_ids=["a", "b"])
        assert [item["session_id"] for item in first] == ["a", "b"]
        assert pipeline.active_sessions == ["a", "b"]

        pipeline("ef", ["x", "y"], session_ids="b")
        assert pipeline.active_sessions == ["b"]
        assert "a" not in pipeline._session_strategies

    def test_duplicate_session_ids_are_rejected_before_cleanup(self):
        pipeline = _make_pipeline()
        pipeline("a", ["x"], session_ids="kept")
        with pytest.raises(ValueError, match="unique"):
            pipeline(["b", "c"], ["x"], session_ids=["dup", "dup"])
        assert pipeline.active_sessions == ["kept"]

    def test_invalid_batch_size_is_rejected_before_cleanup(self):
        pipeline = _make_pipeline()
        pipeline("a", ["x"], session_ids="kept")
        with pytest.raises(ValueError, match="batch_size"):
            pipeline("b", ["x"], session_ids="new", batch_size=0)
        assert pipeline.active_sessions == ["kept"]

    def test_multi_label_threshold_matches_regular_pipeline_behavior(self):
        pipeline = _make_pipeline()
        output = pipeline("a", ["first", "second"], session_ids="s", threshold=0.6)[0]
        assert output["triggered"] is True
        assert [item["label"] for item in output["predictions"]] == ["first"]

    def test_strategy_state_is_independent_per_session(self):
        pipeline = _make_pipeline(default_strategy=EveryNTokensStrategy(3))
        first = pipeline(["a", "a"], ["x"], session_ids=["a", "b"])
        assert [item["triggered"] for item in first] == [False, False]
        second = pipeline(["aa", "a"], ["x"], session_ids=["a", "b"])
        assert [item["triggered"] for item in second] == [True, False]

    def test_prompt_is_cached_only_once(self):
        pipeline = _make_pipeline()
        first = pipeline("a", ["x"], session_ids="s", prompt="p")[0]
        second = pipeline("b", ["x"], session_ids="s", prompt="p")[0]
        assert first["tokens_added"] == 2
        assert second["tokens_added"] == 1
        assert second["cached_length"] == 3

    def test_prefix_only_update_does_not_classify(self):
        pipeline = _make_pipeline()
        output = pipeline("", ["x"], session_ids="s", prompt="prefix")[0]
        assert output["tokens_added"] == len("prefix")
        assert output["triggered"] is False

    def test_truncated_cache_keeps_monotonic_positions(self):
        pipeline = _make_pipeline(max_cache_len=3)
        pipeline("abcde", ["x"], session_ids="s")
        assert pipeline._caches["s"].cached_length == 3
        assert pipeline._caches["s"].next_position == 5
        pipeline("fg", ["x"], session_ids="s")
        assert pipeline.model.position_history[-1][0, :2].tolist() == [5, 6]
        assert pipeline._caches["s"].next_position == 7

    def test_empty_call_clears_sessions(self):
        pipeline = _make_pipeline()
        pipeline("a", ["x"], session_ids="s")
        assert pipeline([], [], session_ids=[]) == []
        assert pipeline.active_sessions == []
