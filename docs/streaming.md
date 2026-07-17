# Streaming classification

`StreamingZeroShotClassificationPipeline` incrementally classifies text with a decoder-KV model. It keeps one text-only KV cache per active session and evaluates label sequences without adding those labels to the persistent cache.

## Installation

```bash
pip install gliclass[streaming]
```

## Basic usage

```python
from gliclass.streaming import EveryNTokensStrategy, StreamingZeroShotClassificationPipeline

pipeline = StreamingZeroShotClassificationPipeline(
    model,
    tokenizer,
    device="cuda",
    max_cache_len=1024,
    default_strategy=EveryNTokensStrategy(50),
)

for chunk in text_chunks:
    output = pipeline(
        chunk,
        ["science", "politics", "finance"],
        session_ids="document-1",
        threshold=0.5,
    )[0]
    if output["triggered"]:
        print(output["predictions"])
```

The call returns one dictionary per input text:

```python
{
    "session_id": "document-1",
    "triggered": True,
    "predictions": [{"label": "science", "score": 0.91}],
    "cached_length": 128,
    "tokens_added": 17,
}
```

## Batched sessions

```python
outputs = pipeline(
    texts=[chunk_a, chunk_b],
    labels=["positive", "negative"],
    session_ids=["session-a", "session-b"],
    strategies=[strategy_a, strategy_b],
    batch_size=8,
)
```

`texts`, `session_ids`, and a per-session `strategies` list must have matching lengths. Labels may be shared, specified per input, or provided as a hierarchical dictionary like the regular zero-shot pipeline. A single strategy is copied independently for every new session; a strategy list assigns one strategy to each new session.

Session IDs must be unique within a call.

## Session lifecycle

The session IDs in each top-level call are the complete active set. A cached session is deleted automatically when it is absent from the next call.

```python
pipeline([chunk_a, chunk_b], labels, session_ids=["a", "b"])
pipeline([next_b, chunk_c], labels, session_ids=["b", "c"])

# Session "a" has been deleted; "b" was extended; "c" was created.
assert pipeline.active_sessions == ["b", "c"]
```

Cleanup is performed once for the complete call, before internal `batch_size` splitting. Therefore, sessions are not accidentally removed merely because they belong to different internal sub-batches.

To retain a session without adding text, include it with an empty string:

```python
pipeline(
    texts=["", next_chunk_b],
    labels=labels,
    session_ids=["a", "b"],
)
```

Session `a` remains active, reports `tokens_added=0`, and is not classified. Omitting `a` would delete it.

Manual lifecycle methods are also available:

```python
pipeline.delete_session("b")
pipeline.clear_sessions()
print(pipeline.active_sessions)
```

Calling the pipeline with empty `texts` and `session_ids` clears all sessions:

```python
pipeline([], [], session_ids=[])
```

## Prompt and examples

Prompt and few-shot examples become a one-time prefix when a session is created:

```python
pipeline(
    first_chunk,
    labels,
    session_ids="document-1",
    prompt="Classify this document: ",
    examples=examples,
)
```

The prefix is not appended again on later calls. Repeating the same prompt/examples is allowed. Once a session contains cached tokens, changing its prompt or examples raises an error; delete the session first when a different prefix is required.

If a new session has an empty text but a non-empty prompt or examples, the prefix is still cached and counted in `tokens_added`. Prefix-only updates never trigger classification.

## Classification strategies

Available strategies:

- `EveryChunkStrategy`: classify every non-empty update.
- `EveryNTokensStrategy(n)`: classify after accumulating `n` tokens.
- `OnDelimiterStrategy(delimiter)`: classify when the incoming text contains the delimiter.
- `NeverStrategy`: update the cache without classifying.
- `SlidingWindowStrategy(window_size)`: classify every chunk using only the newest cached tokens.
- `ComposedStrategy(trigger, window)`: combine a trigger with a classification window.

Each session owns an independent copy of its strategy. Stateful counters are never shared across sessions, even when one strategy is supplied as the pipeline default. `default_strategy=None` selects `EveryChunkStrategy`.

The `strategies` call argument initializes new sessions. Passing another strategy for an already-active session does not reset its state. Replace an active strategy explicitly with:

```python
pipeline.set_session_strategy("document-1", EveryChunkStrategy())
```

## Two-stage execution

Each call has two stages:

1. New text is passed to `model.update_decoder_cache(...)` with `use_cache=True`. The resulting text-only cache replaces the cache for that session.
2. Triggered sessions pass `<<SEP>>label<<LABEL>>...<<SEP>>` to `model.classify_from_decoder_cache(...)`. This stage uses the text cache as context but does not save label KV tensors.

Existing sessions with empty text receive `tokens_added=0` and are not classified. A new prefix-only session may add prompt/example tokens, but it is also not classified until a non-empty text chunk arrives.

## Heterogeneous batching

Sessions may have different cache lengths. Before a decoder call, shorter caches are padded on the left to the longest cache in the batch. The attention mask hides padding, and explicit position IDs preserve each session's logical positions. After updating, the batched KV tensors are copied back into independent per-session caches.

## Cache truncation

Set `max_cache_len` to retain only the newest KV entries:

```python
pipeline = StreamingZeroShotClassificationPipeline(
    model,
    tokenizer,
    max_cache_len=512,
)
```

The physical cache length is capped, while the next absolute position ID continues increasing. This avoids reusing RoPE positions after truncation. Cropping currently requires the Transformers `DynamicCache` format used by supported decoder-KV models.

## CPU offload

```python
pipeline = StreamingZeroShotClassificationPipeline(
    model,
    tokenizer,
    device="cuda",
    offload_to_cpu=True,
    use_pinned_memory=True,
    score_on_cpu=True,
)
```

With CPU offload enabled, caches are loaded for the active internal batch and moved back to CPU afterward. Pinned memory enables non-blocking GPU transfers when CUDA is used.

## Constructor

```python
StreamingZeroShotClassificationPipeline(
    model,
    tokenizer,
    max_classes=25,
    max_length=1024,
    classification_type="multi-label",
    device="cuda:0",
    progress_bar=False,
    label_separator=".",
    max_cache_len=None,
    default_batch_size=8,
    default_strategy=None,
    offload_to_cpu=False,
    use_pinned_memory=False,
    score_on_cpu=False,
)
```

The model must use `architecture_type="decoder-kv"`.

| Parameter | Behavior |
|---|---|
| `model` | A loaded GLiClass decoder-KV model. The pipeline moves it to `device` and enables evaluation mode. |
| `tokenizer` | Matching tokenizer instance or pretrained tokenizer name. |
| `max_classes` | Inherited compatibility setting; decoder-KV streaming does not currently split labels using it. |
| `max_length` | Inherited pipeline configuration value. Streaming cache length is controlled separately by `max_cache_len`. |
| `classification_type` | Default `"multi-label"` or `"single-label"` behavior. |
| `device` | Decoder execution device. Falls back to CPU when CUDA is requested but unavailable. |
| `progress_bar` | Inherited pipeline option; streaming calls do not display a progress bar. |
| `label_separator` | Separator used to flatten and rebuild hierarchical labels. |
| `max_cache_len` | Maximum physically retained tokens per session; `None` is unbounded. Must be positive when provided. |
| `default_batch_size` | Internal session batch size when `batch_size` is omitted from a call. Must be positive. |
| `default_strategy` | Strategy copied into each new session; `None` means `EveryChunkStrategy`. |
| `offload_to_cpu` | Move each processed internal batch's caches back to CPU. |
| `use_pinned_memory` | Use pinned CPU cache memory for CUDA transfers; effective only with CUDA offload. |
| `score_on_cpu` | Move the decoder-KV scorer and its inputs to CPU. |

## Call interface

```python
pipeline(
    texts,
    labels,
    *,
    session_ids,
    threshold=0.5,
    batch_size=None,
    classification_type=None,
    strategies=None,
    examples=None,
    prompt=None,
    return_hierarchical=False,
)
```

Multi-label predictions are filtered by `threshold`. Single-label classification applies softmax and returns the highest-scoring label. `return_hierarchical=True` reconstructs the same hierarchical output format as the regular zero-shot pipeline.

| Argument | Accepted values |
|---|---|
| `texts` | One chunk string or a list of chunk strings. |
| `labels` | Shared label list, per-text label lists, one hierarchical dictionary, or per-text hierarchical dictionaries. At least one label is required whenever the call contains sessions, including sessions with empty chunks. |
| `session_ids` | One non-empty ID or a list matching `texts`. IDs must be unique within the call. |
| `threshold` | One multi-label threshold or a list matching `texts`. |
| `batch_size` | Positive internal batch size override; `None` uses `default_batch_size`. |
| `classification_type` | Default, one supported mode, or one mode per text. Aliases such as `"single"` and `"multi_label"` are normalized. |
| `strategies` | Default, one strategy copied for new sessions, or one strategy per input. Use `set_session_strategy()` for active sessions. |
| `examples` | Shared examples or one examples list per input. Applied once when a session cache is empty. |
| `prompt` | Shared prompt or one prompt per input. Applied once when a session cache is empty. |
| `return_hierarchical` | Rebuild prediction scores into the supplied hierarchical label structure. |

## Return values

The pipeline always returns a list in input order. Each item contains:

| Field | Meaning |
|---|---|
| `session_id` | Session associated with this result. |
| `triggered` | Whether classification ran in this call. |
| `predictions` | Flat prediction list, hierarchical score dictionary, or `None` when not triggered. |
| `cached_length` | Physical text-cache length after update and optional truncation. |
| `tokens_added` | Number of context tokens added in this call, including a new session's one-time prefix. |

For multi-label classification, a triggered call may legitimately return an empty prediction list when every score is below `threshold`.

## Session methods

- `active_sessions` returns active IDs in insertion order.
- `delete_session(*session_ids)` removes cache and strategy state and returns the IDs that actually existed.
- `clear_sessions()` removes all cache and strategy state.
- `set_session_strategy(session_id, strategy)` deep-copies a new strategy into an active session and raises `KeyError` for an unknown ID.
