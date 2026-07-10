# GLiClass Streaming Classification

`gliclass.streaming` provides incremental, multi-session text classification over a decoder-KV model. Instead of classifying a complete document at once, it processes text in arbitrary chunks, maintains a persistent KV cache per session, and triggers classification only when a pluggable strategy decides it is appropriate.

---

## Contents

- [Architecture overview](#architecture-overview)
- [KV cache management](#kv-cache-management)
- [Session lifecycle](#session-lifecycle)
- [Classification strategies](#classification-strategies)
- [StreamingPipeline reference](#streamingpipeline-reference)
- [Data types](#data-types)
- [Usage examples](#usage-examples)
- [Memory and performance considerations](#memory-and-performance-considerations)

---

## Architecture overview

GLiClass Decoder-KV separates the model into two components:

- **`decoder_model`** — a causal language model (e.g. Qwen3.5) that encodes text into a sequence of hidden states and accumulates a KV cache.
- **`scorer`** — a lightweight cross-attention encoder that reads the label hidden states produced by the decoder and outputs per-label logits.

In standard (non-streaming) inference, both components run on the full text each time. In streaming mode, the decoder runs once per chunk to *extend* the KV cache; the scorer runs only when a strategy decides classification should fire. The result is:

```
chunk_1 → decoder → KV₁
chunk_2 → decoder (KV₁ as prefix) → KV₂
chunk_3 → decoder (KV₂ as prefix) → KV₃  ← strategy triggers → scorer → predictions
chunk_4 → ...
```

Each forward pass of the decoder is O(n_chunk × n_cache) rather than O(n_total²), because the attention over cached tokens is handled by the KV cache mechanism and never recomputed.

### Two-stage per call

Every `StreamingPipeline.__call__` runs two sequential stages over the input batch:

**Stage 1 — cache update.**  
New text is tokenized per session, then all sessions in the batch are forwarded through `decoder_model` together. The resulting `past_key_values` are sliced per session and stored in `_caches`.

**Stage 2 — classification.**  
Sessions whose strategy returns `True` from `should_classify` are forwarded again through `decoder_model` with their KV cache as prefix and the label token sequence as input. The decoder's `last_hidden_state` over label positions is then passed to `scorer`, which produces logits. Sigmoid (multi-label) or softmax (single-label) is applied to produce probabilities.

Sessions that did not trigger classification return `triggered=False` with `predictions=None`.

---

## KV cache management

### CacheState

`CacheState` is the per-session container that travels through the pipeline:

```python
@dataclass
class CacheState:
    past_key_values: Any          # transformers DynamicCache or None
    input_ids: torch.Tensor       # flat 1-D tensor of all cached token ids
    attention_mask: torch.Tensor  # flat 1-D tensor of all cached attention mask values
    position_ids: torch.Tensor | None
    cached_length: int            # total number of cached tokens
    session_id: str | None
    metadata: dict                # arbitrary user data
```

A fresh empty cache is created via `create_empty_cache(session_id, device)`.

### Batched stacking with heterogeneous cache lengths

When multiple sessions exist in a batch and have different numbers of cached tokens, their KV caches cannot be naively stacked: different sequence lengths mean different position embeddings baked into the K/V tensors.

`BatchedKVHelper` resolves this with **prepend-padding**:

- KV caches shorter than `max_cached_len` are zero-padded on the **left** (prepend), not the right.
- The attention mask marks these prepended zeros as not-attended, so the model ignores them.
- `position_ids` for the new tokens in each session are set to `[cached_len_i, ..., cached_len_i + new_len_i)`, so RoPE offsets remain correct regardless of padding.

This means a batch of sessions with cache lengths `[150, 400, 300]` is padded to `max_cached=400`, with sessions 0 and 2 receiving 250 and 100 prepend zeros respectively.

After the forward pass, the updated KV is sliced back per session using the real token range `[max_cached - clen_i : max_cached + new_len_i]`.

### Cache truncation

`truncate_cache(cache_state, max_length)` crops the KV to the **last** `max_length` tokens. Only `DynamicCache` is supported for cropping; other cache formats are returned unchanged. The pipeline applies this automatically after each stage-1 update when `max_cache_len` is set.

Truncation discards the oldest context first. This is a simple sliding-window policy; for more selective eviction (e.g. keeping the first few tokens as a persistent prefix), extend or replace the default behavior by subclassing `StreamingPipeline._enforce_max_cache_len`.

### Device offloading

When running more sessions than fit in GPU memory simultaneously, `offload_to_cpu=True` keeps inactive session caches in CPU RAM and moves them to GPU only for the active sub-batch.

With `use_pinned_memory=True`, KV caches are allocated in CPU pinned (page-locked) memory, enabling asynchronous DMA transfers to the GPU via `non_blocking=True`. A `torch.cuda.synchronize` call before the forward pass ensures transfers are complete before the kernel launches.

---

## Session lifecycle

Sessions are identified by a string `session_id`. The pipeline maintains a dict `_caches: dict[str, CacheState]` internally.

**Creation.** A session is created implicitly when a `session_id` appears for the first time in an input batch. `create_empty_cache` is called and stored.

**Updating.** On every call, the session's cache is extended with new tokens if `text` is non-empty. Empty text causes the update stage to be skipped for that session (`tokens_added = 0`), which also skips classification regardless of strategy.

**Cleanup.** Sessions that are absent from the current input batch are deleted. The pipeline does not keep stale sessions alive across calls: any `session_id` not present in the current `inputs` list will have its cache freed. If a session should remain alive without receiving new text, pass it with `text=""`.

**Manual reset.** Delete the cache entry directly:
```python
del pipeline._caches["my_session"]
```

---

## Classification strategies

Strategies implement `ClassificationStrategy` and control:

1. **`should_classify(tokens_added, cached_length, text) -> bool`** — whether to trigger classification this call.
2. **`get_window(cache_state) -> CacheState`** — which portion of the cache to classify over (default: full cache).

### Built-in strategies

#### `EveryChunkStrategy`

Classifies on every call that adds at least one token. Use when latency is acceptable and you want the most up-to-date prediction at every step.

```python
EveryChunkStrategy()
```

#### `EveryNTokensStrategy`

Accumulates tokens across calls and classifies once every `n` tokens. The counter resets after each trigger. Useful for throttling classification frequency over a fast-arriving stream.

```python
EveryNTokensStrategy(n=100)
```

Note: the internal `_accumulated` counter is per-instance. If the same strategy object is shared across sessions, the counter is shared too. Create one instance per session or use `ComposedStrategy` with per-session instantiation.

#### `OnDelimiterStrategy`

Classifies when the incoming text chunk contains a specific string, such as a sentence boundary or a paragraph marker.

```python
OnDelimiterStrategy(delimiter="\n\n")   # paragraph boundary
OnDelimiterStrategy(delimiter=". ")    # sentence boundary (approximate)
```

The delimiter check is a plain substring search on the raw text string, not on tokens.

#### `NeverStrategy`

Never triggers classification. Use this for a pre-filling phase where you want to build up a long context cache before any classification runs.

```python
NeverStrategy()
```

Switch to a different strategy object after pre-filling is complete.

#### `SlidingWindowStrategy`

Classifies on every chunk (same trigger condition as `EveryChunkStrategy`), but passes only the last `window_size` cached tokens to the scorer via `get_window`. This limits the attention span of the classification pass without discarding the full KV cache.

```python
SlidingWindowStrategy(window_size=512)
```

Note that this does not truncate the stored KV cache — the full cache is retained for future updates. Only the classification forward sees the windowed slice.

#### `ComposedStrategy`

Combines a **trigger strategy** (controls `should_classify`) with a **window strategy** (controls `get_window`). This is the standard way to pair a frequency policy with a context window policy.

```python
ComposedStrategy(
    trigger=EveryNTokensStrategy(200),
    window=SlidingWindowStrategy(512),
)
```

Classifies every 200 new tokens, but uses only the last 512 cached tokens for scoring.

### Custom strategies

Subclass `ClassificationStrategy` and implement `should_classify`. Optionally override `get_window` for windowed classification.

```python
class SentenceCountStrategy(ClassificationStrategy):
    def __init__(self, every_n_sentences: int):
        self.n = every_n_sentences
        self._count = 0

    def should_classify(self, tokens_added, cached_length, text):
        self._count += text.count(". ")
        if self._count >= self.n:
            self._count = 0
            return True
        return False
```

---

## StreamingPipeline reference

```python
StreamingPipeline(
    model,
    tokenizer,
    device="cpu",
    max_cache_len=None,
    batch_size=None,
    offload_to_cpu=False,
    use_pinned_memory=False,
    score_on_cpu=False,
)
```

| Parameter | Type | Description |
|---|---|---|
| `model` | `GLiClassModel` | Loaded GLiClass decoder-KV model in eval mode. |
| `tokenizer` | `PreTrainedTokenizer` | Matching tokenizer with `<<LABEL>>`, `<<SEP>>`, `<<EXAMPLE>>` special tokens. |
| `device` | `str \| torch.device` | Device for decoder forward passes. |
| `max_cache_len` | `int \| None` | Hard cap on cached tokens per session. Oldest tokens are dropped when exceeded. `None` means unbounded. |
| `batch_size` | `int \| None` | Maximum sessions per decoder forward. When `len(inputs) > batch_size`, sessions are chunked. With `offload_to_cpu=True`, only the active chunk is on GPU at once. |
| `offload_to_cpu` | `bool` | Move inactive session caches to CPU RAM between sub-batches. |
| `use_pinned_memory` | `bool` | Allocate CPU caches in pinned memory for async GPU uploads. Only effective when `offload_to_cpu=True` and `device` is CUDA. |
| `score_on_cpu` | `bool` | Run the scorer on CPU. Useful when GPU VRAM is the bottleneck and the scorer is small relative to the decoder. |

### `__call__(inputs: list[SessionInput]) -> list[SessionOutput]`

Processes a batch of sessions. Returns one `SessionOutput` per input in the same order. Sessions not present in the current batch have their caches deleted.

---

## Data types

### `SessionInput`

```python
@dataclass
class SessionInput:
    session_id: str                    # unique identifier for this session
    text: str                          # new text chunk to append (empty = no update)
    labels: list[str]                  # candidate label strings
    strategy: ClassificationStrategy   # per-session strategy instance
    classification_type: str           # "multi-label" (sigmoid) or "single-label" (softmax)
```

### `SessionOutput`

```python
@dataclass
class SessionOutput:
    session_id: str
    triggered: bool                    # True if classification ran this call
    predictions: list[dict] | None     # [{"label": str, "score": float}, ...] or None
    cached_length: int                 # total cached tokens after this call
    tokens_added: int                  # tokens added in stage 1 (0 if text was empty)
    metadata: dict                     # pass-through from CacheState.metadata
```

For `multi-label`, `predictions` contains one entry per label with a sigmoid probability. For `single-label`, it contains only the top-1 label with its softmax probability.

---

## Usage examples

### Minimal single-session loop

```python
import torch
from transformers import AutoTokenizer
from gliclass import GLiClassModel
from gliclass.streaming import StreamingPipeline, SessionInput, EveryNTokensStrategy

tokenizer = AutoTokenizer.from_pretrained("path/to/model")
model = GLiClassModel.from_pretrained("path/to/model")
model = model.to("cuda", dtype=torch.bfloat16).eval()

pipeline = StreamingPipeline(model, tokenizer, device="cuda", max_cache_len=1024)

labels = ["science", "politics", "finance", "sports"]
strategy = EveryNTokensStrategy(n=50)

chunks = ["The Fed raised interest rates", " by 25 basis points", " amid inflation concerns."]

for chunk in chunks:
    outputs = pipeline([
        SessionInput(
            session_id="doc_001",
            text=chunk,
            labels=labels,
            strategy=strategy,
            classification_type="multi-label",
        )
    ])
    out = outputs[0]
    print(f"cached={out.cached_length} +{out.tokens_added} triggered={out.triggered}")
    if out.triggered:
        for p in sorted(out.predictions, key=lambda x: -x["score"]):
            print(f"  {p['label']}: {p['score']:.3f}")
```

### Multiple concurrent sessions

```python
session_strategies = {
    "session_A": EveryNTokensStrategy(n=100),
    "session_B": OnDelimiterStrategy(delimiter="\n"),
    "session_C": ComposedStrategy(
        trigger=EveryNTokensStrategy(50),
        window=SlidingWindowStrategy(256),
    ),
}

batch = [
    SessionInput("session_A", chunk_a, labels, session_strategies["session_A"]),
    SessionInput("session_B", chunk_b, labels, session_strategies["session_B"]),
    SessionInput("session_C", chunk_c, labels, session_strategies["session_C"]),
]

outputs = pipeline(batch)
```

All three sessions are updated and classified (if triggered) in a single batched forward pass per stage.

### Pre-filling a long context before classification

```python
from gliclass.streaming import NeverStrategy, EveryChunkStrategy

# Phase 1: ingest background context without any classification overhead
for paragraph in background_paragraphs:
    pipeline([SessionInput("doc", paragraph, labels, NeverStrategy())])

# Phase 2: stream the main document and classify on every chunk
for chunk in main_document_chunks:
    outputs = pipeline([SessionInput("doc", chunk, labels, EveryChunkStrategy())])
    if outputs[0].triggered:
        handle(outputs[0].predictions)
```

### Memory-constrained deployment with CPU offload

```python
pipeline = StreamingPipeline(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    max_cache_len=512,
    batch_size=8,           # process 8 sessions per GPU forward
    offload_to_cpu=True,    # keep inactive caches in RAM
    use_pinned_memory=True, # async DMA transfers
    score_on_cpu=True,      # scorer stays in CPU RAM
)
```

With 32 concurrent sessions and `batch_size=8`, the GPU holds at most 8 KV caches at a time. Inactive caches are evicted to CPU pinned memory and uploaded asynchronously before their sub-batch runs.

---