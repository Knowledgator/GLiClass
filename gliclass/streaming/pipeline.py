"""
StreamingPipeline – batched multi-session streaming classification.

Flow per __call__:
  1. Cleanup sessions absent from current batch.
  2. Create CacheState for new session_ids.
  3. Stage 1 – batch update KV caches with new text (skip empty text).
  4. Stage 2 – resolve strategies; batch classify sessions that triggered.
"""

from __future__ import annotations

import torch

from .cache import BatchedKVHelper, CacheState, create_empty_cache, truncate_cache
from .types import SessionInput, SessionOutput


class StreamingPipeline:
    def __init__(
        self,
        model,
        tokenizer,
        device: str | torch.device = "cpu",
        max_cache_len: int | None = None,
        batch_size: int | None = None,
        offload_to_cpu: bool = False,
        use_pinned_memory: bool = False,
        score_on_cpu: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_cache_len = max_cache_len
        self.batch_size = batch_size
        self.offload_to_cpu = offload_to_cpu
        self.use_pinned_memory = use_pinned_memory and offload_to_cpu and self.device.type == "cuda"
        self.score_on_cpu = score_on_cpu
        self._caches: dict[str, CacheState] = {}

        scorer_target = "cpu" if score_on_cpu else self.device
        self.model.model.scorer.to(scorer_target)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, inputs: list[SessionInput]) -> list[SessionOutput]:
        self._cleanup_dead_sessions(inputs)
        self._ensure_sessions(inputs)

        if self.batch_size is None or len(inputs) <= self.batch_size:
            return self._process_batch(inputs)

        outputs = [None] * len(inputs)
        for start in range(0, len(inputs), self.batch_size):
            chunk = inputs[start : start + self.batch_size]
            chunk_ids = {inp.session_id for inp in chunk}

            if self.offload_to_cpu:
                self._load_to_device(chunk_ids, self.device)

            for i, out in enumerate(self._process_batch(chunk)):
                outputs[start + i] = out

            if self.offload_to_cpu:
                self._load_to_device(chunk_ids, torch.device("cpu"))

        return outputs

    def _process_batch(self, inputs: list[SessionInput]) -> list[SessionOutput]:
        tokens_added = self._stage_update(inputs)
        if self.max_cache_len is not None:
            self._enforce_max_cache_len()
        return self._stage_classify(inputs, tokens_added)

    def _load_to_device(self, session_ids: set[str], device: torch.device) -> None:
        to_cpu = device.type == "cpu"
        for sid in session_ids:
            cache = self._caches.get(sid)
            if cache is None or cache.past_key_values is None:
                continue
            if to_cpu and self.use_pinned_memory:
                self._caches[sid] = _move_cache_pinned(cache)
            else:
                self._caches[sid] = _move_cache_nonblocking(cache, device)

        # synchronize before using on GPU so non_blocking transfers complete
        if not to_cpu and self.use_pinned_memory and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _cleanup_dead_sessions(self, inputs: list[SessionInput]) -> None:
        active = {inp.session_id for inp in inputs}
        for sid in list(self._caches):
            if sid not in active:
                del self._caches[sid]

    def _enforce_max_cache_len(self) -> None:
        for sid, cache in self._caches.items():
            if cache.cached_length > self.max_cache_len:
                self._caches[sid] = truncate_cache(cache, self.max_cache_len)

    def _ensure_sessions(self, inputs: list[SessionInput]) -> None:
        for inp in inputs:
            if inp.session_id not in self._caches:
                self._caches[inp.session_id] = create_empty_cache(
                    session_id=inp.session_id, device=self.device
                )

    # ------------------------------------------------------------------
    # Stage 1: update KV caches
    # ------------------------------------------------------------------

    def _stage_update(self, inputs: list[SessionInput]) -> list[int]:
        """
        Update KV cache for each session. Returns tokens_added per input.
        Sessions with empty text are skipped (tokens_added = 0).
        """
        tokens_added = [0] * len(inputs)
        to_update = [(i, inp) for i, inp in enumerate(inputs) if inp.text]

        if not to_update:
            return tokens_added

        indices = [i for i, _ in to_update]
        active_inputs = [inp for _, inp in to_update]

        # Tokenize new text per session
        tokenized = [
            self.tokenizer(inp.text, return_tensors="pt", add_special_tokens=False)
            for inp in active_inputs
        ]
        new_ids = [t["input_ids"].to(self.device) for t in tokenized]
        new_masks = [t["attention_mask"].to(self.device) for t in tokenized]

        caches = [self._caches[inp.session_id] for inp in active_inputs]

        # Check if any session has a non-empty cache (need KV stacking)
        has_cache = any(c.cached_length > 0 for c in caches)

        if has_cache:
            stacked = BatchedKVHelper.stack_for_update(caches, new_ids, new_masks, self.device)
            decoder = self.model.model.decoder_model

            with torch.no_grad():
                out = decoder(
                    input_ids=stacked["input_ids"],
                    attention_mask=stacked["attention_mask"],
                    position_ids=stacked["position_ids"],
                    past_key_values=stacked["past_key_values"],
                    use_cache=True,
                    return_dict=True,
                )

            updated = BatchedKVHelper.unstack_after_update(
                out.past_key_values, stacked, caches, new_ids, new_masks
            )
        else:
            # All caches empty – simple batched forward without stacking
            updated = self._update_from_scratch(caches, new_ids, new_masks)

        for session_idx, cache, new_len in zip(
            indices, updated, stacked["new_lengths"] if has_cache else [ids.shape[-1] for ids in new_ids]
        ):
            inp = inputs[session_idx]
            self._caches[inp.session_id] = cache
            tokens_added[session_idx] = new_len

        return tokens_added

    def _update_from_scratch(
        self,
        caches: list[CacheState],
        new_ids: list[torch.Tensor],
        new_masks: list[torch.Tensor],
    ) -> list[CacheState]:
        """Batched update when all caches are empty (no KV stacking needed)."""
        max_len = max(ids.shape[-1] for ids in new_ids)
        batch_size = len(caches)

        padded_ids = new_ids[0].new_zeros(batch_size, max_len)
        padded_masks = new_ids[0].new_zeros(batch_size, max_len)
        position_ids = new_ids[0].new_zeros(batch_size, max_len)

        new_lengths = []
        for i, (ids, mask) in enumerate(zip(new_ids, new_masks)):
            flat = ids[0] if ids.dim() == 2 else ids
            flat_m = mask[0] if mask.dim() == 2 else mask
            nlen = flat.shape[0]
            padded_ids[i, :nlen] = flat
            padded_masks[i, :nlen] = flat_m
            position_ids[i, :nlen] = torch.arange(nlen, device=self.device)
            new_lengths.append(nlen)

        decoder = self.model.model.decoder_model
        with torch.no_grad():
            out = decoder(
                input_ids=padded_ids,
                attention_mask=padded_masks,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )

        results = []
        for i, (cache, nlen) in enumerate(zip(caches, new_lengths)):
            sliced_kv = BatchedKVHelper._slice_past_kv(out.past_key_values, i, 0, nlen)
            flat_ids = new_ids[i][0] if new_ids[i].dim() == 2 else new_ids[i]
            flat_mask = new_masks[i][0] if new_masks[i].dim() == 2 else new_masks[i]
            results.append(CacheState(
                past_key_values=sliced_kv,
                input_ids=flat_ids[:nlen],
                attention_mask=flat_mask[:nlen],
                cached_length=nlen,
                session_id=cache.session_id,
                metadata=cache.metadata.copy(),
            ))

        return results

    # ------------------------------------------------------------------
    # Stage 2: classify triggered sessions
    # ------------------------------------------------------------------

    def _stage_classify(
        self, inputs: list[SessionInput], tokens_added: list[int]
    ) -> list[SessionOutput]:
        # Resolve which sessions trigger classification
        triggered_indices = []
        for i, (inp, n_added) in enumerate(zip(inputs, tokens_added)):
            if n_added == 0:
                continue  # no new tokens → cache unchanged, skip classification
            cache = self._caches[inp.session_id]
            if inp.strategy.should_classify(n_added, cache.cached_length, inp.text):
                triggered_indices.append(i)

        outputs: list[SessionOutput | None] = [None] * len(inputs)

        # Fill non-triggered outputs
        for i, inp in enumerate(inputs):
            if i not in triggered_indices:
                cache = self._caches[inp.session_id]
                outputs[i] = SessionOutput(
                    session_id=inp.session_id,
                    triggered=False,
                    predictions=None,
                    cached_length=cache.cached_length,
                    tokens_added=tokens_added[i],
                )

        if not triggered_indices:
            return outputs

        # Prepare label sequences for triggered sessions
        triggered_inputs = [inputs[i] for i in triggered_indices]
        triggered_caches = [
            inputs[i].strategy.get_window(self._caches[inputs[i].session_id])
            for i in triggered_indices
        ]

        label_seqs = [
            self._prepare_label_seq(inp.labels) for inp in triggered_inputs
        ]
        tokenized_labels = [
            self.tokenizer(seq, return_tensors="pt", add_special_tokens=False)
            for seq in label_seqs
        ]
        label_ids = [t["input_ids"].to(self.device) for t in tokenized_labels]
        label_masks = [t["attention_mask"].to(self.device) for t in tokenized_labels]

        stacked = BatchedKVHelper.stack_for_classify(
            triggered_caches, label_ids, label_masks, self.device
        )

        decoder = self.model.model.decoder_model
        with torch.no_grad():
            out = decoder(
                input_ids=stacked["input_ids"],
                attention_mask=stacked["attention_mask"],
                position_ids=stacked["position_ids"],
                past_key_values=stacked["past_key_values"],
                use_cache=False,
                return_dict=True,
            )

        # Slice label hidden states and pass to scorer
        hidden = out.last_hidden_state          # [batch, max_cached + max_label, hidden]
        max_cached = stacked["past_key_values"] and max(c.cached_length for c in triggered_caches) or 0
        label_lengths = stacked["label_lengths"]
        max_label = stacked["max_label_len"]

        # hidden for labels starts at index max_cached (prepend-padded KV)
        # but we need the actual label slice per session
        scorer = self.model.model.scorer
        batch_logits = self._run_scorer_batched(
            scorer, hidden, stacked["input_ids"], stacked["label_mask"],
            label_lengths, max_label,
        )

        for local_idx, global_idx in enumerate(triggered_indices):
            inp = inputs[global_idx]
            cache = self._caches[inp.session_id]
            logits = batch_logits[local_idx]               # [num_labels]
            preds = self._decode_predictions(logits, inp.labels, inp.classification_type)
            outputs[global_idx] = SessionOutput(
                session_id=inp.session_id,
                triggered=True,
                predictions=preds,
                cached_length=cache.cached_length,
                tokens_added=tokens_added[global_idx],
            )

        return outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_label_seq(self, labels: list[str]) -> str:
        label_token = "<<LABEL>>"
        sep_token = "<<SEP>>"
        # Opening <<SEP>> matches training format: text<<SEP>>label1<<LABEL>>...<<SEP>>
        # Cache contains only text; <<SEP>> boundary must be part of label sequence
        parts = [sep_token]
        for label in labels:
            parts.append(label)
            parts.append(label_token)
        parts.append(sep_token)
        return "".join(parts)

    def _run_scorer_batched(
        self,
        scorer,
        hidden: torch.Tensor,
        label_ids: torch.Tensor,
        label_mask: torch.Tensor,
        label_lengths: list[int],
        max_label: int,
    ) -> list[torch.Tensor]:
        """
        hidden: [batch, max_cached + max_label, hidden_size]  (full decoder output)
        We slice out only the label part [max_cached :] per session before scorer.
        label_ids / label_mask are already label-only ([batch, max_label]).
        """
        max_cached_in_hidden = hidden.shape[1] - max_label
        label_hidden = hidden[:, max_cached_in_hidden:, :]   # [batch, max_label, hidden]

        if self.score_on_cpu:
            label_hidden = label_hidden.cpu()
            label_ids = label_ids.cpu()
            label_mask = label_mask.cpu()

        logits = scorer(
            hidden_states=label_hidden,
            input_ids=label_ids,
            attention_mask=label_mask,
        )  # [batch, num_labels]

        return [logits[i] for i in range(logits.shape[0])]

    def _decode_predictions(
        self, logits: torch.Tensor, labels: list[str], classification_type: str
    ) -> list[dict]:
        logits = logits[:len(labels)]
        if classification_type == "single-label":
            scores = torch.softmax(logits, dim=-1)
            best = int(scores.argmax().item())
            return [{"label": labels[best], "score": float(scores[best].detach())}]
        else:
            probs = torch.sigmoid(logits)
            return [
                {"label": label, "score": float(prob.detach())}
                for label, prob in zip(labels, probs)
            ]


def _move_cache_pinned(cache: "CacheState") -> "CacheState":
    """Move KV cache to CPU pinned memory for fast async GPU uploads."""
    from transformers.cache_utils import DynamicCache, DynamicLayer
    from .cache import CacheState

    past_kv = cache.past_key_values
    if not isinstance(past_kv, DynamicCache):
        return cache.to(torch.device("cpu"))

    new_kv = DynamicCache()
    for i, layer in enumerate(past_kv.layers):
        if not layer.is_initialized:
            continue
        pinned_k = layer.keys.cpu().pin_memory()
        pinned_v = layer.values.cpu().pin_memory()
        # bypass update() to avoid torch.cat losing pinned property
        while len(new_kv.layers) <= i:
            new_kv.layers.append(DynamicLayer())
        new_layer = new_kv.layers[i]
        new_layer.dtype = pinned_k.dtype
        new_layer.device = pinned_k.device
        new_layer.keys = pinned_k
        new_layer.values = pinned_v
        new_layer.is_initialized = True

    return CacheState(
        past_key_values=new_kv,
        input_ids=cache.input_ids.cpu(),
        attention_mask=cache.attention_mask.cpu(),
        position_ids=cache.position_ids.cpu() if cache.position_ids is not None else None,
        cached_length=cache.cached_length,
        session_id=cache.session_id,
        metadata=cache.metadata.copy(),
    )


def _move_cache_nonblocking(cache: "CacheState", device: torch.device) -> "CacheState":
    """Move KV cache to device with non_blocking=True (async when src is pinned)."""
    from transformers.cache_utils import DynamicCache
    from .cache import CacheState

    past_kv = cache.past_key_values
    if not isinstance(past_kv, DynamicCache):
        return cache.to(device)

    new_kv = DynamicCache()
    for i, layer in enumerate(past_kv.layers):
        if layer.is_initialized:
            new_kv.update(
                layer.keys.to(device, non_blocking=True),
                layer.values.to(device, non_blocking=True),
                i,
            )

    return CacheState(
        past_key_values=new_kv,
        input_ids=cache.input_ids.to(device, non_blocking=True),
        attention_mask=cache.attention_mask.to(device, non_blocking=True),
        position_ids=cache.position_ids.to(device, non_blocking=True) if cache.position_ids is not None else None,
        cached_length=cache.cached_length,
        session_id=cache.session_id,
        metadata=cache.metadata.copy(),
    )
