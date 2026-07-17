"""Stateful, batched streaming classification for decoder-KV models."""

from __future__ import annotations

import copy
from typing import Any

import torch

from .cache import CacheState, BatchedKVHelper, truncate_cache, create_empty_cache
from ..pipeline import DecoderKVZeroShotClassificationPipeline, build_hierarchical_output
from .strategies import EveryChunkStrategy, ClassificationStrategy
from ..data_processing import format_decoder_kv_labels, format_decoder_kv_context


class StreamingZeroShotClassificationPipeline(DecoderKVZeroShotClassificationPipeline):
    """Incrementally classify multiple decoder-KV sessions with persistent KV caches."""

    def __init__(
        self,
        model,
        tokenizer,
        max_classes: int = 25,
        max_length: int = 1024,
        classification_type: str = "multi-label",
        device: str | torch.device = "cuda:0",
        progress_bar: bool = False,
        label_separator: str = ".",
        max_cache_len: int | None = None,
        default_batch_size: int = 8,
        default_strategy: ClassificationStrategy | None = None,
        offload_to_cpu: bool = False,
        use_pinned_memory: bool = False,
        score_on_cpu: bool = False,
    ):
        if model.config.architecture_type != "decoder-kv":
            raise ValueError("Streaming classification requires architecture_type='decoder-kv'.")

        super().__init__(
            model,
            tokenizer,
            max_classes=max_classes,
            max_length=max_length,
            classification_type=classification_type,
            device=device,
            progress_bar=progress_bar,
            label_separator=label_separator,
        )
        if default_batch_size <= 0:
            raise ValueError("default_batch_size must be greater than zero.")
        if max_cache_len is not None and max_cache_len <= 0:
            raise ValueError("max_cache_len must be greater than zero when provided.")
        if default_strategy is not None and not isinstance(default_strategy, ClassificationStrategy):
            raise TypeError("default_strategy must implement ClassificationStrategy.")

        self.max_cache_len = max_cache_len
        self.default_batch_size = default_batch_size
        self.default_strategy = default_strategy or EveryChunkStrategy()
        self.offload_to_cpu = offload_to_cpu
        self.use_pinned_memory = use_pinned_memory and offload_to_cpu and self.device.type == "cuda"
        self.score_on_cpu = score_on_cpu
        self._caches: dict[str, CacheState] = {}
        self._session_strategies: dict[str, ClassificationStrategy] = {}

        if score_on_cpu:
            self.model.model.scorer.to("cpu")

    @torch.no_grad()
    def __call__(
        self,
        texts: str | list[str],
        labels: list[str] | dict[str, Any] | list[list[str]] | list[dict[str, Any]],
        *,
        session_ids: str | list[str],
        threshold: float | list[float] = 0.5,
        batch_size: int | None = None,
        classification_type: str | list[str] | None = None,
        strategies: ClassificationStrategy | list[ClassificationStrategy] | None = None,
        examples: list[dict[str, Any]] | list[list[dict[str, Any]]] | None = None,
        prompt: str | list[str] | None = None,
        return_hierarchical: bool = False,
    ) -> list[dict[str, Any]]:
        """Append text chunks and classify sessions selected by their strategies."""
        normalized = self._normalize_streaming_inputs(
            texts=texts,
            labels=labels,
            session_ids=session_ids,
            threshold=threshold,
            classification_type=classification_type,
            strategies=strategies,
            examples=examples,
            prompt=prompt,
            return_hierarchical=return_hierarchical,
        )

        batch_size = self.default_batch_size if batch_size is None else batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        active_ids = set(normalized["session_ids"])
        self._cleanup_absent_sessions(active_ids)
        if not active_ids:
            return []

        outputs: list[dict[str, Any] | None] = [None] * len(normalized["texts"])
        for start in range(0, len(normalized["texts"]), batch_size):
            end = start + batch_size
            batch = {key: value[start:end] for key, value in normalized.items()}
            batch_ids = set(batch["session_ids"])
            self._ensure_sessions(batch)

            if self.offload_to_cpu:
                self._load_to_device(batch_ids, self.device)

            batch_outputs = self._process_batch(batch)
            outputs[start:end] = batch_outputs

            if self.offload_to_cpu:
                self._load_to_device(batch_ids, torch.device("cpu"))

        return outputs

    @property
    def active_sessions(self) -> list[str]:
        """Return active session IDs in insertion order."""
        return list(self._caches)

    def delete_session(self, *session_ids: str) -> list[str]:
        """Delete sessions and all cache and strategy state associated with them."""
        deleted = []
        for session_id in session_ids:
            if session_id in self._caches:
                del self._caches[session_id]
                self._session_strategies.pop(session_id, None)
                deleted.append(session_id)
        return deleted

    def clear_sessions(self) -> None:
        """Delete every active session."""
        self._caches.clear()
        self._session_strategies.clear()

    def set_session_strategy(self, session_id: str, strategy: ClassificationStrategy) -> None:
        """Replace the classification strategy for an active session."""
        if session_id not in self._caches:
            raise KeyError(f"Unknown session_id: {session_id}")
        self._session_strategies[session_id] = copy.deepcopy(strategy)

    def _normalize_streaming_inputs(
        self,
        *,
        texts,
        labels,
        session_ids,
        threshold,
        classification_type,
        strategies,
        examples,
        prompt,
        return_hierarchical,
    ) -> dict[str, list]:
        texts = self._normalize_texts(texts)
        if isinstance(session_ids, str):
            session_ids = [session_ids]
        else:
            session_ids = list(session_ids)

        if len(session_ids) != len(texts):
            raise ValueError("session_ids and texts must have the same length.")
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("session_ids must be unique within a streaming call.")
        if any(not isinstance(session_id, str) or not session_id for session_id in session_ids):
            raise ValueError("Every session_id must be a non-empty string.")

        thresholds = self._normalize_thresholds(threshold, len(texts))
        classification_types = self._normalize_classification_types(classification_type, len(texts))
        processed_labels = self._process_labels(labels)
        if texts and not processed_labels:
            raise ValueError("At least one label is required.")

        same_labels = not texts or isinstance(processed_labels[0], str)
        if same_labels:
            labels_per_text = [processed_labels] * len(texts)
            original_labels = [labels] * len(texts)
        else:
            if len(processed_labels) != len(texts):
                raise ValueError("Per-text labels and texts must have the same length.")
            labels_per_text = processed_labels
            original_labels = list(labels)

        if strategies is None:
            strategies_per_text = [None] * len(texts)
        elif isinstance(strategies, ClassificationStrategy):
            strategies_per_text = [strategies] * len(texts)
        else:
            strategies_per_text = list(strategies)
            if len(strategies_per_text) != len(texts):
                raise ValueError("strategies and texts must have the same length.")
            if not all(isinstance(strategy, ClassificationStrategy) for strategy in strategies_per_text):
                raise TypeError("Every strategy must implement ClassificationStrategy.")

        prefixes = []
        for index, session_id in enumerate(session_ids):
            text_prompt = self._format_prompt(prompt, index)
            text_examples = self._get_text_examples(examples, index)
            examples_text = self._format_examples_for_input(text_examples) if text_examples else ""
            prefix = format_decoder_kv_context("", text_prompt, examples_text)
            existing = self._caches.get(session_id)
            if (
                existing is not None
                and existing.cached_length > 0
                and prefix
                and prefix != existing.metadata.get("prefix", "")
            ):
                raise ValueError(
                    f"Prompt or examples changed for active session {session_id!r}; "
                    "delete the session before changing them."
                )
            prefixes.append(prefix)

        return {
            "texts": texts,
            "labels": labels_per_text,
            "original_labels": original_labels,
            "session_ids": session_ids,
            "thresholds": thresholds,
            "classification_types": classification_types,
            "strategies": strategies_per_text,
            "prefixes": prefixes,
            "return_hierarchical": [return_hierarchical] * len(texts),
        }

    def _cleanup_absent_sessions(self, active_ids: set[str]) -> None:
        for session_id in list(self._caches):
            if session_id not in active_ids:
                del self._caches[session_id]
                self._session_strategies.pop(session_id, None)

    def _ensure_sessions(self, batch: dict[str, list]) -> None:
        for session_id, strategy, prefix in zip(
            batch["session_ids"], batch["strategies"], batch["prefixes"], strict=True
        ):
            if session_id in self._caches:
                if self._caches[session_id].cached_length == 0 and prefix:
                    self._caches[session_id].metadata["prefix"] = prefix
                continue
            cache = create_empty_cache(session_id=session_id, device=self.device)
            cache.metadata["prefix"] = prefix
            self._caches[session_id] = cache
            self._session_strategies[session_id] = copy.deepcopy(strategy or self.default_strategy)

    def _process_batch(self, batch: dict[str, list]) -> list[dict[str, Any]]:
        texts = []
        for session_id, text, prefix in zip(batch["session_ids"], batch["texts"], batch["prefixes"], strict=True):
            cache = self._caches[session_id]
            texts.append(f"{prefix}{text}" if cache.cached_length == 0 else text)

        tokens_added = self._update_caches(batch["session_ids"], texts)
        if self.max_cache_len is not None:
            for session_id in batch["session_ids"]:
                cache = self._caches[session_id]
                if cache.cached_length > self.max_cache_len:
                    self._caches[session_id] = truncate_cache(cache, self.max_cache_len)
        return self._classify(batch, tokens_added)

    def _update_caches(self, session_ids: list[str], texts: list[str]) -> list[int]:
        tokens_added = [0] * len(texts)
        active = [
            (index, session_id, text)
            for index, (session_id, text) in enumerate(zip(session_ids, texts, strict=True))
            if text
        ]
        if not active:
            return tokens_added

        indices = [item[0] for item in active]
        active_ids = [item[1] for item in active]
        tokenized = [self.tokenizer(item[2], return_tensors="pt", add_special_tokens=False) for item in active]
        new_ids = [item["input_ids"].to(self.device) for item in tokenized]
        new_masks = [item["attention_mask"].to(self.device) for item in tokenized]
        caches = [self._caches[session_id] for session_id in active_ids]

        if any(cache.cached_length > 0 for cache in caches):
            stacked = BatchedKVHelper.stack_for_update(caches, new_ids, new_masks, self.device)
            model_output = self.model.update_decoder_cache(
                input_ids=stacked["input_ids"],
                attention_mask=stacked["attention_mask"],
                position_ids=stacked["position_ids"],
                past_key_values=stacked["past_key_values"],
            )
            updated = BatchedKVHelper.unstack_after_update(
                model_output.past_key_values,
                stacked,
                caches,
                new_ids,
                new_masks,
            )
            new_lengths = stacked["new_lengths"]
        else:
            updated, new_lengths = self._update_from_scratch(caches, new_ids, new_masks)

        for index, session_id, cache, new_length in zip(indices, active_ids, updated, new_lengths, strict=True):
            self._caches[session_id] = cache
            tokens_added[index] = new_length
        return tokens_added

    def _update_from_scratch(
        self,
        caches: list[CacheState],
        new_ids: list[torch.Tensor],
        new_masks: list[torch.Tensor],
    ) -> tuple[list[CacheState], list[int]]:
        new_lengths = [ids.shape[-1] for ids in new_ids]
        max_length = max(new_lengths)
        batch_size = len(caches)
        padded_ids = new_ids[0].new_zeros(batch_size, max_length)
        padded_masks = new_ids[0].new_zeros(batch_size, max_length)
        position_ids = new_ids[0].new_zeros(batch_size, max_length)

        for index, (ids, mask, cache, length) in enumerate(zip(new_ids, new_masks, caches, new_lengths, strict=True)):
            flat_ids = ids[0] if ids.dim() == 2 else ids
            flat_mask = mask[0] if mask.dim() == 2 else mask
            padded_ids[index, :length] = flat_ids
            padded_masks[index, :length] = flat_mask
            position_ids[index, :length] = torch.arange(
                cache.next_position,
                cache.next_position + length,
                device=self.device,
            )

        model_output = self.model.update_decoder_cache(
            input_ids=padded_ids,
            attention_mask=padded_masks,
            position_ids=position_ids,
            past_key_values=None,
        )

        results = []
        for index, (cache, length) in enumerate(zip(caches, new_lengths, strict=True)):
            flat_ids = new_ids[index][0] if new_ids[index].dim() == 2 else new_ids[index]
            flat_mask = new_masks[index][0] if new_masks[index].dim() == 2 else new_masks[index]
            results.append(
                CacheState(
                    past_key_values=BatchedKVHelper._slice_past_kv(
                        model_output.past_key_values,
                        index,
                        0,
                        length,
                    ),
                    input_ids=flat_ids[:length],
                    attention_mask=flat_mask[:length],
                    cached_length=length,
                    next_position_id=cache.next_position + length,
                    session_id=cache.session_id,
                    metadata=cache.metadata.copy(),
                )
            )
        return results, new_lengths

    def _classify(
        self,
        batch: dict[str, list],
        tokens_added: list[int],
    ) -> list[dict[str, Any]]:
        triggered_indices = []
        for index, (session_id, text, added) in enumerate(
            zip(batch["session_ids"], batch["texts"], tokens_added, strict=True)
        ):
            cache = self._caches[session_id]
            strategy = self._session_strategies[session_id]
            if text and added > 0 and strategy.should_classify(added, cache.cached_length, text):
                triggered_indices.append(index)

        outputs = [
            {
                "session_id": session_id,
                "triggered": False,
                "predictions": None,
                "cached_length": self._caches[session_id].cached_length,
                "tokens_added": tokens_added[index],
            }
            for index, session_id in enumerate(batch["session_ids"])
        ]
        if not triggered_indices:
            return outputs

        triggered_caches = []
        label_ids = []
        label_masks = []
        for index in triggered_indices:
            session_id = batch["session_ids"][index]
            cache = self._session_strategies[session_id].get_window(self._caches[session_id])
            triggered_caches.append(cache)
            sequence = format_decoder_kv_labels(
                batch["labels"][index],
                label_token=self.label_token,
                sep_token=self.sep_token,
            )
            tokenized = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
            label_ids.append(tokenized["input_ids"].to(self.device))
            label_masks.append(tokenized["attention_mask"].to(self.device))

        stacked = BatchedKVHelper.stack_for_classify(triggered_caches, label_ids, label_masks, self.device)
        model_output = self.model.classify_from_decoder_cache(
            input_ids=stacked["input_ids"],
            attention_mask=stacked["attention_mask"],
            label_mask=stacked["label_mask"],
            position_ids=stacked["position_ids"],
            past_key_values=stacked["past_key_values"],
        )

        for local_index, batch_index in enumerate(triggered_indices):
            labels = batch["labels"][batch_index]
            logits = model_output.logits[local_index, : len(labels)]
            predictions, all_scores = self._postprocess_logits(
                logits,
                labels,
                batch["classification_types"][batch_index],
                batch["thresholds"][batch_index],
            )
            if batch["return_hierarchical"][batch_index]:
                predictions = build_hierarchical_output(
                    predictions,
                    batch["original_labels"][batch_index],
                    self.label_separator,
                    all_scores,
                )

            session_id = batch["session_ids"][batch_index]
            outputs[batch_index] = {
                "session_id": session_id,
                "triggered": True,
                "predictions": predictions,
                "cached_length": self._caches[session_id].cached_length,
                "tokens_added": tokens_added[batch_index],
            }
        return outputs

    def _load_to_device(self, session_ids: set[str], device: torch.device) -> None:
        to_cpu = device.type == "cpu"
        for session_id in session_ids:
            cache = self._caches.get(session_id)
            if cache is None or cache.past_key_values is None:
                continue
            if to_cpu and self.use_pinned_memory:
                self._caches[session_id] = _move_cache_pinned(cache)
            else:
                self._caches[session_id] = _move_cache_nonblocking(cache, device)

        if not to_cpu and self.use_pinned_memory and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _move_cache_pinned(cache: CacheState) -> CacheState:
    """Move a DynamicCache to CPU pinned memory."""
    from transformers.cache_utils import DynamicCache, DynamicLayer

    if not isinstance(cache.past_key_values, DynamicCache):
        return cache.to(torch.device("cpu"))

    new_kv = DynamicCache()
    for index, layer in enumerate(cache.past_key_values.layers):
        if not layer.is_initialized:
            continue
        pinned_keys = layer.keys.cpu().pin_memory()
        pinned_values = layer.values.cpu().pin_memory()
        while len(new_kv.layers) <= index:
            new_kv.layers.append(DynamicLayer())
        new_layer = new_kv.layers[index]
        new_layer.dtype = pinned_keys.dtype
        new_layer.device = pinned_keys.device
        new_layer.keys = pinned_keys
        new_layer.values = pinned_values
        new_layer.is_initialized = True

    return CacheState(
        past_key_values=new_kv,
        input_ids=cache.input_ids.cpu(),
        attention_mask=cache.attention_mask.cpu(),
        position_ids=cache.position_ids.cpu() if cache.position_ids is not None else None,
        cached_length=cache.cached_length,
        next_position_id=cache.next_position_id,
        session_id=cache.session_id,
        metadata=cache.metadata.copy(),
    )


def _move_cache_nonblocking(cache: CacheState, device: torch.device) -> CacheState:
    """Move a DynamicCache to a device, asynchronously when possible."""
    from transformers.cache_utils import DynamicCache

    if not isinstance(cache.past_key_values, DynamicCache):
        return cache.to(device)

    new_kv = DynamicCache()
    for index, layer in enumerate(cache.past_key_values.layers):
        if layer.is_initialized:
            new_kv.update(
                layer.keys.to(device, non_blocking=True),
                layer.values.to(device, non_blocking=True),
                index,
            )

    return CacheState(
        past_key_values=new_kv,
        input_ids=cache.input_ids.to(device, non_blocking=True),
        attention_mask=cache.attention_mask.to(device, non_blocking=True),
        position_ids=cache.position_ids.to(device, non_blocking=True) if cache.position_ids is not None else None,
        cached_length=cache.cached_length,
        next_position_id=cache.next_position_id,
        session_id=cache.session_id,
        metadata=cache.metadata.copy(),
    )
