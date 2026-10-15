"""
KV cache management for GLiClass streaming classification.

Includes:
  - CacheState / DynamicKVCacheManager  (low-level per-session ops)
  - BatchedKVHelper                      (stack / unstack heterogeneous KV caches for batched inference)
"""

from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

import torch

# ---------------------------------------------------------------------------
# CacheState
# ---------------------------------------------------------------------------


@dataclass
class CacheState:
    past_key_values: Any
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor | None = None
    cached_length: int = 0
    next_position_id: int | None = None
    session_id: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def next_position(self) -> int:
        """Return the absolute position for the next appended token."""
        return self.cached_length if self.next_position_id is None else self.next_position_id

    def to(self, device: torch.device) -> CacheState:
        return CacheState(
            past_key_values=_move_past_kv(self.past_key_values, device),
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            position_ids=self.position_ids.to(device) if self.position_ids is not None else None,
            cached_length=self.cached_length,
            next_position_id=self.next_position_id,
            session_id=self.session_id,
            metadata=self.metadata.copy(),
        )


def _move_past_kv(past_kv, device):
    if past_kv is None:
        return None
    if hasattr(past_kv, "to"):
        return past_kv.to(device)
    if isinstance(past_kv, (tuple, list)):
        return type(past_kv)(_move_past_kv(item, device) for item in past_kv)
    return past_kv


# ---------------------------------------------------------------------------
def create_empty_cache(session_id: str | None = None, device=None) -> CacheState:
    kw = {"device": device} if device is not None else {}
    return CacheState(
        past_key_values=None,
        input_ids=torch.empty(0, dtype=torch.long, **kw),
        attention_mask=torch.empty(0, dtype=torch.long, **kw),
        position_ids=None,
        cached_length=0,
        next_position_id=0,
        session_id=session_id,
        metadata={},
    )


def truncate_cache(cache_state: CacheState, max_length: int) -> CacheState:
    if cache_state.cached_length <= max_length or cache_state.past_key_values is None:
        return cache_state
    truncated = _crop_past_kv(cache_state.past_key_values, max_length)
    if truncated is None:
        return cache_state
    return CacheState(
        past_key_values=truncated,
        input_ids=cache_state.input_ids[-max_length:],
        attention_mask=cache_state.attention_mask[-max_length:],
        position_ids=cache_state.position_ids[-max_length:] if cache_state.position_ids is not None else None,
        cached_length=max_length,
        next_position_id=cache_state.next_position,
        session_id=cache_state.session_id,
        metadata=cache_state.metadata.copy(),
    )


# ---------------------------------------------------------------------------
# Hybrid caches (e.g. Qwen3.5): linear-attention layers keep fixed-size conv / recurrent states
# instead of per-token keys and values. Those states cannot be sliced or cropped along the sequence,
# only selected along the batch dimension, and every token they see (padding included) changes them.
# ---------------------------------------------------------------------------


def _is_linear_layer(layer) -> bool:
    return hasattr(layer, "recurrent_states")


def _layer_ready(layer) -> bool:
    if _is_linear_layer(layer):
        return any(layer.is_conv_states_initialized.values()) or any(layer.is_recurrent_states_initialized.values())
    return layer.is_initialized


def has_linear_layers(past_kv) -> bool:
    return past_kv is not None and any(_is_linear_layer(layer) for layer in getattr(past_kv, "layers", []))


def _new_cache_like(past_kv):
    """Empty DynamicCache with the same layer types as past_kv (plain DynamicCache for attention-only models)."""
    from transformers.cache_utils import Cache, DynamicCache

    if not has_linear_layers(past_kv):
        return DynamicCache()
    cache = DynamicCache.__new__(DynamicCache)
    Cache.__init__(cache, layers=[type(layer)() for layer in past_kv.layers])
    return cache


def _set_linear_states(layer, conv_states: dict, recurrent_states: dict) -> None:
    """Fill a fresh linear-attention cache layer with already-computed states."""
    for state_idx, conv in conv_states.items():
        if conv is None:
            continue
        layer.conv_states[state_idx] = conv
        layer.conv_kernel_size[state_idx] = conv.shape[-1]
        layer.is_conv_states_initialized[state_idx] = True
        layer.has_previous_state[state_idx] = True
        layer.device, layer.dtype = conv.device, conv.dtype
    for state_idx, recurrent in recurrent_states.items():
        if recurrent is None:
            continue
        layer.recurrent_states[state_idx] = recurrent
        layer.is_recurrent_states_initialized[state_idx] = True
        layer.has_previous_state[state_idx] = True


def _copy_linear_layer(src, dst, select) -> None:
    """Copy src's states into dst, mapping every state tensor through select (clone / batch slice)."""
    _set_linear_states(
        dst,
        {i: select(s) for i, s in src.conv_states.items() if src.is_conv_states_initialized[i]},
        {i: select(s) for i, s in src.recurrent_states.items() if src.is_recurrent_states_initialized[i]},
    )


def _deep_copy_past_kv(past_kv):
    if past_kv is None:
        return None
    from transformers.cache_utils import DynamicCache

    if isinstance(past_kv, DynamicCache):
        new = _new_cache_like(past_kv)
        for i, layer in enumerate(past_kv.layers):
            if not _layer_ready(layer):
                continue
            if _is_linear_layer(layer):
                _copy_linear_layer(layer, new.layers[i], lambda t: t.clone())
            else:
                new.update(layer.keys.clone(), layer.values.clone(), i)
        return new
    if hasattr(past_kv, "clone"):
        return past_kv.clone()
    if isinstance(past_kv, (tuple, list)):
        return type(past_kv)(_deep_copy_past_kv(x) for x in past_kv)
    return past_kv


def _crop_past_kv(past_kv, max_length: int):
    """Crop past_key_values to last max_length tokens. Returns None if not supported.

    Linear-attention states summarize the whole history and cannot be cropped: they are kept as they are,
    so on hybrid models the window applies to the attention layers only.
    """
    if past_kv is None:
        return None
    from transformers.cache_utils import DynamicCache

    if isinstance(past_kv, DynamicCache):
        new = _new_cache_like(past_kv)
        for i, layer in enumerate(past_kv.layers):
            if not _layer_ready(layer):
                continue
            if _is_linear_layer(layer):
                _copy_linear_layer(layer, new.layers[i], lambda t: t.clone())
            else:
                new.update(layer.keys[:, :, -max_length:, :].clone(), layer.values[:, :, -max_length:, :].clone(), i)
        return new
    return None


# ---------------------------------------------------------------------------
# BatchedKVHelper  - stack / unstack heterogeneous KV caches
# ---------------------------------------------------------------------------


class BatchedKVHelper:
    """
    Utilities for batching decoder forwards across sessions with different
    KV cache lengths.

    Stacking strategy:
      - KV caches are prepend-padded with zeros to max_cached_len so that
        position embeddings (RoPE) baked into each K/V tensor remain correct.
      - The attention_mask covers [prepend_zeros | real_cached | new_tokens | new_pad].
      - position_ids for new tokens are set per-session: [cached_len_i .. cached_len_i + new_len_i).
    """

    @staticmethod
    def stack_for_update(
        caches: list[CacheState],
        new_input_ids: list[torch.Tensor],  # one per session, already on device
        new_attention_masks: list[torch.Tensor],
        device: torch.device,
    ) -> dict:
        """
        Prepare a batched decoder forward for the cache-update stage.

        Returns a dict with keys:
          input_ids, attention_mask, position_ids, past_key_values,
          cached_lengths, new_lengths, max_cached_len
        """
        batch_size = len(caches)
        cached_lengths = [c.cached_length for c in caches]
        new_lengths = [ids.shape[-1] for ids in new_input_ids]
        max_cached = max(cached_lengths)
        max_new = max(new_lengths)

        # --- pad new tokens (right-pad) ---
        padded_ids = new_input_ids[0].new_zeros(batch_size, max_new)
        padded_new_mask = new_input_ids[0].new_zeros(batch_size, max_new)
        position_ids = new_input_ids[0].new_zeros(batch_size, max_new)

        for i, (ids, mask, _clen, nlen) in enumerate(
            zip(new_input_ids, new_attention_masks, cached_lengths, new_lengths)
        ):
            padded_ids[i, :nlen] = ids[0] if ids.dim() == 2 else ids
            padded_new_mask[i, :nlen] = mask[0] if mask.dim() == 2 else mask
            next_position = caches[i].next_position
            position_ids[i, :nlen] = torch.arange(next_position, next_position + nlen, device=device)

        # --- build full attention mask: [prepend_zeros | cached | new | new_pad] ---
        full_mask = torch.zeros(batch_size, max_cached + max_new, dtype=torch.long, device=device)
        for i, (clen, nlen) in enumerate(zip(cached_lengths, new_lengths)):
            pad = max_cached - clen
            full_mask[i, pad : pad + clen] = 1  # real cached tokens
            full_mask[i, max_cached : max_cached + nlen] = 1  # real new tokens

        # --- stack past_key_values (prepend-pad to max_cached) ---
        stacked_past_kv = None
        if max_cached > 0:
            stacked_past_kv = BatchedKVHelper._stack_past_kv(caches, max_cached, device)

        return {
            "input_ids": padded_ids,
            "attention_mask": full_mask,
            "position_ids": position_ids,
            "past_key_values": stacked_past_kv,
            "cached_lengths": cached_lengths,
            "new_lengths": new_lengths,
            "max_cached_len": max_cached,
        }

    @staticmethod
    def unstack_after_update(
        new_past_kv,
        stacked_info: dict,
        old_caches: list[CacheState],
        new_input_ids: list[torch.Tensor],
        new_attention_masks: list[torch.Tensor],
    ) -> list[CacheState]:
        """
        Unstack per-session CacheState from a batched decoder output.

        new_past_kv has shape [..., max_cached + max_new, ...] along seq dim.
        We slice [max_cached - clen_i : max_cached + new_len_i] to recover
        only the real tokens for each session.
        """
        cached_lengths = stacked_info["cached_lengths"]
        new_lengths = stacked_info["new_lengths"]
        max_cached = stacked_info["max_cached_len"]

        results = []
        for i, (old, clen, nlen) in enumerate(zip(old_caches, cached_lengths, new_lengths)):
            real_start = max_cached - clen
            real_end = max_cached + nlen
            sliced_kv = BatchedKVHelper._slice_past_kv(new_past_kv, i, real_start, real_end)
            new_clen = clen + nlen

            # rebuild input_ids / attention_mask for CacheState
            new_ids = new_input_ids[i]
            if new_ids.dim() == 2:
                new_ids = new_ids[0]
            new_mask = new_attention_masks[i]
            if new_mask.dim() == 2:
                new_mask = new_mask[0]

            if clen > 0:
                full_ids = torch.cat([old.input_ids, new_ids[:nlen]], dim=0)
                full_mask = torch.cat([old.attention_mask, new_mask[:nlen]], dim=0)
            else:
                full_ids = new_ids[:nlen]
                full_mask = new_mask[:nlen]

            results.append(
                CacheState(
                    past_key_values=sliced_kv,
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    position_ids=None,
                    cached_length=new_clen,
                    next_position_id=old.next_position + nlen,
                    session_id=old.session_id,
                    metadata=old.metadata.copy(),
                )
            )

        return results

    @staticmethod
    def stack_for_classify(
        caches: list[CacheState],
        label_ids: list[torch.Tensor],
        label_masks: list[torch.Tensor],
        device: torch.device,
    ) -> dict:
        """
        Prepare a batched decoder forward for the classification stage.

        Label tokens are right-padded. KV caches are prepend-padded.
        Returns input_ids, attention_mask, position_ids, past_key_values,
        plus metadata needed for slicing scorer inputs afterwards.
        """
        batch_size = len(caches)
        cached_lengths = [c.cached_length for c in caches]
        label_lengths = [ids.shape[-1] for ids in label_ids]
        max_cached = max(cached_lengths)
        max_label = max(label_lengths)

        padded_label_ids = label_ids[0].new_zeros(batch_size, max_label)
        padded_label_mask = label_ids[0].new_zeros(batch_size, max_label)
        position_ids = label_ids[0].new_zeros(batch_size, max_label)

        for i, (ids, mask, _clen, llen) in enumerate(zip(label_ids, label_masks, cached_lengths, label_lengths)):
            flat_ids = ids[0] if ids.dim() == 2 else ids
            flat_mask = mask[0] if mask.dim() == 2 else mask
            padded_label_ids[i, :llen] = flat_ids
            padded_label_mask[i, :llen] = flat_mask
            next_position = caches[i].next_position
            position_ids[i, :llen] = torch.arange(next_position, next_position + llen, device=device)

        full_mask = torch.zeros(batch_size, max_cached + max_label, dtype=torch.long, device=device)
        for i, (clen, llen) in enumerate(zip(cached_lengths, label_lengths)):
            pad = max_cached - clen
            full_mask[i, pad : pad + clen] = 1
            full_mask[i, max_cached : max_cached + llen] = 1

        stacked_past_kv = None
        if max_cached > 0:
            stacked_past_kv = BatchedKVHelper._stack_past_kv(caches, max_cached, device)

        return {
            "input_ids": padded_label_ids,
            "attention_mask": full_mask,  # full: cache + labels (for decoder)
            "label_mask": padded_label_mask,  # labels only (for scorer)
            "position_ids": position_ids,
            "past_key_values": stacked_past_kv,
            "label_lengths": label_lengths,
            "max_label_len": max_label,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_past_kv(caches: list[CacheState], max_cached: int, device: torch.device):
        """Prepend-pad each session's KV to max_cached and stack along batch dim.

        Linear-attention states have no sequence dimension: they are stacked as they are, and a session
        without a cache gets zero states, which is what a fresh prefill starts from.
        """
        non_empty = [c for c in caches if c.past_key_values is not None]
        reference = non_empty[0].past_key_values
        num_layers = len(reference.layers)

        stacked = _new_cache_like(reference)
        for layer_idx in range(num_layers):
            ready = [
                c.past_key_values.layers[layer_idx]
                for c in non_empty
                if _layer_ready(c.past_key_values.layers[layer_idx])
            ]
            if not ready:
                continue
            ref_layer = ready[0]

            if _is_linear_layer(ref_layer):
                conv, recurrent = {}, {}
                for state_idx, ref_state in ref_layer.conv_states.items():
                    if ref_layer.is_conv_states_initialized[state_idx]:
                        conv[state_idx] = BatchedKVHelper._stack_states(
                            caches, layer_idx, "conv_states", "is_conv_states_initialized", state_idx, ref_state, device
                        )
                for state_idx, ref_state in ref_layer.recurrent_states.items():
                    if ref_layer.is_recurrent_states_initialized[state_idx]:
                        recurrent[state_idx] = BatchedKVHelper._stack_states(
                            caches,
                            layer_idx,
                            "recurrent_states",
                            "is_recurrent_states_initialized",
                            state_idx,
                            ref_state,
                            device,
                        )
                _set_linear_states(stacked.layers[layer_idx], conv, recurrent)
                continue

            ref_k = ref_layer.keys  # [1, heads, seq, head_dim]
            keys, values = [], []
            for c in caches:
                kv = c.past_key_values
                if kv is None or not _layer_ready(kv.layers[layer_idx]):
                    k = torch.zeros(1, ref_k.shape[1], max_cached, ref_k.shape[3], device=device, dtype=ref_k.dtype)
                    v = torch.zeros_like(k)
                else:
                    layer = kv.layers[layer_idx]
                    k = layer.keys.to(device)  # [1, heads, clen, head_dim]
                    v = layer.values.to(device)
                    clen = k.shape[2]
                    pad = max_cached - clen
                    if pad > 0:
                        zeros_k = torch.zeros(1, k.shape[1], pad, k.shape[3], device=device, dtype=k.dtype)
                        zeros_v = torch.zeros_like(zeros_k)
                        k = torch.cat([zeros_k, k], dim=2)
                        v = torch.cat([zeros_v, v], dim=2)
                keys.append(k)
                values.append(v)

            stacked.update(torch.cat(keys, dim=0), torch.cat(values, dim=0), layer_idx)

        return stacked

    @staticmethod
    def _stack_states(caches, layer_idx, attr, flag, state_idx, ref_state, device):
        """Concatenate one linear-attention state over sessions (zeros for sessions without it)."""
        states = []
        for c in caches:
            layer = c.past_key_values.layers[layer_idx] if c.past_key_values is not None else None
            if layer is not None and getattr(layer, flag)[state_idx]:
                states.append(getattr(layer, attr)[state_idx].to(device))
            else:
                states.append(torch.zeros_like(ref_state, device=device))
        return torch.cat(states, dim=0)

    @staticmethod
    def _slice_past_kv(past_kv, batch_idx: int, start: int, end: int):
        """Extract single-session KV from a batched past_key_values.

        Linear-attention states are only correct for rows without padding after the real tokens;
        the pipeline batches hybrid-cache updates by equal chunk length to guarantee that.
        """
        if past_kv is None:
            return None

        sliced = _new_cache_like(past_kv)
        for i, layer in enumerate(past_kv.layers):
            if not _layer_ready(layer):
                continue
            if _is_linear_layer(layer):
                _copy_linear_layer(layer, sliced.layers[i], lambda t: t[batch_idx : batch_idx + 1].clone())
                continue
            sliced.update(
                layer.keys[batch_idx : batch_idx + 1, :, start:end, :].clone(),
                layer.values[batch_idx : batch_idx + 1, :, start:end, :].clone(),
                i,
            )
        return sliced
