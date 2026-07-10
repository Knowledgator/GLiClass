from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import CacheState


class ClassificationStrategy(ABC):
    """
    Base class for streaming classification strategies.

    Controls two things:
      - should_classify: whether to trigger classification on this call
      - get_window:      which slice of the cache to classify over
    """

    @abstractmethod
    def should_classify(self, tokens_added: int, cached_length: int, text: str) -> bool: ...

    def get_window(self, cache_state: "CacheState") -> "CacheState":
        """Return the cache slice to classify over. Default: full context."""
        return cache_state


class EveryChunkStrategy(ClassificationStrategy):
    """Classify on every non-empty chunk."""

    def should_classify(self, tokens_added: int, cached_length: int, text: str) -> bool:
        return tokens_added > 0


class EveryNTokensStrategy(ClassificationStrategy):
    """Classify after accumulating at least N new tokens."""

    def __init__(self, n: int):
        self.n = n
        self._accumulated = 0

    def should_classify(self, tokens_added: int, cached_length: int, text: str) -> bool:
        self._accumulated += tokens_added
        if self._accumulated >= self.n:
            self._accumulated = 0
            return True
        return False


class OnDelimiterStrategy(ClassificationStrategy):
    """Classify when the incoming text contains a delimiter."""

    def __init__(self, delimiter: str):
        self.delimiter = delimiter

    def should_classify(self, tokens_added: int, cached_length: int, text: str) -> bool:
        return self.delimiter in text


class NeverStrategy(ClassificationStrategy):
    """Only update cache, never classify. Useful for pre-filling context."""

    def should_classify(self, tokens_added: int, cached_length: int, text: str) -> bool:
        return False


class SlidingWindowStrategy(ClassificationStrategy):
    """
    Classify on every chunk, but only over the last window_size cached tokens.

    Combine with EveryNTokensStrategy behaviour by subclassing or wrapping.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size

    def should_classify(self, tokens_added: int, cached_length: int, text: str) -> bool:
        return tokens_added > 0

    def get_window(self, cache_state: "CacheState") -> "CacheState":
        from .cache import truncate_cache
        return truncate_cache(cache_state, self.window_size)


class ComposedStrategy(ClassificationStrategy):
    """
    Combine a trigger strategy with a window strategy.

    Example:
        ComposedStrategy(trigger=EveryNTokensStrategy(200), window=SlidingWindowStrategy(512))
    """

    def __init__(self, trigger: ClassificationStrategy, window: ClassificationStrategy):
        self.trigger = trigger
        self.window = window

    def should_classify(self, tokens_added: int, cached_length: int, text: str) -> bool:
        return self.trigger.should_classify(tokens_added, cached_length, text)

    def get_window(self, cache_state: "CacheState") -> "CacheState":
        return self.window.get_window(cache_state)
