from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

if TYPE_CHECKING:
    from .strategies import ClassificationStrategy


@dataclass
class SessionInput:
    session_id: str
    text: str
    labels: list[str]
    strategy: ClassificationStrategy
    classification_type: str = "multi-label"  # "multi-label" or "single-label"


@dataclass
class SessionOutput:
    session_id: str
    triggered: bool
    predictions: list[dict] | None
    cached_length: int
    tokens_added: int
    metadata: dict = field(default_factory=dict)
