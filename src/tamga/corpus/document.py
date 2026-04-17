"""The Document class — a single text with metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from tamga.plumbing.hashing import hash_text


@dataclass(frozen=True)
class Document:
    """A single text with optional metadata.

    `hash` is derived lazily from `text` — identical texts hash identically regardless of `id` or
    metadata, which is intentional: the cache key for parsed documents is content-addressed.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def hash(self) -> str:
        return hash_text(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        return cls(id=data["id"], text=data["text"], metadata=dict(data.get("metadata") or {}))
