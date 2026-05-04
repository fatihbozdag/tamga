"""Stable content hashing used throughout the package.

Two requirements every hash must satisfy:

1. **Deterministic across Python processes.** The corpus hash you computed today must match the one
   a collaborator computes tomorrow, on a different machine, in a different Python version.
2. **Stable across equivalent inputs.** A mapping is hashed by its sorted JSON representation so
   key-insertion order is irrelevant.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

_ENCODING = "utf-8"
_HASH = hashlib.sha256


def hash_bytes(data: bytes) -> str:
    """Return the sha256 hex digest of raw bytes."""
    return _HASH(data).hexdigest()


def hash_text(text: str) -> str:
    """Return the sha256 hex digest of a UTF-8 string."""
    return hash_bytes(text.encode(_ENCODING))


def hash_mapping(mapping: Mapping[str, Any]) -> str:
    """Return a stable sha256 hex digest of a JSON-serialisable mapping.

    Keys are sorted before serialisation so insertion order does not affect the result.
    Raises TypeError if any value is not JSON-serialisable.
    """
    try:
        serialised = json.dumps(
            mapping, sort_keys=True, separators=(",", ":"), default=_fail_default
        )
    except TypeError as exc:  # pragma: no cover - covered by _fail_default
        raise TypeError(f"hash_mapping: non-serialisable value: {exc}") from exc
    return hash_text(serialised)


def short_hash(text: str, length: int = 12) -> str:
    """Return the first `length` characters of the hash — useful for directory/cache-key display."""
    return hash_text(text)[:length]


def _fail_default(value: Any) -> Any:
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")
