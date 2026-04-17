"""Content-addressable cache for spaCy `DocBin` blobs.

Keyed by `(document_hash, spacy_model, spacy_version, sorted_excluded_components)`. At this stage
the cache stores raw bytes — Task 13 will wire up `DocBin` serialisation on top.
"""

from __future__ import annotations

from pathlib import Path

from tamga.plumbing.hashing import hash_mapping


def cache_key(
    document_hash: str,
    spacy_model: str,
    spacy_version: str,
    excluded_components: list[str],
) -> str:
    """Return a stable cache key for a (document, spaCy configuration) pair."""
    return hash_mapping(
        {
            "doc": document_hash,
            "model": spacy_model,
            "version": spacy_version,
            "exclude": sorted(excluded_components),
        }
    )


class DocBinCache:
    """Directory-backed cache. One file per key, named `<key>.docbin`."""

    _EXT = ".docbin"

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}{self._EXT}"

    def get(self, key: str) -> bytes | None:
        p = self._path(key)
        if not p.is_file():
            return None
        return p.read_bytes()

    def put(self, key: str, payload: bytes) -> None:
        self._path(key).write_bytes(payload)

    def keys(self) -> list[str]:
        return sorted(f.stem for f in self.directory.glob(f"*{self._EXT}"))

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.directory.glob(f"*{self._EXT}"))

    def clear(self) -> None:
        for f in self.directory.glob(f"*{self._EXT}"):
            f.unlink()
