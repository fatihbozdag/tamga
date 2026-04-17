"""The Corpus collection — an ordered bag of Documents with metadata-aware operations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from tamga.corpus.document import Document
from tamga.plumbing.hashing import hash_mapping, hash_text


@dataclass
class Corpus:
    """An ordered collection of Documents that share a metadata schema.

    Iteration yields Documents in the order provided. Equality and hashing ignore order — two
    Corpora with the same documents in different orders hash identically.
    """

    documents: list[Document] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def __getitem__(self, index: int) -> Document:
        return self.documents[index]

    def filter(self, **query: Any) -> Corpus:
        """Return a new Corpus containing documents whose metadata matches every key in `query`.

        Values may be scalars (exact match) or lists (membership).
        """

        def matches(doc: Document) -> bool:
            for key, expected in query.items():
                actual = doc.metadata.get(key)
                if isinstance(expected, list | tuple | set):
                    if actual not in expected:
                        return False
                elif actual != expected:
                    return False
            return True

        return Corpus(documents=[d for d in self.documents if matches(d)])

    def groupby(self, field_name: str) -> dict[Any, Corpus]:
        """Group documents by a metadata field value.

        Raises KeyError if any document lacks the field.
        """
        groups: dict[Any, list[Document]] = {}
        for doc in self.documents:
            if field_name not in doc.metadata:
                raise KeyError(f"document {doc.id!r} has no metadata field {field_name!r}")
            groups.setdefault(doc.metadata[field_name], []).append(doc)
        return {k: Corpus(documents=v) for k, v in groups.items()}

    def metadata_column(self, field_name: str) -> list[Any]:
        """Return the list of metadata values at `field_name`, in document order.

        Missing values become None; use `filter` first if you want to exclude them.
        """
        return [d.metadata.get(field_name) for d in self.documents]

    def hash(self) -> str:
        """Stable hash — sorted document hashes + sorted metadata."""
        doc_hashes = sorted(d.hash for d in self.documents)
        metadata_summary = sorted((d.id, hash_mapping(d.metadata)) for d in self.documents)
        payload = "|".join(doc_hashes) + "||" + str(metadata_summary)
        return hash_text(payload)

    @classmethod
    def from_iterable(cls, docs: Iterable[Document]) -> Corpus:
        return cls(documents=list(docs))
