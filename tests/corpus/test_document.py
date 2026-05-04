"""Tests for the Document dataclass."""

import pytest

from bitig.corpus.document import Document


def test_document_basic_construction():
    doc = Document(id="doc-1", text="hello world", metadata={"author": "Alice"})
    assert doc.id == "doc-1"
    assert doc.text == "hello world"
    assert doc.metadata == {"author": "Alice"}


def test_document_hash_is_computed_from_text():
    doc = Document(id="doc-1", text="hello world", metadata={})
    assert len(doc.hash) == 64  # sha256 hex
    # Identical text ⇒ identical hash.
    assert doc.hash == Document(id="other-id", text="hello world", metadata={}).hash


def test_document_hash_differs_for_different_text():
    a = Document(id="a", text="hello", metadata={})
    b = Document(id="b", text="world", metadata={})
    assert a.hash != b.hash


def test_document_metadata_defaults_to_empty_mapping():
    doc = Document(id="doc-1", text="abc")
    assert doc.metadata == {}


def test_document_is_immutable():
    doc = Document(id="doc-1", text="abc", metadata={})
    with pytest.raises((AttributeError, TypeError)):
        doc.id = "doc-2"  # type: ignore[misc]


def test_document_round_trips_to_dict():
    doc = Document(id="x", text="hi", metadata={"author": "A", "year": 1984})
    d = doc.to_dict()
    restored = Document.from_dict(d)
    assert restored == doc
