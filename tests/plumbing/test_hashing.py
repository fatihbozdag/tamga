"""Tests for stable content hashing."""

import pytest

from tamga.plumbing.hashing import hash_bytes, hash_mapping, hash_text, short_hash


def test_hash_text_is_stable_across_calls():
    assert hash_text("hello world") == hash_text("hello world")


def test_hash_text_differs_for_different_content():
    assert hash_text("hello") != hash_text("world")


def test_hash_text_returns_hex_digest():
    h = hash_text("abc")
    assert isinstance(h, str)
    assert all(c in "0123456789abcdef" for c in h)
    assert len(h) == 64  # sha256 hex


def test_hash_bytes_matches_hash_text_for_same_content():
    assert hash_bytes(b"hello") == hash_text("hello")


def test_hash_mapping_is_order_independent():
    assert hash_mapping({"a": 1, "b": 2}) == hash_mapping({"b": 2, "a": 1})


def test_hash_mapping_is_stable():
    assert hash_mapping({"a": 1, "b": [1, 2, 3]}) == hash_mapping({"a": 1, "b": [1, 2, 3]})


def test_hash_mapping_differs_for_different_content():
    assert hash_mapping({"a": 1}) != hash_mapping({"a": 2})


def test_hash_mapping_handles_nested_structures():
    v1 = {"a": {"b": {"c": 1}}}
    v2 = {"a": {"b": {"c": 1}}}
    assert hash_mapping(v1) == hash_mapping(v2)


def test_hash_mapping_rejects_non_json_values():
    with pytest.raises(TypeError):
        hash_mapping({"a": object()})


def test_short_hash_is_prefix_of_full():
    full = hash_text("xyz")
    assert short_hash("xyz") == full[:12]
