"""Tests for the Provenance record."""

from datetime import datetime

from tamga.provenance import Provenance


def test_provenance_basic_construction():
    p = Provenance(
        tamga_version="0.1.0.dev0",
        python_version="3.11.7",
        spacy_model="en_core_web_sm",
        spacy_version="3.7.2",
        corpus_hash="deadbeef",
        feature_hash=None,
        seed=42,
        timestamp=datetime(2026, 4, 17, 12, 0, 0),
        resolved_config={"seed": 42},
    )
    assert p.seed == 42
    assert p.feature_hash is None


def test_provenance_round_trips_to_dict():
    p = Provenance(
        tamga_version="0.1.0.dev0",
        python_version="3.11.7",
        spacy_model="en_core_web_sm",
        spacy_version="3.7.2",
        corpus_hash="abc123",
        feature_hash="feat456",
        seed=7,
        timestamp=datetime(2026, 4, 17, 12, 0, 0),
        resolved_config={"seed": 7, "nested": {"k": "v"}},
    )
    d = p.to_dict()
    restored = Provenance.from_dict(d)
    assert restored == p


def test_provenance_current_captures_runtime():
    p = Provenance.current(
        spacy_model="en_core_web_sm",
        spacy_version="3.7.2",
        corpus_hash="h",
        feature_hash=None,
        seed=1,
        resolved_config={},
    )
    assert p.tamga_version
    assert "." in p.python_version
    assert isinstance(p.timestamp, datetime)
