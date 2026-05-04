"""Tests for bootstrap consensus trees."""

from __future__ import annotations

import pytest

from bitig.corpus import Corpus
from bitig.methods.consensus import BootstrapConsensus


def _federalist_mini() -> Corpus:
    # Use the bundled Federalist fixture for a realistic test.
    from bitig.io import load_corpus

    return load_corpus(
        "tests/fixtures/federalist", metadata="tests/fixtures/federalist/metadata.tsv"
    )


pytestmark = pytest.mark.slow  # Consensus is inherently bootstrap-heavy.


def test_consensus_runs_on_federalist() -> None:
    corpus = _federalist_mini().filter(role="train")
    result = BootstrapConsensus(
        mfw_bands=[100, 200, 300],
        replicates=5,  # Small for test speed — production use 100+
        seed=42,
    ).fit_transform(corpus)
    assert "newick" in result.values
    assert "support" in result.values


def test_consensus_newick_is_nonempty_string() -> None:
    corpus = _federalist_mini().filter(role="train")
    result = BootstrapConsensus(mfw_bands=[100, 200], replicates=3, seed=42).fit_transform(corpus)
    newick = result.values["newick"]
    assert isinstance(newick, str)
    assert len(newick) > 10
    assert newick.endswith(";")


def test_consensus_clade_support_is_bounded_0_1() -> None:
    corpus = _federalist_mini().filter(role="train")
    result = BootstrapConsensus(mfw_bands=[100, 200], replicates=3, seed=42).fit_transform(corpus)
    for support in result.values["support"].values():
        assert 0.0 <= support <= 1.0
