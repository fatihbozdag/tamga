"""Tests for DependencyBigramExtractor."""

from pathlib import Path

import pytest

from bitig.corpus import Corpus, Document
from bitig.features.dependency import DependencyBigramExtractor

pytestmark = pytest.mark.spacy

_NLP_MODEL = "en_core_web_sm"


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_dependency_bigram_returns_triples(tmp_path: Path) -> None:
    ex = DependencyBigramExtractor(spacy_model=_NLP_MODEL, cache_dir=tmp_path)
    fm = ex.fit_transform(_corpus("The cat sat on the mat."))
    # Features are "head_lemma|dep|child_lemma" strings.
    assert fm.X.shape[0] == 1
    assert any(f.count("|") == 2 for f in fm.feature_names)


def test_dependency_bigram_counts_are_integers(tmp_path: Path) -> None:
    ex = DependencyBigramExtractor(spacy_model=_NLP_MODEL, cache_dir=tmp_path)
    fm = ex.fit_transform(_corpus("A dog barked. A dog ran."))
    # Integer-valued counts (stored as float by convention).
    assert (fm.X == fm.X.astype(int)).all()  # noqa: SIM300
