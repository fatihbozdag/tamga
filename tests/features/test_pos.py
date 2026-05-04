"""Tests for PosNgramExtractor."""

from pathlib import Path

import pytest

from bitig.corpus import Corpus, Document
from bitig.features.pos import PosNgramExtractor

pytestmark = pytest.mark.spacy

_NLP_MODEL = "en_core_web_sm"


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_pos_ngram_unigrams(tmp_path: Path) -> None:
    ex = PosNgramExtractor(n=1, tagset="coarse", spacy_model=_NLP_MODEL, cache_dir=tmp_path)
    fm = ex.fit_transform(_corpus("The cat sat on the mat."))
    # Coarse tagset (UPOS) → DET, NOUN, VERB, ADP, PUNCT at minimum.
    assert any(t in fm.feature_names for t in ("DET", "NOUN", "VERB"))


def test_pos_ngram_bigrams_count_pairs(tmp_path: Path) -> None:
    ex = PosNgramExtractor(n=2, tagset="coarse", spacy_model=_NLP_MODEL, cache_dir=tmp_path)
    fm = ex.fit_transform(_corpus("The cat sat."))
    # Each bigram is "TAG1|TAG2"; pipe-joined for unambiguous parsing.
    assert any("|" in f for f in fm.feature_names)


def test_pos_ngram_fine_tagset_differs_from_coarse(tmp_path: Path) -> None:
    c = _corpus("The cat ran quickly.")
    coarse = PosNgramExtractor(
        n=1, tagset="coarse", spacy_model=_NLP_MODEL, cache_dir=tmp_path / "c"
    )
    fine = PosNgramExtractor(n=1, tagset="fine", spacy_model=_NLP_MODEL, cache_dir=tmp_path / "f")
    fm_c = coarse.fit_transform(c)
    fm_f = fine.fit_transform(c)
    # Fine tags are generally more numerous than coarse (UPOS has ~17; PTB has ~36).
    assert len(fm_f.feature_names) >= len(fm_c.feature_names)
