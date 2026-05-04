"""Tests for SentenceLengthExtractor."""

from pathlib import Path

import pytest

from bitig.corpus import Corpus, Document
from bitig.features.sentence_length import SentenceLengthExtractor

pytestmark = pytest.mark.spacy


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_sentence_length_returns_mean_sd_skew(tmp_path: Path) -> None:
    ex = SentenceLengthExtractor(spacy_model="en_core_web_sm", cache_dir=tmp_path)
    fm = ex.fit_transform(
        _corpus("Short. A longer sentence. An even longer sentence with more words.")
    )
    assert set(fm.feature_names) == {"mean", "sd", "skew"}
    df = fm.as_dataframe()
    assert df.loc["d0", "mean"] > 0
    assert df.loc["d0", "sd"] >= 0


def test_sentence_length_uniform_has_zero_sd(tmp_path: Path) -> None:
    ex = SentenceLengthExtractor(spacy_model="en_core_web_sm", cache_dir=tmp_path)
    fm = ex.fit_transform(_corpus("Three words here. Three words here. Three words here."))
    assert fm.as_dataframe().loc["d0", "sd"] == 0.0
