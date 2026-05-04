"""Tests for LexicalDiversityExtractor."""

import numpy as np

from bitig.corpus import Corpus, Document
from bitig.features.lexical_diversity import LexicalDiversityExtractor


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_ttr_is_one_for_all_unique_words() -> None:
    ex = LexicalDiversityExtractor(indices=["ttr"])
    fm = ex.fit_transform(_corpus("the quick brown fox"))
    assert fm.as_dataframe().loc["d0", "ttr"] == 1.0


def test_ttr_is_low_for_repetitive_text() -> None:
    ex = LexicalDiversityExtractor(indices=["ttr"])
    fm = ex.fit_transform(_corpus("the the the the the the"))
    # 1 unique / 6 total = 0.1667
    assert fm.as_dataframe().loc["d0", "ttr"] < 0.2


def test_multiple_indices_produce_multiple_columns() -> None:
    ex = LexicalDiversityExtractor(indices=["ttr", "yules_k"])
    fm = ex.fit_transform(_corpus("the quick brown fox jumped over the lazy dog"))
    assert set(fm.feature_names) == {"ttr", "yules_k"}


def test_ldiv_feature_matrix_is_2d_numeric() -> None:
    ex = LexicalDiversityExtractor(indices=["ttr"])
    fm = ex.fit_transform(_corpus("a b c", "a a a"))
    assert fm.X.shape == (2, 1)
    assert np.issubdtype(fm.X.dtype, np.floating)
