"""Tests for CategorizedCharNgramExtractor and the classify_ngram helper."""

from __future__ import annotations

import numpy as np
import pytest

from bitig.corpus import Corpus, Document
from bitig.forensic.char_ngrams import (
    CategorizedCharNgramExtractor,
    classify_ngram,
)


class TestClassifyNgram:
    """The 7-category classifier on individual n-gram occurrences."""

    def test_punct_beats_everything_else(self) -> None:
        # Any punctuation character wins regardless of position.
        assert classify_ngram("th!", left="", right="") == "punct"
        assert classify_ngram("!er", left="t", right="e") == "punct"

    def test_whole_word_requires_both_boundaries_and_no_internal_space(self) -> None:
        assert classify_ngram("the", left=" ", right=" ") == "whole_word"
        assert classify_ngram("the", left="", right=" ") == "whole_word"
        assert classify_ngram("the", left=" ", right="") == "whole_word"

    def test_prefix_is_word_start_plus_internal(self) -> None:
        # "the" inside "there": left = space, right is a letter.
        assert classify_ngram("the", left=" ", right="r") == "prefix"
        assert classify_ngram("the", left="", right="r") == "prefix"

    def test_suffix_is_internal_plus_word_end(self) -> None:
        # "ing" inside "running": left is a letter, right = space.
        assert classify_ngram("ing", left="n", right=" ") == "suffix"
        assert classify_ngram("ing", left="n", right="") == "suffix"

    def test_mid_word_is_both_sides_internal(self) -> None:
        assert classify_ngram("her", left="t", right="e") == "mid_word"
        assert classify_ngram("pqr", left="o", right="s") == "mid_word"

    def test_multi_word_has_internal_space(self) -> None:
        # "e t" — space in the middle.
        assert classify_ngram("e t", left="h", right="h") == "multi_word"
        assert classify_ngram("t h", left=" ", right="e") == "multi_word"


class TestCategoryFiltering:
    """CategorizedCharNgramExtractor must keep only the requested categories."""

    @staticmethod
    def _corpus() -> Corpus:
        return Corpus(
            documents=[
                Document(id="d1", text="the cat sat. running fast! here we go."),
                Document(id="d2", text="a dog ran! running again; slowly moving."),
            ]
        )

    def test_default_keeps_all_categories(self) -> None:
        ex = CategorizedCharNgramExtractor(n=3)
        fm = ex.fit_transform(self._corpus())
        seen = {name.split("|")[1] for name in fm.feature_names}
        # Should include several categories (at minimum prefix, suffix, mid_word).
        assert {"prefix", "suffix", "mid_word"}.issubset(seen)

    def test_affix_only_filter(self) -> None:
        ex = CategorizedCharNgramExtractor(n=3, categories=("prefix", "suffix"))
        fm = ex.fit_transform(self._corpus())
        seen_categories = {name.split("|")[1] for name in fm.feature_names}
        assert seen_categories <= {"prefix", "suffix"}
        assert "mid_word" not in seen_categories
        assert "punct" not in seen_categories

    def test_punct_only_filter(self) -> None:
        ex = CategorizedCharNgramExtractor(n=3, categories=("punct",))
        fm = ex.fit_transform(self._corpus())
        seen = {name.split("|")[1] for name in fm.feature_names}
        assert seen == {"punct"}
        # Every retained feature must contain a punctuation character.
        import re as _re

        for name in fm.feature_names:
            ngram = name.split("|")[0]
            assert _re.search(r"[^\w\s]", ngram), f"{ngram!r} in punct column has no punct"

    def test_unknown_category_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown category"):
            CategorizedCharNgramExtractor(n=3, categories=("bogus",))  # type: ignore[arg-type]

    def test_empty_categories_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CategorizedCharNgramExtractor(n=3, categories=())

    def test_n_less_than_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            CategorizedCharNgramExtractor(n=0)


class TestExtractorContract:
    def test_output_is_feature_matrix_with_correct_shape(self) -> None:
        corpus = Corpus(
            documents=[
                Document(id="d1", text="the quick brown fox"),
                Document(id="d2", text="the slow brown dog"),
            ]
        )
        ex = CategorizedCharNgramExtractor(n=2, categories=("prefix", "suffix"))
        fm = ex.fit_transform(corpus)
        assert fm.X.shape[0] == 2
        assert fm.X.shape[1] == len(fm.feature_names)
        assert fm.document_ids == ["d1", "d2"]
        assert all("|" in name for name in fm.feature_names)

    def test_lowercase_option_merges_case(self) -> None:
        corpus = Corpus(
            documents=[Document(id="d1", text="The quick THE")],
        )
        lower = CategorizedCharNgramExtractor(n=3, lowercase=True).fit_transform(corpus)
        mixed = CategorizedCharNgramExtractor(n=3, lowercase=False).fit_transform(corpus)
        # Lowercased vocabulary is smaller (no duplicate "The|prefix" and "the|prefix").
        assert len(lower.feature_names) <= len(mixed.feature_names)

    def test_scale_zscore_produces_zero_mean(self) -> None:
        corpus = Corpus(
            documents=[Document(id=f"d{i}", text=f"the running fox number {i}") for i in range(5)]
        )
        ex = CategorizedCharNgramExtractor(n=3, scale="zscore", lowercase=True)
        fm = ex.fit_transform(corpus)
        # Training-time z-score means each column has ~zero mean.
        np.testing.assert_allclose(fm.X.mean(axis=0), 0.0, atol=1e-9)

    def test_deterministic_feature_order(self) -> None:
        corpus = Corpus(documents=[Document(id="d1", text="the quick brown fox")])
        ex_a = CategorizedCharNgramExtractor(n=3).fit(corpus)
        ex_b = CategorizedCharNgramExtractor(n=3).fit(corpus)
        assert ex_a._vocabulary == ex_b._vocabulary
