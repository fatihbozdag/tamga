"""Character and word n-gram feature extractors.

Both accept `n` as either an int (fixed order) or a tuple `(min_n, max_n)` (range).
Internally they delegate to sklearn's `CountVectorizer` — a well-tested, fast, battle-hardened
implementation — and present the same FeatureMatrix envelope as our other extractors.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor

Scale = Literal["none", "zscore", "l1", "l2"]


def _coerce_range(n: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(n, int):
        return (n, n)
    return n


def _apply_scale(X: np.ndarray, scale: Scale) -> np.ndarray:  # noqa: N803 (sklearn convention)
    if scale == "none":
        return X
    if scale == "l1":
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return X / row_sums  # type: ignore[no-any-return]
    if scale == "l2":
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        return X / row_norms  # type: ignore[no-any-return]
    # zscore applied by caller (needs fitted stats).
    return X


class CharNgramExtractor(BaseFeatureExtractor):
    feature_type = "char_ngram"

    def __init__(
        self,
        n: int | tuple[int, int] = 3,
        *,
        include_boundaries: bool = False,
        scale: Scale = "none",
    ) -> None:
        self.n = n
        self.include_boundaries = include_boundaries
        self.scale = scale
        self._vectorizer: CountVectorizer | None = None
        self._column_means: np.ndarray | None = None
        self._column_stds: np.ndarray | None = None

    def _fit(self, corpus: Corpus) -> None:
        analyzer = "char_wb" if self.include_boundaries else "char"
        self._vectorizer = CountVectorizer(
            analyzer=analyzer,
            ngram_range=_coerce_range(self.n),
            lowercase=False,
        )
        texts = [d.text for d in corpus.documents]
        X = self._vectorizer.fit_transform(texts).toarray().astype(float)  # noqa: N806
        if self.scale == "zscore":
            self._column_means = X.mean(axis=0)
            stds = X.std(axis=0, ddof=0)
            stds[stds == 0] = 1.0
            self._column_stds = stds

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        assert self._vectorizer is not None
        texts = [d.text for d in corpus.documents]
        X = self._vectorizer.transform(texts).toarray().astype(float)  # noqa: N806
        if self.scale == "zscore":
            assert self._column_means is not None and self._column_stds is not None
            X = (X - self._column_means) / self._column_stds  # noqa: N806
        else:
            X = _apply_scale(X, self.scale)  # noqa: N806
        return X, list(self._vectorizer.get_feature_names_out())


class WordNgramExtractor(BaseFeatureExtractor):
    feature_type = "word_ngram"

    def __init__(
        self,
        n: int | tuple[int, int] = 1,
        *,
        lowercase: bool = False,
        scale: Scale = "none",
    ) -> None:
        self.n = n
        self.lowercase = lowercase
        self.scale = scale
        self._vectorizer: CountVectorizer | None = None
        self._column_means: np.ndarray | None = None
        self._column_stds: np.ndarray | None = None

    def _fit(self, corpus: Corpus) -> None:
        self._vectorizer = CountVectorizer(
            analyzer="word",
            ngram_range=_coerce_range(self.n),
            lowercase=self.lowercase,
            token_pattern=r"(?u)\b\w+\b",
        )
        texts = [d.text for d in corpus.documents]
        X = self._vectorizer.fit_transform(texts).toarray().astype(float)  # noqa: N806
        if self.scale == "zscore":
            self._column_means = X.mean(axis=0)
            stds = X.std(axis=0, ddof=0)
            stds[stds == 0] = 1.0
            self._column_stds = stds

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        assert self._vectorizer is not None
        texts = [d.text for d in corpus.documents]
        X = self._vectorizer.transform(texts).toarray().astype(float)  # noqa: N806
        if self.scale == "zscore":
            assert self._column_means is not None and self._column_stds is not None
            X = (X - self._column_means) / self._column_stds  # noqa: N806
        else:
            X = _apply_scale(X, self.scale)  # noqa: N806
        return X, list(self._vectorizer.get_feature_names_out())
