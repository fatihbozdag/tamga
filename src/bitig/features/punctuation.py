"""Punctuation-symbol frequency extractor."""

from __future__ import annotations

import string

import numpy as np

from bitig.corpus import Corpus
from bitig.features.base import BaseFeatureExtractor

_PUNCT = sorted(string.punctuation)  # deterministic column order


class PunctuationExtractor(BaseFeatureExtractor):
    feature_type = "punctuation"

    def __init__(self) -> None:
        self._symbols: list[str] = list(_PUNCT)

    def _fit(self, corpus: Corpus) -> None:
        # Vocabulary is fixed (the ASCII punctuation set); no fitting needed.
        del corpus

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        index = {s: i for i, s in enumerate(self._symbols)}
        X = np.zeros((len(corpus), len(self._symbols)), dtype=float)  # noqa: N806
        for row, doc in enumerate(corpus.documents):
            for ch in doc.text:
                if ch in index:
                    X[row, index[ch]] += 1
        return X, list(self._symbols)
