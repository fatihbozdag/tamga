"""Readability indices via the `textstat` library: Flesch, Flesch-Kincaid, Gunning Fog, Coleman-Liau, ARI, SMOG."""

from __future__ import annotations

import numpy as np
import textstat

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor

_INDEX_FN = {
    "flesch": textstat.flesch_reading_ease,
    "flesch_kincaid": textstat.flesch_kincaid_grade,
    "gunning_fog": textstat.gunning_fog,
    "coleman_liau": textstat.coleman_liau_index,
    "ari": textstat.automated_readability_index,
    "smog": textstat.smog_index,
}

_DEFAULT_INDICES = ("flesch", "flesch_kincaid", "gunning_fog")


class ReadabilityExtractor(BaseFeatureExtractor):
    feature_type = "readability"

    def __init__(self, indices: list[str] | tuple[str, ...] = _DEFAULT_INDICES) -> None:
        self.indices = list(indices)

    def _fit(self, corpus: Corpus) -> None:
        del corpus
        unknown = [i for i in self.indices if i not in _INDEX_FN]
        if unknown:
            raise ValueError(f"ReadabilityExtractor: unknown indices {unknown}")

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        X = np.zeros((len(corpus), len(self.indices)), dtype=float)  # noqa: N806
        for row, doc in enumerate(corpus.documents):
            for col, index in enumerate(self.indices):
                X[row, col] = float(_INDEX_FN[index](doc.text))
        return X, list(self.indices)
