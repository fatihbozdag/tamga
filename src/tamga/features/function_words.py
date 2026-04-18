"""Function-word frequency extractor with a bundled English word list."""

from __future__ import annotations

import re
from importlib import resources
from typing import Literal

import numpy as np

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor

Scale = Literal["none", "zscore", "l1", "l2"]

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def _load_bundled_list() -> list[str]:
    path = resources.files("tamga.resources.languages.en") / "function_words.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FunctionWordExtractor(BaseFeatureExtractor):
    feature_type = "function_word"

    def __init__(
        self,
        *,
        wordlist: list[str] | None = None,
        scale: Scale = "none",
    ) -> None:
        self.wordlist = wordlist
        self.scale = scale
        self._words: list[str] = []

    def _fit(self, corpus: Corpus) -> None:
        # Vocabulary comes from the wordlist, not the corpus.
        del corpus
        self._words = list(self.wordlist) if self.wordlist is not None else _load_bundled_list()

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        index = {w: i for i, w in enumerate(self._words)}
        X = np.zeros((len(corpus), len(self._words)), dtype=float)  # noqa: N806
        for row, doc in enumerate(corpus.documents):
            for tok in _WORD_RE.findall(doc.text.lower()):
                if tok in index:
                    X[row, index[tok]] += 1
        if self.scale == "l1":
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            X = X / row_sums  # noqa: N806
        elif self.scale == "l2":
            row_norms = np.linalg.norm(X, axis=1, keepdims=True)
            row_norms[row_norms == 0] = 1.0
            X = X / row_norms  # noqa: N806
        # "zscore" scaling for FWs is less common than for MFW — support it but no fitted stats needed for "none"/"l1"/"l2".
        return X, list(self._words)
