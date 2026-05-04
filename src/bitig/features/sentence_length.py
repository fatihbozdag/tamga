"""Sentence-length distribution features: mean, standard deviation, skew."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import skew

from bitig.corpus import Corpus
from bitig.features.base import BaseFeatureExtractor
from bitig.preprocess.pipeline import SpacyPipeline


class SentenceLengthExtractor(BaseFeatureExtractor):
    feature_type = "sentence_length"

    def __init__(
        self,
        *,
        spacy_model: str = "en_core_web_trf",
        cache_dir: str | Path = ".bitig/cache/docbin",
    ) -> None:
        self.spacy_model = spacy_model
        # Stored as str so that sklearn `get_params()` yields a JSON-serialisable config
        # (used by the FeatureMatrix provenance hash).
        self.cache_dir = str(cache_dir)
        self._pipeline: SpacyPipeline | None = None

    def _pipe(self) -> SpacyPipeline:
        if self._pipeline is None:
            self._pipeline = SpacyPipeline(model=self.spacy_model, cache_dir=self.cache_dir)
        return self._pipeline

    def _fit(self, corpus: Corpus) -> None:
        del corpus

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        parsed = self._pipe().parse(corpus)
        X = np.zeros((len(corpus), 3), dtype=float)  # noqa: N806
        for row, spacy_doc in enumerate(parsed.spacy_docs()):
            lengths = [sum(1 for t in sent if not t.is_space) for sent in spacy_doc.sents]
            if not lengths:
                continue
            X[row, 0] = float(np.mean(lengths))
            X[row, 1] = float(np.std(lengths, ddof=0))
            # scipy.stats.skew warns on catastrophic cancellation when values are identical;
            # guard with a variance check to keep the output well-defined (skew is 0 for constant data).
            X[row, 2] = float(skew(lengths)) if len(lengths) > 2 and np.var(lengths) > 0 else 0.0
        return X, ["mean", "sd", "skew"]
