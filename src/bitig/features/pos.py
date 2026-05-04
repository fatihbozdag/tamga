"""POS n-gram feature extractor — spaCy-backed tokens tagged with Universal or fine POS labels."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np

from bitig.corpus import Corpus
from bitig.features.base import BaseFeatureExtractor
from bitig.preprocess.pipeline import SpacyPipeline

Tagset = Literal["coarse", "fine"]


class PosNgramExtractor(BaseFeatureExtractor):
    feature_type = "pos_ngram"

    def __init__(
        self,
        n: int = 2,
        *,
        tagset: Tagset = "coarse",
        spacy_model: str = "en_core_web_trf",
        cache_dir: str | Path = ".bitig/cache/docbin",
    ) -> None:
        self.n = n
        self.tagset = tagset
        self.spacy_model = spacy_model
        # Stored as str so that sklearn `get_params()` yields a JSON-serialisable config
        # (used by the FeatureMatrix provenance hash).
        self.cache_dir = str(cache_dir)
        self._vocabulary: list[str] = []
        self._pipeline: SpacyPipeline | None = None

    def _pipe(self) -> SpacyPipeline:
        if self._pipeline is None:
            self._pipeline = SpacyPipeline(model=self.spacy_model, cache_dir=self.cache_dir)
        return self._pipeline

    def _tag(self, token: object) -> str:
        return token.pos_ if self.tagset == "coarse" else token.tag_  # type: ignore[attr-defined, no-any-return]

    def _ngrams(self, tags: list[str]) -> list[str]:
        if self.n == 1:
            return tags
        return ["|".join(tags[i : i + self.n]) for i in range(len(tags) - self.n + 1)]

    def _fit(self, corpus: Corpus) -> None:
        parsed = self._pipe().parse(corpus)
        vocab_counter: Counter[str] = Counter()
        for spacy_doc in parsed.spacy_docs():
            tags = [self._tag(t) for t in spacy_doc if not t.is_space]
            vocab_counter.update(self._ngrams(tags))
        self._vocabulary = sorted(vocab_counter.keys())

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        index = {ng: i for i, ng in enumerate(self._vocabulary)}
        parsed = self._pipe().parse(corpus)
        X = np.zeros((len(corpus), len(self._vocabulary)), dtype=float)  # noqa: N806
        for row, spacy_doc in enumerate(parsed.spacy_docs()):
            tags = [self._tag(t) for t in spacy_doc if not t.is_space]
            for ng in self._ngrams(tags):
                if ng in index:
                    X[row, index[ng]] += 1
        return X, list(self._vocabulary)
