"""Readability indices, dispatched per-language.

English uses textstat wrappers (unchanged). Non-English languages use native implementations in
tamga.languages.readability_<code>, registered here. The per-language registry is populated in
Phase 5; this task wires only English.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import textstat

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor
from tamga.languages import get_language
from tamga.languages.readability_de import flesch_amstad as _de_flesch_amstad
from tamga.languages.readability_de import wiener_sachtextformel as _de_wst
from tamga.languages.readability_tr import atesman as _tr_atesman
from tamga.languages.readability_tr import bezirci_yilmaz as _tr_bezirci_yilmaz

# {language_code: {index_name: callable(text) -> float}}
_INDEX_REGISTRY: dict[str, dict[str, Callable[[str], float]]] = {
    "en": {
        "flesch": textstat.flesch_reading_ease,
        "flesch_kincaid": textstat.flesch_kincaid_grade,
        "gunning_fog": textstat.gunning_fog,
        "coleman_liau": textstat.coleman_liau_index,
        "ari": textstat.automated_readability_index,
        "smog": textstat.smog_index,
    },
    "tr": {
        "atesman": _tr_atesman,
        "bezirci_yilmaz": _tr_bezirci_yilmaz,
    },
    "de": {
        "flesch_amstad": _de_flesch_amstad,
        "wiener_sachtextformel": _de_wst,
    },
    "es": {},  # Task 5.3
    "fr": {},  # Task 5.4
}


class ReadabilityExtractor(BaseFeatureExtractor):
    feature_type = "readability"

    def __init__(
        self,
        indices: list[str] | tuple[str, ...] | None = None,
        *,
        language: str | None = None,
    ) -> None:
        self.indices = list(indices) if indices is not None else None
        self.language = language
        self._resolved_indices: list[str] = []
        self._fns: list[Callable[[str], float]] = []

    def _fit(self, corpus: Corpus) -> None:
        lang = self.language or corpus.language
        spec = get_language(lang)
        available = _INDEX_REGISTRY.get(lang, {})

        if self.indices is None:
            self._resolved_indices = list(spec.readability_indices)
        else:
            self._resolved_indices = list(self.indices)

        unknown = [i for i in self._resolved_indices if i not in available]
        if unknown:
            raise ValueError(
                f"Readability indices {unknown} not available for language {lang!r}. "
                f"Available for {lang!r}: {sorted(available)}."
            )
        self._fns = [available[i] for i in self._resolved_indices]

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        X = np.zeros((len(corpus), len(self._resolved_indices)), dtype=float)  # noqa: N806
        for row, doc in enumerate(corpus.documents):
            for col, fn in enumerate(self._fns):
                X[row, col] = float(fn(doc.text))
        return X, list(self._resolved_indices)
