"""Language registry: one frozen LanguageSpec per first-class language.

Every language-dependent site in bitig (preprocess pipeline, function-word loading, readability
index selection, embedding model defaults) reads from REGISTRY. Unknown codes raise early and
clearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LanguageSpec:
    """Static metadata for one first-class language."""

    code: str
    name: str
    default_model: str
    backend: Literal["spacy", "spacy_stanza"]
    readability_indices: tuple[str, ...]
    contextual_embedding_default: str
    sentence_embedding_default: str


REGISTRY: dict[str, LanguageSpec] = {
    "en": LanguageSpec(
        code="en",
        name="English",
        default_model="en_core_web_trf",
        backend="spacy",
        readability_indices=("flesch", "flesch_kincaid", "gunning_fog"),
        contextual_embedding_default="bert-base-uncased",
        sentence_embedding_default="sentence-transformers/all-mpnet-base-v2",
    ),
    "tr": LanguageSpec(
        code="tr",
        name="Turkish",
        default_model="tr",
        backend="spacy_stanza",
        readability_indices=("atesman", "bezirci_yilmaz"),
        contextual_embedding_default="dbmdz/bert-base-turkish-cased",
        sentence_embedding_default="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    ),
    "de": LanguageSpec(
        code="de",
        name="German",
        default_model="de_dep_news_trf",
        backend="spacy",
        readability_indices=("flesch_amstad", "wiener_sachtextformel"),
        contextual_embedding_default="deepset/gbert-base",
        sentence_embedding_default="deepset/gbert-base-sts",
    ),
    "es": LanguageSpec(
        code="es",
        name="Spanish",
        default_model="es_dep_news_trf",
        backend="spacy",
        readability_indices=("fernandez_huerta", "szigriszt_pazos"),
        contextual_embedding_default="dccuchile/bert-base-spanish-wwm-cased",
        sentence_embedding_default="hiiamsid/sentence_similarity_spanish_es",
    ),
    "fr": LanguageSpec(
        code="fr",
        name="French",
        default_model="fr_dep_news_trf",
        backend="spacy",
        readability_indices=("kandel_moles", "lix"),
        contextual_embedding_default="almanach/camembert-base",
        sentence_embedding_default="dangvantuan/sentence-camembert-base",
    ),
}


def get(code: str) -> LanguageSpec:
    """Return the LanguageSpec for `code` (case-insensitive). Raises ValueError if unknown."""
    normalized = code.lower()
    if normalized not in REGISTRY:
        supported = sorted(REGISTRY)
        raise ValueError(
            f"Unknown language code: {code!r}. Supported: {supported}. "
            f"To add a new language, extend bitig.languages.registry.REGISTRY."
        )
    return REGISTRY[normalized]
