"""Stamatatos (2013) text distortion for topic-invariant authorship attribution.

Distortion pre-processes text by masking *content* while preserving *style* — function
words, punctuation, and the skeleton of spacing and word lengths. Features extracted from
distorted text (character n-grams, MFW, etc.) are decoupled from topic and therefore much
more robust in cross-domain forensic settings.

Two variants (Stamatatos 2013):

- **DV-MA** (Distortion View — Multiple Asterisks): every alphanumeric character of every
  content word is replaced by ``*``. Spaces, punctuation, and function words are kept
  verbatim. Preserves word *length* — so morphological habits (how long the writer's
  typical nouns are, say) remain visible to downstream features.
- **DV-SA** (Distortion View — Single Asterisk): every content word is collapsed to a
  single ``*``. Word length is lost; only the pattern of function words and punctuation
  remains. More aggressive distortion; useful when topic is strongly confounding.

Both variants target the *document text* — downstream feature extractors operate on the
distorted text without modification. Use :func:`distort_corpus` to produce a new Corpus,
then hand it to any tamga extractor.

References
----------
Stamatatos, E. (2013). On the robustness of authorship attribution based on character
    n-gram features. Journal of Law and Policy, 21(2), 421-439.
"""

from __future__ import annotations

import re
from importlib import resources
from typing import Literal

from tamga.corpus import Corpus, Document

DistortionMode = Literal["dv_ma", "dv_sa"]

# Match a "word" — a letter sequence, optionally followed by an apostrophe and another
# letter sequence, repeated any number of times. Keeps contractions intact: "don't", "you're",
# "it's", "let's", "we'll", "they've", "o'clock" all match as single tokens so the
# function-word lookup can preserve them verbatim.
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", flags=re.UNICODE)


def _load_bundled_function_words() -> frozenset[str]:
    path = resources.files("tamga.resources") / "function_words_en.txt"
    words = (line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines())
    return frozenset(w for w in words if w)


def _ensure_function_words(
    function_words: frozenset[str] | set[str] | list[str] | None,
) -> frozenset[str]:
    if function_words is None:
        return _load_bundled_function_words()
    return frozenset(w.lower() for w in function_words)


def distort_text(
    text: str,
    *,
    mode: DistortionMode = "dv_ma",
    function_words: frozenset[str] | set[str] | list[str] | None = None,
) -> str:
    """Apply Stamatatos distortion to a single string.

    Parameters
    ----------
    text : str
        Input text.
    mode : {"dv_ma", "dv_sa"}
        Distortion variant. ``dv_ma`` preserves word length; ``dv_sa`` collapses each
        content word to one ``*``.
    function_words : iterable of str, optional
        Words to preserve verbatim. If None, uses tamga's bundled English list.

    Returns
    -------
    str
        The distorted text — identical length to the input for DV-MA, shorter for DV-SA.
        All non-word characters (spaces, punctuation, digits) are preserved.
    """
    if mode not in ("dv_ma", "dv_sa"):
        raise ValueError(f"unknown distortion mode {mode!r}; expected dv_ma or dv_sa")
    fw = _ensure_function_words(function_words)

    def _replace(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.lower() in fw:
            return word
        if mode == "dv_ma":
            return "*" * len(word)
        return "*"

    return _TOKEN_RE.sub(_replace, text)


def distort_corpus(
    corpus: Corpus,
    *,
    mode: DistortionMode = "dv_ma",
    function_words: frozenset[str] | set[str] | list[str] | None = None,
) -> Corpus:
    """Produce a new Corpus with each document's text distorted.

    Document ids and metadata are preserved unchanged; ``metadata["distortion_mode"]`` is
    set on each new Document to record how it was produced.

    Parameters
    ----------
    corpus : Corpus
    mode : {"dv_ma", "dv_sa"}
    function_words : iterable of str, optional
    """
    fw = _ensure_function_words(function_words)
    new_docs = []
    for doc in corpus.documents:
        new_text = distort_text(doc.text, mode=mode, function_words=fw)
        new_metadata = dict(doc.metadata)
        new_metadata["distortion_mode"] = mode
        new_docs.append(Document(id=doc.id, text=new_text, metadata=new_metadata))
    return Corpus(documents=new_docs)
