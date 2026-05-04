"""Turkish readability formulas.

Ateşman, E. (1997). Türkçede okunabilirliğin ölçülmesi. Dil Dergisi, 58, 71-74.
Bezirci, B., & Yılmaz, A. E. (2010). Metinlerin okunabilirliğinin ölçülmesi üzerine bir yazılım
kütüphanesi ve Türkçe için yeni bir okunabilirlik ölçütü. Dokuz Eylül Üniversitesi Mühendislik
Fakültesi Fen ve Mühendislik Dergisi, 12(3), 49-62.

Turkish syllables are counted by vowel nuclei. Vowel set: {a, e, ı, i, o, ö, u, ü} and their
uppercase forms. Sentence boundaries are `[.!?…]+`.
"""

from __future__ import annotations

import re
from math import sqrt

_TURKISH_VOWELS = set("aeıioöuüAEIİOÖUÜ")
_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?…]+")


def count_syllables_tr(word: str) -> int:
    """Count Turkish syllables in a single word (= number of vowels)."""
    return sum(1 for c in word if c in _TURKISH_VOWELS)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _sentence_count(text: str) -> int:
    # Count non-empty splits.
    parts = [p for p in _SENTENCE_RE.split(text) if p.strip()]
    return max(1, len(parts))


def atesman(text: str) -> float:
    """Ateşman (1997) — Flesch-analogue for Turkish.

        score = 198.825 - 40.175 * (syllables/words) - 2.610 * (words/sentences)

    Higher = easier. Plausible range ~0-110 for typical Turkish prose.
    """
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_syllables = sum(count_syllables_tr(w) for w in words)
    n_sentences = _sentence_count(text)
    return 198.825 - 40.175 * (n_syllables / n_words) - 2.610 * (n_words / n_sentences)


def bezirci_yilmaz(text: str) -> float:
    """Bezirci & Yılmaz (2010) — weighted polysyllabic measure for Turkish.

        score = sqrt(avg_words_per_sentence * (h3*0.84 + h4*1.5 + h5*3.5 + h6*26.25))

    where `h_k` is the fraction of words with `k` syllables (words with ≥7 syllables are binned
    with 6-syllable words for weighting purposes). Higher = harder to read.
    """
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_sentences = _sentence_count(text)

    counts = {3: 0, 4: 0, 5: 0, 6: 0}
    for w in words:
        syl = count_syllables_tr(w)
        if syl >= 6:
            counts[6] += 1
        elif syl in counts:
            counts[syl] += 1

    h3 = counts[3] / n_words
    h4 = counts[4] / n_words
    h5 = counts[5] / n_words
    h6 = counts[6] / n_words

    avg_wps = n_words / n_sentences
    weighted = h3 * 0.84 + h4 * 1.5 + h5 * 3.5 + h6 * 26.25
    return sqrt(avg_wps * weighted)
