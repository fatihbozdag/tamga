"""German readability formulas.

Amstad, T. (1978). Wie verständlich sind unsere Zeitungen? Zurich: Studenten-Schreib-Service.
Bamberger, R., & Vanecek, E. (1984). Lesen — Verstehen — Lernen — Schreiben: Die
Schwierigkeitsstufen von Texten in deutscher Sprache. Wien: Jugend und Volk.

Syllable count uses pyphen's German hyphenation dictionary (de_DE).
"""

from __future__ import annotations

import re

import pyphen

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?…]+")
_PYPHEN_DE = pyphen.Pyphen(lang="de_DE")


def count_syllables_de(word: str) -> int:
    """Count German syllables via Liang-hyphenation (pyphen de_DE)."""
    if not word:
        return 0
    # Pyphen returns hyphens between syllables; count chunks.
    return len(_PYPHEN_DE.inserted(word).split("-"))


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _sentence_count(text: str) -> int:
    parts = [p for p in _SENTENCE_RE.split(text) if p.strip()]
    return max(1, len(parts))


def flesch_amstad(text: str) -> float:
    """Flesch-Amstad (1978): 180 - ASL - 58.5 * ASW.

    ASL = avg sentence length (words); ASW = avg syllables per word. Higher = easier.
    """
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_sentences = _sentence_count(text)
    n_syllables = sum(count_syllables_de(w) for w in words)
    asl = n_words / n_sentences
    asw = n_syllables / n_words
    return 180.0 - asl - 58.5 * asw


def wiener_sachtextformel(text: str) -> float:
    """Wiener Sachtextformel I (Bamberger & Vanecek 1984).

    Formula: 0.1935 * MS + 0.1672 * SL + 0.1297 * IW - 0.0327 * ES - 0.875
      MS = percent of words with ≥3 syllables
      SL = average sentence length
      IW = percent of words with >6 letters
      ES = percent of monosyllabic words

    Result is a school-grade (roughly 4-15); higher = harder.
    """
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_sentences = _sentence_count(text)
    sl = n_words / n_sentences

    n_ms = sum(1 for w in words if count_syllables_de(w) >= 3)
    n_iw = sum(1 for w in words if len(w) > 6)
    n_es = sum(1 for w in words if count_syllables_de(w) == 1)

    ms_pct = 100.0 * n_ms / n_words
    iw_pct = 100.0 * n_iw / n_words
    es_pct = 100.0 * n_es / n_words

    return 0.1935 * ms_pct + 0.1672 * sl + 0.1297 * iw_pct - 0.0327 * es_pct - 0.875
