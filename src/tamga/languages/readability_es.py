"""Spanish readability formulas.

Fernández-Huerta, J. (1959). Medidas sencillas de lecturabilidad. Consigna, 214, 29-32.
Szigriszt-Pazos, F. (1992). Sistemas predictivos de legibilidad del mensaje escrito: fórmula de
perspicuidad. PhD thesis, Universidad Complutense de Madrid. (a.k.a. INFLESZ)

Syllable counting: count of vowel groups (1+ consecutive vowel characters). This is a standard
approximation sufficient for readability-formula inputs.
"""

from __future__ import annotations

import re

_SPANISH_VOWELS = "aeiouáéíóúüAEIOUÁÉÍÓÚÜ"
_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?…]+")
_VOWEL_GROUP_RE = re.compile(f"[{_SPANISH_VOWELS}]+")


def count_syllables_es(word: str) -> int:
    """Count Spanish syllables (vowel groups)."""
    return len(_VOWEL_GROUP_RE.findall(word))


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _sentence_count(text: str) -> int:
    parts = [p for p in _SENTENCE_RE.split(text) if p.strip()]
    return max(1, len(parts))


def fernandez_huerta(text: str) -> float:
    """Fernández-Huerta (1959): 206.84 - 60*(syll/word) - 1.02*(word/sent)."""
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_syllables = sum(count_syllables_es(w) for w in words)
    n_sentences = _sentence_count(text)
    return 206.84 - 60.0 * (n_syllables / n_words) - 1.02 * (n_words / n_sentences)


def szigriszt_pazos(text: str) -> float:
    """Szigriszt-Pazos (1992), a.k.a. INFLESZ: 206.835 - 62.3*(syll/word) - (word/sent)."""
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_syllables = sum(count_syllables_es(w) for w in words)
    n_sentences = _sentence_count(text)
    return 206.835 - 62.3 * (n_syllables / n_words) - (n_words / n_sentences)
