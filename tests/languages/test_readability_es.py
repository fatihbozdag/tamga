"""Tests for Spanish readability — Fernández-Huerta (1959) and Szigriszt-Pazos (1992)."""

from bitig.languages.readability_es import (
    count_syllables_es,
    fernandez_huerta,
    szigriszt_pazos,
)


def test_count_syllables_es_basic() -> None:
    assert count_syllables_es("casa") == 2  # ca-sa
    assert count_syllables_es("libro") == 2  # li-bro
    assert count_syllables_es("camión") == 2  # ca-mión (diphthong collapsed)
    assert count_syllables_es("sol") == 1


def test_fernandez_huerta_scoring_sense() -> None:
    simple = "El gato duerme. El perro juega."
    complex_ = (
        "La sostenibilidad de las negociaciones diplomáticas internacionales requiere "
        "compromiso por ambas partes."
    )
    assert fernandez_huerta(simple) > fernandez_huerta(complex_)


def test_szigriszt_pazos_scoring_sense() -> None:
    simple = "El gato duerme. El perro juega."
    complex_ = (
        "La complejidad de las interrelaciones socioeconómicas globales requiere "
        "interdisciplinariedad."
    )
    assert szigriszt_pazos(simple) > szigriszt_pazos(complex_)


def test_spanish_readability_wired_into_extractor() -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.readability import ReadabilityExtractor

    c = Corpus(documents=[Document(id="d0", text="El gato duerme.")], language="es")
    fm = ReadabilityExtractor().fit_transform(c)
    assert set(fm.feature_names) == {"fernandez_huerta", "szigriszt_pazos"}


def test_empty_returns_zero() -> None:
    assert fernandez_huerta("") == 0.0
    assert szigriszt_pazos("") == 0.0
