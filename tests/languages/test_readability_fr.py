"""Tests for French readability — Kandel-Moles (1958) and LIX (Björnsson 1968)."""

from bitig.languages.readability_fr import count_syllables_fr, kandel_moles, lix


def test_count_syllables_fr_basic() -> None:
    assert count_syllables_fr("bonjour") >= 2  # bon-jour
    assert count_syllables_fr("chat") == 1
    assert count_syllables_fr("anticonstitutionnellement") > 6


def test_kandel_moles_scoring_sense() -> None:
    simple = "Le chat dort. Le chien court."
    complex_ = (
        "La pérennisation des négociations diplomatiques internationales requiert "
        "des compromis substantiels de toutes les parties prenantes."
    )
    assert kandel_moles(simple) > kandel_moles(complex_)


def test_lix_scoring_sense() -> None:
    simple = "Le chat dort. Le chien joue."
    complex_ = (
        "La pérennisation des négociations diplomatiques internationales contemporaines "
        "constitue un défi multidimensionnel."
    )
    assert lix(complex_) > lix(simple)


def test_french_readability_wired_into_extractor() -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.readability import ReadabilityExtractor

    c = Corpus(documents=[Document(id="d0", text="Le chat dort.")], language="fr")
    fm = ReadabilityExtractor().fit_transform(c)
    assert set(fm.feature_names) == {"kandel_moles", "lix"}


def test_empty_returns_zero() -> None:
    assert kandel_moles("") == 0.0
    assert lix("") == 0.0
