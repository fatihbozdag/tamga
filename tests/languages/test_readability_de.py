"""Tests for German readability — Flesch-Amstad (1978) and Wiener Sachtextformel."""

from tamga.languages.readability_de import (
    count_syllables_de,
    flesch_amstad,
    wiener_sachtextformel,
)


def test_count_syllables_de_basic() -> None:
    assert count_syllables_de("Haus") == 1
    assert count_syllables_de("Computer") == 3  # Com-pu-ter
    assert count_syllables_de("schwimmen") >= 2  # schwim-men


def test_flesch_amstad_scoring_sense() -> None:
    simple = "Der Hund bellt. Die Katze schläft."
    complex_ = (
        "Die Aufrechterhaltung diplomatischer Verhandlungen erfordert "
        "Kompromissbereitschaft auf beiden Seiten."
    )
    assert flesch_amstad(simple) > flesch_amstad(complex_)


def test_wiener_sachtextformel_scoring_sense() -> None:
    simple = "Der Hund bellt. Die Katze schläft. Die Sonne scheint."
    complex_ = (
        "Die Aufrechterhaltung sozialwirtschaftlicher Gleichgewichtsbedingungen "
        "erfordert interdisziplinäre Kooperationsbereitschaft."
    )
    assert wiener_sachtextformel(complex_) > wiener_sachtextformel(simple)


def test_german_readability_wired_into_extractor() -> None:
    from tamga.corpus import Corpus, Document
    from tamga.features.readability import ReadabilityExtractor

    c = Corpus(
        documents=[Document(id="d0", text="Der Hund bellt.")],
        language="de",
    )
    ex = ReadabilityExtractor()
    fm = ex.fit_transform(c)
    assert set(fm.feature_names) == {"flesch_amstad", "wiener_sachtextformel"}


def test_flesch_amstad_empty_returns_zero() -> None:
    assert flesch_amstad("") == 0.0
    assert wiener_sachtextformel("") == 0.0
