"""Tests for Turkish readability — Ateşman (1997) and Bezirci–Yılmaz (2010)."""

from bitig.languages.readability_tr import (
    atesman,
    bezirci_yilmaz,
    count_syllables_tr,
)


def test_count_syllables_simple_word() -> None:
    assert count_syllables_tr("merhaba") == 3  # mer-ha-ba
    assert count_syllables_tr("ev") == 1
    assert count_syllables_tr("öğretmen") == 3  # öğ-ret-men
    assert count_syllables_tr("İstanbul") == 3  # İs-tan-bul


def test_atesman_scoring_sense() -> None:
    """Shorter simpler Turkish prose → higher Ateşman. Reference: paper worked example."""
    simple = "Ali okula gitti. Kedi uyudu. Hava güzeldi."
    complex_ = (
        "Ülkelerarası diplomatik müzakerelerin sürdürülebilirliği, tarafların "
        "uzlaşmacı tutumlarını korumalarına bağlıdır."
    )
    assert atesman(simple) > atesman(complex_)


def test_atesman_range_is_plausible() -> None:
    text = "Ali topu tuttu. Kedi uyudu."
    score = atesman(text)
    assert -50 <= score <= 200  # plausible bounds for the Ateşman scale


def test_bezirci_yilmaz_scoring_sense() -> None:
    """Longer sentences + more polysyllabic words → higher Bezirci-Yılmaz score."""
    simple = "Ev büyük. Ağaç yeşil. Kedi uyur."
    complex_ = "Ülkelerarası diplomatik müzakerelerin sürdürülebilirliği uzun zaman gerektirir."
    assert bezirci_yilmaz(complex_) > bezirci_yilmaz(simple)


def test_atesman_handles_empty_and_single_word() -> None:
    assert atesman("") == 0.0
    assert isinstance(atesman("ev"), float)


def test_turkish_readability_wired_into_extractor() -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.readability import ReadabilityExtractor

    c = Corpus(
        documents=[Document(id="d0", text="Ali topu tuttu. Kedi uyudu.")],
        language="tr",
    )
    ex = ReadabilityExtractor()  # defaults to (atesman, bezirci_yilmaz)
    fm = ex.fit_transform(c)
    assert set(fm.feature_names) == {"atesman", "bezirci_yilmaz"}
    assert fm.X.shape == (1, 2)
