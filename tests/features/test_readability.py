"""Tests for ReadabilityExtractor."""

from tamga.corpus import Corpus, Document
from tamga.features.readability import ReadabilityExtractor


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_flesch_reading_ease_for_simple_text() -> None:
    ex = ReadabilityExtractor(indices=["flesch"])
    fm = ex.fit_transform(_corpus("The cat sat on the mat. It was a warm sunny day."))
    # Simple prose should score high (closer to 100 = very easy).
    assert fm.as_dataframe().loc["d0", "flesch"] > 70


def test_multiple_readability_indices() -> None:
    ex = ReadabilityExtractor(indices=["flesch", "flesch_kincaid", "gunning_fog"])
    fm = ex.fit_transform(_corpus("A simple sentence here. Another simple one."))
    assert set(fm.feature_names) == {"flesch", "flesch_kincaid", "gunning_fog"}


def test_readability_per_document() -> None:
    ex = ReadabilityExtractor(indices=["flesch"])
    fm = ex.fit_transform(_corpus("Simple text.", "Another text."))
    assert fm.X.shape == (2, 1)
