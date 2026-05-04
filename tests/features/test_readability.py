"""Tests for ReadabilityExtractor."""

from bitig.corpus import Corpus, Document
from bitig.features.readability import ReadabilityExtractor


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


def test_readability_resolves_english_defaults_from_registry() -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.readability import ReadabilityExtractor

    c = Corpus(documents=[Document(id="d0", text="A short simple sentence.")], language="en")
    ex = ReadabilityExtractor()  # indices=None
    fm = ex.fit_transform(c)
    assert set(fm.feature_names) == {"flesch", "flesch_kincaid", "gunning_fog"}


def test_readability_rejects_unsupported_index_for_language() -> None:
    import pytest

    from bitig.corpus import Corpus, Document
    from bitig.features.readability import ReadabilityExtractor

    c = Corpus(documents=[Document(id="d0", text="x")], language="en")
    ex = ReadabilityExtractor(indices=["atesman"])  # Turkish index on English corpus
    with pytest.raises(ValueError, match="not available for language 'en'"):
        ex.fit_transform(c)


def test_readability_explicit_language_overrides_corpus() -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.readability import ReadabilityExtractor

    # Even if corpus is stamped 'tr', a user may override by passing language='en'.
    c = Corpus(documents=[Document(id="d0", text="A simple sentence.")], language="tr")
    ex = ReadabilityExtractor(indices=["flesch"], language="en")
    fm = ex.fit_transform(c)
    assert "flesch" in fm.feature_names
