"""Tests for PunctuationExtractor."""

from bitig.corpus import Corpus, Document
from bitig.features.punctuation import PunctuationExtractor


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_punctuation_counts_periods_and_commas() -> None:
    ex = PunctuationExtractor()
    fm = ex.fit_transform(_corpus("Hello, world. Hello, again."))
    df = fm.as_dataframe()
    assert df.loc["d0", ","] == 2
    assert df.loc["d0", "."] == 2


def test_punctuation_ignores_word_characters() -> None:
    ex = PunctuationExtractor()
    fm = ex.fit_transform(_corpus("nowordschars"))
    assert fm.X.sum() == 0


def test_punctuation_includes_question_exclamation() -> None:
    ex = PunctuationExtractor()
    fm = ex.fit_transform(_corpus("What? Really!"))
    df = fm.as_dataframe()
    assert df.loc["d0", "?"] == 1
    assert df.loc["d0", "!"] == 1
