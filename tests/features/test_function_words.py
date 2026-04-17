"""Tests for FunctionWordExtractor."""

from tamga.corpus import Corpus, Document
from tamga.features.function_words import FunctionWordExtractor


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_function_word_counts_the_and_of() -> None:
    ex = FunctionWordExtractor(scale="none")
    fm = ex.fit_transform(_corpus("the cat of the dog"))
    df = fm.as_dataframe()
    assert df.loc["d0", "the"] == 2
    assert df.loc["d0", "of"] == 1


def test_function_word_ignores_content_words() -> None:
    ex = FunctionWordExtractor(scale="none")
    fm = ex.fit_transform(_corpus("cat dog bird"))
    # No function words → all-zero row.
    assert fm.X.sum() == 0


def test_function_word_uses_bundled_list_by_default() -> None:
    ex = FunctionWordExtractor(scale="none")
    fm = ex.fit_transform(_corpus("a test"))
    # Bundled EN list contains "a".
    assert "a" in fm.feature_names


def test_function_word_accepts_custom_list() -> None:
    ex = FunctionWordExtractor(wordlist=["custom"], scale="none")
    fm = ex.fit_transform(_corpus("a custom word"))
    assert list(fm.feature_names) == ["custom"]
