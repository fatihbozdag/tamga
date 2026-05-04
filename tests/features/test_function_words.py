"""Tests for FunctionWordExtractor."""

from bitig.corpus import Corpus, Document
from bitig.features.function_words import FunctionWordExtractor


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


def test_function_word_uses_corpus_language_turkish() -> None:
    """Turkish corpus → Turkish function-word list loaded."""
    from bitig.corpus import Corpus, Document
    from bitig.features.function_words import FunctionWordExtractor

    c = Corpus(documents=[Document(id="d0", text="Ben ve sen gittik.")], language="tr")
    ex = FunctionWordExtractor(scale="none")
    fm = ex.fit_transform(c)
    # "ve" is among the most frequent Turkish function words — must appear.
    assert "ve" in fm.feature_names


def test_function_word_list_bundled_for_all_five_languages() -> None:
    from bitig.features.function_words import _load_bundled_list

    for lang in ["en", "tr", "de", "es", "fr"]:
        words = _load_bundled_list(lang)
        assert len(words) >= 50, f"{lang} list has only {len(words)} entries"


def test_function_word_explicit_language_overrides_corpus() -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.function_words import FunctionWordExtractor

    c = Corpus(documents=[Document(id="d0", text="the cat")], language="tr")
    ex = FunctionWordExtractor(scale="none", language="en")
    fm = ex.fit_transform(c)
    assert "the" in fm.feature_names


def test_function_word_wordlist_overrides_everything() -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.function_words import FunctionWordExtractor

    c = Corpus(documents=[Document(id="d0", text="foo bar")], language="tr")
    ex = FunctionWordExtractor(wordlist=["foo"], language="en", scale="none")
    fm = ex.fit_transform(c)
    assert list(fm.feature_names) == ["foo"]
