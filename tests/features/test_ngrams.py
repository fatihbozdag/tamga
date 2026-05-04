"""Tests for Char/Word n-gram extractors."""

from bitig.corpus import Corpus, Document
from bitig.features.ngrams import CharNgramExtractor, WordNgramExtractor


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_char_ngram_extracts_bigrams():
    corpus = _corpus("abc", "abc")
    ex = CharNgramExtractor(n=2, scale="none")
    fm = ex.fit_transform(corpus)
    # "abc" has char-bigrams: "ab", "bc"
    assert set(fm.feature_names) == {"ab", "bc"}
    assert fm.X.sum() == 4  # 2 bigrams/doc x 2 docs


def test_char_ngram_range_builds_multiple_orders():
    corpus = _corpus("abc")
    ex = CharNgramExtractor(n=(2, 3), scale="none")
    fm = ex.fit_transform(corpus)
    # bigrams: ab, bc; trigrams: abc → 3 distinct n-grams
    assert len(fm.feature_names) == 3


def test_char_ngram_with_word_boundaries():
    corpus = _corpus("ab cd")
    ex = CharNgramExtractor(n=3, include_boundaries=True, scale="none")
    fm = ex.fit_transform(corpus)
    # With boundaries, whitespace becomes a padding char; n-grams span word starts/ends.
    assert len(fm.feature_names) > 0


def test_word_ngram_unigrams_match_mfw_counts():
    corpus = _corpus("the cat sat on the mat", "the cat")
    ex = WordNgramExtractor(n=1, scale="none")
    fm = ex.fit_transform(corpus)
    df = fm.as_dataframe()
    assert df.loc["d0", "the"] == 2
    assert df.loc["d0", "cat"] == 1
    assert df.loc["d1", "the"] == 1


def test_word_ngram_bigrams():
    corpus = _corpus("the cat sat", "the cat ran")
    ex = WordNgramExtractor(n=2, scale="none")
    fm = ex.fit_transform(corpus)
    # Bigrams: "the cat", "cat sat" (d0); "the cat", "cat ran" (d1)
    assert "the cat" in fm.feature_names
    assert "cat sat" in fm.feature_names
    assert "cat ran" in fm.feature_names


def test_word_ngram_range_1_to_2():
    corpus = _corpus("a b c")
    ex = WordNgramExtractor(n=(1, 2), scale="none")
    fm = ex.fit_transform(corpus)
    # Unigrams: a, b, c (3); bigrams: "a b", "b c" (2) → 5
    assert len(fm.feature_names) == 5
