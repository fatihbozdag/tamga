"""Tests for Stamatatos (2013) distortion."""

from __future__ import annotations

import pytest

from bitig.corpus import Corpus, Document
from bitig.forensic.distortion import distort_corpus, distort_text


class TestDistortText:
    def test_dv_ma_masks_content_word_characters_preserves_length(self) -> None:
        """DV-MA replaces each content-word letter with '*' but keeps word length."""
        # "supersede" is not in the default function-word list.
        out = distort_text("the dog supersedes", mode="dv_ma")
        assert out == "the *** **********"

    def test_dv_sa_collapses_each_content_word_to_single_asterisk(self) -> None:
        out = distort_text("the dog supersedes", mode="dv_sa")
        assert out == "the * *"

    def test_function_words_preserved_verbatim(self) -> None:
        # Function words like "the", "of", "and" are in the default list.
        out = distort_text("the cat and the dog", mode="dv_ma")
        # "cat" and "dog" are content; "the" and "and" are function.
        assert out == "the *** and the ***"

    def test_punctuation_preserved(self) -> None:
        out = distort_text("Hello, world! How are you?", mode="dv_ma")
        # "Hello" and "world" are content (not in function list); "How", "are", "you" are typically functional.
        # Regardless of exact mapping, punctuation characters must stay.
        assert "," in out
        assert "!" in out
        assert "?" in out

    def test_whitespace_preserved(self) -> None:
        out = distort_text("a  b\n c", mode="dv_ma")
        # Double-space and newline preserved.
        assert "  " in out
        assert "\n" in out

    def test_digits_preserved(self) -> None:
        """Digits are not matched by the content-word regex, so they stay."""
        out = distort_text("run 123 fast", mode="dv_ma")
        assert "123" in out

    def test_custom_function_word_list(self) -> None:
        """User-supplied function-word list overrides the bundled list."""
        # With an empty function-word list, EVERY word is content.
        out = distort_text("the cat ran", mode="dv_ma", function_words=set())
        assert out == "*** *** ***"

    def test_function_word_matching_is_case_insensitive(self) -> None:
        out_upper = distort_text("THE cat", mode="dv_ma")
        out_lower = distort_text("the cat", mode="dv_ma")
        assert out_upper == "THE ***"
        assert out_lower == "the ***"

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown distortion mode"):
            distort_text("hello", mode="dv_other")  # type: ignore[arg-type]

    def test_contractions_preserved_verbatim(self) -> None:
        """Contractions like "don't", "it's" are in the bundled function-word list and must
        survive DV-MA / DV-SA intact. Regression test for a bug where _TOKEN_RE split the
        apostrophe, producing "***'*" for "don't" — corrupting every common English
        contraction.
        """
        out = distort_text("I don't think it's working", mode="dv_ma")
        assert "don't" in out
        assert "it's" in out
        assert "think" not in out  # content word still masked
        assert "working" not in out

    def test_content_word_with_apostrophe_still_masked_as_single_token(self) -> None:
        """Content words containing apostrophes (e.g., "o'clock") are masked as ONE token
        rather than split into two content-word chunks around the apostrophe."""
        out = distort_text("o'clock chiming", mode="dv_ma")
        # "o'clock" (7 chars including apostrophe) is not in the function-word list, so it's
        # masked. The point is that it's masked as one contiguous "*******" of length 7 —
        # not as "*" + "'" + "*****".
        assert out == "******* *******"


class TestDistortCorpus:
    @staticmethod
    def _corpus() -> Corpus:
        return Corpus(
            documents=[
                Document(
                    id="d1",
                    text="the cat ran fast",
                    metadata={"author": "A", "genre": "fiction"},
                ),
                Document(
                    id="d2",
                    text="and the dog barked",
                    metadata={"author": "B"},
                ),
            ]
        )

    def test_returns_new_corpus_with_same_ids_and_metadata(self) -> None:
        distorted = distort_corpus(self._corpus(), mode="dv_ma")
        assert [d.id for d in distorted.documents] == ["d1", "d2"]
        assert distorted.documents[0].metadata["author"] == "A"
        assert distorted.documents[0].metadata["genre"] == "fiction"
        assert distorted.documents[1].metadata["author"] == "B"

    def test_stamps_distortion_mode_in_metadata(self) -> None:
        distorted = distort_corpus(self._corpus(), mode="dv_ma")
        for doc in distorted.documents:
            assert doc.metadata["distortion_mode"] == "dv_ma"
        sa = distort_corpus(self._corpus(), mode="dv_sa")
        for doc in sa.documents:
            assert doc.metadata["distortion_mode"] == "dv_sa"

    def test_does_not_mutate_source_corpus(self) -> None:
        corpus = self._corpus()
        original_texts = [d.text for d in corpus.documents]
        distort_corpus(corpus, mode="dv_ma")
        assert [d.text for d in corpus.documents] == original_texts
        assert "distortion_mode" not in corpus.documents[0].metadata

    def test_dv_ma_preserves_document_length(self) -> None:
        """DV-MA is length-preserving — critical for features that care about doc length."""
        corpus = self._corpus()
        distorted = distort_corpus(corpus, mode="dv_ma")
        for orig, new in zip(corpus.documents, distorted.documents, strict=True):
            assert len(new.text) == len(orig.text)

    def test_dv_sa_output_is_shorter_when_content_words_present(self) -> None:
        corpus = Corpus(documents=[Document(id="d1", text="the supersedes running")])
        sa = distort_corpus(corpus, mode="dv_sa")
        # "supersedes" (10 chars) and "running" (7 chars) collapse to "*" each.
        assert len(sa.documents[0].text) < len(corpus.documents[0].text)
