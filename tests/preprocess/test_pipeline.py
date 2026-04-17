"""Tests for the spaCy preprocessing pipeline — run against the small English model."""

from pathlib import Path

import pytest

from tamga.corpus import Corpus, Document
from tamga.preprocess.pipeline import ParsedCorpus, SpacyPipeline

pytestmark: pytest.MarkDecorator = pytest.mark.spacy


def _tiny(text: str) -> Document:
    return Document(id=f"d-{hash(text) & 0xFFFF}", text=text, metadata={})


def test_pipeline_parses_documents(tmp_path: Path) -> None:
    corpus = Corpus(documents=[_tiny("Hello world."), _tiny("spaCy parses sentences.")])
    pipe = SpacyPipeline(model="en_core_web_sm", cache_dir=tmp_path)
    parsed = pipe.parse(corpus)
    assert isinstance(parsed, ParsedCorpus)
    assert len(parsed) == 2
    for doc in parsed.spacy_docs():
        assert len(list(doc.sents)) >= 1


def test_pipeline_cache_hit_is_fast(tmp_path: Path) -> None:
    corpus = Corpus(documents=[_tiny("The same sentence every time.")])
    pipe = SpacyPipeline(model="en_core_web_sm", cache_dir=tmp_path)

    pipe.parse(corpus)  # cold
    keys_after_first = set(pipe.cache.keys())

    pipe.parse(corpus)  # warm
    keys_after_second = set(pipe.cache.keys())

    assert keys_after_first == keys_after_second
    assert len(keys_after_first) == 1


def test_pipeline_cache_invalidates_on_text_change(tmp_path: Path) -> None:
    pipe = SpacyPipeline(model="en_core_web_sm", cache_dir=tmp_path)
    pipe.parse(Corpus(documents=[_tiny("original text")]))
    pipe.parse(Corpus(documents=[_tiny("different text")]))
    assert len(pipe.cache.keys()) == 2


def test_parsed_corpus_iteration_preserves_order(tmp_path: Path) -> None:
    corpus = Corpus(documents=[_tiny("A."), _tiny("B."), _tiny("C.")])
    pipe = SpacyPipeline(model="en_core_web_sm", cache_dir=tmp_path)
    parsed = pipe.parse(corpus)
    texts = [doc.text.strip() for doc in parsed.spacy_docs()]
    assert texts == ["A.", "B.", "C."]
