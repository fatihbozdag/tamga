"""Shared pytest fixtures."""

from pathlib import Path

import pytest

from tamga.corpus import Corpus
from tamga.io import load_corpus


@pytest.fixture(scope="session")
def mini_corpus_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "mini_corpus"


@pytest.fixture()
def mini_corpus(mini_corpus_dir: Path) -> Corpus:
    return load_corpus(mini_corpus_dir, metadata=mini_corpus_dir / "metadata.tsv")
