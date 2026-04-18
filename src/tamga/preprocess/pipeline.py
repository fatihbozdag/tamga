"""High-level spaCy parsing wrapper, DocBin-cached."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import spacy
from spacy.language import Language
from spacy.tokens import Doc, DocBin

from tamga.corpus import Corpus, Document
from tamga.plumbing.logging import get_logger
from tamga.preprocess.cache import DocBinCache, cache_key

_log = get_logger(__name__)


class ParsedCorpus:
    """A corpus paired with its parsed spaCy `Doc` objects (index-aligned)."""

    def __init__(self, corpus: Corpus, docs: list[Doc]) -> None:
        if len(corpus) != len(docs):
            raise ValueError("corpus and docs must have the same length")
        self.corpus = corpus
        self._docs = docs

    def __len__(self) -> int:
        return len(self._docs)

    def spacy_docs(self) -> Iterator[Doc]:
        return iter(self._docs)

    def pairs(self) -> Iterator[tuple[Document, Doc]]:
        return zip(self.corpus.documents, self._docs, strict=True)


class SpacyPipeline:
    """Parse a `Corpus` into spaCy `Doc`s, caching results as `DocBin` blobs on disk."""

    def __init__(
        self,
        *,
        model: str = "en_core_web_trf",
        cache_dir: Path | str = ".tamga/cache/docbin",
        exclude: list[str] | None = None,
    ) -> None:
        self.model = model
        self.exclude = list(exclude or [])
        self.cache = DocBinCache(Path(cache_dir))
        self._nlp: Language | None = None

    @property
    def nlp(self) -> Language:
        if self._nlp is None:
            _log.info("loading spaCy model: %s", self.model)
            self._nlp = spacy.load(self.model, exclude=self.exclude)
        return self._nlp

    @property
    def spacy_version(self) -> str:
        return str(spacy.__version__)

    def _key(self, doc: Document) -> str:
        return cache_key(doc.hash, self.model, f"spacy={self.spacy_version}", self.exclude)

    def parse(self, corpus: Corpus) -> ParsedCorpus:
        parsed: list[Doc | None] = []
        to_parse_indices: list[int] = []
        to_parse_texts: list[str] = []

        for i, doc in enumerate(corpus.documents):
            cached = self.cache.get(self._key(doc))
            if cached is not None:
                bin_ = DocBin().from_bytes(cached)
                (spacy_doc,) = list(bin_.get_docs(self.nlp.vocab))
                parsed.append(spacy_doc)
            else:
                parsed.append(None)
                to_parse_indices.append(i)
                to_parse_texts.append(doc.text)

        if to_parse_texts:
            _log.info(
                "parsing %d documents (%d cached)",
                len(to_parse_texts),
                len(corpus) - len(to_parse_texts),
            )
            for i, spacy_doc in zip(to_parse_indices, self.nlp.pipe(to_parse_texts), strict=True):
                parsed[i] = spacy_doc
                bin_ = DocBin(docs=[spacy_doc])
                self.cache.put(self._key(corpus.documents[i]), bin_.to_bytes())

        # At this point, every slot has been populated either from cache or from fresh parsing.
        final_docs: list[Doc] = [d for d in parsed if d is not None]
        assert len(final_docs) == len(parsed), "internal error: some documents were not parsed"
        return ParsedCorpus(corpus=corpus, docs=final_docs)
