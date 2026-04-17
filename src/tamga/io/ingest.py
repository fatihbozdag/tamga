"""Corpus ingestion from a directory of .txt files + optional metadata TSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tamga.corpus import Corpus, Document
from tamga.plumbing.logging import get_logger

_log = get_logger(__name__)

_TEXT_GLOB = "*.txt"
_FILENAME_KEY = "filename"


def load_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Load a TSV metadata file into {filename: {field: value}}.

    The TSV must have a header row with a `filename` column; every other column becomes a metadata
    field on the document whose filename matches.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None or _FILENAME_KEY not in reader.fieldnames:
            raise ValueError(f"{path}: TSV must have a '{_FILENAME_KEY}' column")
        rows: dict[str, dict[str, Any]] = {}
        for row in reader:
            fname = row.pop(_FILENAME_KEY)
            rows[fname] = {k: v for k, v in row.items() if v != ""}
    return rows


def load_corpus(
    path: Path,
    *,
    metadata: Path | None = None,
    strict: bool = True,
    glob: str = _TEXT_GLOB,
    encoding: str = "utf-8",
) -> Corpus:
    """Load every text file under `path` into a Corpus, sorted by filename.

    If `metadata` is provided it must be a TSV readable by `load_metadata`. When `strict` is True
    (default), every file must have a matching metadata row; otherwise, files without metadata
    are included with an empty metadata dict and a warning.
    """
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(path)

    meta_by_filename: dict[str, dict[str, Any]] = load_metadata(metadata) if metadata else {}

    files = sorted(path.glob(glob))
    if not files:
        raise ValueError(f"{path}: no files matching {glob!r}")

    documents: list[Document] = []
    missing: list[str] = []
    for f in files:
        text = f.read_text(encoding=encoding)
        doc_meta = dict(meta_by_filename.get(f.name, {}))
        if not doc_meta and meta_by_filename and strict:
            missing.append(f.name)
        documents.append(Document(id=f.stem, text=text, metadata=doc_meta))

    if missing:
        raise ValueError(f"strict=True: missing metadata for {len(missing)} file(s): {missing}")

    _log.info("loaded corpus: %d documents from %s", len(documents), path)
    return Corpus(documents=documents)
