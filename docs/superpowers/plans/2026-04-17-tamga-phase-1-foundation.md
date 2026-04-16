# tamga — Phase 1: Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundation layer of `tamga`: project skeleton, hashing/seeding plumbing, the `Corpus`/`Document` data model, pydantic `StudyConfig` schema with layered resolution, spaCy preprocessing pipeline with DocBin caching, skeleton CLI with `init`, `ingest`, `info`, and `cache` commands, and CI. End state: `tamga init my-study && tamga ingest corpus/ && tamga info` works end-to-end on a real corpus; all unit + integration tests pass; GitHub Actions CI green.

**Architecture:** `src/tamga/` layout with concentric layers (plumbing → io/corpus → preprocess → config → cli). Every module has one responsibility. Tests live at `tests/` mirroring the package tree. Test-first throughout: each non-trivial unit gets a failing test, a minimal implementation, a passing test, and a commit before the next unit.

**Tech Stack:** Python 3.11+, uv for dependency management, hatchling build backend, pydantic v2 for config, typer + rich + questionary for CLI, pyyaml, spaCy ≥3.7 (with `en_core_web_sm` for tests, `en_core_web_trf` for production), pandas + pyarrow for I/O, pytest + pytest-cov + hypothesis for testing, ruff + mypy + pre-commit for quality.

**Reference spec:** `docs/superpowers/specs/2026-04-17-tamga-stylometry-package-design.md`.

---

## Task 1: Project skeleton — `pyproject.toml`, src layout, smoke test

**Files:**
- Create: `pyproject.toml`
- Create: `src/tamga/__init__.py`
- Create: `src/tamga/_version.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`
- Create: `.python-version`

- [ ] **Step 1.1: Create `.python-version`**

```
3.11
```

- [ ] **Step 1.2: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling>=1.24"]
build-backend = "hatchling.build"

[project]
name = "tamga"
dynamic = ["version"]
description = "Next-generation computational stylometry — a Python replacement for R's Stylo."
readme = "README.md"
requires-python = ">=3.11"
license = "BSD-3-Clause"
authors = [{ name = "Fatih Bozdağ", email = "fbozdag1989@gmail.com" }]
keywords = ["stylometry", "authorship", "corpus-linguistics", "digital-humanities", "nlp"]
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering",
    "Topic :: Text Processing :: Linguistic",
]
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "pandas>=2.2",
    "pyarrow>=15",
    "scikit-learn>=1.4",
    "spacy>=3.7",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "typer[all]>=0.12",
    "rich>=13",
    "questionary>=2",
    "pyyaml>=6",
    "pydantic>=2",
    "jinja2>=3.1",
    "joblib>=1.3",
    "umap-learn>=0.5",
    "hdbscan>=0.8",
]

[project.optional-dependencies]
bayesian = ["pymc>=5.10", "arviz>=0.17"]
embeddings = ["sentence-transformers>=2.6", "torch>=2.2"]
viz = ["plotly>=5.20", "kaleido>=0.2.1", "ete3>=3.1.3"]
reports = ["weasyprint>=61"]
dev = [
    "pytest>=8",
    "pytest-cov>=4",
    "pytest-xdist>=3",
    "hypothesis>=6",
    "ruff>=0.4",
    "mypy>=1.10",
    "pre-commit>=3.7",
    "types-PyYAML",
]

[project.scripts]
tamga = "tamga.cli:app"

[project.urls]
Homepage = "https://github.com/fatihbozdag/tamga"
Documentation = "https://github.com/fatihbozdag/tamga"
Source = "https://github.com/fatihbozdag/tamga"
Issues = "https://github.com/fatihbozdag/tamga/issues"

[tool.hatch.version]
path = "src/tamga/_version.py"

[tool.hatch.build.targets.wheel]
packages = ["src/tamga"]

[tool.hatch.build.targets.sdist]
include = ["src/tamga", "tests", "docs", "README.md", "LICENSE", "CITATION.cff"]
```

- [ ] **Step 1.3: Create `src/tamga/_version.py`**

```python
__version__ = "0.1.0.dev0"
```

- [ ] **Step 1.4: Create `src/tamga/__init__.py`**

```python
"""tamga — next-generation computational stylometry."""

from tamga._version import __version__

__all__ = ["__version__"]
```

- [ ] **Step 1.5: Create `tests/__init__.py` (empty)**

```python
```

- [ ] **Step 1.6: Create `tests/test_smoke.py`**

```python
"""Smoke tests — the absolute minimum that must work."""

import tamga


def test_package_imports():
    assert hasattr(tamga, "__version__")


def test_version_is_string():
    assert isinstance(tamga.__version__, str)
    assert len(tamga.__version__) > 0
```

- [ ] **Step 1.7: Install the package in editable mode with dev deps**

Run: `uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"`
Expected: installation succeeds; `tamga` importable.

- [ ] **Step 1.8: Run the smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 1.9: Commit**

```bash
git add pyproject.toml .python-version src/tamga/__init__.py src/tamga/_version.py tests/__init__.py tests/test_smoke.py
git commit -m "build: project skeleton with hatchling, src layout, smoke test"
```

---

## Task 2: Code-quality config — ruff, mypy, pre-commit

**Files:**
- Modify: `pyproject.toml` (add `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`, `[tool.coverage.run]`)
- Create: `.pre-commit-config.yaml`

- [ ] **Step 2.1: Append quality-tool config to `pyproject.toml`**

```toml

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E", "F", "W",      # pycodestyle + pyflakes
    "I",                # isort
    "UP",               # pyupgrade
    "B",                # bugbear
    "C4",               # comprehensions
    "SIM",              # simplify
    "RUF",              # ruff-specific
    "N",                # pep8-naming
]
ignore = ["E501"]       # handled by formatter

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["N802", "N806"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.mypy]
python_version = "3.11"
strict = false
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "tamga.*"
disallow_untyped_defs = true
no_implicit_optional = true

[tool.pytest.ini_options]
minversion = "8.0"
addopts = "-ra -q --strict-markers"
testpaths = ["tests"]
pythonpath = ["src"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "spacy: marks tests that require a spaCy model download",
    "integration: end-to-end tests that run real subprocesses",
]

[tool.coverage.run]
source = ["src/tamga"]
branch = true

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "raise NotImplementedError", "if TYPE_CHECKING:"]
precision = 1
```

- [ ] **Step 2.2: Create `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=500"]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [pydantic>=2, types-PyYAML]
        args: [--config-file=pyproject.toml]
        files: ^src/
```

- [ ] **Step 2.3: Install pre-commit hooks**

Run: `pre-commit install`
Expected: `pre-commit installed at .git/hooks/pre-commit`.

- [ ] **Step 2.4: Run pre-commit across all files**

Run: `pre-commit run --all-files`
Expected: all checks pass (may auto-fix trivial issues and re-run once).

- [ ] **Step 2.5: Commit**

```bash
git add pyproject.toml .pre-commit-config.yaml
git commit -m "chore: ruff + mypy + pre-commit configuration"
```

---

## Task 3: `plumbing.hashing` — stable content hashes

**Files:**
- Create: `src/tamga/plumbing/__init__.py`
- Create: `src/tamga/plumbing/hashing.py`
- Create: `tests/plumbing/__init__.py`
- Create: `tests/plumbing/test_hashing.py`

- [ ] **Step 3.1: Create `src/tamga/plumbing/__init__.py` (empty)**

```python
"""Low-level cross-cutting utilities: hashing, seeds, logging, paths."""
```

- [ ] **Step 3.2: Create `tests/plumbing/__init__.py` (empty)**

```python
```

- [ ] **Step 3.3: Write failing tests in `tests/plumbing/test_hashing.py`**

```python
"""Tests for stable content hashing."""

import pytest

from tamga.plumbing.hashing import hash_bytes, hash_mapping, hash_text, short_hash


def test_hash_text_is_stable_across_calls():
    assert hash_text("hello world") == hash_text("hello world")


def test_hash_text_differs_for_different_content():
    assert hash_text("hello") != hash_text("world")


def test_hash_text_returns_hex_digest():
    h = hash_text("abc")
    assert isinstance(h, str)
    assert all(c in "0123456789abcdef" for c in h)
    assert len(h) == 64  # sha256 hex


def test_hash_bytes_matches_hash_text_for_same_content():
    assert hash_bytes(b"hello") == hash_text("hello")


def test_hash_mapping_is_order_independent():
    assert hash_mapping({"a": 1, "b": 2}) == hash_mapping({"b": 2, "a": 1})


def test_hash_mapping_is_stable():
    assert hash_mapping({"a": 1, "b": [1, 2, 3]}) == hash_mapping({"a": 1, "b": [1, 2, 3]})


def test_hash_mapping_differs_for_different_content():
    assert hash_mapping({"a": 1}) != hash_mapping({"a": 2})


def test_hash_mapping_handles_nested_structures():
    v1 = {"a": {"b": {"c": 1}}}
    v2 = {"a": {"b": {"c": 1}}}
    assert hash_mapping(v1) == hash_mapping(v2)


def test_hash_mapping_rejects_non_json_values():
    with pytest.raises(TypeError):
        hash_mapping({"a": object()})


def test_short_hash_is_prefix_of_full():
    full = hash_text("xyz")
    assert short_hash("xyz") == full[:12]
```

- [ ] **Step 3.4: Run tests to verify they fail**

Run: `pytest tests/plumbing/test_hashing.py -v`
Expected: all tests FAIL with ModuleNotFoundError on `tamga.plumbing.hashing`.

- [ ] **Step 3.5: Implement `src/tamga/plumbing/hashing.py`**

```python
"""Stable content hashing used throughout the package.

Two requirements every hash must satisfy:

1. **Deterministic across Python processes.** The corpus hash you computed today must match the one
   a collaborator computes tomorrow, on a different machine, in a different Python version.
2. **Stable across equivalent inputs.** A mapping is hashed by its sorted JSON representation so
   key-insertion order is irrelevant.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

_ENCODING = "utf-8"
_HASH = hashlib.sha256


def hash_bytes(data: bytes) -> str:
    """Return the sha256 hex digest of raw bytes."""
    return _HASH(data).hexdigest()


def hash_text(text: str) -> str:
    """Return the sha256 hex digest of a UTF-8 string."""
    return hash_bytes(text.encode(_ENCODING))


def hash_mapping(mapping: Mapping[str, Any]) -> str:
    """Return a stable sha256 hex digest of a JSON-serialisable mapping.

    Keys are sorted before serialisation so insertion order does not affect the result.
    Raises TypeError if any value is not JSON-serialisable.
    """
    try:
        serialised = json.dumps(mapping, sort_keys=True, separators=(",", ":"), default=_fail_default)
    except TypeError as exc:  # pragma: no cover - covered by _fail_default
        raise TypeError(f"hash_mapping: non-serialisable value: {exc}") from exc
    return hash_text(serialised)


def short_hash(text: str, length: int = 12) -> str:
    """Return the first `length` characters of the hash — useful for directory/cache-key display."""
    return hash_text(text)[:length]


def _fail_default(value: Any) -> Any:
    raise TypeError(f"not JSON-serialisable: {type(value).__name__}")
```

- [ ] **Step 3.6: Run tests to verify they pass**

Run: `pytest tests/plumbing/test_hashing.py -v`
Expected: all 10 tests PASS.

- [ ] **Step 3.7: Commit**

```bash
git add src/tamga/plumbing/ tests/plumbing/
git commit -m "feat(plumbing): stable content hashing (hash_text, hash_mapping, short_hash)"
```

---

## Task 4: `plumbing.seeds` — deterministic per-method RNGs

**Files:**
- Create: `src/tamga/plumbing/seeds.py`
- Create: `tests/plumbing/test_seeds.py`

- [ ] **Step 4.1: Write failing tests in `tests/plumbing/test_seeds.py`**

```python
"""Tests for deterministic per-method seed derivation."""

import numpy as np

from tamga.plumbing.seeds import derive_rng, derive_seed


def test_derive_seed_is_deterministic():
    assert derive_seed(42, "burrows") == derive_seed(42, "burrows")


def test_derive_seed_differs_by_method_id():
    assert derive_seed(42, "burrows") != derive_seed(42, "consensus")


def test_derive_seed_differs_by_study_seed():
    assert derive_seed(42, "burrows") != derive_seed(43, "burrows")


def test_derive_seed_is_in_uint32_range():
    for seed in (0, 1, 42, 123456, 2**31 - 1):
        for method in ("a", "b", "consensus-run-0"):
            derived = derive_seed(seed, method)
            assert 0 <= derived < 2**32


def test_derive_rng_returns_numpy_generator():
    rng = derive_rng(42, "burrows")
    assert isinstance(rng, np.random.Generator)


def test_derive_rng_produces_reproducible_draws():
    r1 = derive_rng(42, "burrows").integers(0, 100, size=10)
    r2 = derive_rng(42, "burrows").integers(0, 100, size=10)
    assert np.array_equal(r1, r2)


def test_derive_rng_produces_different_draws_for_different_methods():
    r1 = derive_rng(42, "burrows").integers(0, 10**6, size=10)
    r2 = derive_rng(42, "consensus").integers(0, 10**6, size=10)
    assert not np.array_equal(r1, r2)
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `pytest tests/plumbing/test_seeds.py -v`
Expected: FAIL with ImportError on `tamga.plumbing.seeds`.

- [ ] **Step 4.3: Implement `src/tamga/plumbing/seeds.py`**

```python
"""Per-method RNG derivation.

`numpy.random.Generator` is created from a derived seed combining the study-level seed with the
method id. This gives each method an independent, reproducible random stream — reordering methods
in a study config does not affect any individual method's output.
"""

from __future__ import annotations

import hashlib
import numpy as np


def derive_seed(study_seed: int, method_id: str) -> int:
    """Return a deterministic uint32 seed from a study seed and a method id.

    Implemented via sha256 of `f"{study_seed}:{method_id}"`; takes the first 4 bytes as an
    unsigned 32-bit integer, which is what `numpy.random.default_rng` expects.
    """
    digest = hashlib.sha256(f"{study_seed}:{method_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def derive_rng(study_seed: int, method_id: str) -> np.random.Generator:
    """Return a seeded numpy Generator for a given (study_seed, method_id)."""
    return np.random.default_rng(derive_seed(study_seed, method_id))
```

- [ ] **Step 4.4: Run tests**

Run: `pytest tests/plumbing/test_seeds.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/tamga/plumbing/seeds.py tests/plumbing/test_seeds.py
git commit -m "feat(plumbing): derive_seed and derive_rng for per-method reproducibility"
```

---

## Task 5: `plumbing.logging` — structured logger

**Files:**
- Create: `src/tamga/plumbing/logging.py`
- Create: `tests/plumbing/test_logging.py`

- [ ] **Step 5.1: Write failing tests**

```python
"""Tests for the logger helper."""

import logging

from tamga.plumbing.logging import get_logger, set_verbosity


def test_get_logger_returns_logger():
    log = get_logger("tamga.test")
    assert isinstance(log, logging.Logger)
    assert log.name == "tamga.test"


def test_get_logger_is_idempotent():
    a = get_logger("tamga.test")
    b = get_logger("tamga.test")
    assert a is b


def test_set_verbosity_changes_level():
    set_verbosity("DEBUG")
    log = get_logger("tamga.test")
    assert log.isEnabledFor(logging.DEBUG)

    set_verbosity("WARNING")
    assert not log.isEnabledFor(logging.DEBUG)
    assert log.isEnabledFor(logging.WARNING)
```

- [ ] **Step 5.2: Run — FAIL (module missing)**

Run: `pytest tests/plumbing/test_logging.py -v`
Expected: FAIL.

- [ ] **Step 5.3: Implement `src/tamga/plumbing/logging.py`**

```python
"""Logging helpers.

Loggers are namespaced under `tamga.*`. Verbosity is controlled once at the root and inherited.
Integrates with Rich for readable terminal output when the Rich handler is installed by the CLI.
"""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_ROOT = logging.getLogger("tamga")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. Names not starting with `tamga.` are prefixed."""
    if not name.startswith("tamga"):
        name = f"tamga.{name}"
    _configure_once()
    return logging.getLogger(name)


def set_verbosity(level: str | int) -> None:
    """Set the root `tamga` logger verbosity. Accepts 'DEBUG', 'INFO', 'WARNING', 'ERROR' or int."""
    _configure_once()
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    _ROOT.setLevel(level)


def _configure_once() -> None:
    if _ROOT.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    _ROOT.addHandler(handler)
    _ROOT.setLevel(logging.INFO)
    _ROOT.propagate = False
```

- [ ] **Step 5.4: Run — PASS**

Run: `pytest tests/plumbing/test_logging.py -v`
Expected: 3 passed.

- [ ] **Step 5.5: Commit**

```bash
git add src/tamga/plumbing/logging.py tests/plumbing/test_logging.py
git commit -m "feat(plumbing): namespaced logger helpers"
```

---

## Task 6: `corpus.document` — the `Document` type

**Files:**
- Create: `src/tamga/corpus/__init__.py`
- Create: `src/tamga/corpus/document.py`
- Create: `tests/corpus/__init__.py`
- Create: `tests/corpus/test_document.py`

- [ ] **Step 6.1: Create package `__init__.py` files**

`src/tamga/corpus/__init__.py`:

```python
"""Corpus and Document data model."""

from tamga.corpus.document import Document

__all__ = ["Document"]
```

`tests/corpus/__init__.py`:

```python
```

- [ ] **Step 6.2: Write failing tests in `tests/corpus/test_document.py`**

```python
"""Tests for the Document dataclass."""

import pytest

from tamga.corpus.document import Document


def test_document_basic_construction():
    doc = Document(id="doc-1", text="hello world", metadata={"author": "Alice"})
    assert doc.id == "doc-1"
    assert doc.text == "hello world"
    assert doc.metadata == {"author": "Alice"}


def test_document_hash_is_computed_from_text():
    doc = Document(id="doc-1", text="hello world", metadata={})
    assert len(doc.hash) == 64  # sha256 hex
    # Identical text ⇒ identical hash.
    assert doc.hash == Document(id="other-id", text="hello world", metadata={}).hash


def test_document_hash_differs_for_different_text():
    a = Document(id="a", text="hello", metadata={})
    b = Document(id="b", text="world", metadata={})
    assert a.hash != b.hash


def test_document_metadata_defaults_to_empty_mapping():
    doc = Document(id="doc-1", text="abc")
    assert doc.metadata == {}


def test_document_is_immutable():
    doc = Document(id="doc-1", text="abc", metadata={})
    with pytest.raises((AttributeError, TypeError)):
        doc.id = "doc-2"  # type: ignore[misc]


def test_document_round_trips_to_dict():
    doc = Document(id="x", text="hi", metadata={"author": "A", "year": 1984})
    d = doc.to_dict()
    restored = Document.from_dict(d)
    assert restored == doc
```

- [ ] **Step 6.3: Run — FAIL**

Run: `pytest tests/corpus/test_document.py -v`
Expected: FAIL.

- [ ] **Step 6.4: Implement `src/tamga/corpus/document.py`**

```python
"""The Document class — a single text with metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from tamga.plumbing.hashing import hash_text


@dataclass(frozen=True, slots=True)
class Document:
    """A single text with optional metadata.

    `hash` is derived lazily from `text` — identical texts hash identically regardless of `id` or
    metadata, which is intentional: the cache key for parsed documents is content-addressed.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def hash(self) -> str:
        return hash_text(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        return cls(id=data["id"], text=data["text"], metadata=dict(data.get("metadata") or {}))
```

- [ ] **Step 6.5: Run — PASS**

Run: `pytest tests/corpus/test_document.py -v`
Expected: 6 passed.

- [ ] **Step 6.6: Commit**

```bash
git add src/tamga/corpus/__init__.py src/tamga/corpus/document.py tests/corpus/__init__.py tests/corpus/test_document.py
git commit -m "feat(corpus): Document immutable dataclass with content-hash"
```

---

## Task 7: `corpus.corpus` — the `Corpus` collection

**Files:**
- Create: `src/tamga/corpus/corpus.py`
- Create: `tests/corpus/test_corpus.py`
- Modify: `src/tamga/corpus/__init__.py` (export `Corpus`)

- [ ] **Step 7.1: Write failing tests in `tests/corpus/test_corpus.py`**

```python
"""Tests for the Corpus collection."""

import pytest

from tamga.corpus import Corpus, Document


def _doc(i: int, **meta: object) -> Document:
    return Document(id=f"doc-{i}", text=f"text {i}", metadata=dict(meta))


def test_corpus_wraps_documents():
    docs = [_doc(1, author="A"), _doc(2, author="B")]
    c = Corpus(documents=docs)
    assert len(c) == 2
    assert list(c) == docs


def test_corpus_is_indexable():
    docs = [_doc(1), _doc(2), _doc(3)]
    c = Corpus(documents=docs)
    assert c[0] == docs[0]
    assert c[-1] == docs[-1]


def test_corpus_filter_by_exact_metadata():
    docs = [_doc(1, author="A"), _doc(2, author="B"), _doc(3, author="A")]
    c = Corpus(documents=docs)
    result = c.filter(author="A")
    assert len(result) == 2
    assert all(d.metadata["author"] == "A" for d in result)


def test_corpus_filter_by_list_metadata():
    docs = [_doc(1, group="native"), _doc(2, group="L2"), _doc(3, group="bilingual")]
    c = Corpus(documents=docs)
    result = c.filter(group=["native", "L2"])
    assert len(result) == 2


def test_corpus_groupby_returns_dict_of_corpora():
    docs = [_doc(1, author="A"), _doc(2, author="B"), _doc(3, author="A")]
    c = Corpus(documents=docs)
    groups = c.groupby("author")
    assert set(groups.keys()) == {"A", "B"}
    assert len(groups["A"]) == 2
    assert len(groups["B"]) == 1
    assert all(isinstance(g, Corpus) for g in groups.values())


def test_corpus_groupby_missing_field_raises():
    c = Corpus(documents=[_doc(1, author="A")])
    with pytest.raises(KeyError):
        c.groupby("nonexistent_field")


def test_corpus_hash_is_stable_and_order_independent():
    a = Corpus(documents=[_doc(1), _doc(2), _doc(3)])
    b = Corpus(documents=[_doc(3), _doc(1), _doc(2)])  # different order
    assert a.hash() == b.hash()


def test_corpus_hash_differs_for_different_content():
    a = Corpus(documents=[_doc(1), _doc(2)])
    b = Corpus(documents=[_doc(1), _doc(3)])
    assert a.hash() != b.hash()


def test_corpus_metadata_column_extracts_field_per_document():
    docs = [_doc(1, author="A"), _doc(2, author="B"), _doc(3, author="A")]
    c = Corpus(documents=docs)
    assert c.metadata_column("author") == ["A", "B", "A"]
```

- [ ] **Step 7.2: Run — FAIL**

Run: `pytest tests/corpus/test_corpus.py -v`
Expected: ImportError on `Corpus`.

- [ ] **Step 7.3: Implement `src/tamga/corpus/corpus.py`**

```python
"""The Corpus collection — an ordered bag of Documents with metadata-aware operations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from tamga.corpus.document import Document
from tamga.plumbing.hashing import hash_mapping, hash_text


@dataclass
class Corpus:
    """An ordered collection of Documents that share a metadata schema.

    Iteration yields Documents in the order provided. Equality and hashing ignore order — two
    Corpora with the same documents in different orders hash identically.
    """

    documents: list[Document] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        return iter(self.documents)

    def __getitem__(self, index: int) -> Document:
        return self.documents[index]

    def filter(self, **query: Any) -> Corpus:
        """Return a new Corpus containing documents whose metadata matches every key in `query`.

        Values may be scalars (exact match) or lists (membership).
        """

        def matches(doc: Document) -> bool:
            for key, expected in query.items():
                actual = doc.metadata.get(key)
                if isinstance(expected, (list, tuple, set)):
                    if actual not in expected:
                        return False
                elif actual != expected:
                    return False
            return True

        return Corpus(documents=[d for d in self.documents if matches(d)])

    def groupby(self, field_name: str) -> dict[Any, Corpus]:
        """Group documents by a metadata field value.

        Raises KeyError if any document lacks the field.
        """
        groups: dict[Any, list[Document]] = {}
        for doc in self.documents:
            if field_name not in doc.metadata:
                raise KeyError(f"document {doc.id!r} has no metadata field {field_name!r}")
            groups.setdefault(doc.metadata[field_name], []).append(doc)
        return {k: Corpus(documents=v) for k, v in groups.items()}

    def metadata_column(self, field_name: str) -> list[Any]:
        """Return the list of metadata values at `field_name`, in document order.

        Missing values become None; use `filter` first if you want to exclude them.
        """
        return [d.metadata.get(field_name) for d in self.documents]

    def hash(self) -> str:
        """Stable hash — sorted document hashes + sorted metadata."""
        doc_hashes = sorted(d.hash for d in self.documents)
        metadata_summary = sorted((d.id, hash_mapping(d.metadata)) for d in self.documents)
        payload = "|".join(doc_hashes) + "||" + str(metadata_summary)
        return hash_text(payload)

    @classmethod
    def from_iterable(cls, docs: Iterable[Document]) -> Corpus:
        return cls(documents=list(docs))
```

- [ ] **Step 7.4: Update `src/tamga/corpus/__init__.py`**

```python
"""Corpus and Document data model."""

from tamga.corpus.corpus import Corpus
from tamga.corpus.document import Document

__all__ = ["Corpus", "Document"]
```

- [ ] **Step 7.5: Run — PASS**

Run: `pytest tests/corpus/test_corpus.py -v`
Expected: 9 passed.

- [ ] **Step 7.6: Commit**

```bash
git add src/tamga/corpus/corpus.py src/tamga/corpus/__init__.py tests/corpus/test_corpus.py
git commit -m "feat(corpus): Corpus with filter, groupby, metadata_column, stable hash"
```

---

## Task 8: `io.ingest` — read `.txt` corpus + metadata TSV

**Files:**
- Create: `src/tamga/io/__init__.py`
- Create: `src/tamga/io/ingest.py`
- Create: `tests/io/__init__.py`
- Create: `tests/io/test_ingest.py`
- Create: `tests/fixtures/mini_corpus/alice_one.txt`
- Create: `tests/fixtures/mini_corpus/alice_two.txt`
- Create: `tests/fixtures/mini_corpus/bob_one.txt`
- Create: `tests/fixtures/mini_corpus/bob_two.txt`
- Create: `tests/fixtures/mini_corpus/metadata.tsv`

- [ ] **Step 8.1: Create four fixture text files (10–20 words each, kept trivial for deterministic hashes)**

`tests/fixtures/mini_corpus/alice_one.txt`:

```
The fog came in on little cat feet. It sat looking over harbour and city on silent haunches.
```

`tests/fixtures/mini_corpus/alice_two.txt`:

```
Alice was beginning to get very tired of sitting by her sister on the bank and of having nothing to do.
```

`tests/fixtures/mini_corpus/bob_one.txt`:

```
In the beginning the universe was created. This has made a lot of people very angry.
```

`tests/fixtures/mini_corpus/bob_two.txt`:

```
It was the best of times, it was the worst of times, it was the age of wisdom.
```

- [ ] **Step 8.2: Create `tests/fixtures/mini_corpus/metadata.tsv`**

```
filename	author	group	year
alice_one.txt	Alice	native	2019
alice_two.txt	Alice	native	2020
bob_one.txt	Bob	L2	2019
bob_two.txt	Bob	L2	2021
```

- [ ] **Step 8.3: Create `src/tamga/io/__init__.py`**

```python
"""Filesystem ingestion and serialization."""

from tamga.io.ingest import load_corpus, load_metadata

__all__ = ["load_corpus", "load_metadata"]
```

- [ ] **Step 8.4: Create `tests/io/__init__.py` (empty)**

```python
```

- [ ] **Step 8.5: Write failing tests in `tests/io/test_ingest.py`**

```python
"""Tests for corpus ingestion from a filesystem directory + metadata TSV."""

from pathlib import Path

import pytest

from tamga.corpus import Corpus
from tamga.io import load_corpus, load_metadata

FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini_corpus"


def test_load_metadata_parses_tsv():
    rows = load_metadata(FIXTURES / "metadata.tsv")
    assert len(rows) == 4
    assert rows["alice_one.txt"] == {"author": "Alice", "group": "native", "year": "2019"}


def test_load_metadata_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_metadata(FIXTURES / "does_not_exist.tsv")


def test_load_corpus_from_directory_returns_corpus():
    corpus = load_corpus(FIXTURES)
    assert isinstance(corpus, Corpus)
    assert len(corpus) == 4


def test_load_corpus_attaches_metadata():
    corpus = load_corpus(FIXTURES, metadata=FIXTURES / "metadata.tsv")
    alice_docs = [d for d in corpus if d.metadata.get("author") == "Alice"]
    assert len(alice_docs) == 2


def test_load_corpus_without_metadata_still_works():
    corpus = load_corpus(FIXTURES)
    assert len(corpus) == 4
    for d in corpus:
        assert d.metadata == {}


def test_load_corpus_id_is_filename_stem():
    corpus = load_corpus(FIXTURES)
    ids = {d.id for d in corpus}
    assert ids == {"alice_one", "alice_two", "bob_one", "bob_two"}


def test_load_corpus_sorts_documents_deterministically():
    # Loading the same directory twice yields Documents in the same order → stable corpus hash.
    c1 = load_corpus(FIXTURES)
    c2 = load_corpus(FIXTURES)
    assert [d.id for d in c1] == [d.id for d in c2]


def test_load_corpus_raises_on_missing_metadata_row(tmp_path: Path):
    # Fixture a corpus where metadata does not cover every file.
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    (tmp_path / "meta.tsv").write_text("filename\tauthor\na.txt\tAlice\n")
    with pytest.raises(ValueError, match="missing metadata"):
        load_corpus(tmp_path, metadata=tmp_path / "meta.tsv", strict=True)


def test_load_corpus_non_strict_allows_missing_metadata(tmp_path: Path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    (tmp_path / "meta.tsv").write_text("filename\tauthor\na.txt\tAlice\n")
    corpus = load_corpus(tmp_path, metadata=tmp_path / "meta.tsv", strict=False)
    assert len(corpus) == 2
```

- [ ] **Step 8.6: Run — FAIL**

Run: `pytest tests/io/test_ingest.py -v`
Expected: FAIL.

- [ ] **Step 8.7: Implement `src/tamga/io/ingest.py`**

```python
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
```

- [ ] **Step 8.8: Run — PASS**

Run: `pytest tests/io/test_ingest.py -v`
Expected: 9 passed.

- [ ] **Step 8.9: Commit**

```bash
git add src/tamga/io/ tests/io/ tests/fixtures/
git commit -m "feat(io): load_corpus and load_metadata from directory + TSV"
```

---

## Task 9: `provenance` — the run-record dataclass

**Files:**
- Create: `src/tamga/provenance.py`
- Create: `tests/test_provenance.py`

- [ ] **Step 9.1: Write failing tests in `tests/test_provenance.py`**

```python
"""Tests for the Provenance record."""

from datetime import datetime

from tamga.provenance import Provenance


def test_provenance_basic_construction():
    p = Provenance(
        tamga_version="0.1.0.dev0",
        python_version="3.11.7",
        spacy_model="en_core_web_sm",
        spacy_version="3.7.2",
        corpus_hash="deadbeef",
        feature_hash=None,
        seed=42,
        timestamp=datetime(2026, 4, 17, 12, 0, 0),
        resolved_config={"seed": 42},
    )
    assert p.seed == 42
    assert p.feature_hash is None


def test_provenance_round_trips_to_dict():
    p = Provenance(
        tamga_version="0.1.0.dev0",
        python_version="3.11.7",
        spacy_model="en_core_web_sm",
        spacy_version="3.7.2",
        corpus_hash="abc123",
        feature_hash="feat456",
        seed=7,
        timestamp=datetime(2026, 4, 17, 12, 0, 0),
        resolved_config={"seed": 7, "nested": {"k": "v"}},
    )
    d = p.to_dict()
    restored = Provenance.from_dict(d)
    assert restored == p


def test_provenance_current_captures_runtime():
    p = Provenance.current(
        spacy_model="en_core_web_sm",
        spacy_version="3.7.2",
        corpus_hash="h",
        feature_hash=None,
        seed=1,
        resolved_config={},
    )
    assert p.tamga_version
    assert "." in p.python_version
    assert isinstance(p.timestamp, datetime)
```

- [ ] **Step 9.2: Run — FAIL**

Run: `pytest tests/test_provenance.py -v`
Expected: FAIL.

- [ ] **Step 9.3: Implement `src/tamga/provenance.py`**

```python
"""The Provenance record — captured on every Result so re-runs are fully reproducible."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from tamga._version import __version__


@dataclass
class Provenance:
    tamga_version: str
    python_version: str
    spacy_model: str
    spacy_version: str
    corpus_hash: str
    feature_hash: str | None
    seed: int
    timestamp: datetime
    resolved_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        raw_ts = data["timestamp"]
        ts = datetime.fromisoformat(raw_ts) if isinstance(raw_ts, str) else raw_ts
        return cls(
            tamga_version=data["tamga_version"],
            python_version=data["python_version"],
            spacy_model=data["spacy_model"],
            spacy_version=data["spacy_version"],
            corpus_hash=data["corpus_hash"],
            feature_hash=data.get("feature_hash"),
            seed=int(data["seed"]),
            timestamp=ts,
            resolved_config=dict(data.get("resolved_config") or {}),
        )

    @classmethod
    def current(
        cls,
        *,
        spacy_model: str,
        spacy_version: str,
        corpus_hash: str,
        feature_hash: str | None,
        seed: int,
        resolved_config: dict[str, Any],
    ) -> Provenance:
        return cls(
            tamga_version=__version__,
            python_version=platform.python_version(),
            spacy_model=spacy_model,
            spacy_version=spacy_version,
            corpus_hash=corpus_hash,
            feature_hash=feature_hash,
            seed=seed,
            timestamp=datetime.now(),
            resolved_config=resolved_config,
        )
```

- [ ] **Step 9.4: Run — PASS**

Run: `pytest tests/test_provenance.py -v`
Expected: 3 passed.

- [ ] **Step 9.5: Commit**

```bash
git add src/tamga/provenance.py tests/test_provenance.py
git commit -m "feat: Provenance record with current() factory and round-trip serialization"
```

---

## Task 10: `config.schema` — pydantic `StudyConfig`

**Files:**
- Create: `src/tamga/config/__init__.py`
- Create: `src/tamga/config/schema.py`
- Create: `tests/config/__init__.py`
- Create: `tests/config/test_schema.py`

- [ ] **Step 10.1: Create package `__init__.py` files**

`src/tamga/config/__init__.py`:

```python
"""Configuration schema and resolution."""

from tamga.config.schema import (
    CacheConfig,
    CorpusConfig,
    FeatureConfig,
    MethodConfig,
    OutputConfig,
    PreprocessConfig,
    ReportConfig,
    StudyConfig,
    VizConfig,
)

__all__ = [
    "CacheConfig",
    "CorpusConfig",
    "FeatureConfig",
    "MethodConfig",
    "OutputConfig",
    "PreprocessConfig",
    "ReportConfig",
    "StudyConfig",
    "VizConfig",
]
```

`tests/config/__init__.py`: (empty)

```python
```

- [ ] **Step 10.2: Write failing tests in `tests/config/test_schema.py`**

```python
"""Tests for the StudyConfig pydantic schema."""

import pytest

from tamga.config.schema import StudyConfig


VALID_MINIMAL = {
    "name": "demo",
    "seed": 42,
    "corpus": {"path": "corpus/"},
    "preprocess": {},
    "features": [],
    "methods": [],
    "viz": {},
    "report": {},
    "cache": {},
    "output": {},
}


def test_minimal_config_validates():
    cfg = StudyConfig(**VALID_MINIMAL)
    assert cfg.name == "demo"
    assert cfg.seed == 42
    assert cfg.corpus.path == "corpus/"


def test_feature_config_requires_id_and_type():
    bad = dict(VALID_MINIMAL, features=[{"type": "mfw"}])
    with pytest.raises(Exception):
        StudyConfig(**bad)


def test_feature_config_accepts_full_entry():
    cfg = StudyConfig(**dict(
        VALID_MINIMAL,
        features=[{"id": "mfw1000", "type": "mfw", "n": 1000, "min_df": 2, "scale": "zscore"}],
    ))
    assert cfg.features[0].id == "mfw1000"
    assert cfg.features[0].params["n"] == 1000


def test_method_config_validates_kind():
    cfg = StudyConfig(**dict(
        VALID_MINIMAL,
        methods=[{"id": "d1", "kind": "delta", "method": "burrows", "features": "mfw1000"}],
    ))
    assert cfg.methods[0].kind == "delta"
    assert cfg.methods[0].params["method"] == "burrows"


def test_viz_config_defaults():
    cfg = StudyConfig(**VALID_MINIMAL)
    assert cfg.viz.dpi == 300
    assert "pdf" in cfg.viz.format
    assert "png" in cfg.viz.format


def test_report_offline_default_false():
    cfg = StudyConfig(**VALID_MINIMAL)
    assert cfg.report.offline is False


def test_seed_default_is_42():
    cfg = StudyConfig(**dict(VALID_MINIMAL, seed=None))
    # pydantic should use the default
    assert cfg.seed == 42


def test_round_trip_dict_json():
    cfg = StudyConfig(**VALID_MINIMAL)
    redump = cfg.model_dump()
    again = StudyConfig(**redump)
    assert again == cfg
```

- [ ] **Step 10.3: Run — FAIL**

Run: `pytest tests/config/test_schema.py -v`
Expected: ImportError.

- [ ] **Step 10.4: Implement `src/tamga/config/schema.py`**

```python
"""Pydantic schema for `study.yaml`.

Only the shape of the config is validated here; semantic validation (e.g., that `features:`
references exist, that methods point at real extractors) happens at execution time against the
extractor/method registries — not at parse time.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CvKind = Literal["stratified", "loao", "group_kfold", "leave_one_text_out"]
MethodKind = Literal["delta", "zeta", "reduce", "cluster", "consensus", "classify", "bayesian"]
FeatureType = Literal[
    "mfw",
    "word_ngram",
    "char_ngram",
    "pos_ngram",
    "dependency_bigram",
    "function_word",
    "punctuation",
    "lexical_diversity",
    "readability",
    "sentence_length",
    "sentence_embedding",
    "contextual_embedding",
]

_STRICT_MODEL = ConfigDict(extra="forbid")


def _collect_extras_into_params(values: Any, known: set[str]) -> Any:
    """model_validator(mode='before') helper: move unknown top-level keys into `params`.

    If the input already has an explicit `params` dict, it is respected as-is (no extras
    collection happens). Otherwise, any key outside `known` is treated as a parameter.
    """
    if not isinstance(values, dict):
        return values
    if "params" in values:
        return values
    extras = {k: v for k, v in values.items() if k not in known}
    kept = {k: v for k, v in values.items() if k in known}
    kept["params"] = extras
    return kept


class CorpusConfig(BaseModel):
    model_config = _STRICT_MODEL
    path: str
    metadata: str | None = None
    filter: dict[str, Any] = Field(default_factory=dict)


class SpacyConfig(BaseModel):
    model_config = _STRICT_MODEL
    model: str = "en_core_web_trf"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    exclude: list[str] = Field(default_factory=list)


class NormalizeConfig(BaseModel):
    model_config = _STRICT_MODEL
    lowercase: bool = False
    strip_punct: bool = False
    collapse_numerals: bool = False
    expand_contractions: bool = False


class PreprocessConfig(BaseModel):
    model_config = _STRICT_MODEL
    spacy: SpacyConfig = Field(default_factory=SpacyConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)


class FeatureConfig(BaseModel):
    """A named feature extractor entry.

    `type` selects the extractor; `params` holds the remaining keys (everything except `id` and
    `type`). The extractor registry is responsible for validating `params` against the extractor's
    signature at execution time.
    """

    model_config = _STRICT_MODEL

    id: str
    type: FeatureType
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_params(cls, values: Any) -> Any:
        return _collect_extras_into_params(values, known={"id", "type"})


class CvConfig(BaseModel):
    model_config = _STRICT_MODEL
    kind: CvKind = "stratified"
    groups_from: str | None = None
    folds: int | None = None


class MethodConfig(BaseModel):
    """A named analysis step."""

    model_config = _STRICT_MODEL

    id: str
    kind: MethodKind
    features: str | list[str] | None = None
    group_by: str | None = None
    cv: CvConfig | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_params(cls, values: Any) -> Any:
        return _collect_extras_into_params(
            values, known={"id", "kind", "features", "group_by", "cv"}
        )


class VizConfig(BaseModel):
    model_config = _STRICT_MODEL
    format: list[Literal["pdf", "png", "svg", "eps", "tiff"]] = Field(
        default_factory=lambda: ["pdf", "png"]
    )
    dpi: int = 300
    style: str = "default"
    palette: str = "colorblind"


class ReportConfig(BaseModel):
    model_config = _STRICT_MODEL
    format: Literal["html", "md", "none"] = "none"
    offline: bool = False
    include: list[str] = Field(default_factory=lambda: ["corpus", "config", "provenance", "results"])
    title: str | None = None


class CacheConfig(BaseModel):
    model_config = _STRICT_MODEL
    dir: str = ".tamga/cache"
    reuse: bool = True


class OutputConfig(BaseModel):
    model_config = _STRICT_MODEL
    dir: str = "results/"
    timestamp: bool = True


class StudyConfig(BaseModel):
    model_config = _STRICT_MODEL

    name: str = "unnamed-study"
    seed: int = 42

    corpus: CorpusConfig
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    features: list[FeatureConfig] = Field(default_factory=list)
    methods: list[MethodConfig] = Field(default_factory=list)
    viz: VizConfig = Field(default_factory=VizConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("seed", mode="before")
    @classmethod
    def _default_seed(cls, v: Any) -> int:
        return 42 if v is None else v
```

- [ ] **Step 10.5: Run — PASS**

Run: `pytest tests/config/test_schema.py -v`
Expected: 8 passed.

- [ ] **Step 10.6: Commit**

```bash
git add src/tamga/config/ tests/config/
git commit -m "feat(config): StudyConfig pydantic schema with feature/method param passthrough"
```

---

## Task 11: `config.resolve` — YAML loading + layered resolution

**Files:**
- Create: `src/tamga/config/resolve.py`
- Create: `tests/config/test_resolve.py`
- Modify: `src/tamga/config/__init__.py` (export `load_config`, `resolve_config`)

- [ ] **Step 11.1: Write failing tests in `tests/config/test_resolve.py`**

```python
"""Tests for config YAML loading and precedence layering."""

from pathlib import Path

import pytest

from tamga.config import StudyConfig, load_config, resolve_config


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


def test_load_config_parses_yaml(tmp_path: Path):
    cfg_file = _write(tmp_path / "study.yaml", """
name: t1
seed: 7
corpus: {path: corpus/}
""")
    cfg = load_config(cfg_file)
    assert isinstance(cfg, StudyConfig)
    assert cfg.name == "t1"
    assert cfg.seed == 7


def test_load_config_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_resolve_config_cli_overrides_file(tmp_path: Path):
    cfg_file = _write(tmp_path / "study.yaml", """
name: t1
seed: 7
corpus: {path: corpus/}
""")
    resolved = resolve_config(config_file=cfg_file, cli_overrides={"seed": 99})
    assert resolved.seed == 99
    assert resolved.name == "t1"


def test_resolve_config_deep_merges_nested_overrides(tmp_path: Path):
    cfg_file = _write(tmp_path / "study.yaml", """
name: t1
seed: 7
corpus: {path: corpus/, metadata: meta.tsv}
viz: {dpi: 300, format: [pdf, png]}
""")
    resolved = resolve_config(config_file=cfg_file, cli_overrides={"viz": {"dpi": 600}})
    assert resolved.viz.dpi == 600
    # unspecified nested keys are preserved
    assert "pdf" in resolved.viz.format


def test_resolve_config_with_no_file_uses_defaults(tmp_path: Path):
    resolved = resolve_config(
        config_file=None,
        cli_overrides={"name": "cli-only", "corpus": {"path": "some/"}},
    )
    assert resolved.name == "cli-only"
    assert resolved.corpus.path == "some/"
```

- [ ] **Step 11.2: Run — FAIL**

Run: `pytest tests/config/test_resolve.py -v`
Expected: FAIL.

- [ ] **Step 11.3: Implement `src/tamga/config/resolve.py`**

```python
"""YAML loading + deep-merge config resolution.

Precedence (highest wins): `cli_overrides` > `config_file` > package defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tamga.config.schema import StudyConfig


def load_config(path: Path) -> StudyConfig:
    """Parse a `study.yaml` file into a StudyConfig."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return _validate(data)


def resolve_config(
    *,
    config_file: Path | None,
    cli_overrides: dict[str, Any] | None = None,
) -> StudyConfig:
    """Resolve the effective StudyConfig given a config file and CLI overrides.

    CLI overrides are deep-merged on top of the file contents. Keys unset at both layers fall
    back to the package defaults declared in `StudyConfig`.
    """
    base: dict[str, Any] = {}
    if config_file is not None:
        with Path(config_file).open("r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
    merged = _deep_merge(base, cli_overrides or {})
    return _validate(merged)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, overlay_value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(overlay_value, dict):
            out[key] = _deep_merge(out[key], overlay_value)
        else:
            out[key] = overlay_value
    return out


def _validate(data: dict[str, Any]) -> StudyConfig:
    # StudyConfig requires `corpus`; if absent, fill in a placeholder the caller is expected to
    # override via CLI / subsequent validation. This allows `resolve_config` to be reused for
    # partial inspection commands like `tamga config show`.
    data.setdefault("corpus", {"path": ""})
    return StudyConfig.model_validate(data)
```

- [ ] **Step 11.4: Update `src/tamga/config/__init__.py`**

```python
"""Configuration schema and resolution."""

from tamga.config.resolve import load_config, resolve_config
from tamga.config.schema import (
    CacheConfig,
    CorpusConfig,
    FeatureConfig,
    MethodConfig,
    OutputConfig,
    PreprocessConfig,
    ReportConfig,
    StudyConfig,
    VizConfig,
)

__all__ = [
    "CacheConfig",
    "CorpusConfig",
    "FeatureConfig",
    "MethodConfig",
    "OutputConfig",
    "PreprocessConfig",
    "ReportConfig",
    "StudyConfig",
    "VizConfig",
    "load_config",
    "resolve_config",
]
```

- [ ] **Step 11.5: Run — PASS**

Run: `pytest tests/config/test_resolve.py -v`
Expected: 5 passed.

- [ ] **Step 11.6: Commit**

```bash
git add src/tamga/config/resolve.py src/tamga/config/__init__.py tests/config/test_resolve.py
git commit -m "feat(config): load_config and resolve_config with deep-merge precedence"
```

---

## Task 12: `preprocess.cache` — DocBin cache key + read/write

**Files:**
- Create: `src/tamga/preprocess/__init__.py`
- Create: `src/tamga/preprocess/cache.py`
- Create: `tests/preprocess/__init__.py`
- Create: `tests/preprocess/test_cache.py`

- [ ] **Step 12.1: Create package `__init__.py` files**

`src/tamga/preprocess/__init__.py`:

```python
"""spaCy preprocessing pipeline + DocBin cache."""
```

`tests/preprocess/__init__.py`: (empty)

```python
```

- [ ] **Step 12.2: Write failing tests in `tests/preprocess/test_cache.py`**

```python
"""Tests for the DocBin cache — key derivation and round-trip (mocked spaCy)."""

from pathlib import Path

import pytest

from tamga.preprocess.cache import DocBinCache, cache_key


def test_cache_key_is_deterministic():
    a = cache_key("doc-hash", "en_core_web_sm", "3.7.2", ["ner"])
    b = cache_key("doc-hash", "en_core_web_sm", "3.7.2", ["ner"])
    assert a == b


def test_cache_key_changes_with_any_input():
    base = cache_key("doc-hash", "en_core_web_sm", "3.7.2", [])
    assert cache_key("other-hash", "en_core_web_sm", "3.7.2", []) != base
    assert cache_key("doc-hash", "en_core_web_lg", "3.7.2", []) != base
    assert cache_key("doc-hash", "en_core_web_sm", "3.7.3", []) != base
    assert cache_key("doc-hash", "en_core_web_sm", "3.7.2", ["ner"]) != base


def test_cache_key_is_order_independent_for_excluded_components():
    a = cache_key("doc-hash", "en_core_web_sm", "3.7.2", ["ner", "parser"])
    b = cache_key("doc-hash", "en_core_web_sm", "3.7.2", ["parser", "ner"])
    assert a == b


def test_cache_miss_returns_none(tmp_path: Path):
    c = DocBinCache(tmp_path)
    assert c.get("nonexistent-key") is None


def test_cache_put_get_round_trip_bytes(tmp_path: Path):
    c = DocBinCache(tmp_path)
    payload = b"\x00\x01\x02fake-docbin-bytes"
    c.put("k1", payload)
    assert c.get("k1") == payload


def test_cache_size_bytes_reports_stored_payloads(tmp_path: Path):
    c = DocBinCache(tmp_path)
    c.put("k1", b"x" * 100)
    c.put("k2", b"y" * 50)
    assert c.size_bytes() == 150


def test_cache_clear_removes_all_entries(tmp_path: Path):
    c = DocBinCache(tmp_path)
    c.put("k1", b"x")
    c.put("k2", b"y")
    assert len(c.keys()) == 2
    c.clear()
    assert c.keys() == []
    assert c.size_bytes() == 0
```

- [ ] **Step 12.3: Run — FAIL**

Run: `pytest tests/preprocess/test_cache.py -v`
Expected: FAIL.

- [ ] **Step 12.4: Implement `src/tamga/preprocess/cache.py`**

```python
"""Content-addressable cache for spaCy `DocBin` blobs.

Keyed by `(document_hash, spacy_model, spacy_version, sorted_excluded_components)`. At this stage
the cache stores raw bytes — Task 13 will wire up `DocBin` serialisation on top.
"""

from __future__ import annotations

from pathlib import Path

from tamga.plumbing.hashing import hash_mapping


def cache_key(
    document_hash: str,
    spacy_model: str,
    spacy_version: str,
    excluded_components: list[str],
) -> str:
    """Return a stable cache key for a (document, spaCy configuration) pair."""
    return hash_mapping(
        {
            "doc": document_hash,
            "model": spacy_model,
            "version": spacy_version,
            "exclude": sorted(excluded_components),
        }
    )


class DocBinCache:
    """Directory-backed cache. One file per key, named `<key>.docbin`."""

    _EXT = ".docbin"

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}{self._EXT}"

    def get(self, key: str) -> bytes | None:
        p = self._path(key)
        if not p.is_file():
            return None
        return p.read_bytes()

    def put(self, key: str, payload: bytes) -> None:
        self._path(key).write_bytes(payload)

    def keys(self) -> list[str]:
        return sorted(f.stem for f in self.directory.glob(f"*{self._EXT}"))

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.directory.glob(f"*{self._EXT}"))

    def clear(self) -> None:
        for f in self.directory.glob(f"*{self._EXT}"):
            f.unlink()
```

- [ ] **Step 12.5: Run — PASS**

Run: `pytest tests/preprocess/test_cache.py -v`
Expected: 7 passed.

- [ ] **Step 12.6: Commit**

```bash
git add src/tamga/preprocess/ tests/preprocess/
git commit -m "feat(preprocess): DocBinCache content-addressable cache + deterministic key"
```

---

## Task 13: `preprocess.pipeline` — spaCy wrapper with DocBin caching

**Files:**
- Create: `src/tamga/preprocess/pipeline.py`
- Create: `tests/preprocess/test_pipeline.py`

- [ ] **Step 13.1: Install `en_core_web_sm` once (required for this test file)**

Run: `python -m spacy download en_core_web_sm`
Expected: model installed; verify via `python -c "import spacy; spacy.load('en_core_web_sm')"`.

- [ ] **Step 13.2: Write failing tests in `tests/preprocess/test_pipeline.py`**

```python
"""Tests for the spaCy preprocessing pipeline — run against the small English model."""

from pathlib import Path

import pytest

from tamga.corpus import Corpus, Document
from tamga.preprocess.pipeline import ParsedCorpus, SpacyPipeline


pytestmark = pytest.mark.spacy


def _tiny(text: str) -> Document:
    return Document(id=f"d-{hash(text) & 0xffff}", text=text, metadata={})


def test_pipeline_parses_documents(tmp_path: Path):
    corpus = Corpus(documents=[_tiny("Hello world."), _tiny("spaCy parses sentences.")])
    pipe = SpacyPipeline(model="en_core_web_sm", cache_dir=tmp_path)
    parsed = pipe.parse(corpus)
    assert isinstance(parsed, ParsedCorpus)
    assert len(parsed) == 2
    for doc in parsed.spacy_docs():
        assert len(list(doc.sents)) >= 1


def test_pipeline_cache_hit_is_fast(tmp_path: Path):
    corpus = Corpus(documents=[_tiny("The same sentence every time.")])
    pipe = SpacyPipeline(model="en_core_web_sm", cache_dir=tmp_path)

    pipe.parse(corpus)  # cold
    keys_after_first = set(pipe.cache.keys())

    pipe.parse(corpus)  # warm — must use cache without parsing again
    keys_after_second = set(pipe.cache.keys())

    assert keys_after_first == keys_after_second
    assert len(keys_after_first) == 1


def test_pipeline_cache_invalidates_on_text_change(tmp_path: Path):
    pipe = SpacyPipeline(model="en_core_web_sm", cache_dir=tmp_path)
    pipe.parse(Corpus(documents=[_tiny("original text")]))
    pipe.parse(Corpus(documents=[_tiny("different text")]))
    assert len(pipe.cache.keys()) == 2


def test_parsed_corpus_iteration_preserves_order(tmp_path: Path):
    corpus = Corpus(documents=[_tiny("A."), _tiny("B."), _tiny("C.")])
    pipe = SpacyPipeline(model="en_core_web_sm", cache_dir=tmp_path)
    parsed = pipe.parse(corpus)
    texts = [doc.text.strip() for doc in parsed.spacy_docs()]
    assert texts == ["A.", "B.", "C."]
```

- [ ] **Step 13.3: Run — FAIL**

Run: `pytest tests/preprocess/test_pipeline.py -v`
Expected: ImportError.

- [ ] **Step 13.4: Implement `src/tamga/preprocess/pipeline.py`**

```python
"""High-level spaCy parsing wrapper, DocBin-cached.

Usage:

    pipe = SpacyPipeline(model="en_core_web_sm", cache_dir=".tamga/cache/docbin")
    parsed = pipe.parse(corpus)
    for spacy_doc in parsed.spacy_docs():
        ...

Cache hits skip the spaCy pipeline entirely. Cache keys derive from `(doc_hash, model, version,
excluded_components)` — see `preprocess.cache.cache_key`.
"""

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
    """A Corpus paired with its parsed spaCy Docs, preserving document order."""

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
    """Wraps a spaCy Language with a per-document DocBin cache."""

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
        return spacy.__version__

    def _key(self, doc: Document) -> str:
        return cache_key(doc.hash, self.model, self.spacy_version, self.exclude)

    def parse(self, corpus: Corpus) -> ParsedCorpus:
        """Parse every document, using the cache whenever possible."""
        parsed: list[Doc] = []
        to_parse_indices: list[int] = []
        to_parse_texts: list[str] = []

        for i, doc in enumerate(corpus.documents):
            cached = self.cache.get(self._key(doc))
            if cached is not None:
                bin_ = DocBin().from_bytes(cached)
                (spacy_doc,) = list(bin_.get_docs(self.nlp.vocab))
                parsed.append(spacy_doc)
            else:
                parsed.append(None)  # type: ignore[arg-type]
                to_parse_indices.append(i)
                to_parse_texts.append(doc.text)

        if to_parse_texts:
            _log.info("parsing %d documents (%d cached)", len(to_parse_texts), len(corpus) - len(to_parse_texts))
            for i, spacy_doc in zip(to_parse_indices, self.nlp.pipe(to_parse_texts), strict=True):
                parsed[i] = spacy_doc
                bin_ = DocBin(docs=[spacy_doc])
                self.cache.put(self._key(corpus.documents[i]), bin_.to_bytes())

        return ParsedCorpus(corpus=corpus, docs=parsed)
```

- [ ] **Step 13.5: Run — PASS**

Run: `pytest tests/preprocess/test_pipeline.py -v -m spacy`
Expected: 4 passed.

- [ ] **Step 13.6: Commit**

```bash
git add src/tamga/preprocess/pipeline.py tests/preprocess/test_pipeline.py
git commit -m "feat(preprocess): SpacyPipeline with DocBin cache + ParsedCorpus wrapper"
```

---

## Task 14: CLI skeleton — Typer app + `--version`

**Files:**
- Create: `src/tamga/cli/__init__.py`
- Create: `tests/cli/__init__.py`
- Create: `tests/cli/test_skeleton.py`

- [ ] **Step 14.1: Create `src/tamga/cli/__init__.py`**

```python
"""Typer CLI entry point."""

from __future__ import annotations

import typer
from rich.console import Console

from tamga._version import __version__

console = Console()
app = typer.Typer(
    name="tamga",
    help="tamga — computational stylometry (next-generation Python replacement for R's Stylo).",
    no_args_is_help=True,
    add_completion=True,
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"tamga {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """tamga — computational stylometry."""
```

- [ ] **Step 14.2: Create `tests/cli/__init__.py` (empty)**

```python
```

- [ ] **Step 14.3: Write failing tests in `tests/cli/test_skeleton.py`**

```python
"""Tests for the CLI skeleton."""

from typer.testing import CliRunner

from tamga import __version__
from tamga.cli import app

runner = CliRunner()


def test_cli_help_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "tamga" in result.stdout.lower()


def test_cli_version_flag_prints_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_no_args_shows_help():
    result = runner.invoke(app, [])
    # Typer exits with 0 (help) or 2 (usage-error), both acceptable — check output instead.
    assert "Usage" in result.stdout or "usage" in result.stdout
```

- [ ] **Step 14.4: Run — PASS**

Run: `pytest tests/cli/test_skeleton.py -v`
Expected: 3 passed.

- [ ] **Step 14.5: Verify the console script works**

Run: `tamga --version`
Expected: prints `tamga 0.1.0.dev0`.

Run: `tamga --help`
Expected: prints help text with "tamga — computational stylometry".

- [ ] **Step 14.6: Commit**

```bash
git add src/tamga/cli/__init__.py tests/cli/__init__.py tests/cli/test_skeleton.py
git commit -m "feat(cli): Typer app skeleton with --version and --help"
```

---

## Task 15: Scaffold templates

**Files:**
- Create: `src/tamga/scaffold/__init__.py`
- Create: `src/tamga/scaffold/templates/study.yaml.j2`
- Create: `src/tamga/scaffold/templates/README.md.j2`
- Create: `src/tamga/scaffold/templates/gitignore.tmpl`
- Create: `src/tamga/scaffold/scaffolder.py`
- Create: `tests/scaffold/__init__.py`
- Create: `tests/scaffold/test_scaffolder.py`
- Modify: `pyproject.toml` (ensure templates ship with the wheel)

- [ ] **Step 15.1: Update `pyproject.toml` wheel target to include templates**

Append under `[tool.hatch.build.targets.wheel]`:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/tamga/scaffold/templates" = "tamga/scaffold/templates"
```

- [ ] **Step 15.2: Create `src/tamga/scaffold/__init__.py`**

```python
"""Project-scaffolding logic for `tamga init`."""

from tamga.scaffold.scaffolder import scaffold_project

__all__ = ["scaffold_project"]
```

- [ ] **Step 15.3: Create `src/tamga/scaffold/templates/study.yaml.j2`**

```jinja2
# tamga study config — {{ name }}
name: {{ name }}
seed: 42

corpus:
  path: corpus/
  # metadata: corpus/metadata.tsv    # uncomment when you provide a metadata TSV

preprocess:
  spacy:
    model: en_core_web_trf
    device: auto                     # auto | cpu | mps | cuda

features:
  - {id: mfw1000, type: mfw, n: 1000, min_df: 2, scale: zscore}

methods:
  - id: burrows
    kind: delta
    method: burrows
    features: mfw1000
    group_by: author

viz:
  format: [pdf, png]
  dpi: 300

report:
  format: none                       # 'html' or 'md' to enable reports

cache:
  dir: .tamga/cache

output:
  dir: results/
  timestamp: true
```

- [ ] **Step 15.4: Create `src/tamga/scaffold/templates/README.md.j2`**

```jinja2
# {{ name }}

tamga study project.

## Reproduce

```bash
uv pip install tamga
tamga run study.yaml
```

## Layout

- `study.yaml` — analysis configuration.
- `corpus/` — raw `.txt` files (one document per file).
- `corpus/metadata.tsv` — optional: filename → author, group, year, ... (tab-separated).
- `results/<timestamp>/` — per-run artifacts (generated).
- `reports/<timestamp>.html` — per-run reports (generated, opt-in).
- `.tamga/` — cache and lockfile (gitignored).

Created with tamga {{ tamga_version }} on {{ created_on }}.
```

- [ ] **Step 15.5: Create `src/tamga/scaffold/templates/gitignore.tmpl`** (plain file, not Jinja)

```
.tamga/
results/
reports/
*.docbin
.venv/
__pycache__/
*.py[cod]
```

- [ ] **Step 15.6: Write failing tests in `tests/scaffold/test_scaffolder.py`**

```python
"""Tests for project scaffolding."""

from pathlib import Path

import pytest

from tamga.scaffold import scaffold_project


def test_scaffold_creates_expected_layout(tmp_path: Path):
    target = tmp_path / "my-study"
    created = scaffold_project(name="my-study", target=target)
    assert created == target
    assert (target / "study.yaml").is_file()
    assert (target / "README.md").is_file()
    assert (target / ".gitignore").is_file()
    assert (target / "corpus").is_dir()
    assert (target / ".tamga" / "cache").is_dir()


def test_scaffold_refuses_to_overwrite(tmp_path: Path):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "file").write_text("dont destroy me")
    with pytest.raises(FileExistsError):
        scaffold_project(name="existing", target=target)


def test_scaffold_force_overrides_existing_directory(tmp_path: Path):
    target = tmp_path / "force"
    target.mkdir()
    (target / "preexisting.txt").write_text("hi")
    scaffold_project(name="force", target=target, force=True)
    assert (target / "study.yaml").is_file()
    assert (target / "preexisting.txt").is_file()  # force only fills gaps; never deletes user files


def test_scaffold_study_yaml_contains_project_name(tmp_path: Path):
    target = tmp_path / "named-project"
    scaffold_project(name="named-project", target=target)
    body = (target / "study.yaml").read_text()
    assert "name: named-project" in body
```

- [ ] **Step 15.7: Run — FAIL**

Run: `pytest tests/scaffold/test_scaffolder.py -v`
Expected: FAIL.

- [ ] **Step 15.8: Implement `src/tamga/scaffold/scaffolder.py`**

```python
"""Scaffold a new tamga project directory."""

from __future__ import annotations

from datetime import datetime
from importlib import resources
from pathlib import Path

from jinja2 import Environment

from tamga._version import __version__

_TEMPLATE_PKG = "tamga.scaffold.templates"


def scaffold_project(name: str, target: Path, *, force: bool = False) -> Path:
    """Create a new tamga project at `target`.

    Refuses to create on top of an existing non-empty directory unless `force=True`. When `force`
    is True, existing files are left alone and only missing scaffold files are written.
    """
    target = Path(target)
    if target.exists():
        if any(target.iterdir()) and not force:
            raise FileExistsError(f"{target} exists and is not empty (use force=True to fill in)")
    else:
        target.mkdir(parents=True)

    (target / "corpus").mkdir(exist_ok=True)
    (target / ".tamga" / "cache").mkdir(parents=True, exist_ok=True)
    (target / "results").mkdir(exist_ok=True)
    (target / "reports").mkdir(exist_ok=True)

    env = Environment(trim_blocks=False, lstrip_blocks=False, keep_trailing_newline=True)
    ctx = {
        "name": name,
        "tamga_version": __version__,
        "created_on": datetime.now().strftime("%Y-%m-%d"),
    }

    _render(env, "study.yaml.j2", target / "study.yaml", ctx)
    _render(env, "README.md.j2", target / "README.md", ctx)
    _copy("gitignore.tmpl", target / ".gitignore")
    return target


def _render(env: Environment, template_name: str, dest: Path, ctx: dict[str, object]) -> None:
    if dest.exists():
        return
    src = resources.files(_TEMPLATE_PKG) / template_name
    template = env.from_string(src.read_text(encoding="utf-8"))
    dest.write_text(template.render(**ctx), encoding="utf-8")


def _copy(template_name: str, dest: Path) -> None:
    if dest.exists():
        return
    src = resources.files(_TEMPLATE_PKG) / template_name
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
```

- [ ] **Step 15.9: Create `tests/scaffold/__init__.py` (empty)**

```python
```

- [ ] **Step 15.10: Run — PASS**

Run: `pytest tests/scaffold/test_scaffolder.py -v`
Expected: 4 passed.

- [ ] **Step 15.11: Commit**

```bash
git add src/tamga/scaffold/ tests/scaffold/ pyproject.toml
git commit -m "feat(scaffold): scaffold_project with Jinja templates for study.yaml/README/.gitignore"
```

---

## Task 16: `tamga init` CLI command

**Files:**
- Create: `src/tamga/cli/init_cmd.py`
- Create: `tests/cli/test_init.py`
- Modify: `src/tamga/cli/__init__.py` (register command)

- [ ] **Step 16.1: Write failing tests in `tests/cli/test_init.py`**

```python
"""Tests for `tamga init`."""

from pathlib import Path

from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()


def test_init_creates_project(tmp_path: Path):
    target = tmp_path / "my-study"
    result = runner.invoke(app, ["init", "my-study", "--target", str(target)])
    assert result.exit_code == 0, result.stdout
    assert (target / "study.yaml").is_file()
    assert (target / "corpus").is_dir()


def test_init_refuses_to_clobber(tmp_path: Path):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "file").write_text("hi")
    result = runner.invoke(app, ["init", "existing", "--target", str(target)])
    assert result.exit_code != 0


def test_init_force_fills_in(tmp_path: Path):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "file").write_text("hi")
    result = runner.invoke(app, ["init", "existing", "--target", str(target), "--force"])
    assert result.exit_code == 0
    assert (target / "study.yaml").is_file()
    assert (target / "file").read_text() == "hi"  # untouched
```

- [ ] **Step 16.2: Run — FAIL**

Run: `pytest tests/cli/test_init.py -v`
Expected: FAIL (command not found).

- [ ] **Step 16.3: Implement `src/tamga/cli/init_cmd.py`**

```python
"""`tamga init <name>` — scaffold a new project."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from tamga.scaffold import scaffold_project

console = Console()


def init_command(
    name: str = typer.Argument(..., help="Project name; used as directory name and in study.yaml."),
    target: Path | None = typer.Option(
        None, "--target", "-t", help="Directory to create (default: ./<name>)."
    ),
    force: bool = typer.Option(False, "--force", help="Fill in missing files even if directory is non-empty."),
) -> None:
    """Scaffold a new tamga project directory."""
    dest = target if target is not None else Path.cwd() / name
    try:
        created = scaffold_project(name=name, target=dest, force=force)
    except FileExistsError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]created project[/green] {created}")
    console.print(f"  cd {created}")
    console.print("  # edit study.yaml; drop .txt files in corpus/; then run:")
    console.print("  tamga run study.yaml")
```

- [ ] **Step 16.4: Register in `src/tamga/cli/__init__.py`**

Replace the file contents with:

```python
"""Typer CLI entry point."""

from __future__ import annotations

import typer
from rich.console import Console

from tamga._version import __version__
from tamga.cli.init_cmd import init_command

console = Console()
app = typer.Typer(
    name="tamga",
    help="tamga — computational stylometry (next-generation Python replacement for R's Stylo).",
    no_args_is_help=True,
    add_completion=True,
)

app.command(name="init")(init_command)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"tamga {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """tamga — computational stylometry."""
```

- [ ] **Step 16.5: Run — PASS**

Run: `pytest tests/cli/test_init.py -v`
Expected: 3 passed.

- [ ] **Step 16.6: Commit**

```bash
git add src/tamga/cli/init_cmd.py src/tamga/cli/__init__.py tests/cli/test_init.py
git commit -m "feat(cli): tamga init subcommand"
```

---

## Task 17: `tamga ingest` CLI command

**Files:**
- Create: `src/tamga/cli/ingest_cmd.py`
- Create: `tests/cli/test_ingest.py`
- Modify: `src/tamga/cli/__init__.py` (register `ingest`)

- [ ] **Step 17.1: Write failing tests in `tests/cli/test_ingest.py`**

```python
"""Tests for `tamga ingest`."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()

FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini_corpus"


pytestmark = pytest.mark.spacy


def test_ingest_parses_corpus_without_metadata(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    result = runner.invoke(
        app,
        [
            "ingest",
            str(FIXTURES),
            "--cache-dir", str(cache_dir),
            "--spacy-model", "en_core_web_sm",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # After ingest, the docbin cache should have 4 entries.
    docbin_dir = cache_dir / "docbin"
    assert len(list(docbin_dir.glob("*.docbin"))) == 4


def test_ingest_uses_metadata_when_provided(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "ingest",
            str(FIXTURES),
            "--metadata", str(FIXTURES / "metadata.tsv"),
            "--cache-dir", str(tmp_path / "cache"),
            "--spacy-model", "en_core_web_sm",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "4 documents" in result.stdout


def test_ingest_reports_cache_hits_on_rerun(tmp_path: Path):
    cache_dir = tmp_path / "cache"

    first = runner.invoke(
        app,
        ["ingest", str(FIXTURES), "--cache-dir", str(cache_dir), "--spacy-model", "en_core_web_sm"],
    )
    assert first.exit_code == 0

    second = runner.invoke(
        app,
        ["ingest", str(FIXTURES), "--cache-dir", str(cache_dir), "--spacy-model", "en_core_web_sm"],
    )
    assert second.exit_code == 0
    assert "cached" in second.stdout.lower() or "cache" in second.stdout.lower()
```

- [ ] **Step 17.2: Run — FAIL**

Run: `pytest tests/cli/test_ingest.py -v -m spacy`
Expected: FAIL.

- [ ] **Step 17.3: Implement `src/tamga/cli/ingest_cmd.py`**

```python
"""`tamga ingest <path>` — parse a corpus and populate the DocBin cache."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from tamga.io import load_corpus
from tamga.preprocess.pipeline import SpacyPipeline

console = Console()


def ingest_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    metadata: Path | None = typer.Option(
        None, "--metadata", "-m", exists=True, dir_okay=False,
        help="TSV file mapping filename to metadata fields."
    ),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Every file must have metadata."),
    cache_dir: Path = typer.Option(
        Path(".tamga/cache"), "--cache-dir", help="Directory for the DocBin cache."
    ),
    spacy_model: str = typer.Option(
        "en_core_web_trf", "--spacy-model", help="spaCy model name."
    ),
    exclude: list[str] | None = typer.Option(
        None, "--exclude", help="spaCy pipeline components to skip."
    ),
) -> None:
    """Parse a corpus directory and cache spaCy parses."""
    corpus = load_corpus(path, metadata=metadata, strict=strict)
    console.print(f"[green]loaded[/green] {len(corpus)} documents from {path}")

    pipe = SpacyPipeline(
        model=spacy_model,
        cache_dir=cache_dir / "docbin",
        exclude=exclude or [],
    )

    docbin_before = set(pipe.cache.keys())
    pipe.parse(corpus)
    docbin_after = set(pipe.cache.keys())
    cached_hits = len(corpus) - len(docbin_after - docbin_before)

    console.print(
        f"[green]parsed[/green] {len(corpus)} documents"
        f" ({cached_hits} cached, {len(docbin_after - docbin_before)} newly parsed)"
    )
    console.print(f"  cache: {cache_dir / 'docbin'} ({pipe.cache.size_bytes()} bytes)")
```

- [ ] **Step 17.4: Register in `src/tamga/cli/__init__.py`**

Add alongside `init_command`:

```python
from tamga.cli.ingest_cmd import ingest_command
...
app.command(name="ingest")(ingest_command)
```

Updated full file:

```python
"""Typer CLI entry point."""

from __future__ import annotations

import typer
from rich.console import Console

from tamga._version import __version__
from tamga.cli.ingest_cmd import ingest_command
from tamga.cli.init_cmd import init_command

console = Console()
app = typer.Typer(
    name="tamga",
    help="tamga — computational stylometry (next-generation Python replacement for R's Stylo).",
    no_args_is_help=True,
    add_completion=True,
)

app.command(name="init")(init_command)
app.command(name="ingest")(ingest_command)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"tamga {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """tamga — computational stylometry."""
```

- [ ] **Step 17.5: Run — PASS**

Run: `pytest tests/cli/test_ingest.py -v -m spacy`
Expected: 3 passed.

- [ ] **Step 17.6: Commit**

```bash
git add src/tamga/cli/ingest_cmd.py src/tamga/cli/__init__.py tests/cli/test_ingest.py
git commit -m "feat(cli): tamga ingest — parse corpus and populate DocBin cache"
```

---

## Task 18: `tamga info` and `tamga cache` commands

**Files:**
- Create: `src/tamga/cli/info_cmd.py`
- Create: `src/tamga/cli/cache_cmd.py`
- Create: `tests/cli/test_info.py`
- Create: `tests/cli/test_cache.py`
- Modify: `src/tamga/cli/__init__.py`

- [ ] **Step 18.1: Write failing tests in `tests/cli/test_info.py`**

```python
"""Tests for `tamga info`."""

from typer.testing import CliRunner

from tamga import __version__
from tamga.cli import app

runner = CliRunner()


def test_info_reports_version():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_info_reports_spacy_version():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "spacy" in result.stdout.lower()
```

- [ ] **Step 18.2: Write failing tests in `tests/cli/test_cache.py`**

```python
"""Tests for `tamga cache`."""

from pathlib import Path

from typer.testing import CliRunner

from tamga.cli import app
from tamga.preprocess.cache import DocBinCache

runner = CliRunner()


def _seed(cache_dir: Path, n: int) -> None:
    (cache_dir / "docbin").mkdir(parents=True, exist_ok=True)
    cache = DocBinCache(cache_dir / "docbin")
    for i in range(n):
        cache.put(f"k{i}", b"x" * (i + 1) * 10)


def test_cache_size_reports_bytes(tmp_path: Path):
    _seed(tmp_path, 3)
    result = runner.invoke(app, ["cache", "size", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "bytes" in result.stdout.lower()


def test_cache_list_lists_keys(tmp_path: Path):
    _seed(tmp_path, 2)
    result = runner.invoke(app, ["cache", "list", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "k0" in result.stdout
    assert "k1" in result.stdout


def test_cache_clear_empties_cache(tmp_path: Path):
    _seed(tmp_path, 2)
    result = runner.invoke(app, ["cache", "clear", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0
    cache = DocBinCache(tmp_path / "docbin")
    assert cache.keys() == []
```

- [ ] **Step 18.3: Run — FAIL**

Run: `pytest tests/cli/test_info.py tests/cli/test_cache.py -v`
Expected: FAIL.

- [ ] **Step 18.4: Implement `src/tamga/cli/info_cmd.py`**

```python
"""`tamga info` — show versions and environment."""

from __future__ import annotations

import platform

import spacy
import typer
from rich.console import Console
from rich.table import Table

from tamga._version import __version__

console = Console()


def info_command() -> None:
    """Print versions, paths, and runtime information."""
    table = Table(title="tamga environment", show_header=False)
    table.add_column("key", style="cyan")
    table.add_column("value")
    table.add_row("tamga", __version__)
    table.add_row("python", platform.python_version())
    table.add_row("platform", platform.platform())
    table.add_row("spacy", spacy.__version__)
    console.print(table)
```

- [ ] **Step 18.5: Implement `src/tamga/cli/cache_cmd.py`**

```python
"""`tamga cache [size|list|clear]` — inspect and manage the DocBin cache."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from tamga.preprocess.cache import DocBinCache

console = Console()

cache_app = typer.Typer(name="cache", help="Inspect and manage the DocBin cache.", no_args_is_help=True)


@cache_app.command("size")
def cache_size(
    cache_dir: Path = typer.Option(Path(".tamga/cache"), "--cache-dir"),
) -> None:
    """Show total bytes stored in the DocBin cache."""
    cache = DocBinCache(cache_dir / "docbin")
    console.print(f"{cache.size_bytes()} bytes across {len(cache.keys())} entries")


@cache_app.command("list")
def cache_list(
    cache_dir: Path = typer.Option(Path(".tamga/cache"), "--cache-dir"),
) -> None:
    """List cache keys."""
    cache = DocBinCache(cache_dir / "docbin")
    for key in cache.keys():
        console.print(key)


@cache_app.command("clear")
def cache_clear(
    cache_dir: Path = typer.Option(Path(".tamga/cache"), "--cache-dir"),
) -> None:
    """Delete every entry from the DocBin cache."""
    cache = DocBinCache(cache_dir / "docbin")
    n = len(cache.keys())
    cache.clear()
    console.print(f"cleared {n} entries from {cache_dir / 'docbin'}")
```

- [ ] **Step 18.6: Register in `src/tamga/cli/__init__.py`**

Full file:

```python
"""Typer CLI entry point."""

from __future__ import annotations

import typer
from rich.console import Console

from tamga._version import __version__
from tamga.cli.cache_cmd import cache_app
from tamga.cli.info_cmd import info_command
from tamga.cli.ingest_cmd import ingest_command
from tamga.cli.init_cmd import init_command

console = Console()
app = typer.Typer(
    name="tamga",
    help="tamga — computational stylometry (next-generation Python replacement for R's Stylo).",
    no_args_is_help=True,
    add_completion=True,
)

app.command(name="init")(init_command)
app.command(name="ingest")(ingest_command)
app.command(name="info")(info_command)
app.add_typer(cache_app, name="cache")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"tamga {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """tamga — computational stylometry."""
```

- [ ] **Step 18.7: Run — PASS**

Run: `pytest tests/cli/test_info.py tests/cli/test_cache.py -v`
Expected: 5 passed.

- [ ] **Step 18.8: Commit**

```bash
git add src/tamga/cli/info_cmd.py src/tamga/cli/cache_cmd.py src/tamga/cli/__init__.py tests/cli/test_info.py tests/cli/test_cache.py
git commit -m "feat(cli): tamga info and tamga cache (size/list/clear)"
```

---

## Task 19: End-to-end integration test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 19.1: Write the integration test**

```python
"""End-to-end integration: init → drop fixtures → ingest → info."""

from pathlib import Path
from shutil import copy, copytree

import pytest
from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()

FIXTURES = Path(__file__).parent / "fixtures" / "mini_corpus"


pytestmark = [pytest.mark.integration, pytest.mark.spacy]


def test_end_to_end_init_ingest_info(tmp_path: Path):
    project = tmp_path / "demo-study"

    # 1. init
    r_init = runner.invoke(app, ["init", "demo-study", "--target", str(project)])
    assert r_init.exit_code == 0, r_init.stdout
    assert (project / "study.yaml").is_file()

    # 2. drop fixture corpus files
    copytree(FIXTURES, project / "corpus", dirs_exist_ok=True)

    # 3. ingest
    r_ingest = runner.invoke(
        app,
        [
            "ingest",
            str(project / "corpus"),
            "--metadata", str(project / "corpus" / "metadata.tsv"),
            "--cache-dir", str(project / ".tamga" / "cache"),
            "--spacy-model", "en_core_web_sm",
        ],
    )
    assert r_ingest.exit_code == 0, r_ingest.stdout

    # 4. cache reports 4 entries
    r_size = runner.invoke(
        app, ["cache", "size", "--cache-dir", str(project / ".tamga" / "cache")]
    )
    assert r_size.exit_code == 0
    assert "4 entries" in r_size.stdout

    # 5. info runs
    r_info = runner.invoke(app, ["info"])
    assert r_info.exit_code == 0
```

- [ ] **Step 19.2: Run — PASS**

Run: `pytest tests/test_integration.py -v -m integration`
Expected: 1 passed.

- [ ] **Step 19.3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end init → ingest → cache → info integration test"
```

---

## Task 20: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 20.1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: "0 6 * * 1"   # weekly Monday 06:00 UTC — catch upstream spaCy breakage

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: uv pip install --system -e ".[dev]"
      - run: ruff check src tests
      - run: ruff format --check src tests
      - run: mypy src

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: uv pip install --system -e ".[dev]"
      - name: Download small spaCy model
        run: python -m spacy download en_core_web_sm
      - run: pytest -n auto --cov=tamga --cov-report=term-missing -m "not slow"
```

- [ ] **Step 20.2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: GitHub Actions — lint + tests on py3.11-3.13, ubuntu+macos, weekly spaCy check"
```

---

## Task 21: Public API exports + `__main__`

**Files:**
- Modify: `src/tamga/__init__.py` (re-export public surface)
- Create: `src/tamga/__main__.py`

- [ ] **Step 21.1: Update `src/tamga/__init__.py`**

```python
"""tamga — next-generation computational stylometry."""

from tamga._version import __version__
from tamga.config import StudyConfig, load_config, resolve_config
from tamga.corpus import Corpus, Document
from tamga.io import load_corpus, load_metadata
from tamga.preprocess.pipeline import ParsedCorpus, SpacyPipeline
from tamga.provenance import Provenance

__all__ = [
    "__version__",
    "Corpus",
    "Document",
    "ParsedCorpus",
    "Provenance",
    "SpacyPipeline",
    "StudyConfig",
    "load_config",
    "load_corpus",
    "load_metadata",
    "resolve_config",
]
```

- [ ] **Step 21.2: Create `src/tamga/__main__.py`**

```python
"""Allow `python -m tamga ...` to invoke the CLI."""

from tamga.cli import app

if __name__ == "__main__":
    app()
```

- [ ] **Step 21.3: Verify imports work**

Run:
```bash
python -c "from tamga import Corpus, Document, SpacyPipeline, StudyConfig, __version__; print(__version__)"
```
Expected: prints the version without ImportError.

- [ ] **Step 21.4: Verify `python -m tamga` works**

Run: `python -m tamga --version`
Expected: prints `tamga 0.1.0.dev0`.

- [ ] **Step 21.5: Commit**

```bash
git add src/tamga/__init__.py src/tamga/__main__.py
git commit -m "feat: public API re-exports + python -m tamga entry point"
```

---

## Task 22: Project metadata — LICENSE, CITATION.cff, README

**Files:**
- Create: `LICENSE`
- Create: `CITATION.cff`
- Create: `README.md`

- [ ] **Step 22.1: Create `LICENSE` (BSD-3-Clause)**

```
BSD 3-Clause License

Copyright (c) 2026, Fatih Bozdağ

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

- [ ] **Step 22.2: Create `CITATION.cff`**

```yaml
cff-version: 1.2.0
message: "If you use tamga, please cite it as below."
title: "tamga: next-generation computational stylometry"
authors:
  - family-names: "Bozdağ"
    given-names: "Fatih"
    email: "fbozdag1989@gmail.com"
type: software
license: BSD-3-Clause
url: "https://github.com/fatihbozdag/tamga"
repository-code: "https://github.com/fatihbozdag/tamga"
version: "0.1.0.dev0"
abstract: >
  tamga is a Python package and interactive CLI for computational stylometry.
  It provides feature parity with R's Stylo for authorship attribution,
  author-group comparison, and Digital Humanities analyses, and adds modern
  NLP and ML capabilities via spaCy and scikit-learn.
keywords:
  - stylometry
  - authorship-attribution
  - corpus-linguistics
  - digital-humanities
  - natural-language-processing
```

- [ ] **Step 22.3: Create `README.md`**

```markdown
# tamga

**Next-generation computational stylometry — a Python replacement for R's Stylo.**

`tamga` ("mark, brand, clan-sign" — from Old Turkic) is a Python package and interactive CLI for
authorship attribution, author-group style comparison, and Digital Humanities stylometric
analysis. It reimplements the analytical breadth of R's `Stylo` and adds modern NLP and ML on top.

> Named after the **tamga**, the Turkic clan-mark by which individual and familial identity was
> recognised at a glance — the material-culture counterpart to a stylistic fingerprint.

## Status

**Phase 1 — Foundation.** Currently: corpus ingestion + spaCy parsing + DocBin caching +
project scaffolding + skeleton CLI. Delta family, Zeta, consensus trees, classifiers,
visualisations, reports, and the interactive wizard shell land in Phases 2–5.

See `docs/superpowers/specs/2026-04-17-tamga-stylometry-package-design.md` for the full design.

## Install

```bash
uv pip install tamga
python -m spacy download en_core_web_trf
```

## Quickstart

```bash
tamga init my-study
cd my-study
# drop .txt files into corpus/
# optionally add a metadata.tsv with filename → author/group/year/...
tamga ingest corpus/ --metadata corpus/metadata.tsv
tamga info
```

## License

BSD-3-Clause. See `LICENSE`.

## Citation

If you use tamga in published work, please cite it — see `CITATION.cff`.
```

- [ ] **Step 22.4: Commit**

```bash
git add LICENSE CITATION.cff README.md
git commit -m "docs: LICENSE (BSD-3-Clause), CITATION.cff, README"
```

---

## Task 23: Conftest + shared fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 23.1: Create `tests/conftest.py`**

```python
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
```

- [ ] **Step 23.2: Commit**

```bash
git add tests/conftest.py
git commit -m "test: shared conftest with mini_corpus fixture"
```

---

## Task 24: Run full test suite + coverage gate

- [ ] **Step 24.1: Run all tests with coverage**

Run: `pytest -n auto --cov=tamga --cov-report=term-missing -m "not slow"`
Expected: all tests pass; total coverage ≥85% on `tamga.plumbing`, `tamga.corpus`, `tamga.io`, `tamga.config`, `tamga.preprocess.cache`, `tamga.scaffold`. Coverage on `tamga.cli` and `tamga.preprocess.pipeline` may be lower due to integration-test boundaries; ≥70% acceptable there.

If any module's coverage is below target, inspect `--cov-report=term-missing` output and add the missing tests in the relevant `tests/<module>/test_*.py` file before continuing.

- [ ] **Step 24.2: Run pre-commit on all files**

Run: `pre-commit run --all-files`
Expected: all checks pass (no auto-fixes applied, no warnings).

- [ ] **Step 24.3: Tag Phase 1 complete**

Run:
```bash
git tag -a phase-1-foundation -m "Phase 1 foundation complete: corpus model, config, cache, preprocessing, skeleton CLI, CI"
git log --oneline phase-1-foundation
```

Expected: tag created; log shows the sequence of Phase-1 commits.

---

## Phase 1 — Acceptance Criteria

A reviewer must be able to run the following sequence from a clean checkout and see every step succeed:

```bash
git clone <repo>
cd tamga
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m spacy download en_core_web_sm
pre-commit run --all-files
pytest -n auto --cov=tamga --cov-report=term-missing -m "not slow"

# end-to-end exercise
tamga --version
tamga init /tmp/demo
cp tests/fixtures/mini_corpus/*.txt /tmp/demo/corpus/
cp tests/fixtures/mini_corpus/metadata.tsv /tmp/demo/corpus/
tamga ingest /tmp/demo/corpus/ \
    --metadata /tmp/demo/corpus/metadata.tsv \
    --cache-dir /tmp/demo/.tamga/cache \
    --spacy-model en_core_web_sm
tamga cache size --cache-dir /tmp/demo/.tamga/cache   # → "4 entries"
tamga info
```

Every line above must succeed with zero errors.

---

## Self-Review Notes

- **Spec coverage.** Foundation pieces mapped: corpus model (§3.1–3.2), Provenance (§3.6),
  StudyConfig shape (§3.4, §9), spaCy preprocessing + DocBin cache (§4.1–4.2), project scaffold
  (§12.1), CLI `init`, `ingest`, `info`, `cache` subset (§8.1), config layering (§9.2),
  reproducibility plumbing (§12.3–12.5), CI (§13.3), BSD-3-Clause + CITATION (§14.4).
  Explicitly **deferred** to later phases: `FeatureMatrix` (§3.3), `Result` (§3.5), extractors
  (§5), methods (§6), sklearn protocol adherence (§7), viz (§10), reports (§11), `study.yaml`
  runner (§8.5, `tamga run`), interactive shell (§8.4), full `lock.yaml` writer (§12.4), reporting
  commands.
- **Placeholder scan.** No "TBD"/"TODO"/"implement later" in any step. All code blocks are
  runnable as shown.
- **Type consistency.** `Document` and `Corpus` share the same metadata-dict contract across
  every task; `SpacyPipeline.parse` returns `ParsedCorpus` consistently in tasks 13, 17, 19, 21.
  `DocBinCache.get/put/keys/size_bytes/clear` signatures match across tasks 12 and 18.
- **TDD discipline.** Every unit of behaviour (hashing, seeding, Document, Corpus, ingest,
  config, cache, pipeline, scaffolder, each CLI subcommand) has a failing test written before
  the implementation.
- **Frequent commits.** 22 commits across 24 tasks — one per user-visible unit of work.

---

## Deferred to Later Phase Plans

- **Phase 2:** Feature extractors (MFW, char/word/POS n-grams, deps, function words, punct,
  lexdiv, readability, sentence-length) + Delta family + Federalist Papers parity suite.
- **Phase 3:** Zeta, reducers, clustering, bootstrap consensus trees, sklearn classifiers.
- **Phase 4:** `tamga[embeddings]` and `tamga[bayesian]` optional extras.
- **Phase 5:** Visualisation (matplotlib/seaborn static + plotly interactive), HTML/Markdown
  reports, interactive wizard shell, `tamga run study.yaml`.
- **Phase 6:** MkDocs site, Federalist Papers tutorial, EFCAMDAT-style L2-vs-native tutorial,
  PyPI publishing workflow.
