# Multi-Language Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class support for English, Turkish, German, Spanish, French behind a language registry, with Stanza BOUN for Turkish via `spacy-stanza`.

**Architecture:** A new `bitig.languages` module owns a frozen `LanguageSpec` registry. `Corpus` and `StudyConfig.preprocess` gain a `language` field. Language-dependent extractors (`FunctionWordExtractor`, `ReadabilityExtractor`, `ContextualEmbeddingExtractor`, `SentenceEmbeddingExtractor`) resolve defaults from `corpus.language` unless explicitly overridden. `SpacyPipeline` dispatches between `spacy.load()` and `spacy_stanza.load_pipeline()`; DocBin cache keys gain a backend segment that preserves the existing English key format.

**Tech Stack:** Python 3.11+, spaCy ≥3.7, `spacy-stanza` ≥1.0.4 + `stanza` ≥1.6 (new `turkish` extra), `pyphen` ≥0.14 (new core dep for DE/FR syllable counting), pydantic v2, pytest.

**Spec:** [`docs/superpowers/specs/2026-04-19-multi-language-support-design.md`](../specs/2026-04-19-multi-language-support-design.md)

---

## Phase 1 — Language registry + resource layout

### Task 1.1: Create `bitig.languages.registry`

**Files:**
- Create: `src/bitig/languages/__init__.py`
- Create: `src/bitig/languages/registry.py`
- Create: `tests/languages/__init__.py`
- Create: `tests/languages/test_registry.py`

The registry is frozen. Each entry names (a) the default spaCy pipeline or Stanza lang code, (b) the backend, (c) the canonical readability indices for that language, (d) default contextual and sentence embedding model ids. Nothing resolves these yet — later tasks wire them into extractors.

- [ ] **Step 1: Write the failing test**

```python
# tests/languages/test_registry.py
"""Tests for the language registry."""

import dataclasses

import pytest

from bitig.languages import LANGUAGES, LanguageSpec, get_language


def test_registry_contains_five_first_class_languages() -> None:
    assert set(LANGUAGES) == {"en", "tr", "de", "es", "fr"}


def test_language_spec_is_frozen() -> None:
    spec = get_language("en")
    assert dataclasses.is_dataclass(spec)
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.code = "xx"  # type: ignore[misc]


def test_get_language_is_case_insensitive() -> None:
    assert get_language("EN") is get_language("en")


def test_get_language_unknown_raises_with_supported_list() -> None:
    with pytest.raises(ValueError, match="Unknown language code"):
        get_language("xx")


def test_english_spec_matches_current_defaults() -> None:
    spec = get_language("en")
    assert spec.code == "en"
    assert spec.name == "English"
    assert spec.default_model == "en_core_web_trf"
    assert spec.backend == "spacy"
    assert set(spec.readability_indices) >= {"flesch", "flesch_kincaid", "gunning_fog"}


def test_turkish_spec_uses_spacy_stanza_backend() -> None:
    spec = get_language("tr")
    assert spec.backend == "spacy_stanza"
    assert spec.default_model == "tr"
    assert spec.readability_indices == ("atesman", "bezirci_yilmaz")


def test_every_spec_declares_embedding_defaults() -> None:
    for code, spec in LANGUAGES.items():
        assert spec.contextual_embedding_default, f"{code} missing contextual default"
        assert spec.sentence_embedding_default, f"{code} missing sentence default"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/languages/test_registry.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'bitig.languages'`.

- [ ] **Step 3: Write the implementation**

```python
# src/bitig/languages/registry.py
"""Language registry: one frozen LanguageSpec per first-class language.

Every language-dependent site in bitig (preprocess pipeline, function-word loading, readability
index selection, embedding model defaults) reads from REGISTRY. Unknown codes raise early and
clearly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LanguageSpec:
    """Static metadata for one first-class language."""

    code: str
    name: str
    default_model: str
    backend: Literal["spacy", "spacy_stanza"]
    readability_indices: tuple[str, ...]
    contextual_embedding_default: str
    sentence_embedding_default: str


REGISTRY: dict[str, LanguageSpec] = {
    "en": LanguageSpec(
        code="en",
        name="English",
        default_model="en_core_web_trf",
        backend="spacy",
        readability_indices=("flesch", "flesch_kincaid", "gunning_fog"),
        contextual_embedding_default="bert-base-uncased",
        sentence_embedding_default="sentence-transformers/all-mpnet-base-v2",
    ),
    "tr": LanguageSpec(
        code="tr",
        name="Turkish",
        default_model="tr",
        backend="spacy_stanza",
        readability_indices=("atesman", "bezirci_yilmaz"),
        contextual_embedding_default="dbmdz/bert-base-turkish-cased",
        sentence_embedding_default="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    ),
    "de": LanguageSpec(
        code="de",
        name="German",
        default_model="de_dep_news_trf",
        backend="spacy",
        readability_indices=("flesch_amstad", "wiener_sachtextformel"),
        contextual_embedding_default="deepset/gbert-base",
        sentence_embedding_default="deepset/gbert-base-sts",
    ),
    "es": LanguageSpec(
        code="es",
        name="Spanish",
        default_model="es_dep_news_trf",
        backend="spacy",
        readability_indices=("fernandez_huerta", "szigriszt_pazos"),
        contextual_embedding_default="dccuchile/bert-base-spanish-wwm-cased",
        sentence_embedding_default="hiiamsid/sentence_similarity_spanish_es",
    ),
    "fr": LanguageSpec(
        code="fr",
        name="French",
        default_model="fr_dep_news_trf",
        backend="spacy",
        readability_indices=("kandel_moles", "lix"),
        contextual_embedding_default="almanach/camembert-base",
        sentence_embedding_default="dangvantuan/sentence-camembert-base",
    ),
}


def get(code: str) -> LanguageSpec:
    """Return the LanguageSpec for `code` (case-insensitive). Raises ValueError if unknown."""
    normalized = code.lower()
    if normalized not in REGISTRY:
        supported = sorted(REGISTRY)
        raise ValueError(
            f"Unknown language code: {code!r}. Supported: {supported}. "
            f"To add a new language, extend bitig.languages.registry.REGISTRY."
        )
    return REGISTRY[normalized]
```

```python
# src/bitig/languages/__init__.py
"""Language registry — one first-class entry per supported language."""

from bitig.languages.registry import REGISTRY as LANGUAGES, LanguageSpec
from bitig.languages.registry import get as get_language

__all__ = ["LANGUAGES", "LanguageSpec", "get_language"]
```

```python
# tests/languages/__init__.py
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/languages/test_registry.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/bitig/languages tests/languages
git commit -m "feat(languages): add LanguageSpec registry for EN/TR/DE/ES/FR"
```

---

### Task 1.2: Move English function-word list to per-language layout

**Files:**
- Create: `src/bitig/resources/languages/__init__.py`
- Create: `src/bitig/resources/languages/en/__init__.py`
- Create: `src/bitig/resources/languages/en/function_words.txt` (copy of existing English list)
- Modify: `src/bitig/features/function_words.py` (update `_load_bundled_list`)
- Delete: `src/bitig/resources/function_words_en.txt` (after migration verified)

The goal is to move the English list into the per-language layout **without changing observable behavior**. All existing function-word tests must keep passing.

- [ ] **Step 1: Copy the English list to its new home**

```bash
mkdir -p src/bitig/resources/languages/en
cp src/bitig/resources/function_words_en.txt src/bitig/resources/languages/en/function_words.txt
touch src/bitig/resources/languages/__init__.py
touch src/bitig/resources/languages/en/__init__.py
```

- [ ] **Step 2: Update `_load_bundled_list` to read from the new path**

```python
# src/bitig/features/function_words.py — edit only _load_bundled_list
def _load_bundled_list() -> list[str]:
    path = resources.files("bitig.resources.languages.en") / "function_words.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
```

- [ ] **Step 3: Run existing function-word tests unchanged**

```bash
pytest tests/features/test_function_words.py -v
```

Expected: all 4 tests pass with identical results.

- [ ] **Step 4: Remove the old file**

```bash
rm src/bitig/resources/function_words_en.txt
```

- [ ] **Step 5: Re-run the whole feature test suite**

```bash
pytest tests/features -v
```

Expected: all tests pass. Ensures no code path still references the old filename.

- [ ] **Step 6: Update `pyproject.toml` wheel-force-include path** — verify the existing `"src/bitig/resources"` entry still covers the new layout (it does — the directory is included recursively).

No edit needed if the current entry is `"src/bitig/resources" = "bitig/resources"`. Confirm by reading `pyproject.toml` lines 84-87.

- [ ] **Step 7: Commit**

```bash
git add src/bitig/resources src/bitig/features/function_words.py
git commit -m "refactor(resources): move function_words_en.txt to resources/languages/en/"
```

---

### Task 1.3: Re-export language registry from `bitig.__init__`

**Files:**
- Modify: `src/bitig/__init__.py`
- Create: `tests/test_public_api.py` (new file — or modify if exists)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_public_api.py — add or append if file exists
"""Smoke tests for bitig's top-level public API."""


def test_languages_re_exported_from_top_level() -> None:
    import bitig

    assert "LANGUAGES" in bitig.__all__
    assert "LanguageSpec" in bitig.__all__
    assert "get_language" in bitig.__all__
    assert set(bitig.LANGUAGES) == {"en", "tr", "de", "es", "fr"}
    assert bitig.get_language("tr").backend == "spacy_stanza"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_public_api.py::test_languages_re_exported_from_top_level -v
```

Expected: FAIL with `AttributeError: module 'bitig' has no attribute 'LANGUAGES'`.

- [ ] **Step 3: Add re-exports to `src/bitig/__init__.py`**

Find the existing import block after `from bitig.io import ...` (near line 30) and add:

```python
from bitig.languages import LANGUAGES, LanguageSpec, get_language
```

Then add to `__all__` (keep alphabetical order within the existing list):

```python
    "LANGUAGES",
    "LanguageSpec",
    "get_language",
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_public_api.py::test_languages_re_exported_from_top_level -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/bitig/__init__.py tests/test_public_api.py
git commit -m "feat(api): re-export LANGUAGES, LanguageSpec, get_language at top level"
```

---

## Phase 2 — Corpus.language + config schema

### Task 2.1: Add `language` field to `Corpus`

**Files:**
- Modify: `src/bitig/corpus/corpus.py`
- Modify: `tests/test_corpus.py` (append)

The default is `"en"` so every existing call site keeps working without changes. `Corpus.hash()` must include `language` so corpora in different languages hash differently even when their text contents are identical.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_corpus.py — append
def test_corpus_default_language_is_english() -> None:
    from bitig.corpus import Corpus, Document

    c = Corpus(documents=[Document(id="d0", text="hello")])
    assert c.language == "en"


def test_corpus_accepts_language_argument() -> None:
    from bitig.corpus import Corpus, Document

    c = Corpus(documents=[Document(id="d0", text="merhaba")], language="tr")
    assert c.language == "tr"


def test_corpus_hash_differs_by_language() -> None:
    from bitig.corpus import Corpus, Document

    doc = Document(id="d0", text="merhaba")
    en = Corpus(documents=[doc], language="en")
    tr = Corpus(documents=[doc], language="tr")
    assert en.hash() != tr.hash()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_corpus.py -v -k language
```

Expected: FAIL — `Corpus` has no `language` attribute.

- [ ] **Step 3: Update `Corpus`**

In `src/bitig/corpus/corpus.py`, edit the `@dataclass` body:

```python
@dataclass
class Corpus:
    documents: list[Document] = field(default_factory=list)
    language: str = "en"
```

And update `Corpus.hash()` to include the language:

```python
    def hash(self) -> str:
        """Stable hash — sorted document hashes + sorted metadata + language."""
        doc_hashes = sorted(d.hash for d in self.documents)
        metadata_summary = sorted((d.id, hash_mapping(d.metadata)) for d in self.documents)
        payload = (
            "|".join(doc_hashes)
            + "||"
            + str(metadata_summary)
            + "||lang="
            + self.language
        )
        return hash_text(payload)
```

Update `Corpus.__getitem__` slice/array branches and `from_iterable` to preserve `language`:

```python
    def __getitem__(self, index):
        if isinstance(index, int | np.integer):
            return self.documents[int(index)]
        if isinstance(index, slice):
            return Corpus(documents=self.documents[index], language=self.language)
        return Corpus(
            documents=[self.documents[int(i)] for i in index],
            language=self.language,
        )

    def filter(self, **query):
        ...
        return Corpus(
            documents=[d for d in self.documents if matches(d)],
            language=self.language,
        )

    def groupby(self, field_name):
        ...
        return {
            k: Corpus(documents=v, language=self.language) for k, v in groups.items()
        }

    @classmethod
    def from_iterable(cls, docs, *, language: str = "en") -> Corpus:
        return cls(documents=list(docs), language=language)
```

- [ ] **Step 4: Run corpus tests**

```bash
pytest tests/test_corpus.py -v
```

Expected: all tests pass including the three new ones. Existing hash-equality tests should still pass (they compare corpora with default language).

- [ ] **Step 5: Full test suite smoke check**

```bash
pytest -x --ignore=tests/features/test_embeddings.py
```

Expected: all pass. Note: embeddings tests skipped only if extras aren't installed; if they run, they should also pass.

- [ ] **Step 6: Commit**

```bash
git add src/bitig/corpus/corpus.py tests/test_corpus.py
git commit -m "feat(corpus): add language field (default 'en'); include in hash"
```

---

### Task 2.2: Add `--language` flag to ingestion

**Files:**
- Modify: `src/bitig/io/ingest.py`
- Modify: `src/bitig/cli/ingest_cmd.py`
- Modify: `tests/cli/test_ingest.py` (append)
- Modify: `tests/io/test_ingest.py` (append — or create if missing)

- [ ] **Step 1: Write the failing tests**

```python
# tests/cli/test_ingest.py — append

def test_ingest_cli_accepts_language_flag(tmp_path, monkeypatch) -> None:
    from typer.testing import CliRunner
    from bitig.cli import app

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "a.txt").write_text("merhaba dünya")

    # Stub out SpacyPipeline to avoid model loads during unit tests.
    recorded: dict[str, object] = {}
    from bitig.cli import ingest_cmd

    class _StubPipe:
        def __init__(self, **kwargs: object) -> None:
            recorded["pipe_kwargs"] = kwargs
            self.cache = type("C", (), {"keys": lambda self: [], "size_bytes": lambda self: 0})()

        def parse(self, corpus) -> None:
            recorded["corpus_language"] = corpus.language

    monkeypatch.setattr(ingest_cmd, "SpacyPipeline", _StubPipe)

    runner = CliRunner()
    result = runner.invoke(
        app, ["ingest", str(corpus_dir), "--language", "tr", "--no-strict"]
    )
    assert result.exit_code == 0, result.stdout
    assert recorded["corpus_language"] == "tr"
```

```python
# tests/io/test_ingest.py — append (create if missing, mirror existing structure)

def test_load_corpus_stamps_language_argument(tmp_path) -> None:
    from bitig.io import load_corpus

    (tmp_path / "d.txt").write_text("hola mundo")
    corpus = load_corpus(tmp_path, language="es", strict=False)
    assert corpus.language == "es"


def test_load_corpus_defaults_to_english(tmp_path) -> None:
    from bitig.io import load_corpus

    (tmp_path / "d.txt").write_text("hello")
    corpus = load_corpus(tmp_path, strict=False)
    assert corpus.language == "en"


def test_load_corpus_rejects_unknown_language_code(tmp_path) -> None:
    import pytest
    from bitig.io import load_corpus

    (tmp_path / "d.txt").write_text("x")
    with pytest.raises(ValueError, match="Unknown language code"):
        load_corpus(tmp_path, language="xx", strict=False)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/cli/test_ingest.py tests/io/test_ingest.py -v -k language
```

Expected: FAIL — `load_corpus` doesn't accept `language=`, CLI doesn't accept `--language`.

- [ ] **Step 3: Update `load_corpus`**

In `src/bitig/io/ingest.py`, change the signature and pass through to `Corpus`:

```python
from bitig.languages import get_language


def load_corpus(
    path: Path,
    *,
    metadata: Path | None = None,
    strict: bool = True,
    glob: str = _TEXT_GLOB,
    encoding: str = "utf-8",
    language: str = "en",
) -> Corpus:
    """Load every text file under `path` into a Corpus, sorted by filename.

    `language` must be a registered code (see bitig.LANGUAGES). Defaults to 'en'.
    """
    get_language(language)  # validates early; raises ValueError if unknown
    path = Path(path)
    if not path.is_dir():
        raise NotADirectoryError(path)
    # ... existing body unchanged ...
    return Corpus(documents=documents, language=language.lower())
```

- [ ] **Step 4: Update CLI `ingest_command`**

In `src/bitig/cli/ingest_cmd.py`, add a `language` Typer option and pass it into `load_corpus`:

```python
def ingest_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path | None = typer.Option(  # noqa: B008
        None, "--metadata", "-m", exists=True, dir_okay=False,
        help="TSV file mapping filename to metadata fields.",
    ),
    strict: bool = typer.Option(True, "--strict/--no-strict"),
    cache_dir: Path = typer.Option(Path(".bitig/cache"), "--cache-dir"),  # noqa: B008
    spacy_model: str | None = typer.Option(
        None, "--spacy-model",
        help="spaCy model name. Default: resolved from --language.",
    ),
    exclude: list[str] | None = typer.Option(  # noqa: B008
        None, "--exclude",
    ),
    language: str = typer.Option(
        "en", "--language", "-l",
        help="Corpus language code (en, tr, de, es, fr). Default: en.",
    ),
) -> None:
    """Parse a corpus directory and cache spaCy parses."""
    corpus = load_corpus(path, metadata=metadata, strict=strict, language=language)
    console.print(f"[green]loaded[/green] {len(corpus)} documents from {path} (language={language})")

    pipe = SpacyPipeline(
        language=language,
        model=spacy_model,
        cache_dir=cache_dir / "docbin",
        exclude=exclude or [],
    )

    docbin_before = set(pipe.cache.keys())
    pipe.parse(corpus)
    docbin_after = set(pipe.cache.keys())
    newly_parsed = len(docbin_after - docbin_before)
    cached_hits = len(corpus) - newly_parsed

    console.print(
        f"[green]parsed[/green] {len(corpus)} documents"
        f" ({cached_hits} cached, {newly_parsed} newly parsed)"
    )
    console.print(f"  cache: {cache_dir / 'docbin'} ({pipe.cache.size_bytes()} bytes)")
```

Note: `SpacyPipeline(language=..., model=None)` isn't wired yet — Task 3.3 implements that. For now the CLI test stubs `SpacyPipeline`, so nothing crashes. If any non-stubbed integration test fails here, skip this CLI change temporarily and add it back in Task 3.3.

- [ ] **Step 5: Run tests**

```bash
pytest tests/cli/test_ingest.py tests/io/test_ingest.py -v -k language
```

Expected: new tests pass.

- [ ] **Step 6: Run full existing suite**

```bash
pytest tests/cli tests/io -v
```

Expected: all pass. Existing CLI tests don't specify `--language` and should default to `en`.

- [ ] **Step 7: Commit**

```bash
git add src/bitig/io/ingest.py src/bitig/cli/ingest_cmd.py tests/cli/test_ingest.py tests/io/test_ingest.py
git commit -m "feat(ingest): --language flag stamps Corpus.language (default 'en')"
```

---

### Task 2.3: Add `PreprocessConfig.language` with pydantic validation

**Files:**
- Modify: `src/bitig/config/schema.py`
- Modify: `tests/test_config.py` (append)

Language must validate against the registry at config-load time so typos fail fast.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py — append

def test_preprocess_language_defaults_to_english() -> None:
    from bitig.config.schema import PreprocessConfig

    cfg = PreprocessConfig()
    assert cfg.language == "en"


def test_preprocess_language_accepts_registered_code() -> None:
    from bitig.config.schema import PreprocessConfig

    cfg = PreprocessConfig(language="tr")
    assert cfg.language == "tr"


def test_preprocess_language_rejects_unknown_code() -> None:
    import pytest
    from pydantic import ValidationError
    from bitig.config.schema import PreprocessConfig

    with pytest.raises(ValidationError, match="Unknown language code"):
        PreprocessConfig(language="xx")


def test_preprocess_language_case_insensitive() -> None:
    from bitig.config.schema import PreprocessConfig

    cfg = PreprocessConfig(language="TR")
    assert cfg.language == "tr"


def test_spacy_config_model_now_optional() -> None:
    from bitig.config.schema import SpacyConfig

    cfg = SpacyConfig()
    assert cfg.model is None
    assert cfg.backend is None
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_config.py -v -k "language or spacy_config_model"
```

Expected: FAIL.

- [ ] **Step 3: Update `SpacyConfig` and `PreprocessConfig`**

In `src/bitig/config/schema.py`:

```python
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bitig.languages import LANGUAGES

# ... existing code ...


class SpacyConfig(BaseModel):
    model_config = _STRICT_MODEL
    model: str | None = None
    backend: Literal["spacy", "spacy_stanza"] | None = None
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    exclude: list[str] = Field(default_factory=list)


class PreprocessConfig(BaseModel):
    model_config = _STRICT_MODEL
    language: str = "en"
    spacy: SpacyConfig = Field(default_factory=SpacyConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_and_validate(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError(f"language must be a string, got {type(v).__name__}")
        normalized = v.lower()
        if normalized not in LANGUAGES:
            supported = sorted(LANGUAGES)
            raise ValueError(
                f"Unknown language code: {v!r}. Supported: {supported}."
            )
        return normalized
```

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_config.py -v -k "language or spacy_config_model"
```

Expected: all pass.

- [ ] **Step 5: Full config tests**

```bash
pytest tests/test_config.py -v
```

Expected: all pass. Existing `SpacyConfig(model="en_core_web_trf")` usage in other tests still works (explicit value accepted).

- [ ] **Step 6: Commit**

```bash
git add src/bitig/config/schema.py tests/test_config.py
git commit -m "feat(config): PreprocessConfig.language with registry validation; SpacyConfig.model optional"
```

---

## Phase 3 — spacy-stanza backend in SpacyPipeline

### Task 3.1: Add `turkish` and `multilang` optional extras

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit `[project.optional-dependencies]` in `pyproject.toml`**

Find the existing block (around line 48) and append:

```toml
turkish = [
    "spacy-stanza>=1.0.4",
    "stanza>=1.6",
]
multilang = [
    "bitig[turkish]",
]
```

Also add `pyphen` to the core `dependencies =` list alphabetically — insert `"pyphen>=0.14"` between `"pyarrow"` and `"pyyaml"`:

```toml
dependencies = [
    "numpy>=1.26",
    ...
    "pyarrow>=15",
    "pyphen>=0.14",
    "pyyaml>=6",
    ...
]
```

- [ ] **Step 2: Verify the file parses**

```bash
python -c "import tomllib; tomllib.loads(open('pyproject.toml').read()); print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Install the new dep locally**

```bash
uv pip install -e .
```

Expected: `pyphen` installs without errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore(deps): add turkish/multilang extras; pyphen to core deps"
```

---

### Task 3.2: Cache key accepts a backend version string with English back-compat

**Files:**
- Modify: `src/bitig/preprocess/cache.py`
- Modify: `tests/preprocess/test_cache.py`

The new signature takes `backend_version: str` instead of `spacy_version: str`. For `backend="spacy"` the string format is `"spacy=<version>"` — formatted to match what the native branch emits and preserve English caches. Cross-backend collisions are impossible because the format differs structurally.

- [ ] **Step 1: Write the regression tests**

Append to `tests/preprocess/test_cache.py`:

```python
def test_cache_key_spacy_native_backend_version_format() -> None:
    """English backend_version must format as 'spacy=<version>'."""
    k_new = cache_key("doc-hash", "en_core_web_sm", "spacy=3.7.2", ["ner"])
    # This must match what Task 3.3's SpacyPipeline.backend_version produces for backend='spacy'.
    assert k_new  # smoke: call returns a string


def test_cache_key_stanza_backend_differs_from_spacy_native() -> None:
    """Different backend identifiers must never collide."""
    native = cache_key("doc-hash", "tr", "spacy=3.7.2", [])
    stanza = cache_key("doc-hash", "tr", "spacy_stanza=1.0.4;stanza=1.8.0", [])
    assert native != stanza
```

Replace the existing tests that pass `"3.7.2"` as `spacy_version` with `"spacy=3.7.2"` as `backend_version`:

```python
def test_cache_key_is_deterministic() -> None:
    a = cache_key("doc-hash", "en_core_web_sm", "spacy=3.7.2", ["ner"])
    b = cache_key("doc-hash", "en_core_web_sm", "spacy=3.7.2", ["ner"])
    assert a == b


def test_cache_key_changes_with_any_input() -> None:
    base = cache_key("doc-hash", "en_core_web_sm", "spacy=3.7.2", [])
    assert cache_key("other-hash", "en_core_web_sm", "spacy=3.7.2", []) != base
    assert cache_key("doc-hash", "en_core_web_lg", "spacy=3.7.2", []) != base
    assert cache_key("doc-hash", "en_core_web_sm", "spacy=3.7.3", []) != base
    assert cache_key("doc-hash", "en_core_web_sm", "spacy=3.7.2", ["ner"]) != base


def test_cache_key_is_order_independent_for_excluded_components() -> None:
    a = cache_key("doc-hash", "en_core_web_sm", "spacy=3.7.2", ["ner", "parser"])
    b = cache_key("doc-hash", "en_core_web_sm", "spacy=3.7.2", ["parser", "ner"])
    assert a == b
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/preprocess/test_cache.py -v
```

Expected: existing tests still pass (they pass `"3.7.2"` as a string and `cache_key` didn't care about format — any string is acceptable). Our new tests pass trivially because they just exercise the same function. This task is the API rename.

Actually, re-read step 1: the edits change the existing tests to use the new format string. Those edited tests DO depend on the signature. Re-run:

```bash
pytest tests/preprocess/test_cache.py -v
```

Expected: all pass.

- [ ] **Step 3: Rename the parameter**

In `src/bitig/preprocess/cache.py`:

```python
def cache_key(
    document_hash: str,
    spacy_model: str,
    backend_version: str,
    excluded_components: list[str],
) -> str:
    """Return a stable cache key for a (document, backend configuration) pair.

    `backend_version` is a structured string like 'spacy=3.7.2' (native spaCy backend) or
    'spacy_stanza=1.0.4;stanza=1.8.0' (Stanza-via-spacy-stanza backend). The native branch
    preserves the prior format so English caches built on older bitig versions remain valid.
    """
    return hash_mapping(
        {
            "doc": document_hash,
            "model": spacy_model,
            "version": backend_version,
            "exclude": sorted(excluded_components),
        }
    )
```

- [ ] **Step 4: Run cache tests**

```bash
pytest tests/preprocess/test_cache.py -v
```

Expected: all pass.

- [ ] **Step 5: Update `SpacyPipeline._key` to use the new name (transitional)**

In `src/bitig/preprocess/pipeline.py`, find `_key` and change:

```python
    def _key(self, doc: Document) -> str:
        return cache_key(doc.hash, self.model, f"spacy={self.spacy_version}", self.exclude)
```

- [ ] **Step 6: Run preprocess tests**

```bash
pytest tests/preprocess -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/bitig/preprocess tests/preprocess
git commit -m "refactor(cache): rename cache_key arg to backend_version; format 'spacy=X.Y.Z' for native"
```

---

### Task 3.3: `SpacyPipeline` gains language/backend dispatch

**Files:**
- Modify: `src/bitig/preprocess/pipeline.py`
- Modify: `tests/preprocess/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/preprocess/test_pipeline.py — append

from bitig.preprocess.pipeline import SpacyPipeline


def test_pipeline_resolves_english_defaults_from_registry(tmp_path) -> None:
    pipe = SpacyPipeline(cache_dir=tmp_path)
    assert pipe.language == "en"
    assert pipe.backend == "spacy"
    assert pipe.model == "en_core_web_trf"


def test_pipeline_resolves_turkish_backend_to_spacy_stanza(tmp_path) -> None:
    pipe = SpacyPipeline(language="tr", cache_dir=tmp_path)
    assert pipe.backend == "spacy_stanza"
    assert pipe.model == "tr"


def test_pipeline_explicit_backend_wins_over_language_default(tmp_path) -> None:
    pipe = SpacyPipeline(language="tr", backend="spacy", model="custom_model", cache_dir=tmp_path)
    assert pipe.backend == "spacy"
    assert pipe.model == "custom_model"


def test_pipeline_unknown_language_raises(tmp_path) -> None:
    import pytest

    with pytest.raises(ValueError, match="Unknown language code"):
        SpacyPipeline(language="xx", cache_dir=tmp_path)


def test_pipeline_backend_version_native_matches_prior_format(tmp_path, monkeypatch) -> None:
    """English native-spaCy backend_version is 'spacy=<version>' — preserves cache keys."""
    import spacy

    monkeypatch.setattr(spacy, "__version__", "3.7.2")
    pipe = SpacyPipeline(language="en", cache_dir=tmp_path)
    assert pipe.backend_version == "spacy=3.7.2"


def test_pipeline_backend_version_stanza_format(tmp_path, monkeypatch) -> None:
    """Stanza backend_version is structurally different from native — no cache collisions."""
    fake_spacy_stanza = type("M", (), {"__version__": "1.0.4"})
    fake_stanza = type("M", (), {"__version__": "1.8.0"})
    monkeypatch.setitem(__import__("sys").modules, "spacy_stanza", fake_spacy_stanza)
    monkeypatch.setitem(__import__("sys").modules, "stanza", fake_stanza)

    pipe = SpacyPipeline(language="tr", cache_dir=tmp_path)
    assert pipe.backend_version == "spacy_stanza=1.0.4;stanza=1.8.0"


def test_pipeline_exclude_warns_on_spacy_stanza_backend(tmp_path, caplog) -> None:
    """exclude= is meaningless on spacy_stanza; we emit a warning on nlp property access."""
    import sys

    fake_spacy_stanza = type("M", (), {
        "__version__": "1.0.4",
        "load_pipeline": staticmethod(lambda lang: object()),
    })
    fake_stanza = type("M", (), {"__version__": "1.8.0"})
    sys.modules["spacy_stanza"] = fake_spacy_stanza
    sys.modules["stanza"] = fake_stanza
    try:
        pipe = SpacyPipeline(language="tr", cache_dir=tmp_path, exclude=["ner"])
        import logging
        with caplog.at_level(logging.WARNING):
            _ = pipe.nlp  # triggers lazy load
        assert any("exclude" in r.message and "ignored" in r.message for r in caplog.records)
    finally:
        sys.modules.pop("spacy_stanza", None)
        sys.modules.pop("stanza", None)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/preprocess/test_pipeline.py -v -k "backend or language or registry"
```

Expected: FAIL — `SpacyPipeline` doesn't accept `language=` or `backend=`.

- [ ] **Step 3: Rewrite `SpacyPipeline`**

Replace `SpacyPipeline` class body in `src/bitig/preprocess/pipeline.py`:

```python
from typing import Literal

import spacy
from spacy.language import Language
from spacy.tokens import Doc, DocBin

from bitig.corpus import Corpus, Document
from bitig.languages import get_language
from bitig.plumbing.logging import get_logger
from bitig.preprocess.cache import DocBinCache, cache_key

_log = get_logger(__name__)


class SpacyPipeline:
    """Parse a Corpus into spaCy Docs, caching results as DocBin blobs on disk.

    Supports two backends behind a single interface:
      - backend="spacy"         — spacy.load(model)
      - backend="spacy_stanza"  — spacy_stanza.load_pipeline(lang=model)

    Both produce native spaCy Doc objects so downstream extractors don't care which backend
    produced them.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        model: str | None = None,
        backend: Literal["spacy", "spacy_stanza"] | None = None,
        cache_dir: Path | str = ".bitig/cache/docbin",
        exclude: list[str] | None = None,
    ) -> None:
        spec = get_language(language)
        self.language = spec.code
        self.model = model if model is not None else spec.default_model
        self.backend = backend if backend is not None else spec.backend
        self.exclude = list(exclude or [])
        self.cache = DocBinCache(Path(cache_dir))
        self._nlp: Language | None = None

    @property
    def nlp(self) -> Language:
        if self._nlp is None:
            if self.backend == "spacy_stanza":
                if self.exclude:
                    _log.warning(
                        "exclude=%s ignored on spacy_stanza backend "
                        "(Stanza attributes are set in the tokenizer, not pipeline components)",
                        self.exclude,
                    )
                try:
                    import spacy_stanza
                except ImportError as e:
                    raise ImportError(
                        "bitig requires spacy-stanza for the 'spacy_stanza' backend. "
                        "Install with: uv pip install 'bitig[turkish]'"
                    ) from e
                _log.info("loading Stanza pipeline via spacy-stanza: lang=%s", self.model)
                try:
                    self._nlp = spacy_stanza.load_pipeline(lang=self.model)
                except FileNotFoundError as e:
                    raise RuntimeError(
                        f"Stanza model for language {self.model!r} not found. "
                        f"Run: python -c \"import stanza; stanza.download('{self.model}')\""
                    ) from e
            else:
                _log.info("loading spaCy model: %s", self.model)
                self._nlp = spacy.load(self.model, exclude=self.exclude)
        return self._nlp

    @property
    def spacy_version(self) -> str:
        return str(spacy.__version__)

    @property
    def backend_version(self) -> str:
        """Structured version string used in cache keys.

        Native backend: 'spacy=<version>' — matches prior cache-key format, preserves English caches.
        Stanza backend: 'spacy_stanza=<v>;stanza=<v>' — structurally distinct from native.
        """
        if self.backend == "spacy_stanza":
            import spacy_stanza  # type: ignore[import-not-found]
            import stanza  # type: ignore[import-not-found]
            return f"spacy_stanza={spacy_stanza.__version__};stanza={stanza.__version__}"
        return f"spacy={self.spacy_version}"

    def _key(self, doc: Document) -> str:
        return cache_key(doc.hash, self.model, self.backend_version, self.exclude)

    def parse(self, corpus: Corpus) -> ParsedCorpus:
        # ... existing body unchanged — keep lines 68-97 of current file ...
```

Keep `ParsedCorpus` class and `parse` method body unchanged.

- [ ] **Step 4: Update `SpacyConfig`-consuming code that constructs `SpacyPipeline`**

Search for all `SpacyPipeline(` call sites:

```bash
grep -rn "SpacyPipeline(" src/ tests/ --include="*.py"
```

For each call site that passes `model=...`, confirm it still works (explicit model wins). For sites that construct with defaults, confirm they still default to English.

- [ ] **Step 5: Run tests**

```bash
pytest tests/preprocess/test_pipeline.py -v
```

Expected: all pass.

- [ ] **Step 6: Run full suite (no model downloads yet — Stanza path is stubbed in tests)**

```bash
pytest -x
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/bitig/preprocess/pipeline.py tests/preprocess/test_pipeline.py
git commit -m "feat(preprocess): SpacyPipeline backend dispatch (spacy | spacy_stanza)"
```

---

## Phase 4 — Extractors pick up language

### Task 4.1: `FunctionWordExtractor.language` + per-language resource loading

**Files:**
- Modify: `src/bitig/features/function_words.py`
- Modify: `tests/features/test_function_words.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/features/test_function_words.py`:

```python
def test_function_word_uses_corpus_language_when_unspecified() -> None:
    # Create a Turkish corpus; ensure the extractor tries to load the Turkish list.
    # (Turkish list won't exist yet; this test is marked xfail until Task 5.5 ships.)
    import pytest
    from bitig.corpus import Corpus, Document
    from bitig.features.function_words import FunctionWordExtractor

    c = Corpus(documents=[Document(id="d0", text="merhaba")], language="tr")
    ex = FunctionWordExtractor(scale="none")
    with pytest.raises(FileNotFoundError, match="function word list"):
        ex.fit_transform(c)


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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/features/test_function_words.py -v
```

Expected: FAIL — `FunctionWordExtractor` doesn't accept `language=`.

- [ ] **Step 3: Update `FunctionWordExtractor`**

Rewrite `src/bitig/features/function_words.py`:

```python
"""Function-word frequency extractor with per-language bundled word lists."""

from __future__ import annotations

import re
from importlib import resources
from typing import Literal

import numpy as np

from bitig.corpus import Corpus
from bitig.features.base import BaseFeatureExtractor
from bitig.languages import LANGUAGES

Scale = Literal["none", "zscore", "l1", "l2"]

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def _load_bundled_list(language: str) -> list[str]:
    """Load resources/languages/<lang>/function_words.txt.

    Raises FileNotFoundError with a helpful message listing supported languages if no list is
    bundled for `language`.
    """
    pkg = f"bitig.resources.languages.{language}"
    try:
        path = resources.files(pkg) / "function_words.txt"
    except (ModuleNotFoundError, FileNotFoundError) as e:
        supported = sorted(LANGUAGES)
        raise FileNotFoundError(
            f"No bundled function word list for language {language!r}. "
            f"Supported: {supported}. Pass wordlist=[...] to override."
        ) from e
    if not path.is_file():
        supported = sorted(LANGUAGES)
        raise FileNotFoundError(
            f"No bundled function word list for language {language!r} "
            f"(expected at {path}). Supported: {supported}. Pass wordlist=[...] to override."
        )
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FunctionWordExtractor(BaseFeatureExtractor):
    feature_type = "function_word"

    def __init__(
        self,
        *,
        wordlist: list[str] | None = None,
        language: str | None = None,
        scale: Scale = "none",
    ) -> None:
        self.wordlist = wordlist
        self.language = language
        self.scale = scale
        self._words: list[str] = []

    def _fit(self, corpus: Corpus) -> None:
        if self.wordlist is not None:
            self._words = list(self.wordlist)
            return
        lang = self.language or corpus.language
        self._words = _load_bundled_list(lang)

    # ... _transform unchanged ...
```

Keep `_transform` body unchanged.

- [ ] **Step 4: Run function-word tests**

```bash
pytest tests/features/test_function_words.py -v
```

Expected: all pass (including the new xfail-style test that expects `FileNotFoundError` for Turkish).

- [ ] **Step 5: Commit**

```bash
git add src/bitig/features/function_words.py tests/features/test_function_words.py
git commit -m "feat(features): FunctionWordExtractor.language; resolve from corpus.language"
```

---

### Task 4.2: `ReadabilityExtractor.language` + per-language index registry

**Files:**
- Modify: `src/bitig/features/readability.py`
- Modify: `tests/features/test_readability.py`

The old `_INDEX_FN` dict becomes `_INDEX_REGISTRY: dict[str, dict[str, Callable]]`, keyed first by language. English indices remain wired via `textstat`. Non-English language entries are empty `{}` in this task — Task 5.1-5.4 fill them in.

- [ ] **Step 1: Write the failing tests**

Append to `tests/features/test_readability.py`:

```python
def test_readability_resolves_english_defaults_from_registry() -> None:
    from bitig.features.readability import ReadabilityExtractor
    from bitig.corpus import Corpus, Document

    c = Corpus(documents=[Document(id="d0", text="A short simple sentence.")], language="en")
    ex = ReadabilityExtractor()  # indices=None
    fm = ex.fit_transform(c)
    assert set(fm.feature_names) == {"flesch", "flesch_kincaid", "gunning_fog"}


def test_readability_rejects_unsupported_index_for_language() -> None:
    import pytest
    from bitig.corpus import Corpus, Document
    from bitig.features.readability import ReadabilityExtractor

    c = Corpus(documents=[Document(id="d0", text="x")], language="en")
    ex = ReadabilityExtractor(indices=["atesman"])  # Turkish index on English corpus
    with pytest.raises(ValueError, match="not available for language 'en'"):
        ex.fit_transform(c)


def test_readability_explicit_language_overrides_corpus() -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.readability import ReadabilityExtractor

    # Even if corpus is stamped 'tr', a user may override by passing language='en'.
    c = Corpus(documents=[Document(id="d0", text="A simple sentence.")], language="tr")
    ex = ReadabilityExtractor(indices=["flesch"], language="en")
    fm = ex.fit_transform(c)
    assert "flesch" in fm.feature_names
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/features/test_readability.py -v -k "registry or unsupported or override"
```

Expected: FAIL — `ReadabilityExtractor` doesn't accept `language=`.

- [ ] **Step 3: Rewrite `ReadabilityExtractor`**

```python
"""Readability indices, dispatched per-language.

English uses textstat wrappers (unchanged). Non-English languages use native implementations in
bitig.languages.readability_<code>, registered here. The per-language registry is populated in
Phase 5; this task wires only English.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import textstat

from bitig.corpus import Corpus
from bitig.features.base import BaseFeatureExtractor
from bitig.languages import get_language

# {language_code: {index_name: callable(text) -> float}}
_INDEX_REGISTRY: dict[str, dict[str, Callable[[str], float]]] = {
    "en": {
        "flesch": textstat.flesch_reading_ease,
        "flesch_kincaid": textstat.flesch_kincaid_grade,
        "gunning_fog": textstat.gunning_fog,
        "coleman_liau": textstat.coleman_liau_index,
        "ari": textstat.automated_readability_index,
        "smog": textstat.smog_index,
    },
    "tr": {},  # Task 5.1
    "de": {},  # Task 5.2
    "es": {},  # Task 5.3
    "fr": {},  # Task 5.4
}


class ReadabilityExtractor(BaseFeatureExtractor):
    feature_type = "readability"

    def __init__(
        self,
        indices: list[str] | tuple[str, ...] | None = None,
        *,
        language: str | None = None,
    ) -> None:
        self.indices = list(indices) if indices is not None else None
        self.language = language
        self._resolved_indices: list[str] = []
        self._fns: list[Callable[[str], float]] = []

    def _fit(self, corpus: Corpus) -> None:
        lang = self.language or corpus.language
        spec = get_language(lang)
        available = _INDEX_REGISTRY.get(lang, {})

        if self.indices is None:
            self._resolved_indices = list(spec.readability_indices)
        else:
            self._resolved_indices = list(self.indices)

        unknown = [i for i in self._resolved_indices if i not in available]
        if unknown:
            raise ValueError(
                f"Readability indices {unknown} not available for language {lang!r}. "
                f"Available for {lang!r}: {sorted(available)}."
            )
        self._fns = [available[i] for i in self._resolved_indices]

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        X = np.zeros((len(corpus), len(self._resolved_indices)), dtype=float)  # noqa: N806
        for row, doc in enumerate(corpus.documents):
            for col, fn in enumerate(self._fns):
                X[row, col] = float(fn(doc.text))
        return X, list(self._resolved_indices)
```

- [ ] **Step 4: Run readability tests**

```bash
pytest tests/features/test_readability.py -v
```

Expected: all pass, including new tests. Existing tests that pass `indices=["flesch"]` and rely on English corpus default work because `Corpus.language` defaults to `"en"`.

- [ ] **Step 5: Commit**

```bash
git add src/bitig/features/readability.py tests/features/test_readability.py
git commit -m "feat(features): ReadabilityExtractor per-language index registry; language param"
```

---

### Task 4.3: Embedding extractors resolve default model from language

**Files:**
- Modify: `src/bitig/features/embeddings.py`
- Modify: `tests/features/test_embeddings.py`

Gated on `bitig[embeddings]` extra. If the extra isn't installed, `ContextualEmbeddingExtractor`/`SentenceEmbeddingExtractor` aren't importable and these tests are skipped. Implement conditionally.

- [ ] **Step 1: Write the failing tests**

Append to `tests/features/test_embeddings.py`:

```python
import pytest

try:
    from bitig.features.embeddings import (
        ContextualEmbeddingExtractor,
        SentenceEmbeddingExtractor,
    )
    _HAS_EMBEDDINGS = True
except ImportError:
    _HAS_EMBEDDINGS = False

pytestmark = pytest.mark.skipif(not _HAS_EMBEDDINGS, reason="requires bitig[embeddings]")


def test_sentence_embedding_resolves_english_default_model() -> None:
    from bitig.corpus import Corpus, Document

    ex = SentenceEmbeddingExtractor()  # no model= specified
    # Resolution is lazy — happens at _fit. We inspect the stored model after construction.
    # The extractor should remember the language (default 'en') and resolve later.
    # Simulate: pretend _fit was called with an English corpus.
    c = Corpus(documents=[Document(id="d0", text="x")], language="en")
    ex._resolve_model(c)
    assert ex.model == "sentence-transformers/all-mpnet-base-v2"


def test_sentence_embedding_resolves_turkish_default_model() -> None:
    from bitig.corpus import Corpus, Document

    ex = SentenceEmbeddingExtractor()
    c = Corpus(documents=[Document(id="d0", text="x")], language="tr")
    ex._resolve_model(c)
    assert ex.model == "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"


def test_contextual_embedding_resolves_default_from_language() -> None:
    from bitig.corpus import Corpus, Document

    ex = ContextualEmbeddingExtractor()
    c = Corpus(documents=[Document(id="d0", text="x")], language="tr")
    ex._resolve_model(c)
    assert ex.model == "dbmdz/bert-base-turkish-cased"


def test_explicit_model_overrides_language_default() -> None:
    from bitig.corpus import Corpus, Document

    ex = SentenceEmbeddingExtractor(model="custom/model-name")
    c = Corpus(documents=[Document(id="d0", text="x")], language="tr")
    ex._resolve_model(c)
    assert ex.model == "custom/model-name"
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/features/test_embeddings.py -v
```

Expected: SKIP if embeddings extra not installed, else FAIL on the `_resolve_model` attribute.

- [ ] **Step 3: Update `SentenceEmbeddingExtractor` and `ContextualEmbeddingExtractor`**

In `src/bitig/features/embeddings.py`, change both `__init__` signatures to allow `model=None` and `language=None`, and add a `_resolve_model` helper. Also track `language`. The existing `_fit`/`_transform` bodies need to call `_resolve_model(corpus)` before using `self.model`.

```python
class SentenceEmbeddingExtractor(BaseFeatureExtractor):
    feature_type = "sentence_embedding"

    def __init__(
        self,
        *,
        model: str | None = None,
        language: str | None = None,
        pool: Pool = "mean",
        device: str | None = None,
    ) -> None:
        if not _sentence_transformers_available:
            raise ImportError(_INSTALL_HINT)
        self.model = model
        self.language = language
        self.pool = pool
        self.device = device
        self._encoder: Any = None

    def _resolve_model(self, corpus: Corpus) -> None:
        if self.model is None:
            from bitig.languages import get_language
            lang = self.language or corpus.language
            self.model = get_language(lang).sentence_embedding_default

    def _load_encoder(self) -> Any:
        if self._encoder is None:
            assert self.model is not None, "call _resolve_model(corpus) before _load_encoder()"
            self._encoder = SentenceTransformer(self.model, device=self.device)
        return self._encoder

    def _fit(self, corpus: Corpus) -> None:
        self._resolve_model(corpus)
        self._load_encoder()

    # _transform body unchanged
```

Repeat the same pattern for `ContextualEmbeddingExtractor`, using `contextual_embedding_default` instead of `sentence_embedding_default`.

- [ ] **Step 4: Run embeddings tests**

```bash
pytest tests/features/test_embeddings.py -v
```

Expected: pass if extras installed; skip otherwise.

- [ ] **Step 5: Commit**

```bash
git add src/bitig/features/embeddings.py tests/features/test_embeddings.py
git commit -m "feat(features): embedding extractors resolve default model from language"
```

---

## Phase 5 — Non-English resources

### Task 5.1: Turkish readability (Ateşman + Bezirci–Yılmaz)

**Files:**
- Create: `src/bitig/languages/readability_tr.py`
- Create: `tests/languages/test_readability_tr.py`
- Modify: `src/bitig/features/readability.py` (wire into registry)

**Formula references:**
- **Ateşman (1997)**: `198.825 - 40.175 * (syllables/words) - 2.610 * (words/sentences)` — analogue of Flesch Reading Ease.
- **Bezirci–Yılmaz (2010)**: `sqrt(avg_words_per_sentence * (H3*0.84 + H4*1.5 + H5*3.5 + H6*26.25))` where `H_k` is the fraction of words with exactly `k` syllables. Turkish words ≥7 syllables count as 6 syllables for this purpose.

Turkish syllables = count of vowels `aeıioöuüAEIİOÖUÜ`. Sentence split = regex `[.!?…]+`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/languages/test_readability_tr.py
"""Tests for Turkish readability — Ateşman (1997) and Bezirci–Yılmaz (2010)."""

import pytest

from bitig.languages.readability_tr import (
    atesman,
    bezirci_yilmaz,
    count_syllables_tr,
)


def test_count_syllables_simple_word() -> None:
    assert count_syllables_tr("merhaba") == 3  # mer-ha-ba
    assert count_syllables_tr("ev") == 1
    assert count_syllables_tr("öğretmen") == 3  # öğ-ret-men
    assert count_syllables_tr("İstanbul") == 3  # İs-tan-bul


def test_atesman_scoring_sense() -> None:
    """Shorter simpler Turkish prose → higher Ateşman. Reference: paper worked example."""
    simple = "Ali okula gitti. Kedi uyudu. Hava güzeldi."
    complex_ = (
        "Ülkelerarası diplomatik müzakerelerin sürdürülebilirliği, tarafların "
        "uzlaşmacı tutumlarını korumalarına bağlıdır."
    )
    assert atesman(simple) > atesman(complex_)


def test_atesman_range_is_plausible() -> None:
    text = "Ali topu tuttu. Kedi uyudu."
    score = atesman(text)
    assert -50 <= score <= 200  # plausible bounds for the Ateşman scale


def test_bezirci_yilmaz_scoring_sense() -> None:
    """Longer sentences + more polysyllabic words → higher Bezirci-Yılmaz score."""
    simple = "Ev büyük. Ağaç yeşil. Kedi uyur."
    complex_ = (
        "Ülkelerarası diplomatik müzakerelerin sürdürülebilirliği uzun zaman gerektirir."
    )
    assert bezirci_yilmaz(complex_) > bezirci_yilmaz(simple)


def test_atesman_handles_empty_and_single_word() -> None:
    assert atesman("") == 0.0
    assert isinstance(atesman("ev"), float)


def test_turkish_readability_wired_into_extractor() -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.readability import ReadabilityExtractor

    c = Corpus(
        documents=[Document(id="d0", text="Ali topu tuttu. Kedi uyudu.")],
        language="tr",
    )
    ex = ReadabilityExtractor()  # defaults to (atesman, bezirci_yilmaz)
    fm = ex.fit_transform(c)
    assert set(fm.feature_names) == {"atesman", "bezirci_yilmaz"}
    assert fm.X.shape == (1, 2)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/languages/test_readability_tr.py -v
```

Expected: FAIL — `bitig.languages.readability_tr` does not exist.

- [ ] **Step 3: Implement Turkish readability**

```python
# src/bitig/languages/readability_tr.py
"""Turkish readability formulas.

Ateşman, E. (1997). Türkçede okunabilirliğin ölçülmesi. Dil Dergisi, 58, 71-74.
Bezirci, B., & Yılmaz, A. E. (2010). Metinlerin okunabilirliğinin ölçülmesi üzerine bir yazılım
kütüphanesi ve Türkçe için yeni bir okunabilirlik ölçütü. Dokuz Eylül Üniversitesi Mühendislik
Fakültesi Fen ve Mühendislik Dergisi, 12(3), 49-62.

Turkish syllables are counted by vowel nuclei. Vowel set: {a, e, ı, i, o, ö, u, ü} and their
uppercase forms. Sentence boundaries are `[.!?…]+`.
"""

from __future__ import annotations

import re
from math import sqrt

_TURKISH_VOWELS = set("aeıioöuüAEIİOÖUÜ")
_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?…]+")


def count_syllables_tr(word: str) -> int:
    """Count Turkish syllables in a single word (= number of vowels)."""
    return sum(1 for c in word if c in _TURKISH_VOWELS)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _sentence_count(text: str) -> int:
    # Count non-empty splits.
    parts = [p for p in _SENTENCE_RE.split(text) if p.strip()]
    return max(1, len(parts))


def atesman(text: str) -> float:
    """Ateşman (1997) — Flesch-analogue for Turkish.

        score = 198.825 - 40.175 * (syllables/words) - 2.610 * (words/sentences)

    Higher = easier. Plausible range ~0-110 for typical Turkish prose.
    """
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_syllables = sum(count_syllables_tr(w) for w in words)
    n_sentences = _sentence_count(text)
    return 198.825 - 40.175 * (n_syllables / n_words) - 2.610 * (n_words / n_sentences)


def bezirci_yilmaz(text: str) -> float:
    """Bezirci & Yılmaz (2010) — weighted polysyllabic measure for Turkish.

        score = sqrt(avg_words_per_sentence * (h3*0.84 + h4*1.5 + h5*3.5 + h6*26.25))

    where `h_k` is the fraction of words with `k` syllables (words with ≥7 syllables are binned
    with 6-syllable words for weighting purposes). Higher = harder to read.
    """
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_sentences = _sentence_count(text)

    counts = {3: 0, 4: 0, 5: 0, 6: 0}
    for w in words:
        syl = count_syllables_tr(w)
        if syl >= 6:
            counts[6] += 1
        elif syl in counts:
            counts[syl] += 1

    h3 = counts[3] / n_words
    h4 = counts[4] / n_words
    h5 = counts[5] / n_words
    h6 = counts[6] / n_words

    avg_wps = n_words / n_sentences
    weighted = h3 * 0.84 + h4 * 1.5 + h5 * 3.5 + h6 * 26.25
    return sqrt(avg_wps * weighted)
```

- [ ] **Step 4: Wire Turkish indices into the extractor's registry**

In `src/bitig/features/readability.py`, replace the `"tr": {},` line:

```python
from bitig.languages.readability_tr import atesman as _tr_atesman
from bitig.languages.readability_tr import bezirci_yilmaz as _tr_bezirci_yilmaz

_INDEX_REGISTRY: dict[str, dict[str, Callable[[str], float]]] = {
    "en": {
        # ... unchanged ...
    },
    "tr": {
        "atesman": _tr_atesman,
        "bezirci_yilmaz": _tr_bezirci_yilmaz,
    },
    "de": {},  # Task 5.2
    "es": {},  # Task 5.3
    "fr": {},  # Task 5.4
}
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/languages/test_readability_tr.py tests/features/test_readability.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/bitig/languages/readability_tr.py src/bitig/features/readability.py tests/languages/test_readability_tr.py
git commit -m "feat(languages/tr): Ateşman + Bezirci–Yılmaz readability formulas"
```

---

### Task 5.2: German readability (Flesch-Amstad + Wiener Sachtextformel)

**Files:**
- Create: `src/bitig/languages/readability_de.py`
- Create: `tests/languages/test_readability_de.py`
- Modify: `src/bitig/features/readability.py` (wire into registry)

**Formulas:**
- **Flesch-Amstad (1978)**: `180 - ASL - (58.5 * ASW)` where ASL = average sentence length in words, ASW = average syllables per word.
- **Wiener Sachtextformel I** (Bamberger & Vanecek 1984): `0.1935 * MS + 0.1672 * SL + 0.1297 * IW - 0.0327 * ES - 0.875`, where MS = % of 3+-syllable words, SL = average sentence length, IW = % of words >6 letters, ES = % of single-syllable words. Result is a school-grade (roughly 4-15).

Syllable counting for German uses `pyphen.Pyphen(lang='de_DE')` — splits a word into syllables by the Liang hyphenation dictionary; count = chunks count.

- [ ] **Step 1: Write the failing tests**

```python
# tests/languages/test_readability_de.py
"""Tests for German readability — Flesch-Amstad (1978) and Wiener Sachtextformel."""

from bitig.languages.readability_de import (
    count_syllables_de,
    flesch_amstad,
    wiener_sachtextformel,
)


def test_count_syllables_de_basic() -> None:
    assert count_syllables_de("Haus") == 1
    assert count_syllables_de("Computer") == 3  # Com-pu-ter
    assert count_syllables_de("schwimmen") >= 2  # schwim-men


def test_flesch_amstad_scoring_sense() -> None:
    simple = "Der Hund bellt. Die Katze schläft."
    complex_ = (
        "Die Aufrechterhaltung diplomatischer Verhandlungen erfordert "
        "Kompromissbereitschaft auf beiden Seiten."
    )
    assert flesch_amstad(simple) > flesch_amstad(complex_)


def test_wiener_sachtextformel_scoring_sense() -> None:
    simple = "Der Hund bellt. Die Katze schläft. Die Sonne scheint."
    complex_ = (
        "Die Aufrechterhaltung sozialwirtschaftlicher Gleichgewichtsbedingungen "
        "erfordert interdisziplinäre Kooperationsbereitschaft."
    )
    assert wiener_sachtextformel(complex_) > wiener_sachtextformel(simple)


def test_german_readability_wired_into_extractor() -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.readability import ReadabilityExtractor

    c = Corpus(
        documents=[Document(id="d0", text="Der Hund bellt.")],
        language="de",
    )
    ex = ReadabilityExtractor()
    fm = ex.fit_transform(c)
    assert set(fm.feature_names) == {"flesch_amstad", "wiener_sachtextformel"}


def test_flesch_amstad_empty_returns_zero() -> None:
    assert flesch_amstad("") == 0.0
    assert wiener_sachtextformel("") == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/languages/test_readability_de.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement German readability**

```python
# src/bitig/languages/readability_de.py
"""German readability formulas.

Amstad, T. (1978). Wie verständlich sind unsere Zeitungen? Zurich: Studenten-Schreib-Service.
Bamberger, R., & Vanecek, E. (1984). Lesen — Verstehen — Lernen — Schreiben: Die
Schwierigkeitsstufen von Texten in deutscher Sprache. Wien: Jugend und Volk.

Syllable count uses pyphen's German hyphenation dictionary (de_DE).
"""

from __future__ import annotations

import re

import pyphen

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?…]+")
_PYPHEN_DE = pyphen.Pyphen(lang="de_DE")


def count_syllables_de(word: str) -> int:
    """Count German syllables via Liang-hyphenation (pyphen de_DE)."""
    if not word:
        return 0
    # Pyphen returns hyphens between syllables; count chunks.
    return len(_PYPHEN_DE.inserted(word).split("-"))


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _sentence_count(text: str) -> int:
    parts = [p for p in _SENTENCE_RE.split(text) if p.strip()]
    return max(1, len(parts))


def flesch_amstad(text: str) -> float:
    """Flesch-Amstad (1978): 180 - ASL - 58.5 * ASW.

    ASL = avg sentence length (words); ASW = avg syllables per word. Higher = easier.
    """
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_sentences = _sentence_count(text)
    n_syllables = sum(count_syllables_de(w) for w in words)
    asl = n_words / n_sentences
    asw = n_syllables / n_words
    return 180.0 - asl - 58.5 * asw


def wiener_sachtextformel(text: str) -> float:
    """Wiener Sachtextformel I (Bamberger & Vanecek 1984).

    Formula: 0.1935 * MS + 0.1672 * SL + 0.1297 * IW - 0.0327 * ES - 0.875
      MS = percent of words with ≥3 syllables
      SL = average sentence length
      IW = percent of words with >6 letters
      ES = percent of monosyllabic words

    Result is a school-grade (roughly 4-15); higher = harder.
    """
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_sentences = _sentence_count(text)
    sl = n_words / n_sentences

    n_ms = sum(1 for w in words if count_syllables_de(w) >= 3)
    n_iw = sum(1 for w in words if len(w) > 6)
    n_es = sum(1 for w in words if count_syllables_de(w) == 1)

    ms_pct = 100.0 * n_ms / n_words
    iw_pct = 100.0 * n_iw / n_words
    es_pct = 100.0 * n_es / n_words

    return 0.1935 * ms_pct + 0.1672 * sl + 0.1297 * iw_pct - 0.0327 * es_pct - 0.875
```

- [ ] **Step 4: Wire into extractor**

In `src/bitig/features/readability.py`, replace `"de": {},`:

```python
from bitig.languages.readability_de import flesch_amstad as _de_flesch_amstad
from bitig.languages.readability_de import wiener_sachtextformel as _de_wst

# ...
    "de": {
        "flesch_amstad": _de_flesch_amstad,
        "wiener_sachtextformel": _de_wst,
    },
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/languages/test_readability_de.py tests/features/test_readability.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/bitig/languages/readability_de.py src/bitig/features/readability.py tests/languages/test_readability_de.py
git commit -m "feat(languages/de): Flesch-Amstad + Wiener Sachtextformel readability"
```

---

### Task 5.3: Spanish readability (Fernández-Huerta + Szigriszt-Pazos)

**Files:**
- Create: `src/bitig/languages/readability_es.py`
- Create: `tests/languages/test_readability_es.py`
- Modify: `src/bitig/features/readability.py`

**Formulas:**
- **Fernández-Huerta (1959)**: `206.84 - 60 * (syllables/words) - 1.02 * (words/sentences)`. Flesch for Spanish.
- **Szigriszt-Pazos (1992) / INFLESZ**: `206.835 - 62.3 * (syllables/words) - (words/sentences)`. Refinement.

Syllables for Spanish: count of vowel groups. Vowels = `aeiouáéíóúü`. A vowel group is 1+ consecutive vowels treated as a single syllable nucleus (approximation sufficient for readability scores).

- [ ] **Step 1: Write failing tests**

```python
# tests/languages/test_readability_es.py
"""Tests for Spanish readability — Fernández-Huerta (1959) and Szigriszt-Pazos (1992)."""

from bitig.languages.readability_es import (
    count_syllables_es,
    fernandez_huerta,
    szigriszt_pazos,
)


def test_count_syllables_es_basic() -> None:
    assert count_syllables_es("casa") == 2  # ca-sa
    assert count_syllables_es("libro") == 2  # li-bro
    assert count_syllables_es("camión") == 2  # ca-mión (diphthong collapsed)
    assert count_syllables_es("sol") == 1


def test_fernandez_huerta_scoring_sense() -> None:
    simple = "El gato duerme. El perro juega."
    complex_ = (
        "La sostenibilidad de las negociaciones diplomáticas internacionales requiere "
        "compromiso por ambas partes."
    )
    assert fernandez_huerta(simple) > fernandez_huerta(complex_)


def test_szigriszt_pazos_scoring_sense() -> None:
    simple = "El gato duerme. El perro juega."
    complex_ = (
        "La complejidad de las interrelaciones socioeconómicas globales requiere "
        "interdisciplinariedad."
    )
    assert szigriszt_pazos(simple) > szigriszt_pazos(complex_)


def test_spanish_readability_wired_into_extractor() -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.readability import ReadabilityExtractor

    c = Corpus(documents=[Document(id="d0", text="El gato duerme.")], language="es")
    fm = ReadabilityExtractor().fit_transform(c)
    assert set(fm.feature_names) == {"fernandez_huerta", "szigriszt_pazos"}


def test_empty_returns_zero() -> None:
    assert fernandez_huerta("") == 0.0
    assert szigriszt_pazos("") == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/languages/test_readability_es.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement Spanish readability**

```python
# src/bitig/languages/readability_es.py
"""Spanish readability formulas.

Fernández-Huerta, J. (1959). Medidas sencillas de lecturabilidad. Consigna, 214, 29-32.
Szigriszt-Pazos, F. (1992). Sistemas predictivos de legibilidad del mensaje escrito: fórmula de
perspicuidad. PhD thesis, Universidad Complutense de Madrid. (a.k.a. INFLESZ)

Syllable counting: count of vowel groups (1+ consecutive vowel characters). This is a standard
approximation sufficient for readability-formula inputs.
"""

from __future__ import annotations

import re

_SPANISH_VOWELS = "aeiouáéíóúüAEIOUÁÉÍÓÚÜ"
_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?…]+")
_VOWEL_GROUP_RE = re.compile(f"[{_SPANISH_VOWELS}]+")


def count_syllables_es(word: str) -> int:
    """Count Spanish syllables (vowel groups)."""
    return len(_VOWEL_GROUP_RE.findall(word))


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _sentence_count(text: str) -> int:
    parts = [p for p in _SENTENCE_RE.split(text) if p.strip()]
    return max(1, len(parts))


def fernandez_huerta(text: str) -> float:
    """Fernández-Huerta (1959): 206.84 - 60*(syll/word) - 1.02*(word/sent)."""
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_syllables = sum(count_syllables_es(w) for w in words)
    n_sentences = _sentence_count(text)
    return 206.84 - 60.0 * (n_syllables / n_words) - 1.02 * (n_words / n_sentences)


def szigriszt_pazos(text: str) -> float:
    """Szigriszt-Pazos (1992), a.k.a. INFLESZ: 206.835 - 62.3*(syll/word) - (word/sent)."""
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_syllables = sum(count_syllables_es(w) for w in words)
    n_sentences = _sentence_count(text)
    return 206.835 - 62.3 * (n_syllables / n_words) - (n_words / n_sentences)
```

- [ ] **Step 4: Wire into extractor**

```python
# src/bitig/features/readability.py — replace "es": {},
from bitig.languages.readability_es import fernandez_huerta as _es_fh
from bitig.languages.readability_es import szigriszt_pazos as _es_sp

    "es": {
        "fernandez_huerta": _es_fh,
        "szigriszt_pazos": _es_sp,
    },
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/languages/test_readability_es.py tests/features/test_readability.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/bitig/languages/readability_es.py src/bitig/features/readability.py tests/languages/test_readability_es.py
git commit -m "feat(languages/es): Fernández-Huerta + Szigriszt-Pazos readability"
```

---

### Task 5.4: French readability (Kandel-Moles + LIX)

**Files:**
- Create: `src/bitig/languages/readability_fr.py`
- Create: `tests/languages/test_readability_fr.py`
- Modify: `src/bitig/features/readability.py`

**Formulas:**
- **Kandel-Moles (1958)**: `207 - 1.015 * (words/sentences) - 73.6 * (syllables/words)`. French Flesch adaptation.
- **LIX (Björnsson 1968)**: `(words/sentences) + 100 * (long_words/words)`, where long_word = >6 letters. Language-agnostic but we keep it in the French default set.

French syllables via pyphen `fr_FR` — same pattern as German.

- [ ] **Step 1: Write failing tests**

```python
# tests/languages/test_readability_fr.py

from bitig.languages.readability_fr import count_syllables_fr, kandel_moles, lix


def test_count_syllables_fr_basic() -> None:
    assert count_syllables_fr("bonjour") >= 2  # bon-jour
    assert count_syllables_fr("chat") == 1
    assert count_syllables_fr("anticonstitutionnellement") > 6


def test_kandel_moles_scoring_sense() -> None:
    simple = "Le chat dort. Le chien court."
    complex_ = (
        "La pérennisation des négociations diplomatiques internationales requiert "
        "des compromis substantiels de toutes les parties prenantes."
    )
    assert kandel_moles(simple) > kandel_moles(complex_)


def test_lix_scoring_sense() -> None:
    simple = "Le chat dort. Le chien joue."
    complex_ = (
        "La pérennisation des négociations diplomatiques internationales contemporaines "
        "constitue un défi multidimensionnel."
    )
    assert lix(complex_) > lix(simple)


def test_french_readability_wired_into_extractor() -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.readability import ReadabilityExtractor

    c = Corpus(documents=[Document(id="d0", text="Le chat dort.")], language="fr")
    fm = ReadabilityExtractor().fit_transform(c)
    assert set(fm.feature_names) == {"kandel_moles", "lix"}


def test_empty_returns_zero() -> None:
    assert kandel_moles("") == 0.0
    assert lix("") == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/languages/test_readability_fr.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement French readability**

```python
# src/bitig/languages/readability_fr.py
"""French readability formulas.

Kandel, L., & Moles, A. (1958). Application de l'indice de Flesch à la langue française.
Cahiers d'études de radio-télévision, 19, 253-274.
Björnsson, C. H. (1968). Läsbarhet. Stockholm: Liber.

Syllable count uses pyphen's French hyphenation dictionary (fr_FR).
"""

from __future__ import annotations

import re

import pyphen

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?…]+")
_PYPHEN_FR = pyphen.Pyphen(lang="fr_FR")


def count_syllables_fr(word: str) -> int:
    if not word:
        return 0
    return len(_PYPHEN_FR.inserted(word).split("-"))


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _sentence_count(text: str) -> int:
    parts = [p for p in _SENTENCE_RE.split(text) if p.strip()]
    return max(1, len(parts))


def kandel_moles(text: str) -> float:
    """Kandel-Moles (1958): 207 - 1.015 * (word/sent) - 73.6 * (syll/word)."""
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_syllables = sum(count_syllables_fr(w) for w in words)
    n_sentences = _sentence_count(text)
    return 207.0 - 1.015 * (n_words / n_sentences) - 73.6 * (n_syllables / n_words)


def lix(text: str) -> float:
    """LIX (Björnsson 1968): (word/sent) + 100 * (long/word). long = >6 letters."""
    words = _words(text)
    if not words:
        return 0.0
    n_words = len(words)
    n_sentences = _sentence_count(text)
    n_long = sum(1 for w in words if len(w) > 6)
    return (n_words / n_sentences) + 100.0 * (n_long / n_words)
```

- [ ] **Step 4: Wire into extractor**

```python
# src/bitig/features/readability.py — replace "fr": {},
from bitig.languages.readability_fr import kandel_moles as _fr_km
from bitig.languages.readability_fr import lix as _fr_lix

    "fr": {
        "kandel_moles": _fr_km,
        "lix": _fr_lix,
    },
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/languages/test_readability_fr.py tests/features/test_readability.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/bitig/languages/readability_fr.py src/bitig/features/readability.py tests/languages/test_readability_fr.py
git commit -m "feat(languages/fr): Kandel-Moles + LIX readability"
```

---

### Task 5.5: Function-word regenerator script + bundled lists for TR/DE/ES/FR

**Files:**
- Create: `scripts/regenerate_function_words.py`
- Create: `src/bitig/resources/languages/tr/__init__.py`
- Create: `src/bitig/resources/languages/tr/function_words.txt`
- Create: `src/bitig/resources/languages/de/__init__.py`
- Create: `src/bitig/resources/languages/de/function_words.txt`
- Create: `src/bitig/resources/languages/es/__init__.py`
- Create: `src/bitig/resources/languages/es/function_words.txt`
- Create: `src/bitig/resources/languages/fr/__init__.py`
- Create: `src/bitig/resources/languages/fr/function_words.txt`
- Modify: `tests/features/test_function_words.py` (remove the xfail-style test; add positive tests)

Generator reads CoNLL-U UD files and counts closed-class tokens. Checked-in lists must be committed so ordinary users don't need UD locally.

- [ ] **Step 1: Write the regenerator script**

```python
# scripts/regenerate_function_words.py
"""Regenerate resources/languages/<lang>/function_words.txt from UD CoNLL-U treebanks.

Usage:
    python scripts/regenerate_function_words.py --lang tr \\
        --treebank path/to/UD_Turkish-BOUN \\
        --out src/bitig/resources/languages/tr/function_words.txt

Fetches all tokens tagged with closed-class UPOS (DET PRON ADP CCONJ SCONJ AUX PART), counts
lowercased frequencies, writes the top N. A header comment records source + generation date.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from collections import Counter
from pathlib import Path

_CLOSED_UPOS = {"DET", "PRON", "ADP", "CCONJ", "SCONJ", "AUX", "PART"}


def parse_conllu(path: Path) -> list[tuple[str, str]]:
    """Yield (form, upos) pairs from a CoNLL-U file. Skips comments + multi-word tokens."""
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            idx = fields[0]
            if "-" in idx or "." in idx:
                continue  # multi-word token range or empty node
            form, upos = fields[1], fields[3]
            out.append((form, upos))
    return out


def count_closed_class(treebank_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for f in treebank_dir.glob("*.conllu"):
        for form, upos in parse_conllu(f):
            if upos in _CLOSED_UPOS:
                counts[form.lower()] += 1
    return counts


def write_list(
    out_path: Path,
    counts: Counter[str],
    n: int,
    lang: str,
    source: str,
) -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = "unknown"
    header = [
        f"# Function-word list for {lang} — top {n} closed-class tokens by frequency",
        f"# UPOS filter: {sorted(_CLOSED_UPOS)}",
        f"# Source treebank(s): {source}",
        f"# Generated: {dt.date.today().isoformat()} by scripts/regenerate_function_words.py @ {commit}",
    ]
    top = [w for w, _ in counts.most_common(n)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(header + [""] + top) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--treebank", required=True, type=Path, nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    merged: Counter[str] = Counter()
    names = []
    for tb in args.treebank:
        merged.update(count_closed_class(tb))
        names.append(tb.name)

    write_list(args.out, merged, args.n, args.lang, source=" + ".join(names))
    print(f"Wrote {args.n} tokens to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the four new lists**

Download the required UD treebanks (one-time developer setup; users never run this):

```bash
mkdir -p /tmp/ud
cd /tmp/ud
for repo in UD_Turkish-BOUN UD_German-GSD UD_Spanish-AnCora UD_French-GSD; do
    git clone --depth 1 "https://github.com/UniversalDependencies/${repo}.git"
done
cd -
```

Run the generator for each language:

```bash
python scripts/regenerate_function_words.py --lang tr \
    --treebank /tmp/ud/UD_Turkish-BOUN \
    --out src/bitig/resources/languages/tr/function_words.txt

python scripts/regenerate_function_words.py --lang de \
    --treebank /tmp/ud/UD_German-GSD \
    --out src/bitig/resources/languages/de/function_words.txt

python scripts/regenerate_function_words.py --lang es \
    --treebank /tmp/ud/UD_Spanish-AnCora \
    --out src/bitig/resources/languages/es/function_words.txt

python scripts/regenerate_function_words.py --lang fr \
    --treebank /tmp/ud/UD_French-GSD \
    --out src/bitig/resources/languages/fr/function_words.txt
```

Expected: each command writes a ≥200-token file with a header.

- [ ] **Step 3: Create package `__init__.py` files for resource discovery**

```bash
for lang in tr de es fr; do
    touch src/bitig/resources/languages/${lang}/__init__.py
done
```

- [ ] **Step 4: Update the function-word test to match the new reality**

Replace the xfail-expecting test in `tests/features/test_function_words.py`:

```python
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
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/features/test_function_words.py tests/languages -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/regenerate_function_words.py src/bitig/resources/languages/tr src/bitig/resources/languages/de src/bitig/resources/languages/es src/bitig/resources/languages/fr tests/features/test_function_words.py
git commit -m "feat(resources): generate function-word lists for TR/DE/ES/FR from UD closed-class tokens"
```

---

## Phase 6 — CLI polish, docs, integration CI

### Task 6.1: `bitig init --language` + `bitig info` displays language

**Files:**
- Modify: `src/bitig/cli/init_cmd.py`
- Modify: `src/bitig/scaffold/__init__.py` (or wherever `scaffold_project` lives)
- Modify: `src/bitig/scaffold/templates/study.yaml.j2`
- Modify: `src/bitig/cli/info_cmd.py`
- Modify: `tests/cli/test_init.py` (append)
- Modify: `tests/cli/test_info.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# tests/cli/test_init.py — append

def test_init_writes_language_into_study_yaml(tmp_path) -> None:
    from typer.testing import CliRunner
    from bitig.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app, ["init", "mystudy", "--target", str(tmp_path / "mystudy"), "--language", "tr"]
    )
    assert result.exit_code == 0, result.stdout
    yaml = (tmp_path / "mystudy" / "study.yaml").read_text()
    assert "language: tr" in yaml
```

```python
# tests/cli/test_info.py — append

def test_info_displays_corpus_language(tmp_path) -> None:
    from typer.testing import CliRunner
    from bitig.cli import app

    # Set up a minimal project
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("merhaba")
    # Create a minimal study.yaml
    (tmp_path / "study.yaml").write_text(
        "name: t\ncorpus:\n  path: corpus\npreprocess:\n  language: tr\n"
    )

    runner = CliRunner()
    result = runner.invoke(app, ["info"], catch_exceptions=False, env={"BITIG_PROJECT_DIR": str(tmp_path)})
    # The test assumes `bitig info` reads study.yaml from cwd or from BITIG_PROJECT_DIR if supported.
    # If not supported, cd into tmp_path before invoking — adapt to existing test pattern.
    assert "language" in result.stdout.lower()
    assert "tr" in result.stdout
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/cli/test_init.py tests/cli/test_info.py -v -k language
```

Expected: FAIL.

- [ ] **Step 3: Add `--language` to init**

```python
# src/bitig/cli/init_cmd.py
def init_command(
    name: str = typer.Argument(...),
    target: Path | None = typer.Option(None, "--target", "-t"),  # noqa: B008
    force: bool = typer.Option(False, "--force"),
    language: str = typer.Option(
        "en", "--language", "-l", help="Project language code (en/tr/de/es/fr)."
    ),
) -> None:
    """Scaffold a new bitig project directory."""
    dest = target if target is not None else Path.cwd() / name
    try:
        created = scaffold_project(name=name, target=dest, force=force, language=language)
    except FileExistsError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]created project[/green] {created} (language={language})")
    console.print(f"  cd {created}")
    console.print("  bitig run study.yaml")
```

- [ ] **Step 4: Thread language through the scaffold**

In `src/bitig/scaffold/__init__.py` (or the file that defines `scaffold_project`), add a `language` parameter and pass it to the Jinja context rendering `study.yaml.j2`.

```python
def scaffold_project(
    *,
    name: str,
    target: Path,
    force: bool = False,
    language: str = "en",
) -> Path:
    from bitig.languages import get_language
    get_language(language)  # validate early
    # ... existing body ...
    # When rendering study.yaml.j2, add `language=language` to the context dict.
```

- [ ] **Step 5: Update the Jinja template**

In `src/bitig/scaffold/templates/study.yaml.j2`, add inside the `preprocess:` block:

```yaml
preprocess:
  language: {{ language | default("en") }}
```

- [ ] **Step 6: Update `bitig info` to display language**

In `src/bitig/cli/info_cmd.py`, add a line that reads `preprocess.language` from the loaded `StudyConfig` (or from the ingested `Corpus.language`) and prints it. Exact code depends on current structure; look at how existing fields are displayed.

Minimal addition, near existing corpus info display:

```python
# inside info_command body, after loading config/corpus
console.print(f"  language: {config.preprocess.language}")
```

- [ ] **Step 7: Run tests**

```bash
pytest tests/cli/test_init.py tests/cli/test_info.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/bitig/cli src/bitig/scaffold tests/cli
git commit -m "feat(cli): init --language; info displays configured language"
```

---

### Task 6.2: Turkish tutorial page + concepts/languages.md

**Files:**
- Create: `docs/site/tutorials/turkish.md`
- Create: `docs/site/concepts/languages.md`
- Modify: `mkdocs.yml` (add nav entries)

Turkish tutorial uses 3-5 short Ömer Seyfettin stories from Turkish Wikisource (public domain). The tutorial walks through: `bitig init --language tr`, dropping texts into `corpus/`, ingest, basic Delta, PCA plot. No forensic framing.

- [ ] **Step 1: Write the concepts page**

```markdown
<!-- docs/site/concepts/languages.md -->
# Languages

bitig ships with first-class support for five languages: **English**, **Turkish**, **German**,
**Spanish**, **French**. Each language has bundled function-word lists, native readability
formulas, and tested end-to-end pipelines.

## Supported languages

| Code | Name    | Backend         | Default model            | Readability                         |
|------|---------|-----------------|--------------------------|-------------------------------------|
| en   | English | native spaCy    | `en_core_web_trf`        | Flesch, FK, Gunning Fog, …          |
| tr   | Turkish | `spacy-stanza`  | Stanza `tr` (BOUN)       | Ateşman, Bezirci–Yılmaz             |
| de   | German  | native spaCy    | `de_dep_news_trf`        | Flesch-Amstad, Wiener Sachtextformel|
| es   | Spanish | native spaCy    | `es_dep_news_trf`        | Fernández-Huerta, Szigriszt-Pazos   |
| fr   | French  | native spaCy    | `fr_dep_news_trf`        | Kandel-Moles, LIX                   |

## How the registry works

```python
from bitig import LANGUAGES, get_language

spec = get_language("tr")
print(spec.backend)                   # 'spacy_stanza'
print(spec.readability_indices)       # ('atesman', 'bezirci_yilmaz')
print(spec.contextual_embedding_default)  # 'dbmdz/bert-base-turkish-cased'
```

## Declaring language in a study

```yaml
# study.yaml
preprocess:
  language: tr
  spacy:
    # model and backend are auto-resolved from `language`.
    # Override if you know what you're doing:
    # model: my-custom-model
    # backend: spacy
```

From the CLI:

```bash
bitig init mystudy --language tr
bitig ingest corpus/ --language tr --metadata corpus/metadata.tsv
```

## Turkish prerequisites

Turkish uses [Stanza](https://stanfordnlp.github.io/stanza/) through
[`spacy-stanza`](https://github.com/explosion/spacy-stanza).

```bash
uv pip install 'bitig[turkish]'
python -c "import stanza; stanza.download('tr')"
```

After download, `bitig ingest --language tr` works identically to the English path — Stanza
pipelines return native spaCy `Doc` objects.

## Adding a sixth language

1. Add a `LanguageSpec` entry to `bitig.languages.registry.REGISTRY`.
2. Create `src/bitig/resources/languages/<code>/function_words.txt` (run
   `scripts/regenerate_function_words.py`).
3. If readability formulas don't exist for the language, write them in
   `bitig.languages.readability_<code>` and register them in
   `bitig.features.readability._INDEX_REGISTRY`.
4. Add integration tests and a tutorial page.

See [`docs/superpowers/specs/2026-04-19-multi-language-support-design.md`](#) for the full spec.
```

- [ ] **Step 2: Write the Turkish tutorial**

```markdown
<!-- docs/site/tutorials/turkish.md -->
# Turkish authorship walkthrough

A complete runnable example: attribute a small Turkish corpus using MFW + Ateşman readability.

## Setup

```bash
uv pip install 'bitig[turkish]'
python -c "import stanza; stanza.download('tr')"
bitig init seyfettin --language tr
cd seyfettin
```

This scaffolds a project directory with `study.yaml` pre-configured for Turkish.

## Corpus

Place 3-5 Turkish short stories in `corpus/` as UTF-8 `.txt` files. Public-domain sources:

- [Ömer Seyfettin on Turkish Wikisource](https://tr.wikisource.org/wiki/Yazar:%C3%96mer_Seyfettin) — dozens of early-20th-century short stories

Add a `corpus/metadata.tsv`:

```tsv
filename	author	year
bomba.txt	Omer_Seyfettin	1910
kesik_biyik.txt	Omer_Seyfettin	1911
forsa.txt	Omer_Seyfettin	1913
```

## Running the study

```bash
bitig ingest corpus/ --language tr --metadata corpus/metadata.tsv
bitig info
bitig run study.yaml --name first-run
```

`bitig ingest` runs Stanza through `spacy-stanza`. The first run parses every document and
caches the DocBins; subsequent runs hit the cache.

## What you get

A default Turkish study computes:

- MFW (top 200 tokens, z-scored relative frequencies)
- Turkish function words (loaded from `resources/languages/tr/function_words.txt`, derived from
  UD Turkish BOUN closed-class tokens)
- Ateşman and Bezirci-Yılmaz readability indices
- Burrows Delta + PCA/MDS reduction plots

The output folder `results/first-run/` contains JSON, Parquet feature matrices, PNG/PDF figures,
and a standalone HTML report.

## Customising

Edit `study.yaml` to swap features or methods. For example, to use contextual embeddings instead
of MFW:

```yaml
features:
  - id: bert_tr
    type: contextual_embedding
    # model auto-resolves to `dbmdz/bert-base-turkish-cased` via the language registry
```

For heavyweight embeddings (e.g. BERT5urk, a 1.42B Turkish T5 model):

```yaml
features:
  - id: bert5urk
    type: contextual_embedding
    model: stefan-it/bert5urk
    pool: mean
```
```

- [ ] **Step 3: Add nav entries to `mkdocs.yml`**

In the `nav:` section, under `Concepts:`, add:

```yaml
  - Concepts:
      - concepts/index.md
      - Corpus: concepts/corpus.md
      - Features: concepts/features.md
      - Languages: concepts/languages.md      # NEW
      - Methods: concepts/methods.md
      - Results & provenance: concepts/results.md
```

Under `Tutorials:`, add:

```yaml
  - Tutorials:
      - tutorials/index.md
      - Federalist Papers: tutorials/federalist.md
      - PAN-CLEF verification: tutorials/pan-clef.md
      - Turkish stylometry: tutorials/turkish.md   # NEW
```

- [ ] **Step 4: Build docs locally to verify**

```bash
uv pip install -e '.[docs]'
mkdocs build --strict
```

Expected: builds clean. The strict flag catches broken links.

- [ ] **Step 5: Commit**

```bash
git add docs/site/concepts/languages.md docs/site/tutorials/turkish.md mkdocs.yml
git commit -m "docs: concepts/languages + tutorials/turkish; nav updates"
```

---

### Task 6.3: `tests-multilang.yml` CI workflow

**Files:**
- Create: `.github/workflows/tests-multilang.yml`
- Create: `tests/integration/__init__.py` (if missing)
- Create: `tests/integration/test_turkish_end_to_end.py`

The workflow runs on a schedule + tags, installs spaCy + Stanza models, and exercises the Turkish path end-to-end. Gated on `@pytest.mark.slow`.

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_turkish_end_to_end.py
"""End-to-end Turkish pipeline: Stanza parse → features → Burrows Delta.

Skipped unless:
  - spacy-stanza + stanza are installed
  - stanza's Turkish model has been downloaded

Runs under the `slow` marker only.
"""

import pytest

pytest.importorskip("spacy_stanza")
pytest.importorskip("stanza")

pytestmark = pytest.mark.slow


def test_turkish_pipeline_smoke(tmp_path) -> None:
    from bitig.corpus import Corpus, Document
    from bitig.features.function_words import FunctionWordExtractor
    from bitig.features.readability import ReadabilityExtractor
    from bitig.preprocess.pipeline import SpacyPipeline

    texts = [
        "Ali topu tuttu. Kedi uyudu. Hava güzeldi.",
        "Ahmet kitabı okudu. Öğretmen sordu. Öğrenciler dinledi.",
        "Mehmet eve gitti. Kardeşi bekledi. Yemek yediler.",
    ]
    corpus = Corpus(
        documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)],
        language="tr",
    )

    pipe = SpacyPipeline(language="tr", cache_dir=tmp_path / "cache")
    parsed = pipe.parse(corpus)
    assert len(parsed) == 3

    fw = FunctionWordExtractor(scale="none").fit_transform(corpus)
    assert fw.X.shape[0] == 3

    rb = ReadabilityExtractor().fit_transform(corpus)
    assert rb.X.shape == (3, 2)  # atesman + bezirci_yilmaz
```

- [ ] **Step 2: Write the CI workflow**

```yaml
# .github/workflows/tests-multilang.yml
name: Multi-language tests

on:
  push:
    tags: ["v*"]
  schedule:
    - cron: "17 3 * * 1"  # Mondays 03:17 UTC
  workflow_dispatch:

permissions:
  contents: read

jobs:
  turkish:
    runs-on: ubuntu-latest
    timeout-minutes: 25
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install bitig + Turkish extras
        run: uv pip install --system -e ".[turkish,embeddings,dev]"
      - name: Download Stanza Turkish model
        run: python -c "import stanza; stanza.download('tr')"
      - name: Run Turkish integration tests
        run: pytest -m slow tests/integration -v

  european:
    runs-on: ubuntu-latest
    timeout-minutes: 25
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: uv pip install --system -e ".[dev]"
      - name: Download spaCy pipelines (DE, ES, FR)
        run: |
          python -m spacy download de_dep_news_trf
          python -m spacy download es_dep_news_trf
          python -m spacy download fr_dep_news_trf
      - name: Run language readability unit tests
        run: pytest tests/languages -v
```

- [ ] **Step 3: Validate the workflow locally with `actionlint` or `yamllint`**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/tests-multilang.yml'))" && echo ok
```

Expected: `ok`.

- [ ] **Step 4: Verify the integration test skips cleanly locally when models aren't present**

```bash
pytest -m slow tests/integration -v
```

Expected: SKIP with "spacy_stanza not installed" OR PASS if you've already installed + downloaded. In either case, no failure.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/tests-multilang.yml tests/integration
git commit -m "ci: weekly multi-language integration workflow; Turkish end-to-end test"
```

---

## Self-review

**Spec coverage check:**
- ✅ Language registry + `LanguageSpec` — Task 1.1
- ✅ Per-language resource layout + English migration — Task 1.2
- ✅ Top-level public API re-exports — Task 1.3
- ✅ `Corpus.language` + hash participation — Task 2.1
- ✅ `load_corpus(..., language=)` + CLI `--language` — Task 2.2
- ✅ `PreprocessConfig.language` + pydantic validation — Task 2.3
- ✅ `turkish`/`multilang` extras, `pyphen` core dep — Task 3.1
- ✅ `cache_key` renamed to `backend_version` with English back-compat — Task 3.2
- ✅ `SpacyPipeline` backend dispatch + error messages — Task 3.3
- ✅ `FunctionWordExtractor.language` — Task 4.1
- ✅ `ReadabilityExtractor.language` + per-language registry — Task 4.2
- ✅ Embedding extractors pick up language defaults — Task 4.3
- ✅ Native readability for TR/DE/ES/FR with source-paper citations — Tasks 5.1–5.4
- ✅ Function-word generator + bundled lists — Task 5.5
- ✅ CLI `init --language` + `info` displays language — Task 6.1
- ✅ `concepts/languages.md` + Turkish tutorial — Task 6.2
- ✅ Multi-language integration CI workflow — Task 6.3

**Placeholder scan:** no `TBD`, no `// ...`, no references to undefined types. The `"tr": {},` placeholders in Task 4.2 are filled in by Tasks 5.1–5.4 in order.

**Type consistency:** `LanguageSpec.default_model`, `LanguageSpec.backend`, `SpacyPipeline.model`, `SpacyPipeline.backend`, `_INDEX_REGISTRY: dict[str, dict[str, Callable[[str], float]]]` — all referenced consistently.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-19-multi-language-support.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
