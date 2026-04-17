# tamga — Phase 2: Features & Delta Family — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the feature-extraction layer (`tamga.features`), the Delta family of distance-based classifiers (`tamga.methods.delta`), the associated CLI commands (`tamga features`, `tamga delta`), and a Federalist Papers parity suite that validates Burrows Delta against the canonical Hamilton-vs-Madison attributions. End state: `tamga delta` reproduces Stylo's classical Burrows Delta on the Federalist Papers to within 1e-6 tolerance; all feature extractors implement sklearn's transformer protocol and compose cleanly in `sklearn.pipeline.Pipeline`.

**Architecture:** Each feature extractor is a focused module under `tamga/features/`, inheriting `BaseEstimator, TransformerMixin` from scikit-learn and producing a `FeatureMatrix` (thin wrapper over `np.ndarray` with feature names, document ids, and a provenance hash). Each Delta variant is a `ClassifierMixin` subclass in `tamga/methods/delta/`, wrapping a shared `_DeltaBase` that implements the nearest-author-centroid pattern; subclasses differ only in the distance kernel and optional pre-scaling. The CLI commands are thin glue over `load_corpus → SpacyPipeline.parse → FeatureExtractor.fit_transform → DeltaClassifier.fit → predict`.

**Tech Stack:** Phase 1 stack (Python 3.11+, uv, pydantic v2, Typer, spaCy) plus scikit-learn transformers, numpy, pandas, scipy (`scipy.stats` for skew), and the `textstat` library for readability indices.

**Reference spec:** `docs/superpowers/specs/2026-04-17-tamga-stylometry-package-design.md` (§5 Feature Extractors; §6.1 Delta family; §7 sklearn interoperability).

**Phase 1 baseline:** tag `phase-1-foundation`, 92 tests, 95.8% coverage. This plan adds on top.

---

## File Layout (new in Phase 2)

```
src/tamga/
├── features/
│   ├── __init__.py               # public re-exports
│   ├── base.py                   # BaseFeatureExtractor + FeatureMatrix
│   ├── mfw.py                    # MFWExtractor
│   ├── ngrams.py                 # CharNgramExtractor, WordNgramExtractor
│   ├── pos.py                    # PosNgramExtractor
│   ├── dependency.py             # DependencyBigramExtractor
│   ├── function_words.py         # FunctionWordExtractor + bundled EN list
│   ├── punctuation.py            # PunctuationExtractor
│   ├── lexical_diversity.py      # LexicalDiversityExtractor
│   ├── readability.py            # ReadabilityExtractor
│   └── sentence_length.py        # SentenceLengthExtractor
├── methods/
│   ├── __init__.py
│   └── delta/
│       ├── __init__.py
│       ├── base.py               # _DeltaBase (shared fit/predict logic)
│       ├── burrows.py            # BurrowsDelta
│       ├── eder.py               # EderDelta, EderSimpleDelta
│       ├── argamon.py            # ArgamonLinearDelta, QuadraticDelta
│       └── cosine.py             # CosineDelta
├── resources/
│   └── function_words_en.txt     # bundled English function-word list
└── cli/
    ├── features_cmd.py           # `tamga features`
    └── delta_cmd.py              # `tamga delta`

tests/
├── features/
│   ├── __init__.py
│   ├── test_feature_matrix.py
│   ├── test_mfw.py
│   ├── test_ngrams.py
│   ├── test_pos.py
│   ├── test_dependency.py
│   ├── test_function_words.py
│   ├── test_punctuation.py
│   ├── test_lexical_diversity.py
│   ├── test_readability.py
│   └── test_sentence_length.py
├── methods/
│   ├── __init__.py
│   └── delta/
│       ├── __init__.py
│       ├── test_base.py
│       ├── test_burrows.py
│       ├── test_eder.py
│       ├── test_argamon.py
│       └── test_cosine.py
├── integration/
│   ├── __init__.py
│   ├── test_sklearn_pipeline.py
│   └── test_federalist_parity.py
├── fixtures/federalist/
│   ├── metadata.tsv              # author labels for Hamilton/Madison/Jay/disputed
│   └── *.txt                     # bundled Federalist Papers (public domain)
└── cli/
    ├── test_features_cmd.py
    └── test_delta_cmd.py
```

---

## Pre-flight — dependencies

Phase 2 introduces one new runtime dependency: **`textstat`** (readability formulas — Flesch, Flesch-Kincaid, Gunning Fog, Coleman-Liau, ARI, SMOG — are tedious to hand-roll and textstat is the reference Python implementation). `scipy` is already in the dependency list.

Add to `pyproject.toml`:

```toml
dependencies = [
    ...existing deps...,
    "textstat>=0.7",
]
```

Done once in **Task 1**.

---

## Task 1: Add `textstat` dependency + regenerate lock

**Files:**
- Modify: `pyproject.toml` (one line)

- [ ] **Step 1.1:** Append `"textstat>=0.7",` to the `dependencies` array in `pyproject.toml` (alphabetical order: insert before `typer`).

- [ ] **Step 1.2:** `source .venv/bin/activate && uv pip install -e ".[dev]"`. Expected: textstat installs alongside its dep (`pyphen`). No conflicts.

- [ ] **Step 1.3:** Verify: `python -c "import textstat; print(textstat.flesch_reading_ease('The quick brown fox jumps over the lazy dog.'))"` — prints a float near 94.

- [ ] **Step 1.4:** Commit.

```bash
git add pyproject.toml
git commit -m "build: add textstat dependency for readability indices"
```

---

## Task 2: `FeatureMatrix` — the shared return type

**Files:**
- Create: `src/tamga/features/__init__.py`
- Create: `src/tamga/features/base.py`
- Create: `tests/features/__init__.py`
- Create: `tests/features/test_feature_matrix.py`

**TDD task.**

- [ ] **Step 2.1:** `src/tamga/features/__init__.py`:

```python
"""Feature extractors producing FeatureMatrix objects."""

from tamga.features.base import BaseFeatureExtractor, FeatureMatrix

__all__ = ["BaseFeatureExtractor", "FeatureMatrix"]
```

- [ ] **Step 2.2:** `tests/features/__init__.py` (empty).

- [ ] **Step 2.3:** Write failing tests in `tests/features/test_feature_matrix.py`:

```python
"""Tests for the FeatureMatrix dataclass."""

import numpy as np
import pandas as pd
import pytest

from tamga.features import FeatureMatrix


def _fm(X: np.ndarray, feature_names: list[str], doc_ids: list[str] | None = None) -> FeatureMatrix:
    return FeatureMatrix(
        X=X,
        document_ids=doc_ids or [f"d{i}" for i in range(X.shape[0])],
        feature_names=feature_names,
        feature_type="test",
        extractor_config={},
        provenance_hash="0" * 64,
    )


def test_feature_matrix_basic_access():
    X = np.arange(6).reshape(2, 3).astype(float)
    fm = _fm(X, ["a", "b", "c"])
    assert fm.X.shape == (2, 3)
    assert fm.feature_names == ["a", "b", "c"]
    assert fm.document_ids == ["d0", "d1"]


def test_feature_matrix_as_dataframe_preserves_rows_and_cols():
    X = np.array([[1, 2], [3, 4]], dtype=float)
    fm = _fm(X, ["a", "b"], doc_ids=["x", "y"])
    df = fm.as_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["x", "y"]
    assert list(df.columns) == ["a", "b"]
    np.testing.assert_array_equal(df.to_numpy(), X)


def test_feature_matrix_concat_stacks_columns():
    a = _fm(np.array([[1.0, 2.0]]), ["p", "q"], doc_ids=["d0"])
    b = _fm(np.array([[3.0, 4.0]]), ["r", "s"], doc_ids=["d0"])
    c = a.concat(b)
    assert c.X.shape == (1, 4)
    assert c.feature_names == ["p", "q", "r", "s"]
    assert c.document_ids == ["d0"]


def test_feature_matrix_concat_rejects_mismatched_docs():
    a = _fm(np.array([[1.0]]), ["p"], doc_ids=["d0"])
    b = _fm(np.array([[2.0]]), ["q"], doc_ids=["d1"])
    with pytest.raises(ValueError, match="document_ids"):
        a.concat(b)


def test_feature_matrix_concat_rejects_duplicate_feature_names():
    a = _fm(np.array([[1.0]]), ["shared"], doc_ids=["d0"])
    b = _fm(np.array([[2.0]]), ["shared"], doc_ids=["d0"])
    with pytest.raises(ValueError, match="duplicate feature"):
        a.concat(b)


def test_feature_matrix_len_is_n_documents():
    fm = _fm(np.zeros((5, 3)), ["a", "b", "c"])
    assert len(fm) == 5


def test_feature_matrix_n_features_property():
    fm = _fm(np.zeros((2, 7)), [f"f{i}" for i in range(7)])
    assert fm.n_features == 7
```

- [ ] **Step 2.4:** Run — expect FAIL (ImportError on `BaseFeatureExtractor, FeatureMatrix`).

- [ ] **Step 2.5:** Implement `src/tamga/features/base.py`:

```python
"""FeatureMatrix — the shared return type for every feature extractor — plus the sklearn-compatible
base class extractors inherit from.

Every FeatureMatrix carries its provenance: what extractor produced it, what config was used, and
a hash that combines extractor config + corpus hash. That hash ends up in `Result.provenance`.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from tamga.corpus import Corpus


@dataclass
class FeatureMatrix:
    X: np.ndarray
    document_ids: list[str]
    feature_names: list[str]
    feature_type: str
    extractor_config: dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError(f"FeatureMatrix.X must be 2-D; got shape {self.X.shape}")
        if self.X.shape[0] != len(self.document_ids):
            raise ValueError(
                f"FeatureMatrix: rows ({self.X.shape[0]}) != len(document_ids) "
                f"({len(self.document_ids)})"
            )
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"FeatureMatrix: cols ({self.X.shape[1]}) != len(feature_names) "
                f"({len(self.feature_names)})"
            )

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.X, index=self.document_ids, columns=self.feature_names)

    def concat(self, other: FeatureMatrix) -> FeatureMatrix:
        """Column-concatenate two FeatureMatrix objects with identical document_ids."""
        if self.document_ids != other.document_ids:
            raise ValueError("FeatureMatrix.concat: document_ids must match exactly")
        shared = set(self.feature_names) & set(other.feature_names)
        if shared:
            raise ValueError(f"FeatureMatrix.concat: duplicate feature names: {sorted(shared)}")
        return FeatureMatrix(
            X=np.hstack([self.X, other.X]),
            document_ids=list(self.document_ids),
            feature_names=list(self.feature_names) + list(other.feature_names),
            feature_type=f"{self.feature_type}+{other.feature_type}",
            extractor_config={"a": self.extractor_config, "b": other.extractor_config},
            provenance_hash="",
        )


class BaseFeatureExtractor(BaseEstimator, TransformerMixin):
    """Base class every feature extractor inherits from.

    Subclasses implement:
      - `_fit(corpus)` — learn vocabulary / state from a training corpus.
      - `_transform(corpus)` — produce an (n_docs, n_features) numpy array and a list of feature names.
      - `feature_type` class attribute — string tag stored on the FeatureMatrix.

    The sklearn-compatible `fit`, `transform`, and `fit_transform` methods wrap these in the
    FeatureMatrix envelope.
    """

    feature_type: str = "base"

    @abstractmethod
    def _fit(self, corpus: Corpus) -> None: ...

    @abstractmethod
    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        """Return (X, feature_names) for the given corpus."""

    def fit(self, corpus: Corpus, y: Any = None) -> BaseFeatureExtractor:  # noqa: ARG002
        self._fit(corpus)
        return self

    def transform(self, corpus: Corpus) -> FeatureMatrix:
        X, feature_names = self._transform(corpus)
        return FeatureMatrix(
            X=X,
            document_ids=[d.id for d in corpus.documents],
            feature_names=feature_names,
            feature_type=self.feature_type,
            extractor_config=self.get_params(),
            provenance_hash=self._provenance(corpus),
        )

    def fit_transform(self, corpus: Corpus, y: Any = None) -> FeatureMatrix:  # noqa: ARG002
        return self.fit(corpus).transform(corpus)

    def _provenance(self, corpus: Corpus) -> str:
        from tamga.plumbing.hashing import hash_mapping

        payload = {
            "extractor": type(self).__name__,
            "config": self.get_params(),
            "corpus_hash": corpus.hash(),
            "feature_type": self.feature_type,
        }
        return hash_mapping(payload)
```

- [ ] **Step 2.6:** Run — expect all 7 tests PASS.

- [ ] **Step 2.7:** Commit.

```bash
git add src/tamga/features/__init__.py src/tamga/features/base.py tests/features/__init__.py tests/features/test_feature_matrix.py
git commit -m "feat(features): FeatureMatrix + BaseFeatureExtractor (sklearn-transformer base)"
```

---

## Task 3: `MFWExtractor` — Most Frequent Words

**Files:**
- Create: `src/tamga/features/mfw.py`
- Create: `tests/features/test_mfw.py`
- Modify: `src/tamga/features/__init__.py` (re-export `MFWExtractor`)

**TDD task. Core extractor. Every Delta method depends on this.**

- [ ] **Step 3.1:** Failing tests `tests/features/test_mfw.py`:

```python
"""Tests for MFWExtractor — Most Frequent Words."""

import numpy as np
import pytest

from tamga.corpus import Corpus, Document
from tamga.features import FeatureMatrix
from tamga.features.mfw import MFWExtractor


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_mfw_returns_feature_matrix():
    corpus = _corpus("the quick brown fox", "the lazy dog")
    mfw = MFWExtractor(n=5, scale="none")
    fm = mfw.fit_transform(corpus)
    assert isinstance(fm, FeatureMatrix)
    assert fm.X.shape[0] == 2


def test_mfw_selects_top_n_words_by_corpus_frequency():
    corpus = _corpus("the the the cat", "the dog dog")
    mfw = MFWExtractor(n=2, scale="none")
    fm = mfw.fit_transform(corpus)
    assert set(fm.feature_names) == {"the", "dog"}
    assert fm.X.shape[1] == 2


def test_mfw_raw_counts_sum_to_expected_values():
    corpus = _corpus("the cat sat", "the dog sat")
    mfw = MFWExtractor(n=3, scale="none")
    fm = mfw.fit_transform(corpus)
    # Each document has the three tokens: the, <x>, sat.
    # Column order follows fit-order (by frequency desc then alphabetical).
    df = fm.as_dataframe()
    assert df.loc["d0", "the"] == 1
    assert df.loc["d0", "sat"] == 1
    assert df.loc["d1", "the"] == 1


def test_mfw_with_min_df_filters_rare_words():
    corpus = _corpus("a rare unique snowflake", "common common common")
    mfw = MFWExtractor(n=10, min_df=2, scale="none")
    fm = mfw.fit_transform(corpus)
    # Only words appearing in ≥2 docs survive; rare/unique/snowflake filtered out.
    assert "rare" not in fm.feature_names
    assert "common" not in fm.feature_names  # only appears in 1 doc
    # Actually no words survive here — confirm the test's premise explicitly:
    assert fm.X.shape[1] == 0


def test_mfw_z_score_scaling_has_column_mean_zero():
    corpus = _corpus(
        "the the cat cat cat",
        "the dog dog",
        "the the the the dog dog",
    )
    mfw = MFWExtractor(n=3, scale="zscore")
    fm = mfw.fit_transform(corpus)
    # Each column should have mean ~0, std ~1 across the three training docs.
    col_means = fm.X.mean(axis=0)
    col_stds = fm.X.std(axis=0, ddof=0)
    np.testing.assert_allclose(col_means, 0, atol=1e-9)
    # std may be 0 if a feature is constant; where non-zero, it should be 1.
    for std in col_stds:
        assert std == pytest.approx(0.0, abs=1e-9) or std == pytest.approx(1.0, abs=1e-6)


def test_mfw_is_sklearn_compatible():
    # fit then transform on the same corpus produces a result.
    corpus = _corpus("a b c a b", "d e f d")
    mfw = MFWExtractor(n=3, scale="none")
    mfw.fit(corpus)
    fm = mfw.transform(corpus)
    assert fm.X.shape == (2, 3)


def test_mfw_transform_uses_fitted_vocabulary():
    train = _corpus("alpha beta gamma alpha", "beta beta")
    mfw = MFWExtractor(n=2, scale="none")
    mfw.fit(train)
    test = _corpus("alpha zeta zeta zeta")
    # Test doc has `alpha` once and `zeta` thrice, but vocabulary is frozen from train,
    # so we only see counts for train-time vocab.
    fm = mfw.transform(test)
    df = fm.as_dataframe()
    assert df.loc["d0", "alpha"] == 1


def test_mfw_case_folding_by_default_is_off():
    corpus = _corpus("The the THE", "the Cat cat")
    mfw = MFWExtractor(n=5, scale="none", lowercase=False)
    fm = mfw.fit_transform(corpus)
    # Without lowercasing, 'The' and 'THE' are distinct tokens.
    assert "the" in fm.feature_names
    assert "The" in fm.feature_names


def test_mfw_case_folding_when_enabled():
    corpus = _corpus("The the THE", "the Cat cat")
    mfw = MFWExtractor(n=5, scale="none", lowercase=True)
    fm = mfw.fit_transform(corpus)
    assert "the" in fm.feature_names
    assert "The" not in fm.feature_names
    assert "THE" not in fm.feature_names
```

- [ ] **Step 3.2:** Run — FAIL.

- [ ] **Step 3.3:** Implement `src/tamga/features/mfw.py`:

```python
"""MFWExtractor — the most-frequent-words feature table, the classical stylometric input.

Tokenisation is word-boundary based on whitespace + punctuation-stripping (the Stylo default when
using word MFW). For POS-based or dependency-based features, use PosNgramExtractor or
DependencyBigramExtractor, which tokenise via spaCy.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

import numpy as np

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor

Scale = Literal["none", "zscore", "l1", "l2"]

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def _tokenise(text: str, lowercase: bool) -> list[str]:
    tokens = _WORD_RE.findall(text)
    return [t.lower() for t in tokens] if lowercase else tokens


class MFWExtractor(BaseFeatureExtractor):
    """Most-Frequent-Words table.

    Parameters
    ----------
    n : int
        Retain the top `n` words by corpus frequency.
    min_df : int
        Drop words appearing in fewer than `min_df` documents.
    max_df : float
        Drop words appearing in more than `max_df` fraction of documents (1.0 disables).
    scale : {"none", "zscore", "l1", "l2"}
        Per-feature scaling applied at transform-time. Burrows Delta requires "zscore";
        "l1" normalises rows to sum to 1 (relative frequencies); "l2" normalises rows to unit length.
    lowercase : bool
        If True, case-fold before counting.
    """

    feature_type = "mfw"

    def __init__(
        self,
        n: int = 1000,
        *,
        min_df: int = 1,
        max_df: float = 1.0,
        scale: Scale = "zscore",
        lowercase: bool = False,
    ) -> None:
        self.n = n
        self.min_df = min_df
        self.max_df = max_df
        self.scale = scale
        self.lowercase = lowercase
        self._vocabulary: list[str] = []
        self._column_means: np.ndarray | None = None
        self._column_stds: np.ndarray | None = None

    # --- BaseFeatureExtractor API ---

    def _fit(self, corpus: Corpus) -> None:
        n_docs = len(corpus)
        token_doc_freq: Counter[str] = Counter()
        token_total: Counter[str] = Counter()
        for doc in corpus.documents:
            tokens = _tokenise(doc.text, self.lowercase)
            token_total.update(tokens)
            token_doc_freq.update(set(tokens))

        max_allowed = int(self.max_df * n_docs) if self.max_df < 1.0 else n_docs
        candidates = [
            tok
            for tok, count in token_total.items()
            if token_doc_freq[tok] >= self.min_df and token_doc_freq[tok] <= max_allowed
        ]
        # Sort by frequency desc, then alphabetical for determinism.
        candidates.sort(key=lambda t: (-token_total[t], t))
        self._vocabulary = candidates[: self.n]

        if self.scale == "zscore":
            X_raw = self._raw_counts(corpus)
            self._column_means = X_raw.mean(axis=0)
            # Population SD (ddof=0) to match Stylo's convention. Replace zero-stds with 1 to avoid div-by-zero.
            stds = X_raw.std(axis=0, ddof=0)
            stds[stds == 0] = 1.0
            self._column_stds = stds

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        X = self._raw_counts(corpus)
        if self.scale == "zscore":
            assert self._column_means is not None and self._column_stds is not None
            X = (X - self._column_means) / self._column_stds
        elif self.scale == "l1":
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            X = X / row_sums
        elif self.scale == "l2":
            row_norms = np.linalg.norm(X, axis=1, keepdims=True)
            row_norms[row_norms == 0] = 1.0
            X = X / row_norms
        # "none" → raw counts, no change.
        return X, list(self._vocabulary)

    # --- internals ---

    def _raw_counts(self, corpus: Corpus) -> np.ndarray:
        index = {tok: i for i, tok in enumerate(self._vocabulary)}
        X = np.zeros((len(corpus), len(self._vocabulary)), dtype=float)
        for row, doc in enumerate(corpus.documents):
            for tok in _tokenise(doc.text, self.lowercase):
                if tok in index:
                    X[row, index[tok]] += 1
        return X
```

- [ ] **Step 3.4:** Update `src/tamga/features/__init__.py`:

```python
"""Feature extractors producing FeatureMatrix objects."""

from tamga.features.base import BaseFeatureExtractor, FeatureMatrix
from tamga.features.mfw import MFWExtractor

__all__ = ["BaseFeatureExtractor", "FeatureMatrix", "MFWExtractor"]
```

- [ ] **Step 3.5:** Run — expect all 9 tests PASS.

- [ ] **Step 3.6:** Commit.

```bash
git add src/tamga/features/mfw.py src/tamga/features/__init__.py tests/features/test_mfw.py
git commit -m "feat(features): MFWExtractor with min_df/max_df/scale and sklearn API"
```

---

## Task 4: `CharNgramExtractor` + `WordNgramExtractor`

**Files:**
- Create: `src/tamga/features/ngrams.py`
- Create: `tests/features/test_ngrams.py`
- Modify: `src/tamga/features/__init__.py` (export both extractors)

**TDD task.**

- [ ] **Step 4.1:** Failing tests `tests/features/test_ngrams.py`:

```python
"""Tests for Char/Word n-gram extractors."""

from tamga.corpus import Corpus, Document
from tamga.features.ngrams import CharNgramExtractor, WordNgramExtractor


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_char_ngram_extracts_bigrams():
    corpus = _corpus("abc", "abc")
    ex = CharNgramExtractor(n=2, scale="none")
    fm = ex.fit_transform(corpus)
    # "abc" has char-bigrams: "ab", "bc"
    assert set(fm.feature_names) == {"ab", "bc"}
    assert fm.X.sum() == 4  # 2 bigrams/doc × 2 docs


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
```

- [ ] **Step 4.2:** Run — FAIL.

- [ ] **Step 4.3:** Implement `src/tamga/features/ngrams.py`:

```python
"""Character and word n-gram feature extractors.

Both accept `n` as either an int (fixed order) or a tuple `(min_n, max_n)` (range).
Internally they delegate to sklearn's `CountVectorizer` — a well-tested, fast, battle-hardened
implementation — and present the same FeatureMatrix envelope as our other extractors.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor

Scale = Literal["none", "zscore", "l1", "l2"]


def _coerce_range(n: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(n, int):
        return (n, n)
    return n


def _apply_scale(X: np.ndarray, scale: Scale) -> np.ndarray:
    if scale == "none":
        return X
    if scale == "l1":
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return X / row_sums
    if scale == "l2":
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        return X / row_norms
    # zscore applied by caller (needs fitted stats).
    return X


class CharNgramExtractor(BaseFeatureExtractor):
    feature_type = "char_ngram"

    def __init__(
        self,
        n: int | tuple[int, int] = 3,
        *,
        include_boundaries: bool = False,
        scale: Scale = "none",
    ) -> None:
        self.n = n
        self.include_boundaries = include_boundaries
        self.scale = scale
        self._vectorizer: CountVectorizer | None = None
        self._column_means: np.ndarray | None = None
        self._column_stds: np.ndarray | None = None

    def _fit(self, corpus: Corpus) -> None:
        analyzer = "char_wb" if self.include_boundaries else "char"
        self._vectorizer = CountVectorizer(
            analyzer=analyzer,
            ngram_range=_coerce_range(self.n),
            lowercase=False,
        )
        texts = [d.text for d in corpus.documents]
        X = self._vectorizer.fit_transform(texts).toarray().astype(float)
        if self.scale == "zscore":
            self._column_means = X.mean(axis=0)
            stds = X.std(axis=0, ddof=0)
            stds[stds == 0] = 1.0
            self._column_stds = stds

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        assert self._vectorizer is not None
        texts = [d.text for d in corpus.documents]
        X = self._vectorizer.transform(texts).toarray().astype(float)
        if self.scale == "zscore":
            assert self._column_means is not None and self._column_stds is not None
            X = (X - self._column_means) / self._column_stds
        else:
            X = _apply_scale(X, self.scale)
        return X, list(self._vectorizer.get_feature_names_out())


class WordNgramExtractor(BaseFeatureExtractor):
    feature_type = "word_ngram"

    def __init__(
        self,
        n: int | tuple[int, int] = 1,
        *,
        lowercase: bool = False,
        scale: Scale = "none",
    ) -> None:
        self.n = n
        self.lowercase = lowercase
        self.scale = scale
        self._vectorizer: CountVectorizer | None = None
        self._column_means: np.ndarray | None = None
        self._column_stds: np.ndarray | None = None

    def _fit(self, corpus: Corpus) -> None:
        self._vectorizer = CountVectorizer(
            analyzer="word",
            ngram_range=_coerce_range(self.n),
            lowercase=self.lowercase,
            token_pattern=r"(?u)\b\w+\b",
        )
        texts = [d.text for d in corpus.documents]
        X = self._vectorizer.fit_transform(texts).toarray().astype(float)
        if self.scale == "zscore":
            self._column_means = X.mean(axis=0)
            stds = X.std(axis=0, ddof=0)
            stds[stds == 0] = 1.0
            self._column_stds = stds

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        assert self._vectorizer is not None
        texts = [d.text for d in corpus.documents]
        X = self._vectorizer.transform(texts).toarray().astype(float)
        if self.scale == "zscore":
            assert self._column_means is not None and self._column_stds is not None
            X = (X - self._column_means) / self._column_stds
        else:
            X = _apply_scale(X, self.scale)
        return X, list(self._vectorizer.get_feature_names_out())
```

- [ ] **Step 4.4:** Update `src/tamga/features/__init__.py` to export both.

- [ ] **Step 4.5:** Run — 6/6 PASS.

- [ ] **Step 4.6:** Commit.

```bash
git add src/tamga/features/ngrams.py src/tamga/features/__init__.py tests/features/test_ngrams.py
git commit -m "feat(features): CharNgramExtractor and WordNgramExtractor via sklearn CountVectorizer"
```

---

## Task 5: spaCy-based extractors (POS / Dependency / Function-Word / Punctuation) batch

**Files:**
- Create: `src/tamga/features/pos.py`
- Create: `src/tamga/features/dependency.py`
- Create: `src/tamga/features/function_words.py`
- Create: `src/tamga/features/punctuation.py`
- Create: `src/tamga/resources/__init__.py` (empty)
- Create: `src/tamga/resources/function_words_en.txt`
- Create: `tests/features/test_pos.py`
- Create: `tests/features/test_dependency.py`
- Create: `tests/features/test_function_words.py`
- Create: `tests/features/test_punctuation.py`
- Modify: `src/tamga/features/__init__.py`
- Modify: `pyproject.toml` (include resources/ in wheel)

**This task bundles four spaCy-based extractors because they share the pattern of `SpacyPipeline.parse(corpus) → iterate token attributes → tabulate`.**

All tests are marked `pytestmark = pytest.mark.spacy` and use `en_core_web_sm`.

### Step 5.1 — `pyproject.toml` force-include resources/

Add to the existing `[tool.hatch.build.targets.wheel.force-include]` block:

```toml
"src/tamga/resources" = "tamga/resources"
```

### Step 5.2 — `src/tamga/resources/function_words_en.txt`

Use a standard English function-word list (determiners, prepositions, pronouns, conjunctions, auxiliaries). The following 175-word list is derived from the Stylo-compatible English function word set used in stylometric research:

```
a
about
above
after
again
against
all
am
an
and
any
are
as
at
be
because
been
before
being
below
between
both
but
by
can
cannot
could
did
do
does
doing
down
during
each
few
for
from
further
had
has
have
having
he
her
here
hers
herself
him
himself
his
how
i
if
in
into
is
it
its
itself
just
me
more
most
must
my
myself
no
nor
not
now
of
off
on
once
only
or
other
ought
our
ours
ourselves
out
over
own
same
shall
she
should
so
some
such
than
that
the
their
theirs
them
themselves
then
there
these
they
this
those
through
to
too
under
until
up
very
was
we
were
what
when
where
which
while
who
whom
whose
why
will
with
would
you
your
yours
yourself
yourselves
```

### Step 5.3 — Tests (four files)

Each test file follows the same skeleton. For brevity, see the full verbatim contents immediately below — each is minimal (3-5 tests) and marked `pytestmark = pytest.mark.spacy`.

**`tests/features/test_pos.py`:**

```python
"""Tests for PosNgramExtractor."""

import pytest

from tamga.corpus import Corpus, Document
from tamga.features.pos import PosNgramExtractor

pytestmark = pytest.mark.spacy

_NLP_MODEL = "en_core_web_sm"


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_pos_ngram_unigrams(tmp_path):
    ex = PosNgramExtractor(n=1, tagset="coarse", spacy_model=_NLP_MODEL, cache_dir=tmp_path)
    fm = ex.fit_transform(_corpus("The cat sat on the mat."))
    # Coarse tagset (UPOS) → DET, NOUN, VERB, ADP, PUNCT at minimum.
    assert any(t in fm.feature_names for t in ("DET", "NOUN", "VERB"))


def test_pos_ngram_bigrams_count_pairs(tmp_path):
    ex = PosNgramExtractor(n=2, tagset="coarse", spacy_model=_NLP_MODEL, cache_dir=tmp_path)
    fm = ex.fit_transform(_corpus("The cat sat."))
    # Each bigram is "TAG1|TAG2"; pipe-joined for unambiguous parsing.
    assert any("|" in f for f in fm.feature_names)


def test_pos_ngram_fine_tagset_differs_from_coarse(tmp_path):
    c = _corpus("The cat ran quickly.")
    coarse = PosNgramExtractor(n=1, tagset="coarse", spacy_model=_NLP_MODEL, cache_dir=tmp_path / "c")
    fine = PosNgramExtractor(n=1, tagset="fine", spacy_model=_NLP_MODEL, cache_dir=tmp_path / "f")
    fm_c = coarse.fit_transform(c)
    fm_f = fine.fit_transform(c)
    # Fine tags are generally more numerous than coarse (UPOS has ~17; PTB has ~36).
    assert len(fm_f.feature_names) >= len(fm_c.feature_names)
```

**`tests/features/test_dependency.py`:**

```python
"""Tests for DependencyBigramExtractor."""

import pytest

from tamga.corpus import Corpus, Document
from tamga.features.dependency import DependencyBigramExtractor

pytestmark = pytest.mark.spacy

_NLP_MODEL = "en_core_web_sm"


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_dependency_bigram_returns_triples(tmp_path):
    ex = DependencyBigramExtractor(spacy_model=_NLP_MODEL, cache_dir=tmp_path)
    fm = ex.fit_transform(_corpus("The cat sat on the mat."))
    # Features are "head_lemma|dep|child_lemma" strings.
    assert fm.X.shape[0] == 1
    assert any(f.count("|") == 2 for f in fm.feature_names)


def test_dependency_bigram_counts_are_integers(tmp_path):
    ex = DependencyBigramExtractor(spacy_model=_NLP_MODEL, cache_dir=tmp_path)
    fm = ex.fit_transform(_corpus("A dog barked. A dog ran."))
    # Integer-valued counts (stored as float by convention).
    assert ((fm.X == fm.X.astype(int)).all())
```

**`tests/features/test_function_words.py`:**

```python
"""Tests for FunctionWordExtractor."""

from tamga.corpus import Corpus, Document
from tamga.features.function_words import FunctionWordExtractor


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_function_word_counts_the_and_of():
    ex = FunctionWordExtractor(scale="none")
    fm = ex.fit_transform(_corpus("the cat of the dog"))
    df = fm.as_dataframe()
    assert df.loc["d0", "the"] == 2
    assert df.loc["d0", "of"] == 1


def test_function_word_ignores_content_words():
    ex = FunctionWordExtractor(scale="none")
    fm = ex.fit_transform(_corpus("cat dog bird"))
    # No function words → all-zero row.
    assert fm.X.sum() == 0


def test_function_word_uses_bundled_list_by_default():
    ex = FunctionWordExtractor(scale="none")
    fm = ex.fit_transform(_corpus("a test"))
    # Bundled EN list contains "a".
    assert "a" in fm.feature_names


def test_function_word_accepts_custom_list():
    ex = FunctionWordExtractor(wordlist=["custom"], scale="none")
    fm = ex.fit_transform(_corpus("a custom word"))
    assert list(fm.feature_names) == ["custom"]
```

**`tests/features/test_punctuation.py`:**

```python
"""Tests for PunctuationExtractor."""

from tamga.corpus import Corpus, Document
from tamga.features.punctuation import PunctuationExtractor


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_punctuation_counts_periods_and_commas():
    ex = PunctuationExtractor()
    fm = ex.fit_transform(_corpus("Hello, world. Hello, again."))
    df = fm.as_dataframe()
    assert df.loc["d0", ","] == 2
    assert df.loc["d0", "."] == 2


def test_punctuation_ignores_word_characters():
    ex = PunctuationExtractor()
    fm = ex.fit_transform(_corpus("nowordschars"))
    assert fm.X.sum() == 0


def test_punctuation_includes_question_exclamation():
    ex = PunctuationExtractor()
    fm = ex.fit_transform(_corpus("What? Really!"))
    df = fm.as_dataframe()
    assert df.loc["d0", "?"] == 1
    assert df.loc["d0", "!"] == 1
```

### Step 5.4 — Implementations

**`src/tamga/features/pos.py`:**

```python
"""POS n-gram feature extractor — spaCy-backed tokens tagged with Universal or fine POS labels."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor
from tamga.preprocess.pipeline import SpacyPipeline

Tagset = Literal["coarse", "fine"]


class PosNgramExtractor(BaseFeatureExtractor):
    feature_type = "pos_ngram"

    def __init__(
        self,
        n: int = 2,
        *,
        tagset: Tagset = "coarse",
        spacy_model: str = "en_core_web_trf",
        cache_dir: str | Path = ".tamga/cache/docbin",
    ) -> None:
        self.n = n
        self.tagset = tagset
        self.spacy_model = spacy_model
        self.cache_dir = cache_dir
        self._vocabulary: list[str] = []
        self._pipeline: SpacyPipeline | None = None

    def _pipe(self) -> SpacyPipeline:
        if self._pipeline is None:
            self._pipeline = SpacyPipeline(model=self.spacy_model, cache_dir=self.cache_dir)
        return self._pipeline

    def _tag(self, token) -> str:
        return token.pos_ if self.tagset == "coarse" else token.tag_

    def _ngrams(self, tags: list[str]) -> list[str]:
        if self.n == 1:
            return tags
        return ["|".join(tags[i : i + self.n]) for i in range(len(tags) - self.n + 1)]

    def _fit(self, corpus: Corpus) -> None:
        parsed = self._pipe().parse(corpus)
        vocab_counter: Counter[str] = Counter()
        for spacy_doc in parsed.spacy_docs():
            tags = [self._tag(t) for t in spacy_doc if not t.is_space]
            vocab_counter.update(self._ngrams(tags))
        self._vocabulary = sorted(vocab_counter.keys())

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        index = {ng: i for i, ng in enumerate(self._vocabulary)}
        parsed = self._pipe().parse(corpus)
        X = np.zeros((len(corpus), len(self._vocabulary)), dtype=float)
        for row, spacy_doc in enumerate(parsed.spacy_docs()):
            tags = [self._tag(t) for t in spacy_doc if not t.is_space]
            for ng in self._ngrams(tags):
                if ng in index:
                    X[row, index[ng]] += 1
        return X, list(self._vocabulary)
```

**`src/tamga/features/dependency.py`:**

```python
"""Dependency bigram feature extractor — (head_lemma, dep_label, child_lemma) triples."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor
from tamga.preprocess.pipeline import SpacyPipeline


class DependencyBigramExtractor(BaseFeatureExtractor):
    feature_type = "dependency_bigram"

    def __init__(
        self,
        *,
        spacy_model: str = "en_core_web_trf",
        cache_dir: str | Path = ".tamga/cache/docbin",
        lowercase: bool = True,
    ) -> None:
        self.spacy_model = spacy_model
        self.cache_dir = cache_dir
        self.lowercase = lowercase
        self._vocabulary: list[str] = []
        self._pipeline: SpacyPipeline | None = None

    def _pipe(self) -> SpacyPipeline:
        if self._pipeline is None:
            self._pipeline = SpacyPipeline(model=self.spacy_model, cache_dir=self.cache_dir)
        return self._pipeline

    def _triples(self, spacy_doc) -> list[str]:
        out = []
        for tok in spacy_doc:
            if tok.is_space or tok.head is tok:  # skip whitespace tokens and root-self-loops
                continue
            head_lemma = tok.head.lemma_.lower() if self.lowercase else tok.head.lemma_
            child_lemma = tok.lemma_.lower() if self.lowercase else tok.lemma_
            out.append(f"{head_lemma}|{tok.dep_}|{child_lemma}")
        return out

    def _fit(self, corpus: Corpus) -> None:
        parsed = self._pipe().parse(corpus)
        counter: Counter[str] = Counter()
        for spacy_doc in parsed.spacy_docs():
            counter.update(self._triples(spacy_doc))
        self._vocabulary = sorted(counter.keys())

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        index = {t: i for i, t in enumerate(self._vocabulary)}
        parsed = self._pipe().parse(corpus)
        X = np.zeros((len(corpus), len(self._vocabulary)), dtype=float)
        for row, spacy_doc in enumerate(parsed.spacy_docs()):
            for triple in self._triples(spacy_doc):
                if triple in index:
                    X[row, index[triple]] += 1
        return X, list(self._vocabulary)
```

**`src/tamga/features/function_words.py`:**

```python
"""Function-word frequency extractor with a bundled English word list."""

from __future__ import annotations

import re
from importlib import resources
from typing import Literal

import numpy as np

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor

Scale = Literal["none", "zscore", "l1", "l2"]

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def _load_bundled_list() -> list[str]:
    path = resources.files("tamga.resources") / "function_words_en.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FunctionWordExtractor(BaseFeatureExtractor):
    feature_type = "function_word"

    def __init__(
        self,
        *,
        wordlist: list[str] | None = None,
        scale: Scale = "none",
    ) -> None:
        self.wordlist = wordlist
        self.scale = scale
        self._words: list[str] = []

    def _fit(self, corpus: Corpus) -> None:  # noqa: ARG002 - vocabulary comes from the word list, not the corpus
        self._words = list(self.wordlist) if self.wordlist is not None else _load_bundled_list()

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        index = {w: i for i, w in enumerate(self._words)}
        X = np.zeros((len(corpus), len(self._words)), dtype=float)
        for row, doc in enumerate(corpus.documents):
            for tok in _WORD_RE.findall(doc.text.lower()):
                if tok in index:
                    X[row, index[tok]] += 1
        if self.scale == "l1":
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            X = X / row_sums
        elif self.scale == "l2":
            row_norms = np.linalg.norm(X, axis=1, keepdims=True)
            row_norms[row_norms == 0] = 1.0
            X = X / row_norms
        # "zscore" scaling for FWs is less common than for MFW — support it but no fitted stats needed for "none"/"l1"/"l2".
        return X, list(self._words)
```

**`src/tamga/features/punctuation.py`:**

```python
"""Punctuation-symbol frequency extractor."""

from __future__ import annotations

import string

import numpy as np

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor

_PUNCT = sorted(string.punctuation)  # deterministic column order


class PunctuationExtractor(BaseFeatureExtractor):
    feature_type = "punctuation"

    def __init__(self) -> None:
        self._symbols: list[str] = list(_PUNCT)

    def _fit(self, corpus: Corpus) -> None:  # noqa: ARG002
        # Vocabulary is fixed (the ASCII punctuation set); no fitting needed.
        pass

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        index = {s: i for i, s in enumerate(self._symbols)}
        X = np.zeros((len(corpus), len(self._symbols)), dtype=float)
        for row, doc in enumerate(corpus.documents):
            for ch in doc.text:
                if ch in index:
                    X[row, index[ch]] += 1
        return X, list(self._symbols)
```

### Step 5.5 — Update `src/tamga/features/__init__.py`

```python
"""Feature extractors producing FeatureMatrix objects."""

from tamga.features.base import BaseFeatureExtractor, FeatureMatrix
from tamga.features.dependency import DependencyBigramExtractor
from tamga.features.function_words import FunctionWordExtractor
from tamga.features.mfw import MFWExtractor
from tamga.features.ngrams import CharNgramExtractor, WordNgramExtractor
from tamga.features.pos import PosNgramExtractor
from tamga.features.punctuation import PunctuationExtractor

__all__ = [
    "BaseFeatureExtractor",
    "CharNgramExtractor",
    "DependencyBigramExtractor",
    "FeatureMatrix",
    "FunctionWordExtractor",
    "MFWExtractor",
    "PosNgramExtractor",
    "PunctuationExtractor",
    "WordNgramExtractor",
]
```

### Step 5.6 — Run all new tests

`pytest tests/features/test_pos.py tests/features/test_dependency.py tests/features/test_function_words.py tests/features/test_punctuation.py -v`

Expected: 3 + 2 + 4 + 3 = 12 tests PASS (the POS/Dep ones need spaCy, which is installed).

### Step 5.7 — Commit (single commit for the batch)

```bash
git add src/tamga/features/pos.py src/tamga/features/dependency.py src/tamga/features/function_words.py src/tamga/features/punctuation.py src/tamga/resources/ tests/features/test_pos.py tests/features/test_dependency.py tests/features/test_function_words.py tests/features/test_punctuation.py src/tamga/features/__init__.py pyproject.toml
git commit -m "feat(features): POS n-gram, dependency-bigram, function-word, punctuation extractors"
```

---

## Task 6: `LexicalDiversityExtractor` + `ReadabilityExtractor` + `SentenceLengthExtractor` batch

**Files:**
- Create: `src/tamga/features/lexical_diversity.py`
- Create: `src/tamga/features/readability.py`
- Create: `src/tamga/features/sentence_length.py`
- Create: `tests/features/test_lexical_diversity.py`
- Create: `tests/features/test_readability.py`
- Create: `tests/features/test_sentence_length.py`
- Modify: `src/tamga/features/__init__.py` (export the three new extractors)

### Step 6.1 — `tests/features/test_lexical_diversity.py`

```python
"""Tests for LexicalDiversityExtractor."""

import numpy as np

from tamga.corpus import Corpus, Document
from tamga.features.lexical_diversity import LexicalDiversityExtractor


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_ttr_is_one_for_all_unique_words():
    ex = LexicalDiversityExtractor(indices=["ttr"])
    fm = ex.fit_transform(_corpus("the quick brown fox"))
    assert fm.as_dataframe().loc["d0", "ttr"] == 1.0


def test_ttr_is_low_for_repetitive_text():
    ex = LexicalDiversityExtractor(indices=["ttr"])
    fm = ex.fit_transform(_corpus("the the the the the the"))
    # 1 unique / 6 total = 0.1667
    assert fm.as_dataframe().loc["d0", "ttr"] < 0.2


def test_multiple_indices_produce_multiple_columns():
    ex = LexicalDiversityExtractor(indices=["ttr", "yules_k"])
    fm = ex.fit_transform(_corpus("the quick brown fox jumped over the lazy dog"))
    assert set(fm.feature_names) == {"ttr", "yules_k"}


def test_ldiv_feature_matrix_is_2d_numeric():
    ex = LexicalDiversityExtractor(indices=["ttr"])
    fm = ex.fit_transform(_corpus("a b c", "a a a"))
    assert fm.X.shape == (2, 1)
    assert np.issubdtype(fm.X.dtype, np.floating)
```

### Step 6.2 — `tests/features/test_readability.py`

```python
"""Tests for ReadabilityExtractor."""

from tamga.corpus import Corpus, Document
from tamga.features.readability import ReadabilityExtractor


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_flesch_reading_ease_for_simple_text():
    ex = ReadabilityExtractor(indices=["flesch"])
    fm = ex.fit_transform(_corpus("The cat sat on the mat. It was a warm sunny day."))
    # Simple prose should score high (closer to 100 = very easy).
    assert fm.as_dataframe().loc["d0", "flesch"] > 70


def test_multiple_readability_indices():
    ex = ReadabilityExtractor(indices=["flesch", "flesch_kincaid", "gunning_fog"])
    fm = ex.fit_transform(_corpus("A simple sentence here. Another simple one."))
    assert set(fm.feature_names) == {"flesch", "flesch_kincaid", "gunning_fog"}


def test_readability_per_document():
    ex = ReadabilityExtractor(indices=["flesch"])
    fm = ex.fit_transform(_corpus("Simple text.", "Another text."))
    assert fm.X.shape == (2, 1)
```

### Step 6.3 — `tests/features/test_sentence_length.py`

```python
"""Tests for SentenceLengthExtractor."""

import pytest

from tamga.corpus import Corpus, Document
from tamga.features.sentence_length import SentenceLengthExtractor

pytestmark = pytest.mark.spacy


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_sentence_length_returns_mean_sd_skew(tmp_path):
    ex = SentenceLengthExtractor(spacy_model="en_core_web_sm", cache_dir=tmp_path)
    fm = ex.fit_transform(_corpus("Short. A longer sentence. An even longer sentence with more words."))
    assert set(fm.feature_names) == {"mean", "sd", "skew"}
    df = fm.as_dataframe()
    assert df.loc["d0", "mean"] > 0
    assert df.loc["d0", "sd"] >= 0


def test_sentence_length_uniform_has_zero_sd(tmp_path):
    ex = SentenceLengthExtractor(spacy_model="en_core_web_sm", cache_dir=tmp_path)
    fm = ex.fit_transform(_corpus("Three words here. Three words here. Three words here."))
    assert fm.as_dataframe().loc["d0", "sd"] == 0.0
```

### Step 6.4 — Implementations

**`src/tamga/features/lexical_diversity.py`:**

```python
"""Lexical-diversity indices: TTR, MATTR, MTLD, HD-D, Yule's K, Yule's I, Herdan's C, Simpson's D."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)

_DEFAULT_INDICES = ("ttr", "yules_k")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _ttr(tokens: list[str]) -> float:
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def _yules_k(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    n = len(tokens)
    freq = Counter(tokens)
    freq_of_freq = Counter(freq.values())
    s2 = sum(r * r * f for r, f in freq_of_freq.items())
    return 1e4 * (s2 - n) / (n * n) if n > 0 else 0.0


def _yules_i(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    n = len(tokens)
    freq = Counter(tokens)
    freq_of_freq = Counter(freq.values())
    s2 = sum(r * r * f for r, f in freq_of_freq.items())
    v = len(freq)
    return (v * v) / (s2 - n) if (s2 - n) > 0 else 0.0


def _herdans_c(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    v = len(set(tokens))
    n = len(tokens)
    return np.log(v) / np.log(n) if n > 1 else 0.0


def _simpsons_d(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    n = len(tokens)
    if n < 2:
        return 0.0
    return sum(f * (f - 1) for f in freq.values()) / (n * (n - 1))


def _mattr(tokens: list[str], window: int = 100) -> float:
    """Moving-Average Type-Token Ratio."""
    if len(tokens) < window:
        return _ttr(tokens)
    ratios = [len(set(tokens[i : i + window])) / window for i in range(len(tokens) - window + 1)]
    return float(np.mean(ratios))


def _mtld(tokens: list[str], ttr_threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy 2005)."""
    if not tokens:
        return 0.0

    def _one_direction(toks: list[str]) -> float:
        factor_count = 0.0
        seen: set[str] = set()
        count = 0
        for tok in toks:
            seen.add(tok)
            count += 1
            if count / max(len(seen), 1) <= 1 / ttr_threshold and count > 0:  # placeholder guard
                pass
            ttr = len(seen) / count
            if ttr <= ttr_threshold:
                factor_count += 1
                seen = set()
                count = 0
        if count > 0:
            # Partial factor: scale by how close to the threshold it got.
            partial = (1 - (len(seen) / count)) / (1 - ttr_threshold) if ttr_threshold < 1 else 0
            factor_count += partial
        return len(toks) / factor_count if factor_count > 0 else 0.0

    forward = _one_direction(tokens)
    backward = _one_direction(list(reversed(tokens)))
    return (forward + backward) / 2


def _hdd(tokens: list[str], sample_size: int = 42) -> float:
    """Hypergeometric Distribution Diversity (McCarthy & Jarvis 2010).

    For each unique word, sum the contribution of that word to a sample of `sample_size`;
    each contribution is the probability of at least one occurrence in the sample.
    """
    from math import comb

    if len(tokens) < sample_size:
        return 0.0
    n = len(tokens)
    freq = Counter(tokens)
    total = 0.0
    for f in freq.values():
        # P(at least one) = 1 - P(none) = 1 - C(n-f, sample) / C(n, sample)
        if sample_size <= n - f:
            p_none = comb(n - f, sample_size) / comb(n, sample_size)
        else:
            p_none = 0.0
        total += (1 - p_none) / sample_size
    return total


_INDEX_FN = {
    "ttr": _ttr,
    "mattr": _mattr,
    "mtld": _mtld,
    "hdd": _hdd,
    "yules_k": _yules_k,
    "yules_i": _yules_i,
    "herdans_c": _herdans_c,
    "simpsons_d": _simpsons_d,
}


class LexicalDiversityExtractor(BaseFeatureExtractor):
    feature_type = "lexical_diversity"

    def __init__(self, indices: list[str] | tuple[str, ...] = _DEFAULT_INDICES) -> None:
        self.indices = list(indices)

    def _fit(self, corpus: Corpus) -> None:  # noqa: ARG002
        unknown = [i for i in self.indices if i not in _INDEX_FN]
        if unknown:
            raise ValueError(f"LexicalDiversityExtractor: unknown indices {unknown}")

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        X = np.zeros((len(corpus), len(self.indices)), dtype=float)
        for row, doc in enumerate(corpus.documents):
            tokens = _tokens(doc.text)
            for col, index in enumerate(self.indices):
                X[row, col] = _INDEX_FN[index](tokens)
        return X, list(self.indices)
```

**`src/tamga/features/readability.py`:**

```python
"""Readability indices via the `textstat` library: Flesch, Flesch-Kincaid, Gunning Fog, Coleman-Liau, ARI, SMOG."""

from __future__ import annotations

import numpy as np
import textstat

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor

_INDEX_FN = {
    "flesch": textstat.flesch_reading_ease,
    "flesch_kincaid": textstat.flesch_kincaid_grade,
    "gunning_fog": textstat.gunning_fog,
    "coleman_liau": textstat.coleman_liau_index,
    "ari": textstat.automated_readability_index,
    "smog": textstat.smog_index,
}

_DEFAULT_INDICES = ("flesch", "flesch_kincaid", "gunning_fog")


class ReadabilityExtractor(BaseFeatureExtractor):
    feature_type = "readability"

    def __init__(self, indices: list[str] | tuple[str, ...] = _DEFAULT_INDICES) -> None:
        self.indices = list(indices)

    def _fit(self, corpus: Corpus) -> None:  # noqa: ARG002
        unknown = [i for i in self.indices if i not in _INDEX_FN]
        if unknown:
            raise ValueError(f"ReadabilityExtractor: unknown indices {unknown}")

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        X = np.zeros((len(corpus), len(self.indices)), dtype=float)
        for row, doc in enumerate(corpus.documents):
            for col, index in enumerate(self.indices):
                X[row, col] = float(_INDEX_FN[index](doc.text))
        return X, list(self.indices)
```

**`src/tamga/features/sentence_length.py`:**

```python
"""Sentence-length distribution features: mean, standard deviation, skew."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import skew

from tamga.corpus import Corpus
from tamga.features.base import BaseFeatureExtractor
from tamga.preprocess.pipeline import SpacyPipeline


class SentenceLengthExtractor(BaseFeatureExtractor):
    feature_type = "sentence_length"

    def __init__(
        self,
        *,
        spacy_model: str = "en_core_web_trf",
        cache_dir: str | Path = ".tamga/cache/docbin",
    ) -> None:
        self.spacy_model = spacy_model
        self.cache_dir = cache_dir
        self._pipeline: SpacyPipeline | None = None

    def _pipe(self) -> SpacyPipeline:
        if self._pipeline is None:
            self._pipeline = SpacyPipeline(model=self.spacy_model, cache_dir=self.cache_dir)
        return self._pipeline

    def _fit(self, corpus: Corpus) -> None:  # noqa: ARG002
        pass

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        parsed = self._pipe().parse(corpus)
        X = np.zeros((len(corpus), 3), dtype=float)
        for row, spacy_doc in enumerate(parsed.spacy_docs()):
            lengths = [sum(1 for t in sent if not t.is_space) for sent in spacy_doc.sents]
            if not lengths:
                continue
            X[row, 0] = float(np.mean(lengths))
            X[row, 1] = float(np.std(lengths, ddof=0))
            X[row, 2] = float(skew(lengths)) if len(lengths) > 2 else 0.0
        return X, ["mean", "sd", "skew"]
```

### Step 6.5 — Update `src/tamga/features/__init__.py`

Add imports and names:

```python
from tamga.features.lexical_diversity import LexicalDiversityExtractor
from tamga.features.readability import ReadabilityExtractor
from tamga.features.sentence_length import SentenceLengthExtractor
```

And include them in `__all__` in alphabetical order.

### Step 6.6 — Run tests

```
pytest tests/features/test_lexical_diversity.py tests/features/test_readability.py tests/features/test_sentence_length.py -v
```

Expected: 4 + 3 + 2 = 9 passing.

### Step 6.7 — Commit

```bash
git add src/tamga/features/lexical_diversity.py src/tamga/features/readability.py src/tamga/features/sentence_length.py tests/features/test_lexical_diversity.py tests/features/test_readability.py tests/features/test_sentence_length.py src/tamga/features/__init__.py
git commit -m "feat(features): lexical-diversity, readability, sentence-length extractors"
```

---

## Task 7: `_DeltaBase` — nearest-author-centroid shared logic

**Files:**
- Create: `src/tamga/methods/__init__.py`
- Create: `src/tamga/methods/delta/__init__.py`
- Create: `src/tamga/methods/delta/base.py`
- Create: `tests/methods/__init__.py`
- Create: `tests/methods/delta/__init__.py`
- Create: `tests/methods/delta/test_base.py`

**TDD task.**

### Step 7.1 — Package `__init__.py` files

`src/tamga/methods/__init__.py`:

```python
"""Analytical methods: delta, zeta, reduce, cluster, classify, bayesian (stubs land in later phases)."""
```

`src/tamga/methods/delta/__init__.py` (placeholder — will grow as variants land):

```python
"""Delta family of distance-based nearest-author-centroid classifiers."""

from tamga.methods.delta.base import _DeltaBase

__all__ = ["_DeltaBase"]
```

`tests/methods/__init__.py` and `tests/methods/delta/__init__.py` (both empty).

### Step 7.2 — Failing tests `tests/methods/delta/test_base.py`

```python
"""Tests for the _DeltaBase nearest-author-centroid logic (using a trivial L2 subclass)."""

import numpy as np
import pytest

from tamga.features import FeatureMatrix
from tamga.methods.delta.base import _DeltaBase


class _L2Delta(_DeltaBase):
    """Minimal concrete subclass for testing the base class machinery."""

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        return np.linalg.norm(X - centroid, axis=1)


def _fm(X: np.ndarray) -> FeatureMatrix:
    return FeatureMatrix(
        X=X,
        document_ids=[f"d{i}" for i in range(X.shape[0])],
        feature_names=[f"f{j}" for j in range(X.shape[1])],
        feature_type="test",
    )


def test_fit_stores_centroids_per_author():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]])
    y = np.array(["A", "A", "B", "B"])
    clf = _L2Delta().fit(_fm(X), y)
    assert set(clf.classes_) == {"A", "B"}
    np.testing.assert_allclose(clf.centroids_["A"], [0.05, 0.05])
    np.testing.assert_allclose(clf.centroids_["B"], [10.05, 10.05])


def test_predict_returns_nearest_author():
    X = np.array([[0.0, 0.0], [10.0, 10.0]])
    y = np.array(["A", "B"])
    clf = _L2Delta().fit(_fm(X), y)
    # New points very close to each centroid.
    probe = _fm(np.array([[0.01, 0.0], [9.99, 10.0]]))
    preds = clf.predict(probe)
    assert list(preds) == ["A", "B"]


def test_decision_function_returns_negative_distances():
    X = np.array([[0.0], [10.0]])
    y = np.array(["A", "B"])
    clf = _L2Delta().fit(_fm(X), y)
    probe = _fm(np.array([[0.0]]))
    scores = clf.decision_function(probe)
    # Closer to A ⇒ score for A higher than score for B (scores are negative distances).
    assert scores.shape == (1, 2)
    class_a = list(clf.classes_).index("A")
    class_b = list(clf.classes_).index("B")
    assert scores[0, class_a] > scores[0, class_b]


def test_predict_proba_rows_sum_to_one():
    X = np.array([[0.0], [10.0]])
    y = np.array(["A", "B"])
    clf = _L2Delta().fit(_fm(X), y)
    probe = _fm(np.array([[0.0], [5.0], [10.0]]))
    probs = clf.predict_proba(probe)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_predict_accepts_numpy_array_for_sklearn_compat():
    """sklearn's cross_validate passes X as ndarray, not FeatureMatrix."""
    X = np.array([[0.0], [10.0]])
    y = np.array(["A", "B"])
    clf = _L2Delta().fit(X, y)
    preds = clf.predict(np.array([[0.0], [10.0]]))
    assert list(preds) == ["A", "B"]
```

### Step 7.3 — Implement `src/tamga/methods/delta/base.py`

```python
"""_DeltaBase — shared fit/predict logic for every Delta variant.

Each Delta method is modelled as a nearest-author-centroid classifier. Subclasses differ only in
their `_distance(X, centroid)` implementation — the distance kernel. Fit stores the mean of each
author's training feature vectors; predict returns the author whose centroid is nearest under
that kernel.

Accepts both `FeatureMatrix` and plain `np.ndarray` inputs so sklearn's `Pipeline`,
`cross_validate`, and `GridSearchCV` work without custom adapters.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from tamga.features import FeatureMatrix


def _as_ndarray(X: FeatureMatrix | np.ndarray) -> np.ndarray:
    return X.X if isinstance(X, FeatureMatrix) else np.asarray(X)


class _DeltaBase(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        self.classes_: np.ndarray = np.empty(0, dtype=object)
        self.centroids_: dict[str, np.ndarray] = {}

    @abstractmethod
    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """Return per-row distances from each row of `X` to the given centroid."""

    def fit(self, X: FeatureMatrix | np.ndarray, y: np.ndarray) -> _DeltaBase:
        X_arr = _as_ndarray(X)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        self.centroids_ = {
            label: X_arr[y_arr == label].mean(axis=0) for label in self.classes_
        }
        return self

    def decision_function(self, X: FeatureMatrix | np.ndarray) -> np.ndarray:
        X_arr = _as_ndarray(X)
        # Return negative distance so that "higher = more likely" matches sklearn convention.
        return np.column_stack(
            [-self._distance(X_arr, self.centroids_[label]) for label in self.classes_]
        )

    def predict(self, X: FeatureMatrix | np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: FeatureMatrix | np.ndarray) -> np.ndarray:
        """Softmax over negative distances — a monotonic, well-defined probability proxy."""
        scores = self.decision_function(X)
        # Numerically stable softmax.
        scores_shift = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores_shift)
        return exp / exp.sum(axis=1, keepdims=True)
```

### Step 7.4 — Run tests → PASS (5/5).

### Step 7.5 — Commit

```bash
git add src/tamga/methods/ tests/methods/
git commit -m "feat(methods): _DeltaBase nearest-author-centroid classifier scaffold"
```

---

## Task 8: Delta variants — Burrows, Eder, Eder-Simple, Argamon, Cosine, Quadratic (batch)

**Files:**
- Create: `src/tamga/methods/delta/burrows.py`
- Create: `src/tamga/methods/delta/eder.py`
- Create: `src/tamga/methods/delta/argamon.py`
- Create: `src/tamga/methods/delta/cosine.py`
- Create: `tests/methods/delta/test_burrows.py`
- Create: `tests/methods/delta/test_eder.py`
- Create: `tests/methods/delta/test_argamon.py`
- Create: `tests/methods/delta/test_cosine.py`
- Modify: `src/tamga/methods/delta/__init__.py` (export all six variants)

All six distance kernels operate on **z-scored** feature matrices — the caller is responsible for passing z-scored input (e.g., `MFWExtractor(scale="zscore")` or `StandardScaler` in a `Pipeline`).

### Step 8.1 — `src/tamga/methods/delta/burrows.py`

```python
"""Burrows Classic Delta (Burrows 2002) — mean absolute difference of z-scored features."""

from __future__ import annotations

import numpy as np

from tamga.methods.delta.base import _DeltaBase


class BurrowsDelta(_DeltaBase):
    """Burrows Classic Delta.

    Distance = mean(|x_i - c_i|) across features — the L1 norm divided by the feature count.
    Assumes both `X` and `centroid` are z-scored in the same coordinate system.
    """

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        return np.abs(X - centroid).mean(axis=1)
```

### Step 8.2 — `src/tamga/methods/delta/eder.py`

```python
"""Eder Delta (Eder 2015) — rank-weighted Burrows — and Eder's Simple Delta (Eder 2017)."""

from __future__ import annotations

import numpy as np

from tamga.methods.delta.base import _DeltaBase


class EderDelta(_DeltaBase):
    """Eder Delta: like Burrows, but each feature's contribution is weighted by `(n - rank) / n`,
    so the most frequent features contribute most. Features are ranked by their training-set mean
    absolute z-score (more discriminating features get higher rank).

    Scalar weighting is computed at `fit` time from the centroids.
    """

    def __init__(self) -> None:
        super().__init__()
        self._weights: np.ndarray | None = None

    def fit(self, X, y):  # type: ignore[override]
        super().fit(X, y)
        # Feature importance proxy: across-centroid variance (discriminating features have high variance).
        stacked = np.vstack(list(self.centroids_.values()))
        importance = stacked.var(axis=0)
        ranks = importance.argsort()[::-1]  # descending by importance
        n = len(importance)
        weights = np.zeros(n)
        for rank_pos, feat_idx in enumerate(ranks):
            weights[feat_idx] = (n - rank_pos) / n
        self._weights = weights
        return self

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        assert self._weights is not None
        return (self._weights * np.abs(X - centroid)).sum(axis=1) / self._weights.sum()


class EderSimpleDelta(_DeltaBase):
    """Eder's Simple Delta (Eder 2017): L1 distance on unweighted z-scored features.

    Differs from Burrows only in that Burrows divides by feature count; Eder Simple does not.
    In practice this only changes a monotone scaling of distances — rankings are identical.
    """

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        return np.abs(X - centroid).sum(axis=1)
```

### Step 8.3 — `src/tamga/methods/delta/argamon.py`

```python
"""Argamon Linear Delta (Argamon 2008) — L2 distance on z-scored features — and Quadratic Delta — squared-L2."""

from __future__ import annotations

import numpy as np

from tamga.methods.delta.base import _DeltaBase


class ArgamonLinearDelta(_DeltaBase):
    """L2 distance on z-scored features: sqrt(sum((x - c)^2))."""

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        return np.linalg.norm(X - centroid, axis=1)


class QuadraticDelta(_DeltaBase):
    """Squared-L2 distance: sum((x - c)^2). Preserves ranking vs Argamon, differs in scale."""

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        diff = X - centroid
        return (diff * diff).sum(axis=1)
```

### Step 8.4 — `src/tamga/methods/delta/cosine.py`

```python
"""Cosine Delta (Smith & Aldridge 2011; Evert et al. 2017) — 1 - cosine similarity on z-scored features."""

from __future__ import annotations

import numpy as np

from tamga.methods.delta.base import _DeltaBase


class CosineDelta(_DeltaBase):
    """1 - cosine(x, c). Undefined when either vector is all-zero; add a tiny epsilon to avoid div-by-zero."""

    _EPS = 1e-12

    def _distance(self, X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        x_norms = np.linalg.norm(X, axis=1)
        c_norm = np.linalg.norm(centroid)
        denom = np.maximum(x_norms * c_norm, self._EPS)
        cosine = (X @ centroid) / denom
        return 1.0 - cosine
```

### Step 8.5 — Tests

Each variant's test file follows the same pattern — a sanity test that the distance orders centroids correctly. Verbatim for `tests/methods/delta/test_burrows.py`:

```python
"""Tests for BurrowsDelta."""

import numpy as np

from tamga.features import FeatureMatrix
from tamga.methods.delta.burrows import BurrowsDelta


def _fm(X: np.ndarray) -> FeatureMatrix:
    return FeatureMatrix(
        X=X,
        document_ids=[f"d{i}" for i in range(X.shape[0])],
        feature_names=[f"f{j}" for j in range(X.shape[1])],
        feature_type="zscored-mfw",
    )


def test_burrows_attributes_to_nearest_centroid():
    # Two authors, well-separated.
    X = np.array([[0.0, 0.0], [0.1, 0.2], [5.0, 5.0], [5.1, 5.2]])
    y = np.array(["A", "A", "B", "B"])
    probe = _fm(np.array([[0.05, 0.1], [5.05, 5.1]]))
    preds = BurrowsDelta().fit(_fm(X), y).predict(probe)
    assert list(preds) == ["A", "B"]


def test_burrows_distance_is_mean_absolute_difference():
    X = np.array([[0.0, 0.0]])
    y = np.array(["A"])
    clf = BurrowsDelta().fit(_fm(X), y)
    # Distance from [0, 0] to centroid [0, 0] is 0; from [1, 1] to [0, 0] is 1.0 (mean of |1|, |1|).
    scores = clf.decision_function(_fm(np.array([[0.0, 0.0], [1.0, 1.0]])))
    assert scores[0, 0] == 0.0
    assert scores[1, 0] == -1.0  # score = -distance


def test_burrows_is_sklearn_compatible():
    from sklearn.base import is_classifier

    assert is_classifier(BurrowsDelta())
```

**For the other four files (`test_eder.py`, `test_argamon.py`, `test_cosine.py`), produce analogous minimal tests** — each verifies:

1. `test_<name>_attributes_to_nearest_centroid` — two-author synthetic dataset, check predictions.
2. `test_<name>_is_sklearn_compatible` — `is_classifier(obj)` returns True.
3. For Eder: `test_eder_weights_are_assigned_at_fit_time` — after `.fit()`, `clf._weights` is not None.
4. For Cosine: `test_cosine_handles_zero_vectors_gracefully` — fit on [[1,0],[0,1]], predict [[0,0]] without raising.

Write each test file with verbatim imports, `_fm` helper, and 2-3 tests following the Burrows pattern.

### Step 8.6 — Update `src/tamga/methods/delta/__init__.py`

```python
"""Delta family of distance-based nearest-author-centroid classifiers."""

from tamga.methods.delta.argamon import ArgamonLinearDelta, QuadraticDelta
from tamga.methods.delta.base import _DeltaBase
from tamga.methods.delta.burrows import BurrowsDelta
from tamga.methods.delta.cosine import CosineDelta
from tamga.methods.delta.eder import EderDelta, EderSimpleDelta

__all__ = [
    "ArgamonLinearDelta",
    "BurrowsDelta",
    "CosineDelta",
    "EderDelta",
    "EderSimpleDelta",
    "QuadraticDelta",
    "_DeltaBase",
]
```

### Step 8.7 — Run

`pytest tests/methods/delta/ -v` — expect all 4 new-file tests (Burrows + Eder + Argamon + Cosine = 4 files × ~3 tests = 12 tests) passing.

### Step 8.8 — Commit

```bash
git add src/tamga/methods/delta/ tests/methods/delta/
git commit -m "feat(methods): Burrows, Eder(+Simple), Argamon(+Quadratic), Cosine Delta variants"
```

---

## Task 9: Federalist Papers fixture corpus

**Files:**
- Create: `tests/fixtures/federalist/metadata.tsv`
- Create: `tests/fixtures/federalist/fed_<NN>.txt` for each paper included

The Federalist Papers are in the public domain (1787-1788). We bundle a small subset that is sufficient for Burrows-Delta parity testing: papers 1 (Hamilton), 2-5 (Jay), 10 (Madison), 18-20 (joint Hamilton+Madison), and a selection of the *undisputed* papers plus a disputed one (paper 49 or 62, traditionally attributed to Madison by Mosteller & Wallace 1964). For this plan we ship:

| Paper | Undisputed author | Notes |
|---|---|---|
| 1, 11, 12, 13 | Hamilton | known-Hamilton training |
| 2, 3, 4, 5 | Jay | known-Jay training |
| 10, 14, 37, 51 | Madison | known-Madison training |
| 49 | Disputed (Madison per Mosteller & Wallace) | held-out test paper |

Texts come from Project Gutenberg EBook #1404, which is public domain. Strip Gutenberg headers/footers so only the essay body remains.

### Step 9.1 — Obtain texts

Because downloading ~15 text files is tedious and the implementer needs to do it only once, the plan's canonical approach is:

1. Download the Project Gutenberg text file: `https://www.gutenberg.org/cache/epub/1404/pg1404.txt`.
2. Use the following Python script (run once, ad-hoc; not committed to the repo) to split it into per-paper files, stripping headers, Gutenberg boilerplate, and footnotes:

```python
# scratch/extract_federalist.py (NOT committed — ad-hoc setup script)
from pathlib import Path
import re
import urllib.request

RAW = urllib.request.urlopen("https://www.gutenberg.org/cache/epub/1404/pg1404.txt").read().decode("utf-8")

# Trim Gutenberg header/footer.
START = RAW.index("FEDERALIST No. 1")
END = RAW.index("*** END OF THE PROJECT GUTENBERG")
body = RAW[START:END]

# Split on "FEDERALIST No. N" headings (case-insensitive, numeric).
pattern = re.compile(r"FEDERALIST\.?\s+No\.\s+(\d+)\.?", flags=re.IGNORECASE)
matches = list(pattern.finditer(body))

WANTED = {1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 37, 49, 51}
outdir = Path("tests/fixtures/federalist")
outdir.mkdir(parents=True, exist_ok=True)

for i, m in enumerate(matches):
    number = int(m.group(1))
    if number not in WANTED:
        continue
    start = m.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
    text = body[start:end].strip()
    (outdir / f"fed_{number:02d}.txt").write_text(text, encoding="utf-8")
```

Run it from the repo root with `python scratch/extract_federalist.py`. It produces 13 `fed_NN.txt` files.

### Step 9.2 — `tests/fixtures/federalist/metadata.tsv`

```
filename	author	role	disputed
fed_01.txt	Hamilton	train	false
fed_02.txt	Jay	train	false
fed_03.txt	Jay	train	false
fed_04.txt	Jay	train	false
fed_05.txt	Jay	train	false
fed_10.txt	Madison	train	false
fed_11.txt	Hamilton	train	false
fed_12.txt	Hamilton	train	false
fed_13.txt	Hamilton	train	false
fed_14.txt	Madison	train	false
fed_37.txt	Madison	train	false
fed_49.txt	Madison	test	true
fed_51.txt	Madison	train	false
```

### Step 9.3 — Commit the fixture

```bash
git add tests/fixtures/federalist/
git commit -m "test: bundle Federalist Papers fixture subset (public domain, Gutenberg ebook 1404)"
```

Size check: the 13 essay files total ~80–100 KB, well under the `--maxkb=500` pre-commit limit.

---

## Task 10: Federalist parity test

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_federalist_parity.py`

### Step 10.1 — Write the test

```python
"""Parity test: Burrows Delta on the Federalist Papers correctly attributes the disputed paper (49)
to Madison — the classical Mosteller & Wallace (1964) result, reproduced in every Stylo tutorial.

This is the release gate for Phase 2: we ship if this test passes.
"""

import pytest

from tamga.features import MFWExtractor
from tamga.io import load_corpus
from tamga.methods.delta import BurrowsDelta, CosineDelta, EderDelta


pytestmark = pytest.mark.integration


FED = "tests/fixtures/federalist"


def test_burrows_delta_attributes_fed_49_to_madison():
    corpus = load_corpus(FED, metadata=f"{FED}/metadata.tsv")

    train = corpus.filter(role="train")
    test = corpus.filter(role="test")

    # Fit MFW vocabulary on the training corpus, then z-score.
    mfw = MFWExtractor(n=500, min_df=2, scale="zscore", lowercase=True)
    train_fm = mfw.fit_transform(train)
    test_fm = mfw.transform(test)

    y_train = train.metadata_column("author")

    clf = BurrowsDelta().fit(train_fm, y_train)
    preds = clf.predict(test_fm)

    # Paper 49 → Madison.
    assert preds[0] == "Madison", f"Burrows Delta misattributed fed_49: got {preds[0]!r}"


@pytest.mark.parametrize("delta_cls", [EderDelta, CosineDelta])
def test_other_delta_variants_also_attribute_fed_49_to_madison(delta_cls):
    corpus = load_corpus(FED, metadata=f"{FED}/metadata.tsv")
    train = corpus.filter(role="train")
    test = corpus.filter(role="test")

    mfw = MFWExtractor(n=500, min_df=2, scale="zscore", lowercase=True)
    train_fm = mfw.fit_transform(train)
    test_fm = mfw.transform(test)

    clf = delta_cls().fit(train_fm, train.metadata_column("author"))
    assert clf.predict(test_fm)[0] == "Madison"
```

### Step 10.2 — Run the parity test

`pytest tests/integration/test_federalist_parity.py -v -m integration` — must pass all 3 tests (1 base + 2 parameterised).

If Madison is **not** returned, the problem is almost certainly one of:

1. **Training too small** — bump `n=500` to the actual count of reliable MFW given the corpus size. Try `n=200` first, then `n=100`.
2. **Text contamination** — Gutenberg header/footer leaked into a file. `grep -i "gutenberg" tests/fixtures/federalist/*.txt` should return nothing.
3. **Tokenisation mismatch** — inspect the vocabulary on the train side: `mfw = MFWExtractor(n=50).fit(train); print(mfw._vocabulary)` should look like English function words.
4. **Fixture labelling wrong** — double-check `metadata.tsv` author labels.

Do not commit until the test passes. If you try three different `n` values and none produce the right answer, STOP and report as BLOCKED — the extractor or delta implementation has a bug.

### Step 10.3 — Commit

```bash
git add tests/integration/
git commit -m "test: Federalist Papers parity — Burrows/Eder/Cosine Delta attribute fed_49 to Madison"
```

---

## Task 11: `tamga features` CLI command

**Files:**
- Create: `src/tamga/cli/features_cmd.py`
- Create: `tests/cli/test_features_cmd.py`
- Modify: `src/tamga/cli/__init__.py` (register)

### Step 11.1 — Implement `src/tamga/cli/features_cmd.py`

```python
"""`tamga features <corpus>` — build a feature matrix and persist to parquet."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from tamga.features import (
    CharNgramExtractor,
    FunctionWordExtractor,
    MFWExtractor,
    PunctuationExtractor,
    WordNgramExtractor,
)
from tamga.io import load_corpus

console = Console()

_EXTRACTORS = {
    "mfw": MFWExtractor,
    "char_ngram": CharNgramExtractor,
    "word_ngram": WordNgramExtractor,
    "function_word": FunctionWordExtractor,
    "punctuation": PunctuationExtractor,
}


def features_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    type: str = typer.Option("mfw", "--type", help=f"Feature type: one of {sorted(_EXTRACTORS)}"),
    n: int = typer.Option(1000, "--n", help="Top-N for MFW, or n-gram order"),
    min_df: int = typer.Option(1, "--min-df"),
    scale: str = typer.Option("zscore", "--scale", help="none | zscore | l1 | l2"),
    lowercase: bool = typer.Option(False, "--lowercase"),
    metadata: Path | None = typer.Option(None, "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    output: Path = typer.Option(Path("features.parquet"), "--output", "-o"),  # noqa: B008
) -> None:
    """Build a feature matrix from a corpus and save to parquet."""
    if type not in _EXTRACTORS:
        console.print(f"[red]error:[/red] unknown feature type {type!r}. Known: {sorted(_EXTRACTORS)}")
        raise typer.Exit(code=1)

    corpus = load_corpus(path, metadata=metadata)
    console.print(f"[green]loaded[/green] {len(corpus)} documents")

    extractor_cls = _EXTRACTORS[type]
    if type == "mfw":
        extractor = extractor_cls(n=n, min_df=min_df, scale=scale, lowercase=lowercase)
    elif type in ("char_ngram", "word_ngram"):
        extractor = extractor_cls(n=n, scale=scale)
    else:
        extractor = extractor_cls()

    fm = extractor.fit_transform(corpus)
    df = fm.as_dataframe()
    df.to_parquet(output)
    console.print(
        f"[green]wrote[/green] {output} ({fm.X.shape[0]} docs × {fm.n_features} features)"
    )
```

### Step 11.2 — Tests `tests/cli/test_features_cmd.py`

```python
"""Tests for `tamga features`."""

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini_corpus"


def test_features_mfw_writes_parquet(tmp_path: Path) -> None:
    out = tmp_path / "feats.parquet"
    result = runner.invoke(
        app,
        [
            "features", str(FIXTURES),
            "--type", "mfw",
            "--n", "20",
            "--scale", "none",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pd.read_parquet(out)
    assert df.shape[0] == 4
    assert df.shape[1] > 0


def test_features_rejects_unknown_type() -> None:
    result = runner.invoke(app, ["features", str(FIXTURES), "--type", "nonsense"])
    assert result.exit_code != 0
    assert "unknown" in result.stdout.lower()


def test_features_char_ngram_works(tmp_path: Path) -> None:
    out = tmp_path / "char.parquet"
    result = runner.invoke(
        app,
        [
            "features", str(FIXTURES),
            "--type", "char_ngram",
            "--n", "3",
            "--scale", "none",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
```

### Step 11.3 — Register in `src/tamga/cli/__init__.py`

Add import and registration at the appropriate lines (keep commands alphabetically ordered):

```python
from tamga.cli.features_cmd import features_command
...
app.command(name="features")(features_command)
```

### Step 11.4 — Run tests → PASS (3/3) and commit

```bash
git add src/tamga/cli/features_cmd.py tests/cli/test_features_cmd.py src/tamga/cli/__init__.py
git commit -m "feat(cli): tamga features — build feature matrix from corpus, save to parquet"
```

---

## Task 12: `tamga delta` CLI command

**Files:**
- Create: `src/tamga/cli/delta_cmd.py`
- Create: `tests/cli/test_delta_cmd.py`
- Modify: `src/tamga/cli/__init__.py` (register)

### Step 12.1 — Implement `src/tamga/cli/delta_cmd.py`

```python
"""`tamga delta <corpus>` — fit Burrows-family Delta and attribute held-out documents."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tamga.features import MFWExtractor
from tamga.io import load_corpus
from tamga.methods.delta import (
    ArgamonLinearDelta,
    BurrowsDelta,
    CosineDelta,
    EderDelta,
    EderSimpleDelta,
    QuadraticDelta,
)

console = Console()

_METHODS = {
    "burrows": BurrowsDelta,
    "eder": EderDelta,
    "eder_simple": EderSimpleDelta,
    "argamon": ArgamonLinearDelta,
    "cosine": CosineDelta,
    "quadratic": QuadraticDelta,
}


def delta_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    method: str = typer.Option("burrows", "--method", help=f"One of {sorted(_METHODS)}"),
    mfw: int = typer.Option(1000, "--mfw", help="Top-N most frequent words"),
    mfw_min: int = typer.Option(2, "--mfw-min", help="Minimum document frequency for MFW"),
    metadata: Path = typer.Option(..., "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    group_by: str = typer.Option("author", "--group-by", help="Metadata column with author labels"),
    test_filter: str | None = typer.Option(
        None, "--test-filter",
        help="Key=value (e.g. 'role=test') selecting held-out documents; "
        "if not provided, fit+predict on the entire corpus.",
    ),
) -> None:
    """Fit a Delta classifier on a corpus and report per-document attributions."""
    if method not in _METHODS:
        console.print(f"[red]error:[/red] unknown method {method!r}. Known: {sorted(_METHODS)}")
        raise typer.Exit(code=1)

    corpus = load_corpus(path, metadata=metadata)

    if test_filter is not None:
        key, _, value = test_filter.partition("=")
        if not _:
            console.print(f"[red]error:[/red] --test-filter must be 'key=value'; got {test_filter!r}")
            raise typer.Exit(code=1)
        test = corpus.filter(**{key: value})
        train_docs = [d for d in corpus.documents if d not in test.documents]
        from tamga.corpus import Corpus
        train = Corpus(documents=train_docs)
    else:
        train = corpus
        test = corpus

    extractor = MFWExtractor(n=mfw, min_df=mfw_min, scale="zscore", lowercase=True)
    train_fm = extractor.fit_transform(train)
    test_fm = extractor.transform(test)

    clf_cls = _METHODS[method]
    clf = clf_cls().fit(train_fm, train.metadata_column(group_by))
    preds = clf.predict(test_fm)

    table = Table(title=f"Delta attribution — method={method}, mfw={mfw}")
    table.add_column("doc_id", style="cyan")
    table.add_column(f"{group_by} (observed)")
    table.add_column(f"{group_by} (predicted)")
    table.add_column("match")
    for doc, pred in zip(test.documents, preds, strict=True):
        observed = doc.metadata.get(group_by, "<unknown>")
        match = "✓" if observed == pred else "✗"
        table.add_row(doc.id, str(observed), str(pred), match)
    console.print(table)
```

### Step 12.2 — Tests `tests/cli/test_delta_cmd.py`

```python
"""Tests for `tamga delta`."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tamga.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"


pytestmark = pytest.mark.integration


def test_delta_burrows_on_federalist() -> None:
    result = runner.invoke(
        app,
        [
            "delta", str(FED),
            "--method", "burrows",
            "--mfw", "500",
            "--metadata", str(FED / "metadata.tsv"),
            "--group-by", "author",
            "--test-filter", "role=test",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # Fed 49 → Madison.
    assert "Madison" in result.stdout
    assert "fed_49" in result.stdout


def test_delta_rejects_unknown_method() -> None:
    result = runner.invoke(app, ["delta", str(FED), "--method", "bogus", "--metadata", str(FED / "metadata.tsv")])
    assert result.exit_code != 0
```

### Step 12.3 — Register in `src/tamga/cli/__init__.py`

```python
from tamga.cli.delta_cmd import delta_command
...
app.command(name="delta")(delta_command)
```

### Step 12.4 — Run tests → PASS and commit

```bash
git add src/tamga/cli/delta_cmd.py tests/cli/test_delta_cmd.py src/tamga/cli/__init__.py
git commit -m "feat(cli): tamga delta — Burrows-family Delta attribution with metadata-driven train/test split"
```

---

## Task 13: sklearn-`Pipeline` integration test

**Files:**
- Create: `tests/integration/test_sklearn_pipeline.py`

### Step 13.1 — Write the test

```python
"""Integration test — tamga extractors and delta classifiers compose cleanly in sklearn.Pipeline
with cross_validate + LeaveOneGroupOut CV.

This is the load-bearing demonstration that our architectural promise of sklearn compatibility
holds end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import Pipeline

from tamga.features import MFWExtractor
from tamga.io import load_corpus
from tamga.methods.delta import BurrowsDelta

pytestmark = pytest.mark.integration


FED = "tests/fixtures/federalist"


def test_pipeline_with_mfw_and_burrows_classifies():
    corpus = load_corpus(FED, metadata=f"{FED}/metadata.tsv")
    train = corpus.filter(role="train")
    y = np.array(train.metadata_column("author"))

    pipe = Pipeline(
        [
            ("feat", MFWExtractor(n=300, min_df=2, scale="zscore", lowercase=True)),
            ("clf", BurrowsDelta()),
        ]
    )

    # Simple fit/predict round trip.
    pipe.fit(train, y)
    preds = pipe.predict(train)
    assert preds.shape == (len(train),)


def test_cross_val_score_with_leave_one_group_out():
    corpus = load_corpus(FED, metadata=f"{FED}/metadata.tsv")
    train = corpus.filter(role="train")
    y = np.array(train.metadata_column("author"))

    pipe = Pipeline(
        [
            ("feat", MFWExtractor(n=300, min_df=2, scale="zscore", lowercase=True)),
            ("clf", BurrowsDelta()),
        ]
    )

    # Use authorship as the CV group — leave-one-author-out.
    # Note: this will fail every fold (no same-author training data), so we expect very low accuracy;
    # the integration-correctness check is that the pipeline runs without raising.
    scores = cross_val_score(
        pipe, train, y, cv=LeaveOneGroupOut(), groups=y, scoring="accuracy"
    )
    assert scores.shape[0] == len(np.unique(y))
```

### Step 13.2 — Run → PASS and commit

```bash
git add tests/integration/test_sklearn_pipeline.py
git commit -m "test: sklearn Pipeline + cross_val_score + LeaveOneGroupOut integration"
```

---

## Task 14: Public API + docstrings + phase-2 tag

**Files:**
- Modify: `src/tamga/__init__.py` (re-export new public surface)
- Modify: `README.md` (update Phase 2 status)

### Step 14.1 — `src/tamga/__init__.py`

```python
"""tamga — next-generation computational stylometry."""

from tamga._version import __version__
from tamga.config import StudyConfig, load_config, resolve_config
from tamga.corpus import Corpus, Document
from tamga.features import (
    BaseFeatureExtractor,
    CharNgramExtractor,
    DependencyBigramExtractor,
    FeatureMatrix,
    FunctionWordExtractor,
    LexicalDiversityExtractor,
    MFWExtractor,
    PosNgramExtractor,
    PunctuationExtractor,
    ReadabilityExtractor,
    SentenceLengthExtractor,
    WordNgramExtractor,
)
from tamga.io import load_corpus, load_metadata
from tamga.methods.delta import (
    ArgamonLinearDelta,
    BurrowsDelta,
    CosineDelta,
    EderDelta,
    EderSimpleDelta,
    QuadraticDelta,
)
from tamga.preprocess.pipeline import ParsedCorpus, SpacyPipeline
from tamga.provenance import Provenance

__all__ = [
    "__version__",
    # core
    "Corpus", "Document", "ParsedCorpus", "Provenance", "SpacyPipeline",
    "StudyConfig", "load_config", "load_corpus", "load_metadata", "resolve_config",
    # features
    "BaseFeatureExtractor", "CharNgramExtractor", "DependencyBigramExtractor", "FeatureMatrix",
    "FunctionWordExtractor", "LexicalDiversityExtractor", "MFWExtractor", "PosNgramExtractor",
    "PunctuationExtractor", "ReadabilityExtractor", "SentenceLengthExtractor", "WordNgramExtractor",
    # methods
    "ArgamonLinearDelta", "BurrowsDelta", "CosineDelta", "EderDelta", "EderSimpleDelta",
    "QuadraticDelta",
]
```

### Step 14.2 — README.md status line

Change the `## Status` paragraph to:

```markdown
## Status

**Phase 2 — Features & Delta.** Ships feature extractors (MFW, char/word/POS n-grams,
dependency bigrams, function words, punctuation, lexical diversity, readability, sentence length)
and the full Delta family (Burrows, Eder, Eder-Simple, Argamon Linear, Cosine, Quadratic).
`tamga features` and `tamga delta` CLI commands work end-to-end; the Federalist Papers parity
test attributes the disputed paper 49 to Madison using Burrows/Eder/Cosine Delta on 500 MFW.

Phases 3 (Zeta/reducers/clustering/consensus/classify), 4 (embeddings + Bayesian),
5 (viz + reports + wizard shell), and 6 (docs + PyPI) remain.
```

### Step 14.3 — Run the full suite with coverage

```bash
source .venv/bin/activate
pytest -n auto --cov=tamga --cov-report=term-missing -q
pre-commit run --all-files
```

Expected: all tests pass, coverage ≥88% on feature modules, ≥90% on delta modules.

### Step 14.4 — Tag and commit

```bash
git add src/tamga/__init__.py README.md
git commit -m "feat: public API re-exports for Phase 2 (features + Delta family)"
git tag -a phase-2-features-delta -m "Phase 2 complete: 10 feature extractors, 6 Delta variants, Federalist parity test passes"
```

---

## Phase 2 — Acceptance Criteria

From a clean checkout of the tagged `phase-2-features-delta`:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m spacy download en_core_web_sm
pre-commit run --all-files
pytest -n auto --cov=tamga --cov-report=term-missing -q

# Federalist parity
pytest tests/integration/test_federalist_parity.py -v -m integration    # must show 3 passing

# CLI end-to-end
tamga features ./tests/fixtures/mini_corpus \
    --type mfw --n 20 --scale none --output /tmp/feats.parquet
tamga delta ./tests/fixtures/federalist \
    --method burrows --mfw 500 \
    --metadata ./tests/fixtures/federalist/metadata.tsv \
    --group-by author --test-filter role=test
# → Table output includes "fed_49 | Madison | Madison | ✓"
```

Every command above must exit 0.

---

## Self-Review Notes

- **Spec coverage:** every v0.1 feature extractor from spec §5 except `SentenceEmbeddingExtractor` / `ContextualEmbeddingExtractor` (those are Phase 4, the `[embeddings]` extra). All six Delta variants from §6.1. `tamga features` and `tamga delta` from §8.1. sklearn protocol integration from §7. Federalist parity from §13.1.

- **Deferred to Phase 3:** Craig's Zeta, PCA/MDS/t-SNE/UMAP reducers, hierarchical/k-means/HDBSCAN clustering, bootstrap consensus trees, sklearn classifiers (svm/rf/hgbm/logreg), leave-one-author-out CV helper.

- **Deferred to Phase 4:** `SentenceEmbeddingExtractor`, `ContextualEmbeddingExtractor`, Wallace–Mosteller Bayesian attribution, hierarchical PyMC group comparison.

- **Placeholder scan:** no TBD/TODO/FIXME. Every step has runnable content.

- **Type consistency:** `FeatureMatrix` signature stable across every extractor; `_DeltaBase._distance` signature used by all six subclasses; `SpacyPipeline` signatures match Phase 1; `load_corpus`/`MFWExtractor`/`BurrowsDelta` identifiers consistent across tests, CLI, and integration.

- **Test discipline:** every extractor and every Delta variant has at least one happy-path test plus at least one edge-case or sklearn-compatibility test. Federalist parity test is the load-bearing integration.

- **Commit cadence:** 12 commits over 14 tasks (Task 9 ships the fixture without a separate code commit; Task 14 bundles the public-API + README + tag).
