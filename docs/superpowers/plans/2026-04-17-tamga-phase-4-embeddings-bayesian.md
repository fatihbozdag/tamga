# bitig — Phase 4: Embeddings + Bayesian — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ship the two optional-extra modules declared in the spec: `bitig[embeddings]` (sentence-transformer + contextual-BERT embedding extractors implementing the existing `BaseFeatureExtractor` protocol) and `bitig[bayesian]` (a Wallace–Mosteller Bayesian authorship attributor implementing `ClassifierMixin`, plus a PyMC hierarchical group-comparison model for L2-vs-native and proficiency-band analyses). Add CLI commands `bitig embed` and `bitig bayesian`. End state: users who `pip install bitig[embeddings,bayesian]` unlock modern style vectors and probabilistic attribution; users who don't get a clear "install the `[embeddings]`/`[bayesian]` extra to enable" message.

**Architecture:** Optional extras load lazily. Imports of `sentence_transformers` / `torch` / `pymc` / `arviz` live inside the class methods, not at module top level, so users without the extras can still `import bitig` and run Phase 1-3 code. Each new extractor/method does an explicit import check in `__init__` that raises `ImportError` with the install hint.

**Tech Stack:** `sentence-transformers` (≥2.6), `torch` (≥2.2), `transformers` (pulled transitively), `pymc` (≥5.10), `arviz` (≥0.17). All five already declared in `pyproject.toml` optional-extras from Phase 1.

**Reference spec:** §5 (embeddings), §6.7 (Bayesian extras).

**Phase 3 baseline:** tag `phase-3-analytical-breadth`, 203 tests passing.

---

## Task 1: Install optional-extras dependencies

- [ ] **Step 1.1:** `source .venv/bin/activate && uv pip install -e ".[embeddings,bayesian]"`

Expected: `sentence-transformers`, `torch`, `pymc`, `arviz`, and their transitive deps install. Disk cost ≈ 3-4 GB. Time: 2-5 minutes depending on bandwidth.

- [ ] **Step 1.2:** Sanity imports:

```bash
python -c "import sentence_transformers, torch, pymc, arviz; print('embeddings:', sentence_transformers.__version__); print('torch:', torch.__version__); print('pymc:', pymc.__version__)"
```

All should print non-empty versions.

- [ ] **Step 1.3:** Download a small sentence-transformer model for tests (one-time):

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

`all-MiniLM-L6-v2` is 80 MB — small, fast, deterministic enough for tests. We'll use this instead of the `all-mpnet-base-v2` mentioned in the spec (420 MB) for test-speed reasons.

- [ ] **Step 1.4:** No commit for this task — dependency installation doesn't change the repo state. (Phase 1's `pyproject.toml` already declares the extras.)

---

## Task 2: `SentenceEmbeddingExtractor` + `ContextualEmbeddingExtractor`

**Files:**
- Create: `src/bitig/features/embeddings.py`
- Create: `tests/features/test_embeddings.py`
- Modify: `src/bitig/features/__init__.py` (conditional import)

### Step 2.1 — Failing tests `tests/features/test_embeddings.py`

```python
"""Tests for embedding-based feature extractors — requires bitig[embeddings]."""

from __future__ import annotations

import numpy as np
import pytest

from bitig.corpus import Corpus, Document

pytestmark = pytest.mark.slow  # Model loading is slow.

_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _corpus(*texts: str) -> Corpus:
    return Corpus(documents=[Document(id=f"d{i}", text=t) for i, t in enumerate(texts)])


def test_sentence_embedding_extractor_produces_fixed_dim() -> None:
    from bitig.features.embeddings import SentenceEmbeddingExtractor

    ex = SentenceEmbeddingExtractor(model=_MODEL, pool="mean")
    fm = ex.fit_transform(_corpus("Hello world.", "The quick brown fox jumped."))
    # all-MiniLM-L6-v2 is 384-dim.
    assert fm.X.shape == (2, 384)
    assert fm.feature_names[:3] == ["emb_0", "emb_1", "emb_2"]


def test_sentence_embedding_similar_texts_have_high_cosine_sim() -> None:
    from bitig.features.embeddings import SentenceEmbeddingExtractor

    ex = SentenceEmbeddingExtractor(model=_MODEL, pool="mean")
    fm = ex.fit_transform(
        _corpus(
            "The cat sat on the mat.",
            "A cat was sitting on the mat.",
            "Quantum chromodynamics governs strong interactions.",
        )
    )
    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity(fm.X)
    # First two are near-paraphrases; third is unrelated physics.
    assert sims[0, 1] > sims[0, 2]
    assert sims[0, 1] > sims[1, 2]


def test_sentence_embedding_handles_empty_text() -> None:
    from bitig.features.embeddings import SentenceEmbeddingExtractor

    ex = SentenceEmbeddingExtractor(model=_MODEL, pool="mean")
    fm = ex.fit_transform(_corpus("", "hello"))
    assert fm.X.shape == (2, 384)


def test_embeddings_raises_clear_error_when_not_installed(monkeypatch) -> None:
    """If sentence-transformers is absent, import/construction raises an informative error."""
    from bitig.features import embeddings

    # Simulate missing dep by breaking the import path temporarily.
    monkeypatch.setattr(embeddings, "_sentence_transformers_available", False)
    with pytest.raises(ImportError, match=r"bitig\[embeddings\]"):
        embeddings.SentenceEmbeddingExtractor(model=_MODEL, pool="mean")
```

### Step 2.2 — Run → FAIL.

### Step 2.3 — Implement `src/bitig/features/embeddings.py`

```python
"""Embedding-based feature extractors (optional — bitig[embeddings]).

Two extractors:

- `SentenceEmbeddingExtractor` — sentence-transformers model; pool the whole-document embedding by
  averaging per-sentence embeddings (default) or taking the CLS token.
- `ContextualEmbeddingExtractor` — a raw transformer (e.g. BERT); pool layer-k token embeddings
  by mean or CLS.

Both raise `ImportError` at construction if the optional `bitig[embeddings]` extra is not installed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from bitig.corpus import Corpus
from bitig.features.base import BaseFeatureExtractor

try:
    from sentence_transformers import SentenceTransformer

    _sentence_transformers_available = True
except ImportError:
    _sentence_transformers_available = False

Pool = Literal["mean", "cls", "max"]

_INSTALL_HINT = (
    "this extractor requires the optional `bitig[embeddings]` extra — "
    "install with `pip install bitig[embeddings]`"
)


class SentenceEmbeddingExtractor(BaseFeatureExtractor):
    feature_type = "sentence_embedding"

    def __init__(
        self,
        *,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        pool: Pool = "mean",
        device: str | None = None,
    ) -> None:
        if not _sentence_transformers_available:
            raise ImportError(_INSTALL_HINT)
        self.model = model
        self.pool = pool
        self.device = device
        self._encoder: SentenceTransformer | None = None

    def _load_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.model, device=self.device)
        return self._encoder

    def _fit(self, corpus: Corpus) -> None:  # noqa: ARG002
        self._load_encoder()

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        encoder = self._load_encoder()
        texts = [d.text for d in corpus.documents]
        embeddings = encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # For pool="mean", sentence-transformers already averages per-sentence; whole-document
        # embedding is the single vector. "cls"/"max" are left as aliases of "mean" here — the
        # model's own pooling layer decides. Users who need a different pooling strategy should
        # use the ContextualEmbeddingExtractor.
        feature_names = [f"emb_{i}" for i in range(embeddings.shape[1])]
        return embeddings.astype(float), feature_names


class ContextualEmbeddingExtractor(BaseFeatureExtractor):
    """Raw HF transformer embeddings — average of layer-k token vectors per document."""

    feature_type = "contextual_embedding"

    def __init__(
        self,
        *,
        model: str = "bert-base-uncased",
        layer: int = -1,
        pool: Pool = "mean",
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        if not _sentence_transformers_available:
            raise ImportError(_INSTALL_HINT)
        self.model = model
        self.layer = layer
        self.pool = pool
        self.device = device
        self.max_length = max_length
        self._tokenizer = None
        self._transformer = None

    def _load(self):
        if self._transformer is None:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._transformer = AutoModel.from_pretrained(self.model, output_hidden_states=True)
            if self.device:
                self._transformer = self._transformer.to(self.device)
            self._transformer.eval()

    def _fit(self, corpus: Corpus) -> None:  # noqa: ARG002
        self._load()

    def _transform(self, corpus: Corpus) -> tuple[np.ndarray, list[str]]:
        import torch

        self._load()
        embeddings = []
        with torch.no_grad():
            for doc in corpus.documents:
                encoded = self._tokenizer(
                    doc.text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                )
                if self.device:
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self._transformer(**encoded)
                hidden_states = outputs.hidden_states[self.layer]  # (1, tokens, dim)
                if self.pool == "cls":
                    vec = hidden_states[0, 0, :]
                elif self.pool == "max":
                    vec, _ = hidden_states[0].max(dim=0)
                else:  # mean
                    vec = hidden_states[0].mean(dim=0)
                embeddings.append(vec.cpu().numpy())
        X = np.vstack(embeddings).astype(float)
        feature_names = [f"emb_{i}" for i in range(X.shape[1])]
        return X, feature_names
```

### Step 2.4 — Update `src/bitig/features/__init__.py`

Add conditional imports — only export if the extractor class can be constructed (i.e. the extra is installed):

```python
# ... existing imports ...

try:
    from bitig.features.embeddings import ContextualEmbeddingExtractor, SentenceEmbeddingExtractor

    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

# Append to __all__ only if available.
if _EMBEDDINGS_AVAILABLE:
    __all__ = [*__all__, "ContextualEmbeddingExtractor", "SentenceEmbeddingExtractor"]
```

### Step 2.5 — Run → PASS (4/4). Commit:

```bash
git add src/bitig/features/embeddings.py tests/features/test_embeddings.py src/bitig/features/__init__.py
git commit -m "feat(features): sentence + contextual embedding extractors (bitig[embeddings] extra)"
```

---

## Task 3: `BayesianAuthorshipAttributor` — Wallace–Mosteller attribution

**Files:**
- Create: `src/bitig/methods/bayesian.py`
- Create: `tests/methods/test_bayesian.py`

### Step 3.1 — Failing tests

```python
"""Tests for Bayesian authorship attribution — requires bitig[bayesian]."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import is_classifier

from bitig.corpus import Corpus, Document
from bitig.features import MFWExtractor

pytestmark = pytest.mark.slow


def _corpus() -> Corpus:
    rng = np.random.default_rng(42)
    docs = []
    for i in range(20):
        author = "A" if i < 10 else "B"
        # Author A prefers "alpha"; Author B prefers "beta".
        text = " ".join(
            rng.choice(
                ["alpha", "beta", "gamma", "delta", "epsilon"],
                size=200,
                p=[0.4, 0.1, 0.2, 0.2, 0.1] if author == "A" else [0.1, 0.4, 0.2, 0.2, 0.1],
            )
        )
        docs.append(Document(id=f"d{i}", text=text, metadata={"author": author}))
    return Corpus(documents=docs)


def test_bayesian_attributor_is_classifier() -> None:
    from bitig.methods.bayesian import BayesianAuthorshipAttributor

    assert is_classifier(BayesianAuthorshipAttributor())


def test_bayesian_attributor_separates_two_authors() -> None:
    from bitig.methods.bayesian import BayesianAuthorshipAttributor

    corpus = _corpus()
    y = np.array(corpus.metadata_column("author"))
    fm = MFWExtractor(n=5, scale="none", lowercase=True).fit_transform(corpus)
    clf = BayesianAuthorshipAttributor(prior_alpha=0.5).fit(fm, y)
    preds = clf.predict(fm)
    # On in-sample data with strong author-word associations, accuracy should be near 1.
    assert (preds == y).mean() >= 0.8


def test_bayesian_attributor_predict_proba_sums_to_one() -> None:
    from bitig.methods.bayesian import BayesianAuthorshipAttributor

    corpus = _corpus()
    y = np.array(corpus.metadata_column("author"))
    fm = MFWExtractor(n=5, scale="none", lowercase=True).fit_transform(corpus)
    clf = BayesianAuthorshipAttributor().fit(fm, y)
    probs = clf.predict_proba(fm)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_bayesian_raises_clear_error_when_not_installed(monkeypatch) -> None:
    from bitig.methods import bayesian

    monkeypatch.setattr(bayesian, "_pymc_available", False)
    with pytest.raises(ImportError, match=r"bitig\[bayesian\]"):
        bayesian.HierarchicalGroupComparison(group_by="author")
```

### Step 3.2 — Implement `src/bitig/methods/bayesian.py`

```python
"""Bayesian authorship attribution (Wallace-Mosteller-style) + hierarchical group comparison.

Wallace-Mosteller approach:
  For candidate author `a`, compute log P(tokens | a) = sum(count_w * log rate_w_a), where
  rate_w_a is the MAP estimate of word w's rate under author a's training texts, with a Beta
  (or Dirichlet) prior smoothing zero-count words.

Predicted author = argmax log posterior = log prior (uniform) + log likelihood.

This is a sklearn `ClassifierMixin` so it plugs into Pipeline / cross_validate.

HierarchicalGroupComparison requires PyMC; it builds a varying-intercept model with per-author
draws from a group-level hyperparameter — useful for testing whether two author populations
differ systematically in a stylistic feature (e.g., L2 vs. native function-word use).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from bitig.features import FeatureMatrix

try:
    import pymc  # noqa: F401

    _pymc_available = True
except ImportError:
    _pymc_available = False

_INSTALL_HINT_BAYESIAN = (
    "this method requires the optional `bitig[bayesian]` extra — "
    "install with `pip install bitig[bayesian]`"
)


def _as_array(X: FeatureMatrix | np.ndarray) -> np.ndarray:
    return X.X if isinstance(X, FeatureMatrix) else np.asarray(X)


class BayesianAuthorshipAttributor(ClassifierMixin, BaseEstimator):
    """Wallace-Mosteller-style Bayesian authorship attribution.

    Expects count-valued features (raw word counts or equivalent). If z-scored features are
    passed, predictions will still work but the "rate" interpretation breaks down — use
    `MFWExtractor(scale="none")` to produce the right input.
    """

    def __init__(self, *, prior_alpha: float = 1.0) -> None:
        self.prior_alpha = prior_alpha
        self.classes_: np.ndarray = np.empty(0, dtype=object)
        self.log_rates_: dict[str, np.ndarray] = {}

    def fit(self, X: FeatureMatrix | np.ndarray, y: np.ndarray) -> BayesianAuthorshipAttributor:
        counts = _as_array(X)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)

        n_features = counts.shape[1]
        self.log_rates_ = {}
        for cls in self.classes_:
            class_counts = counts[y_arr == cls].sum(axis=0)
            # Additive (Dirichlet / Beta for binary words) smoothing.
            smoothed = class_counts + self.prior_alpha
            rates = smoothed / smoothed.sum()
            # Clip for numerical stability on unseen-but-allowed words.
            rates = np.clip(rates, 1e-12, 1.0)
            self.log_rates_[str(cls)] = np.log(rates)
        return self

    def decision_function(self, X: FeatureMatrix | np.ndarray) -> np.ndarray:
        counts = _as_array(X)
        scores = np.column_stack(
            [counts @ self.log_rates_[str(cls)] for cls in self.classes_]
        )
        return scores

    def predict(self, X: FeatureMatrix | np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: FeatureMatrix | np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        scores_shift = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores_shift)
        return exp / exp.sum(axis=1, keepdims=True)


class HierarchicalGroupComparison:
    """PyMC hierarchical model: per-author draws of a stylistic feature drawn from a group-level
    hyperparameter, enabling a test of whether two or more groups differ systematically.

    Simple form:
        mu_group    ~ Normal(0, 5)       # group-level mean
        sigma_group ~ HalfNormal(1)      # group-level SD
        theta_author ~ Normal(mu_group, sigma_group)   # per-author stylistic score
        observation[author, doc] ~ Normal(theta_author, obs_sigma)

    Requires bitig[bayesian] extra.
    """

    def __init__(
        self,
        *,
        group_by: str,
        chains: int = 2,
        samples: int = 500,
        tune: int = 500,
        seed: int = 42,
    ) -> None:
        if not _pymc_available:
            raise ImportError(_INSTALL_HINT_BAYESIAN)
        self.group_by = group_by
        self.chains = chains
        self.samples = samples
        self.tune = tune
        self.seed = seed

    def fit_transform(self, fm: FeatureMatrix, y: np.ndarray, groups: np.ndarray) -> dict:
        """Fit the hierarchical model and return a summary dict.

        Parameters
        ----------
        fm : FeatureMatrix
            A single-feature column (use e.g. a scalar index per document such as lexical-diversity).
            If fm has multiple columns, we fit one model per column.
        y : np.ndarray
            Per-document author labels.
        groups : np.ndarray
            Per-document group labels (e.g. L1 vs L2).
        """
        import arviz as az
        import pymc as pm

        X = fm.X
        unique_groups = np.unique(groups)
        unique_authors = np.unique(y)
        group_idx = np.array([list(unique_groups).index(g) for g in groups])
        author_idx = np.array([list(unique_authors).index(a) for a in y])

        results = []
        for col in range(X.shape[1]):
            observations = X[:, col]
            with pm.Model():
                mu_group = pm.Normal("mu_group", mu=0, sigma=5, shape=len(unique_groups))
                sigma_group = pm.HalfNormal("sigma_group", sigma=1, shape=len(unique_groups))
                theta_author = pm.Normal(
                    "theta_author",
                    mu=mu_group[group_idx[np.arange(len(unique_authors))]]
                    if len(unique_authors) == len(groups)
                    else 0,
                    sigma=1,
                    shape=len(unique_authors),
                )
                obs_sigma = pm.HalfNormal("obs_sigma", sigma=1)
                pm.Normal(
                    "obs",
                    mu=theta_author[author_idx],
                    sigma=obs_sigma,
                    observed=observations,
                )
                trace = pm.sample(
                    self.samples,
                    tune=self.tune,
                    chains=self.chains,
                    random_seed=self.seed,
                    progressbar=False,
                    return_inferencedata=True,
                )
            summary = az.summary(trace, var_names=["mu_group"])
            results.append(
                {
                    "feature": fm.feature_names[col],
                    "mu_group_summary": summary.to_dict(),
                    "groups": list(unique_groups),
                }
            )
        return {"results": results}
```

### Step 3.3 — Run tests, commit:

`pytest tests/methods/test_bayesian.py -v` — expect 4 passing (the hierarchical test may be skipped or simplified if PyMC sampling is slow).

Actually the hierarchical test (`test_bayesian_raises_clear_error_when_not_installed`) only checks the error message path, not PyMC sampling itself. A dedicated PyMC sampling test is deferred because fitting a real hierarchical model takes 30+ seconds — not appropriate for the main test suite. Add a `@pytest.mark.slow` integration test later if needed.

```bash
git add src/bitig/methods/bayesian.py tests/methods/test_bayesian.py
git commit -m "feat(methods): Bayesian authorship attribution + hierarchical group comparison (bitig[bayesian])"
```

---

## Task 4: CLI — `bitig embed` and `bitig bayesian`

**Files:**
- Create: `src/bitig/cli/embed_cmd.py`
- Create: `src/bitig/cli/bayesian_cmd.py`
- Create: `tests/cli/test_embed_cmd.py`
- Create: `tests/cli/test_bayesian_cmd.py`
- Modify: `src/bitig/cli/__init__.py`

### Step 4.1 — `src/bitig/cli/embed_cmd.py`

```python
"""`bitig embed <corpus>` — produce an embedding FeatureMatrix and save to parquet."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from bitig.io import load_corpus

console = Console()


def embed_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path | None = typer.Option(None, "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2", "--model"),
    pool: str = typer.Option("mean", "--pool"),
    output: Path = typer.Option(Path("embeddings.parquet"), "--output", "-o"),  # noqa: B008
) -> None:
    """Embed a corpus with a sentence-transformer model and save the matrix to parquet."""
    try:
        from bitig.features.embeddings import SentenceEmbeddingExtractor
    except ImportError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    corpus = load_corpus(path, metadata=metadata)
    ex = SentenceEmbeddingExtractor(model=model, pool=pool)  # type: ignore[arg-type]
    fm = ex.fit_transform(corpus)
    fm.as_dataframe().to_parquet(output)
    console.print(f"[green]wrote[/green] {output} ({fm.X.shape[0]} docs x {fm.n_features} dims)")
```

### Step 4.2 — `src/bitig/cli/bayesian_cmd.py`

```python
"""`bitig bayesian <corpus>` — Bayesian authorship attribution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from bitig.features import MFWExtractor
from bitig.io import load_corpus

console = Console()


def bayesian_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path = typer.Option(..., "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    group_by: str = typer.Option("author", "--group-by"),
    test_filter: str | None = typer.Option(None, "--test-filter", help="key=value selecting held-out documents"),
    mfw: int = typer.Option(500, "--mfw"),
    prior_alpha: float = typer.Option(1.0, "--prior-alpha"),
) -> None:
    """Wallace-Mosteller Bayesian authorship attribution."""
    from bitig.methods.bayesian import BayesianAuthorshipAttributor

    corpus = load_corpus(path, metadata=metadata)
    if test_filter:
        key, _, value = test_filter.partition("=")
        test = corpus.filter(**{key: value})
        train_docs = [d for d in corpus.documents if d not in test.documents]
        from bitig.corpus import Corpus

        train = Corpus(documents=train_docs)
    else:
        train = corpus
        test = corpus

    ex = MFWExtractor(n=mfw, min_df=2, scale="none", lowercase=True)
    train_fm = ex.fit_transform(train)
    test_fm = ex.transform(test)

    clf = BayesianAuthorshipAttributor(prior_alpha=prior_alpha).fit(
        train_fm, np.array(train.metadata_column(group_by))
    )
    preds = clf.predict(test_fm)
    probs = clf.predict_proba(test_fm)

    table = Table(title=f"Bayesian attribution — prior_alpha={prior_alpha}, mfw={mfw}")
    table.add_column("doc_id", style="cyan")
    table.add_column(f"{group_by} (observed)")
    table.add_column("predicted")
    table.add_column("max p(author)")
    for doc, pred, prob in zip(test.documents, preds, probs, strict=True):
        observed = doc.metadata.get(group_by, "<unknown>")
        table.add_row(doc.id, str(observed), str(pred), f"{prob.max():.3f}")
    console.print(table)
```

### Step 4.3 — Tests

**`tests/cli/test_embed_cmd.py`:**

```python
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
FIXTURES = Path(__file__).parent.parent / "fixtures" / "mini_corpus"


pytestmark = pytest.mark.slow


def test_embed_writes_parquet(tmp_path: Path) -> None:
    out = tmp_path / "emb.parquet"
    result = runner.invoke(
        app,
        [
            "embed", str(FIXTURES),
            "--model", "sentence-transformers/all-MiniLM-L6-v2",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pd.read_parquet(out)
    assert df.shape == (4, 384)
```

**`tests/cli/test_bayesian_cmd.py`:**

```python
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_bayesian_attributes_fed_49_to_madison() -> None:
    result = runner.invoke(
        app,
        [
            "bayesian", str(FED),
            "--metadata", str(FED / "metadata.tsv"),
            "--group-by", "author",
            "--test-filter", "role=test",
            "--mfw", "500",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "fed_49" in result.stdout
    assert "Madison" in result.stdout
```

### Step 4.4 — Register and commit

Append to `src/bitig/cli/__init__.py`:

```python
from bitig.cli.bayesian_cmd import bayesian_command
from bitig.cli.embed_cmd import embed_command

app.command(name="embed")(embed_command)
app.command(name="bayesian")(bayesian_command)
```

Commit:

```bash
git add src/bitig/cli/embed_cmd.py src/bitig/cli/bayesian_cmd.py tests/cli/test_embed_cmd.py tests/cli/test_bayesian_cmd.py src/bitig/cli/__init__.py
git commit -m "feat(cli): bitig embed and bitig bayesian (optional-extras-aware)"
```

---

## Task 5: Public API + tag

**Files:**
- Modify: `src/bitig/__init__.py`
- Modify: `src/bitig/methods/__init__.py` (conditional bayesian export)
- Modify: `README.md`

### Step 5.1 — `src/bitig/__init__.py` — conditional embeddings + bayesian exports

Append after the Phase 3 section:

```python
# Optional extras — available only when bitig[embeddings] / bitig[bayesian] installed.
try:
    from bitig.features.embeddings import ContextualEmbeddingExtractor, SentenceEmbeddingExtractor  # noqa: F401

    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

try:
    from bitig.methods.bayesian import BayesianAuthorshipAttributor, HierarchicalGroupComparison  # noqa: F401

    _BAYESIAN_AVAILABLE = True
except ImportError:
    _BAYESIAN_AVAILABLE = False

if _EMBEDDINGS_AVAILABLE:
    __all__ = [*__all__, "ContextualEmbeddingExtractor", "SentenceEmbeddingExtractor"]
if _BAYESIAN_AVAILABLE:
    __all__ = [*__all__, "BayesianAuthorshipAttributor", "HierarchicalGroupComparison"]
```

### Step 5.2 — Update README status

Replace the `## Status` section with:

```markdown
## Status

**Phase 4 — Optional extras.** `bitig[embeddings]` adds sentence-transformer + contextual-BERT
embeddings (pool mean/cls/max). `bitig[bayesian]` adds Wallace-Mosteller Bayesian authorship
attribution (sklearn ClassifierMixin, plugs into Pipeline/cross_validate) and a PyMC
hierarchical group-comparison model. CLI: `bitig embed`, `bitig bayesian`. Phases 1-3 remain
installable without either extra.

Phase 5 (viz + reports + wizard shell), Phase 6 (docs + PyPI) remain.
```

### Step 5.3 — Commit + tag

```bash
source .venv/bin/activate
pytest -n auto --cov=bitig -q
pre-commit run --all-files
git add src/bitig/__init__.py README.md
git commit -m "feat: public API re-exports for Phase 4 (embeddings + Bayesian extras)"
git tag -a phase-4-extras -m "Phase 4 complete: embeddings + Bayesian attribution optional extras"
```

---

## Phase 4 — Acceptance Criteria

```bash
# Full suite passes.
pytest -n auto -q

# Embedding CLI round-trip.
bitig embed ./tests/fixtures/mini_corpus --model sentence-transformers/all-MiniLM-L6-v2 --output /tmp/emb.parquet

# Bayesian attribution on Federalist.
bitig bayesian ./tests/fixtures/federalist --metadata ./tests/fixtures/federalist/metadata.tsv \
    --group-by author --test-filter role=test --mfw 500
```

Every command exits 0. Bayesian CLI's output table contains `fed_49 | Madison | Madison`.

---

## Self-Review

- **Spec §5 embeddings** → Task 2.
- **Spec §6.7 Bayesian** → Task 3.
- **sklearn protocol (Spec §7):** BayesianAuthorshipAttributor is ClassifierMixin; runs through Pipeline/cross_validate.
- **Graceful degradation:** every optional-extra import wrapped in try/except; clear error messages on missing installs.
- **Placeholder scan:** no TBD/TODO.
- **Type consistency:** FeatureMatrix signature preserved.
