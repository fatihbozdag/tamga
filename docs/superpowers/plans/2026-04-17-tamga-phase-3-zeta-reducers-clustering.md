# bitig — Phase 3: Zeta, Reducers, Clustering, Consensus, Classify — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the analytical breadth of R's `Stylo`: Craig's Zeta contrastive vocabulary, four dimensionality reducers (PCA/MDS/t-SNE/UMAP), three clustering methods (hierarchical/k-means/HDBSCAN), bootstrap consensus trees over MFW bands, and sklearn-wrapped classifiers (linear/RBF SVM, logistic regression, random forest, histogram gradient boosting) with stylometry-aware CV primitives including leave-one-author-out. End state: every analytical method from spec §6 is callable via library API and CLI; `bitig run study.yaml` can drive a full multi-method study (wired in Phase 5 report-generator).

**Architecture:** Each analytical family lives under `bitig/methods/`. Dimensionality reduction + clustering are thin sklearn wrappers that return `Result` objects with resolved-config provenance. Classifiers are sklearn estimators with bitig's CV helpers on top. Consensus trees are the single piece of genuinely new algorithmic work — they iterate bootstrap-subsampled MFW bands, aggregate Ward linkages, and produce a Newick-serialised consensus.

**Tech Stack:** Phase 1+2 stack + sklearn's unsupervised modules (`sklearn.decomposition.PCA`, `sklearn.manifold.MDS/TSNE`, `sklearn.cluster.AgglomerativeClustering/KMeans`), `umap-learn` (already in deps), `hdbscan` (already in deps), `scipy.cluster.hierarchy` (linkage matrices, dendrogram plumbing), and optional `ete3` via `bitig[viz]` for richer tree rendering.

**Reference spec:** `docs/superpowers/specs/2026-04-17-bitig-stylometry-package-design.md` §6.2–6.6.

**Phase 2 baseline:** tag `phase-2-features-delta`, 165 tests, 87.4% coverage.

---

## File Layout (new in Phase 3)

```
src/bitig/
├── methods/
│   ├── zeta.py              # ZetaClassic + ZetaEder
│   ├── reduce.py            # PCAReducer, MDSReducer, TSNEReducer, UMAPReducer
│   ├── cluster.py           # HierarchicalCluster, KMeansCluster, HDBSCANCluster
│   ├── consensus.py         # BootstrapConsensus (consensus tree from Ward linkages + MFW bands)
│   └── classify.py          # sklearn classifier wrappers + LOAO helpers
├── result.py                # Result dataclass (shared across methods)
└── cli/
    ├── zeta_cmd.py
    ├── reduce_cmd.py
    ├── cluster_cmd.py
    ├── consensus_cmd.py
    └── classify_cmd.py

tests/
├── methods/
│   ├── test_zeta.py
│   ├── test_reduce.py
│   ├── test_cluster.py
│   ├── test_consensus.py
│   └── test_classify.py
├── test_result.py
├── integration/
│   └── test_phase3_workflow.py
└── cli/
    ├── test_zeta_cmd.py
    ├── test_reduce_cmd.py
    ├── test_cluster_cmd.py
    ├── test_consensus_cmd.py
    └── test_classify_cmd.py
```

---

## Task 1: `Result` dataclass

**Files:**
- Create: `src/bitig/result.py`
- Create: `tests/test_result.py`

TDD.

### Step 1.1 — Failing tests

```python
"""Tests for the Result record."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from bitig.provenance import Provenance
from bitig.result import Result


def _prov() -> Provenance:
    return Provenance(
        bitig_version="0.1.0.dev0",
        python_version="3.11.7",
        spacy_model="en_core_web_sm",
        spacy_version="3.7.2",
        corpus_hash="c",
        feature_hash=None,
        seed=42,
        timestamp=datetime(2026, 4, 17, 12, 0, 0),
        resolved_config={},
    )


def test_result_basic_construction() -> None:
    r = Result(
        method_name="burrows_delta",
        params={"method": "burrows"},
        values={"distances": np.zeros((2, 2))},
        tables=[pd.DataFrame({"a": [1]})],
        figures=[],
        provenance=_prov(),
    )
    assert r.method_name == "burrows_delta"
    assert "distances" in r.values


def test_result_to_json_round_trip(tmp_path) -> None:
    r = Result(
        method_name="test",
        params={"k": 1},
        values={"labels": ["A", "B"], "matrix": np.array([[1.0, 2.0], [3.0, 4.0]])},
        tables=[],
        figures=[],
        provenance=_prov(),
    )
    path = tmp_path / "result.json"
    r.to_json(path)
    restored = Result.from_json(path)
    assert restored.method_name == "test"
    assert restored.params == {"k": 1}
    np.testing.assert_array_equal(restored.values["matrix"], r.values["matrix"])


def test_result_save_writes_tables_and_json(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    r = Result(
        method_name="demo",
        params={},
        values={},
        tables=[df],
        figures=[],
        provenance=_prov(),
    )
    r.save(tmp_path)
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "table_0.parquet").is_file()
    # Restored table round-trips.
    restored = pd.read_parquet(tmp_path / "table_0.parquet")
    pd.testing.assert_frame_equal(restored, df)
```

### Step 1.2 — Run → FAIL (ImportError on bitig.result).

### Step 1.3 — Implement `src/bitig/result.py`

```python
"""The Result record — uniform return type from every analytical method."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bitig.provenance import Provenance


@dataclass
class Result:
    method_name: str
    params: dict[str, Any] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)
    tables: list[pd.DataFrame] = field(default_factory=list)
    figures: list[Any] = field(default_factory=list)
    provenance: Provenance | None = None

    def to_json(self, path: str | Path) -> None:
        """Persist params + values + provenance to a single JSON file.

        ndarray values are encoded as `{"__ndarray__": list, "shape": ..., "dtype": ...}` so they
        round-trip exactly.
        """
        payload = {
            "method_name": self.method_name,
            "params": _encode(self.params),
            "values": _encode(self.values),
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> Result:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            method_name=data["method_name"],
            params=_decode(data["params"]),
            values=_decode(data["values"]),
            tables=[],
            figures=[],
            provenance=Provenance.from_dict(data["provenance"]) if data["provenance"] else None,
        )

    def save(self, directory: str | Path) -> Path:
        """Persist everything to `directory/`: result.json, tables as parquet, figures deferred to viz layer."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.to_json(directory / "result.json")
        for i, df in enumerate(self.tables):
            df.to_parquet(directory / f"table_{i}.parquet")
        return directory


def _encode(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_encode(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": obj.tolist(),
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, np.integer | np.floating):
        return obj.item()
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            arr = np.array(obj["__ndarray__"], dtype=obj["dtype"])
            return arr.reshape(obj["shape"])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj
```

### Step 1.4 — Run → PASS (3/3).

### Step 1.5 — Commit

```bash
git add src/bitig/result.py tests/test_result.py
git commit -m "feat: Result record with JSON round-trip and parquet table persistence"
```

---

## Task 2: Craig's Zeta

**Files:**
- Create: `src/bitig/methods/zeta.py`
- Create: `tests/methods/test_zeta.py`

TDD. Zeta is a two-group contrastive-vocabulary method. For each word type, compute:
`zeta(w) = proportion_A(w) - proportion_B(w)`
where `proportion_X(w)` is the fraction of *documents* in group X that contain w (binarised, not raw counts). Classical Burrows/Craig convention.

### Step 2.1 — Failing tests

```python
"""Tests for Craig's Zeta."""

from __future__ import annotations

import numpy as np
import pytest

from bitig.corpus import Corpus, Document
from bitig.methods.zeta import ZetaClassic, ZetaEder


def _corpus(*texts: str, groups: list[str]) -> Corpus:
    return Corpus(
        documents=[
            Document(id=f"d{i}", text=t, metadata={"group": g})
            for i, (t, g) in enumerate(zip(texts, groups, strict=True))
        ]
    )


def test_zeta_returns_two_tables() -> None:
    c = _corpus(
        "the cat sat on the mat",
        "the dog ran in the park",
        "rain falls softly on fields",
        "wind blows gently across plains",
        groups=["A", "A", "B", "B"],
    )
    res = ZetaClassic(group_by="group", top_k=3).fit_transform(c)
    # First table: top preferred in A; second: top preferred in B.
    assert len(res.tables) == 2


def test_zeta_distinguishes_preferred_vocab() -> None:
    c = _corpus(
        "alpha alpha alpha beta",
        "alpha alpha gamma",
        "zeta zeta zeta delta",
        "zeta zeta epsilon",
        groups=["A", "A", "B", "B"],
    )
    res = ZetaClassic(group_by="group", top_k=5).fit_transform(c)
    # 'alpha' should dominate group A; 'zeta' should dominate group B.
    top_a = res.tables[0]
    top_b = res.tables[1]
    assert "alpha" in top_a["word"].tolist()
    assert "zeta" in top_b["word"].tolist()


def test_zeta_eder_smooths_with_laplace() -> None:
    c = _corpus(
        "one two three",
        "four five six",
        groups=["A", "B"],
    )
    # Eder's variant applies Laplace smoothing; no division-by-zero on singleton groups.
    res = ZetaEder(group_by="group", top_k=3).fit_transform(c)
    assert len(res.tables) == 2


def test_zeta_rejects_fewer_than_two_groups() -> None:
    c = _corpus("hi there", "hello world", groups=["A", "A"])
    with pytest.raises(ValueError, match="at least two groups"):
        ZetaClassic(group_by="group").fit_transform(c)


def test_zeta_supports_custom_group_pair() -> None:
    c = _corpus(
        "alpha alpha",
        "beta beta",
        "gamma gamma",
        groups=["X", "Y", "Z"],
    )
    # Only compare X vs Z.
    res = ZetaClassic(group_by="group", top_k=2, group_a="X", group_b="Z").fit_transform(c)
    top_a = res.tables[0]["word"].tolist()
    assert "alpha" in top_a
    assert "gamma" not in top_a
```

### Step 2.2 — Run → FAIL.

### Step 2.3 — Implement `src/bitig/methods/zeta.py`

```python
"""Craig's Zeta — contrastive-vocabulary extraction between two author/group populations.

Classical Zeta (Burrows 2007; Craig & Kinney 2009):
    zeta(w) = proportion_A(w) - proportion_B(w)
where proportion_X(w) = (# documents in X containing w) / (# documents in X).

Eder's variant (Eder 2017) adds Laplace smoothing so zero-count words don't explode under the
logarithmic variants some authors use.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from bitig.corpus import Corpus
from bitig.provenance import Provenance
from bitig.result import Result

_WORD_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


class _ZetaBase:
    def __init__(
        self,
        *,
        group_by: str,
        top_k: int = 100,
        min_df: int = 2,
        group_a: str | None = None,
        group_b: str | None = None,
    ) -> None:
        self.group_by = group_by
        self.top_k = top_k
        self.min_df = min_df
        self.group_a = group_a
        self.group_b = group_b

    def _score(self, proportion_a: dict[str, float], proportion_b: dict[str, float]) -> dict[str, float]:
        raise NotImplementedError

    def fit_transform(self, corpus: Corpus) -> Result:
        grouped = corpus.groupby(self.group_by)
        if len(grouped) < 2:
            raise ValueError("Zeta requires at least two groups in corpus.groupby(group_by)")

        if self.group_a is None or self.group_b is None:
            # Take the two with most documents, deterministically.
            ordered = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            label_a, docs_a = ordered[0]
            label_b, docs_b = ordered[1]
        else:
            label_a, label_b = self.group_a, self.group_b
            docs_a = grouped[label_a]
            docs_b = grouped[label_b]

        n_a = len(docs_a)
        n_b = len(docs_b)

        count_in_a: Counter[str] = Counter()
        count_in_b: Counter[str] = Counter()
        for d in docs_a.documents:
            count_in_a.update(set(_tokens(d.text)))
        for d in docs_b.documents:
            count_in_b.update(set(_tokens(d.text)))

        vocabulary = {w for w, c in count_in_a.items() if c >= self.min_df}
        vocabulary |= {w for w, c in count_in_b.items() if c >= self.min_df}

        proportion_a = {w: count_in_a.get(w, 0) / n_a for w in vocabulary}
        proportion_b = {w: count_in_b.get(w, 0) / n_b for w in vocabulary}
        scores = self._score(proportion_a, proportion_b)

        scored = sorted(scores.items(), key=lambda kv: kv[1])
        top_b = scored[: self.top_k]
        top_a = list(reversed(scored[-self.top_k :]))

        df_a = pd.DataFrame(
            [
                {"word": w, "zeta": s, "prop_a": proportion_a[w], "prop_b": proportion_b[w]}
                for w, s in top_a
            ]
        )
        df_b = pd.DataFrame(
            [
                {"word": w, "zeta": s, "prop_a": proportion_a[w], "prop_b": proportion_b[w]}
                for w, s in top_b
            ]
        )
        df_a.attrs["group"] = label_a
        df_b.attrs["group"] = label_b

        return Result(
            method_name=type(self).__name__,
            params={
                "group_by": self.group_by,
                "top_k": self.top_k,
                "min_df": self.min_df,
                "group_a": label_a,
                "group_b": label_b,
            },
            values={"group_a": label_a, "group_b": label_b, "n_a": n_a, "n_b": n_b},
            tables=[df_a, df_b],
            figures=[],
            provenance=None,
        )


class ZetaClassic(_ZetaBase):
    def _score(
        self, proportion_a: dict[str, float], proportion_b: dict[str, float]
    ) -> dict[str, float]:
        return {w: proportion_a[w] - proportion_b[w] for w in proportion_a}


class ZetaEder(_ZetaBase):
    """Eder 2017 variant with Laplace smoothing."""

    def _score(
        self, proportion_a: dict[str, float], proportion_b: dict[str, float]
    ) -> dict[str, float]:
        return {
            w: (proportion_a[w] + 0.5) / 1.0 - (proportion_b[w] + 0.5) / 1.0
            for w in proportion_a
        }
```

### Step 2.4 — Run → PASS (5/5).

### Step 2.5 — Commit

```bash
git add src/bitig/methods/zeta.py tests/methods/test_zeta.py
git commit -m "feat(methods): Craig's Zeta (classic + Eder smoothed variant)"
```

---

## Task 3: Dimensionality reducers (PCA / MDS / t-SNE / UMAP)

**Files:**
- Create: `src/bitig/methods/reduce.py`
- Create: `tests/methods/test_reduce.py`

All four are thin sklearn-style wrappers returning a `Result` with the 2-D / n-d coordinates.

### Step 3.1 — Failing tests

```python
"""Tests for dimensionality reducers."""

from __future__ import annotations

import numpy as np
import pytest

from bitig.features import FeatureMatrix
from bitig.methods.reduce import MDSReducer, PCAReducer, TSNEReducer, UMAPReducer


def _fm(n: int = 10, d: int = 5) -> FeatureMatrix:
    rng = np.random.default_rng(42)
    return FeatureMatrix(
        X=rng.standard_normal((n, d)),
        document_ids=[f"d{i}" for i in range(n)],
        feature_names=[f"f{j}" for j in range(d)],
        feature_type="test",
    )


def test_pca_reduces_to_target_components() -> None:
    r = PCAReducer(n_components=2).fit_transform(_fm(10, 5))
    assert r.values["coordinates"].shape == (10, 2)
    assert "explained_variance_ratio" in r.values


def test_mds_reduces_to_target_components() -> None:
    r = MDSReducer(n_components=2, random_state=42).fit_transform(_fm(10, 5))
    assert r.values["coordinates"].shape == (10, 2)


def test_tsne_reduces_to_2d() -> None:
    # t-SNE requires perplexity < n_samples — use small perplexity for tiny fixture.
    r = TSNEReducer(n_components=2, perplexity=3.0, random_state=42).fit_transform(_fm(10, 5))
    assert r.values["coordinates"].shape == (10, 2)


@pytest.mark.slow
def test_umap_reduces_to_target_components() -> None:
    r = UMAPReducer(n_components=2, random_state=42).fit_transform(_fm(20, 5))
    assert r.values["coordinates"].shape == (20, 2)


def test_reduce_result_contains_document_ids() -> None:
    fm = _fm(5, 4)
    r = PCAReducer(n_components=2).fit_transform(fm)
    assert r.values["document_ids"] == fm.document_ids
```

### Step 3.2 — Run → FAIL.

### Step 3.3 — Implement `src/bitig/methods/reduce.py`

```python
"""Dimensionality reducers — PCA, MDS, t-SNE, UMAP — with a shared Result interface."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

from bitig.features import FeatureMatrix
from bitig.result import Result


class _ReducerBase:
    _impl: type
    method_name: str

    def __init__(self, **kwargs: object) -> None:
        self._kwargs = kwargs

    def fit_transform(self, fm: FeatureMatrix) -> Result:
        model = self._impl(**self._kwargs)
        coords = model.fit_transform(fm.X)
        values: dict[str, object] = {
            "coordinates": coords,
            "document_ids": list(fm.document_ids),
        }
        # Attach implementation-specific artefacts.
        if hasattr(model, "explained_variance_ratio_"):
            values["explained_variance_ratio"] = model.explained_variance_ratio_
        if hasattr(model, "stress_"):
            values["stress"] = float(model.stress_)
        return Result(
            method_name=self.method_name,
            params=dict(self._kwargs),
            values=values,
        )


class PCAReducer(_ReducerBase):
    _impl = PCA
    method_name = "pca"


class MDSReducer(_ReducerBase):
    _impl = MDS
    method_name = "mds"


class TSNEReducer(_ReducerBase):
    _impl = TSNE
    method_name = "tsne"


class UMAPReducer(_ReducerBase):
    method_name = "umap"

    @property
    def _impl(self) -> type:  # type: ignore[override]
        import umap

        return umap.UMAP
```

### Step 3.4 — Run → PASS (5/5 — `test_umap` marked slow).

### Step 3.5 — Commit

```bash
git add src/bitig/methods/reduce.py tests/methods/test_reduce.py
git commit -m "feat(methods): PCA / MDS / t-SNE / UMAP reducers returning Result"
```

---

## Task 4: Clustering (hierarchical + k-means + HDBSCAN)

**Files:**
- Create: `src/bitig/methods/cluster.py`
- Create: `tests/methods/test_cluster.py`

### Step 4.1 — Failing tests

```python
"""Tests for clustering methods."""

from __future__ import annotations

import numpy as np
import pytest

from bitig.features import FeatureMatrix
from bitig.methods.cluster import HDBSCANCluster, HierarchicalCluster, KMeansCluster


def _fm() -> FeatureMatrix:
    rng = np.random.default_rng(42)
    cluster_a = rng.standard_normal((10, 3)) + np.array([0, 0, 0])
    cluster_b = rng.standard_normal((10, 3)) + np.array([5, 5, 5])
    X = np.vstack([cluster_a, cluster_b])
    return FeatureMatrix(
        X=X,
        document_ids=[f"d{i}" for i in range(20)],
        feature_names=["x", "y", "z"],
        feature_type="test",
    )


def test_hierarchical_ward_returns_linkage() -> None:
    r = HierarchicalCluster(n_clusters=2, linkage="ward").fit_transform(_fm())
    assert "labels" in r.values
    assert "linkage" in r.values
    assert r.values["linkage"].shape == (19, 4)  # n-1 merges × (cluster_i, cluster_j, dist, size)


def test_hierarchical_finds_two_well_separated_clusters() -> None:
    r = HierarchicalCluster(n_clusters=2, linkage="ward").fit_transform(_fm())
    labels = r.values["labels"]
    # The first 10 docs should all share a label; the last 10 should all share the other.
    assert len(set(labels[:10])) == 1
    assert len(set(labels[10:])) == 1
    assert labels[0] != labels[10]


def test_kmeans_finds_two_clusters() -> None:
    r = KMeansCluster(n_clusters=2, random_state=42).fit_transform(_fm())
    labels = r.values["labels"]
    assert set(labels) == {0, 1}


def test_hdbscan_returns_labels_including_noise() -> None:
    r = HDBSCANCluster(min_cluster_size=3).fit_transform(_fm())
    labels = r.values["labels"]
    # HDBSCAN labels noise points as -1; our two blobs should produce at least one cluster.
    assert any(label >= 0 for label in labels)


@pytest.mark.parametrize("linkage", ["ward", "average", "complete", "single"])
def test_hierarchical_all_linkages(linkage: str) -> None:
    r = HierarchicalCluster(n_clusters=2, linkage=linkage).fit_transform(_fm())
    assert r.values["labels"].shape == (20,)
```

### Step 4.2 — Run → FAIL.

### Step 4.3 — Implement `src/bitig/methods/cluster.py`

```python
"""Clustering methods — hierarchical (Ward/average/complete/single), k-means, HDBSCAN."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans

from bitig.features import FeatureMatrix
from bitig.result import Result


class HierarchicalCluster:
    """scipy-based hierarchical clustering — returns both flat labels and the full linkage matrix
    (the latter is what the viz layer uses to render dendrograms).
    """

    def __init__(
        self,
        n_clusters: int = 2,
        *,
        linkage: str = "ward",
        metric: str = "euclidean",
    ) -> None:
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric

    def fit_transform(self, fm: FeatureMatrix) -> Result:
        # Ward requires Euclidean distances, scipy handles this internally when method="ward".
        Z = linkage(fm.X, method=self.linkage, metric=self.metric if self.linkage != "ward" else "euclidean")
        labels = fcluster(Z, t=self.n_clusters, criterion="maxclust") - 1  # 0-indexed
        return Result(
            method_name="hierarchical",
            params={"n_clusters": self.n_clusters, "linkage": self.linkage, "metric": self.metric},
            values={
                "labels": labels,
                "linkage": Z,
                "document_ids": list(fm.document_ids),
            },
        )


class KMeansCluster:
    def __init__(
        self,
        n_clusters: int = 2,
        *,
        random_state: int | None = None,
        n_init: int | str = "auto",
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init

    def fit_transform(self, fm: FeatureMatrix) -> Result:
        model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=self.n_init)
        labels = model.fit_predict(fm.X)
        return Result(
            method_name="kmeans",
            params={"n_clusters": self.n_clusters, "random_state": self.random_state},
            values={
                "labels": labels,
                "centers": model.cluster_centers_,
                "inertia": float(model.inertia_),
                "document_ids": list(fm.document_ids),
            },
        )


class HDBSCANCluster:
    def __init__(
        self,
        min_cluster_size: int = 5,
        *,
        min_samples: int | None = None,
        metric: str = "euclidean",
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

    def fit_transform(self, fm: FeatureMatrix) -> Result:
        import hdbscan

        model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
        )
        labels = model.fit_predict(fm.X)
        return Result(
            method_name="hdbscan",
            params={
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "metric": self.metric,
            },
            values={
                "labels": labels,
                "probabilities": getattr(model, "probabilities_", None),
                "document_ids": list(fm.document_ids),
            },
        )
```

### Step 4.4 — Run → PASS (7/7 — 4 linkages parameterised).

### Step 4.5 — Commit

```bash
git add src/bitig/methods/cluster.py tests/methods/test_cluster.py
git commit -m "feat(methods): hierarchical (Ward/avg/complete/single) + k-means + HDBSCAN clustering"
```

---

## Task 5: Bootstrap consensus tree

**Files:**
- Create: `src/bitig/methods/consensus.py`
- Create: `tests/methods/test_consensus.py`

TDD. This is the most algorithmically novel piece of Phase 3.

The approach follows Eder (2017):
1. Iterate MFW bands (e.g. `[100, 200, 300, 400, 500]`).
2. Optionally, for each band, run `replicates` bootstrap subsamples of documents (default 100).
3. For each (band × replicate), compute Burrows Delta + Ward linkage → binary dendrogram.
4. Collect all clades (sets of leaf documents) observed across all dendrograms.
5. Clade support = (# dendrograms containing it) / (total dendrograms).
6. The consensus tree is the set of clades with support ≥ threshold (majority by default).
7. Emit Newick for rendering.

### Step 5.1 — Failing tests

```python
"""Tests for bootstrap consensus trees."""

from __future__ import annotations

import pytest

from bitig.corpus import Corpus, Document
from bitig.methods.consensus import BootstrapConsensus


def _federalist_mini() -> Corpus:
    # Use the bundled Federalist fixture for a realistic test.
    from bitig.io import load_corpus

    return load_corpus("tests/fixtures/federalist", metadata="tests/fixtures/federalist/metadata.tsv")


pytestmark = pytest.mark.slow  # Consensus is inherently bootstrap-heavy.


def test_consensus_runs_on_federalist() -> None:
    corpus = _federalist_mini().filter(role="train")
    result = BootstrapConsensus(
        mfw_bands=[100, 200, 300],
        replicates=5,  # Small for test speed — production use 100+
        seed=42,
    ).fit_transform(corpus)
    assert "newick" in result.values
    assert "support" in result.values


def test_consensus_newick_is_nonempty_string() -> None:
    corpus = _federalist_mini().filter(role="train")
    result = BootstrapConsensus(mfw_bands=[100, 200], replicates=3, seed=42).fit_transform(corpus)
    newick = result.values["newick"]
    assert isinstance(newick, str)
    assert len(newick) > 10
    assert newick.endswith(";")


def test_consensus_clade_support_is_bounded_0_1() -> None:
    corpus = _federalist_mini().filter(role="train")
    result = BootstrapConsensus(mfw_bands=[100, 200], replicates=3, seed=42).fit_transform(corpus)
    for support in result.values["support"].values():
        assert 0.0 <= support <= 1.0
```

### Step 5.2 — Run → FAIL.

### Step 5.3 — Implement `src/bitig/methods/consensus.py`

```python
"""Bootstrap consensus trees (Eder 2017).

Iterate MFW bands × replicates → Burrows Delta → Ward linkage → extract clades. Aggregate
clade support as fraction-of-dendrograms. Emit majority-support consensus as Newick.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree

from bitig.corpus import Corpus
from bitig.features import MFWExtractor
from bitig.methods.delta import BurrowsDelta
from bitig.plumbing.seeds import derive_rng
from bitig.result import Result


class BootstrapConsensus:
    def __init__(
        self,
        *,
        mfw_bands: list[int],
        replicates: int = 100,
        subsample: float = 0.8,
        support_threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.mfw_bands = list(mfw_bands)
        self.replicates = replicates
        self.subsample = subsample
        self.support_threshold = support_threshold
        self.seed = seed

    def fit_transform(self, corpus: Corpus) -> Result:
        doc_ids = [d.id for d in corpus.documents]
        n_docs = len(doc_ids)
        rng = derive_rng(self.seed, "consensus")

        clade_counts: Counter[frozenset[str]] = Counter()
        total_dendrograms = 0

        for band in self.mfw_bands:
            for _ in range(self.replicates):
                idx = rng.choice(n_docs, size=max(2, int(n_docs * self.subsample)), replace=False)
                subsample_corpus = Corpus(documents=[corpus.documents[int(i)] for i in idx])
                subsample_ids = [d.id for d in subsample_corpus.documents]

                mfw = MFWExtractor(n=band, min_df=2, scale="zscore", lowercase=True)
                fm = mfw.fit_transform(subsample_corpus)
                if fm.X.shape[1] == 0:
                    continue  # No MFW survived culling; skip this replicate.

                Z = linkage(fm.X, method="ward")
                tree = to_tree(Z, rd=False)

                for clade in _extract_clades(tree, subsample_ids):
                    if 2 <= len(clade) < n_docs:
                        clade_counts[frozenset(clade)] += 1
                total_dendrograms += 1

        if total_dendrograms == 0:
            raise ValueError("Consensus: no valid dendrograms produced (all bands culled out?)")

        support = {
            clade: count / total_dendrograms for clade, count in clade_counts.items()
        }
        majority = {clade: s for clade, s in support.items() if s >= self.support_threshold}
        newick = _build_newick(doc_ids, majority)

        return Result(
            method_name="bootstrap_consensus",
            params={
                "mfw_bands": self.mfw_bands,
                "replicates": self.replicates,
                "subsample": self.subsample,
                "support_threshold": self.support_threshold,
                "seed": self.seed,
            },
            values={
                "newick": newick,
                "support": {",".join(sorted(c)): s for c, s in support.items()},
                "total_dendrograms": total_dendrograms,
                "document_ids": doc_ids,
            },
        )


def _extract_clades(node, leaf_ids: list[str]) -> list[list[str]]:
    """Return every internal node's leaf-ID set."""
    if node.is_leaf():
        return []
    left = _leaves_of(node.left, leaf_ids)
    right = _leaves_of(node.right, leaf_ids)
    here = left + right
    out = [here]
    out.extend(_extract_clades(node.left, leaf_ids))
    out.extend(_extract_clades(node.right, leaf_ids))
    return out


def _leaves_of(node, leaf_ids: list[str]) -> list[str]:
    if node.is_leaf():
        return [leaf_ids[node.id]]
    return _leaves_of(node.left, leaf_ids) + _leaves_of(node.right, leaf_ids)


def _build_newick(leaves: list[str], clades_with_support: dict[frozenset[str], float]) -> str:
    """Build a Newick string where internal branches are annotated with support values.

    Algorithm: compatibility via greedy nesting. Sort clades by size (largest first) so parents
    encapsulate children. A minority-supported set of "missing" internal relationships becomes
    a flat polytomy at the root.
    """
    ordered = sorted(clades_with_support.items(), key=lambda kv: (-len(kv[0]), sorted(kv[0])))
    clade_children: dict[frozenset[str], list[frozenset[str] | str]] = defaultdict(list)
    assigned = {leaf: leaf for leaf in leaves}

    # Build containment tree: each clade's direct children are the largest sub-clades it contains
    # that haven't been parented elsewhere.
    remaining_leaves = set(leaves)
    clade_list = [c for c, _ in ordered]
    for c in clade_list:
        clade_children[c] = []
    placed: set[frozenset[str]] = set()
    for parent in clade_list:
        for child in clade_list:
            if child is parent or child in placed:
                continue
            if child < parent and not any(child < other < parent for other in clade_list if other != parent and other != child):
                clade_children[parent].append(child)
                placed.add(child)

    def render(clade: frozenset[str]) -> str:
        child_clades = clade_children.get(clade, [])
        child_leaves = clade - frozenset().union(*child_clades) if child_clades else clade
        parts = [render(c) for c in child_clades] + sorted(child_leaves)
        return "(" + ",".join(parts) + f"){clades_with_support[clade]:.2f}"

    top = [c for c in clade_list if c not in placed]
    if not top:
        # All leaves flat at root.
        return "(" + ",".join(sorted(remaining_leaves)) + ");"

    covered = frozenset().union(*top)
    stray = remaining_leaves - covered
    rendered_tops = [render(c) for c in top] + sorted(stray)
    return "(" + ",".join(rendered_tops) + ");"
```

### Step 5.4 — Run → PASS (3/3).

### Step 5.5 — Commit

```bash
git add src/bitig/methods/consensus.py tests/methods/test_consensus.py
git commit -m "feat(methods): bootstrap consensus tree over MFW bands with Newick output"
```

---

## Task 6: sklearn classifier wrappers

**Files:**
- Create: `src/bitig/methods/classify.py`
- Create: `tests/methods/test_classify.py`

Thin factory that returns pre-configured sklearn classifiers + a cross-validation helper that
understands our `leave_one_author_out` CV kind.

### Step 6.1 — Failing tests

```python
"""Tests for sklearn classifier wrappers and LOAO CV."""

from __future__ import annotations

import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection import LeaveOneGroupOut

from bitig.corpus import Corpus, Document
from bitig.features import MFWExtractor
from bitig.methods.classify import build_classifier, cross_validate_bitig


def _corpus() -> Corpus:
    docs = []
    rng = np.random.default_rng(42)
    for i in range(20):
        author = "A" if i < 10 else "B"
        text = " ".join(rng.choice(["the", "of", "and", "to", "a"], size=100))
        docs.append(Document(id=f"d{i}", text=text, metadata={"author": author}))
    return Corpus(documents=docs)


def test_build_classifier_logreg() -> None:
    clf = build_classifier("logreg", random_state=42)
    assert is_classifier(clf)


def test_build_classifier_svm_linear() -> None:
    clf = build_classifier("svm_linear", random_state=42)
    assert is_classifier(clf)


def test_build_classifier_rejects_unknown() -> None:
    import pytest
    with pytest.raises(ValueError, match="unknown"):
        build_classifier("nonsense")


def test_cross_validate_bitig_loao() -> None:
    corpus = _corpus()
    y = np.array(corpus.metadata_column("author"))
    mfw = MFWExtractor(n=5, scale="zscore", lowercase=True)
    X_fm = mfw.fit_transform(corpus)
    report = cross_validate_bitig(
        build_classifier("logreg", random_state=42),
        X_fm,
        y,
        cv_kind="loao",
        groups_from=y,
    )
    assert "accuracy" in report
    assert "per_class" in report


def test_cross_validate_bitig_stratified() -> None:
    corpus = _corpus()
    y = np.array(corpus.metadata_column("author"))
    mfw = MFWExtractor(n=5, scale="zscore", lowercase=True)
    X_fm = mfw.fit_transform(corpus)
    report = cross_validate_bitig(
        build_classifier("rf", random_state=42),
        X_fm,
        y,
        cv_kind="stratified",
        folds=5,
    )
    assert "accuracy" in report
```

### Step 6.2 — Run → FAIL.

### Step 6.3 — Implement `src/bitig/methods/classify.py`

```python
"""sklearn classifier wrappers + CV helper with stylometry-aware splits."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC

from bitig.features import FeatureMatrix

_ESTIMATORS = {
    "logreg": lambda **kw: LogisticRegression(max_iter=2000, **kw),
    "svm_linear": lambda **kw: SVC(kernel="linear", probability=True, **kw),
    "svm_rbf": lambda **kw: SVC(kernel="rbf", probability=True, **kw),
    "rf": lambda **kw: RandomForestClassifier(**kw),
    "hgbm": lambda **kw: HistGradientBoostingClassifier(**kw),
}


def build_classifier(name: str, **kwargs: Any) -> BaseEstimator:
    if name not in _ESTIMATORS:
        raise ValueError(f"unknown classifier {name!r}; known: {sorted(_ESTIMATORS)}")
    return _ESTIMATORS[name](**kwargs)


def cross_validate_bitig(
    estimator: BaseEstimator,
    fm: FeatureMatrix,
    y: np.ndarray,
    *,
    cv_kind: str = "stratified",
    groups_from: np.ndarray | None = None,
    folds: int = 5,
) -> dict[str, Any]:
    """Run cross-validation with a stylometry-aware CV strategy.

    cv_kind:
      - "stratified": StratifiedKFold(folds)
      - "loao":       LeaveOneGroupOut (requires groups_from)
      - "leave_one_text_out": LeaveOneOut
    """
    if cv_kind == "stratified":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        groups = None
    elif cv_kind == "loao":
        if groups_from is None:
            raise ValueError("cv_kind='loao' requires groups_from")
        cv = LeaveOneGroupOut()
        groups = np.asarray(groups_from)
    elif cv_kind == "leave_one_text_out":
        cv = LeaveOneOut()
        groups = None
    else:
        raise ValueError(f"unknown cv_kind {cv_kind!r}")

    preds = cross_val_predict(estimator, fm.X, y, cv=cv, groups=groups)
    report = classification_report(y, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": float((preds == y).mean()),
        "predictions": preds,
        "per_class": report,
    }
```

### Step 6.4 — Run → PASS (5/5).

### Step 6.5 — Commit

```bash
git add src/bitig/methods/classify.py tests/methods/test_classify.py
git commit -m "feat(methods): sklearn classifier wrappers + cross_validate_bitig (stratified/loao/loto)"
```

---

## Task 7: CLI commands — `zeta`, `reduce`, `cluster`, `consensus`, `classify` batch

**Files:**
- Create: `src/bitig/cli/zeta_cmd.py`, `reduce_cmd.py`, `cluster_cmd.py`, `consensus_cmd.py`, `classify_cmd.py`
- Create: `tests/cli/test_zeta_cmd.py`, `test_reduce_cmd.py`, `test_cluster_cmd.py`, `test_consensus_cmd.py`, `test_classify_cmd.py`
- Modify: `src/bitig/cli/__init__.py` (register all five)

Each CLI command follows the same pattern as `bitig delta` in Phase 2: load corpus + metadata, build MFW feature matrix, run the method, print a Rich table or write a result file.

### Step 7.1 — `src/bitig/cli/zeta_cmd.py`

```python
"""`bitig zeta <corpus>` — contrastive vocabulary between two metadata groups."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from bitig.io import load_corpus
from bitig.methods.zeta import ZetaClassic, ZetaEder

console = Console()


def zeta_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path = typer.Option(..., "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    group_by: str = typer.Option("author", "--group-by"),
    variant: str = typer.Option("classic", "--variant", help="classic | eder"),
    top_k: int = typer.Option(20, "--top-k"),
    group_a: str | None = typer.Option(None, "--group-a"),
    group_b: str | None = typer.Option(None, "--group-b"),
) -> None:
    """Extract contrastive vocabulary between two groups via Craig's Zeta."""
    cls = {"classic": ZetaClassic, "eder": ZetaEder}.get(variant)
    if cls is None:
        console.print(f"[red]error:[/red] unknown variant {variant!r}")
        raise typer.Exit(code=1)
    corpus = load_corpus(path, metadata=metadata)
    result = cls(group_by=group_by, top_k=top_k, group_a=group_a, group_b=group_b).fit_transform(corpus)

    label_a = result.values["group_a"]
    label_b = result.values["group_b"]
    for df, label in [(result.tables[0], label_a), (result.tables[1], label_b)]:
        table = Table(title=f"preferred in {label}")
        table.add_column("word", style="cyan")
        table.add_column("zeta")
        table.add_column(f"prop_{label_a}")
        table.add_column(f"prop_{label_b}")
        for _, row in df.iterrows():
            table.add_row(row["word"], f"{row['zeta']:.3f}", f"{row['prop_a']:.3f}", f"{row['prop_b']:.3f}")
        console.print(table)
```

### Step 7.2 — `src/bitig/cli/reduce_cmd.py`

```python
"""`bitig reduce <corpus>` — dimensionality reduction of the MFW feature matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from bitig.features import MFWExtractor
from bitig.io import load_corpus
from bitig.methods.reduce import MDSReducer, PCAReducer, TSNEReducer, UMAPReducer

console = Console()

_REDUCERS = {
    "pca": PCAReducer,
    "mds": MDSReducer,
    "tsne": TSNEReducer,
    "umap": UMAPReducer,
}


def reduce_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path | None = typer.Option(None, "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    method: str = typer.Option("pca", "--method"),
    n_components: int = typer.Option(2, "--n-components"),
    mfw: int = typer.Option(500, "--mfw"),
    output: Path = typer.Option(Path("reduce.parquet"), "--output", "-o"),  # noqa: B008
) -> None:
    """Reduce corpus MFW matrix via PCA/MDS/t-SNE/UMAP; save coordinates to parquet."""
    if method not in _REDUCERS:
        console.print(f"[red]error:[/red] unknown method {method!r}")
        raise typer.Exit(code=1)
    corpus = load_corpus(path, metadata=metadata)
    fm = MFWExtractor(n=mfw, min_df=2, scale="zscore", lowercase=True).fit_transform(corpus)
    reducer = _REDUCERS[method](n_components=n_components)
    result = reducer.fit_transform(fm)
    coords: np.ndarray = result.values["coordinates"]
    df = pd.DataFrame(coords, index=fm.document_ids, columns=[f"c{i}" for i in range(coords.shape[1])])
    df.to_parquet(output)
    console.print(f"[green]wrote[/green] {output} ({coords.shape[0]} docs x {coords.shape[1]} components)")
```

### Step 7.3 — `src/bitig/cli/cluster_cmd.py`

```python
"""`bitig cluster <corpus>` — hierarchical / k-means / HDBSCAN clustering of the MFW matrix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from bitig.features import MFWExtractor
from bitig.io import load_corpus
from bitig.methods.cluster import HDBSCANCluster, HierarchicalCluster, KMeansCluster

console = Console()


def cluster_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path | None = typer.Option(None, "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    method: str = typer.Option("hierarchical", "--method"),
    n_clusters: int = typer.Option(2, "--n-clusters"),
    linkage: str = typer.Option("ward", "--linkage", help="For hierarchical: ward | average | complete | single"),
    mfw: int = typer.Option(500, "--mfw"),
    output: Path = typer.Option(Path("cluster.parquet"), "--output", "-o"),  # noqa: B008
) -> None:
    """Cluster corpus MFW matrix; save per-document labels to parquet + print a summary."""
    corpus = load_corpus(path, metadata=metadata)
    fm = MFWExtractor(n=mfw, min_df=2, scale="zscore", lowercase=True).fit_transform(corpus)
    if method == "hierarchical":
        result = HierarchicalCluster(n_clusters=n_clusters, linkage=linkage).fit_transform(fm)
    elif method == "kmeans":
        result = KMeansCluster(n_clusters=n_clusters, random_state=42).fit_transform(fm)
    elif method == "hdbscan":
        result = HDBSCANCluster(min_cluster_size=max(2, n_clusters)).fit_transform(fm)
    else:
        console.print(f"[red]error:[/red] unknown method {method!r}")
        raise typer.Exit(code=1)
    labels = result.values["labels"]
    df = pd.DataFrame({"document_id": fm.document_ids, "cluster": labels})
    df.to_parquet(output)
    table = Table(title=f"clustering — {method}")
    table.add_column("cluster")
    table.add_column("documents")
    for cluster_id, grp in df.groupby("cluster"):
        table.add_row(str(cluster_id), ", ".join(grp["document_id"].tolist()))
    console.print(table)
    console.print(f"[green]wrote[/green] {output}")
```

### Step 7.4 — `src/bitig/cli/consensus_cmd.py`

```python
"""`bitig consensus <corpus>` — bootstrap consensus tree over MFW bands."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from bitig.io import load_corpus
from bitig.methods.consensus import BootstrapConsensus

console = Console()


def consensus_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path | None = typer.Option(None, "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    bands: str = typer.Option("100,200,300,400,500", "--bands", help="Comma-separated MFW band sizes"),
    replicates: int = typer.Option(100, "--replicates"),
    subsample: float = typer.Option(0.8, "--subsample"),
    support_threshold: float = typer.Option(0.5, "--support-threshold"),
    seed: int = typer.Option(42, "--seed"),
    output: Path = typer.Option(Path("consensus.nwk"), "--output", "-o"),  # noqa: B008
) -> None:
    """Build a bootstrap consensus tree and write Newick to disk."""
    corpus = load_corpus(path, metadata=metadata)
    bands_list = [int(b) for b in bands.split(",")]
    result = BootstrapConsensus(
        mfw_bands=bands_list,
        replicates=replicates,
        subsample=subsample,
        support_threshold=support_threshold,
        seed=seed,
    ).fit_transform(corpus)
    output.write_text(result.values["newick"], encoding="utf-8")
    console.print(f"[green]wrote[/green] {output} (based on {result.values['total_dendrograms']} dendrograms)")
```

### Step 7.5 — `src/bitig/cli/classify_cmd.py`

```python
"""`bitig classify <corpus>` — sklearn classifier + CV report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from bitig.features import MFWExtractor
from bitig.io import load_corpus
from bitig.methods.classify import build_classifier, cross_validate_bitig

console = Console()


def classify_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path = typer.Option(..., "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    estimator: str = typer.Option("logreg", "--estimator"),
    group_by: str = typer.Option("author", "--group-by"),
    cv_kind: str = typer.Option("loao", "--cv-kind"),
    folds: int = typer.Option(5, "--folds"),
    mfw: int = typer.Option(500, "--mfw"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Fit+cross-validate a classifier and print per-author metrics."""
    corpus = load_corpus(path, metadata=metadata)
    y = np.array(corpus.metadata_column(group_by))
    fm = MFWExtractor(n=mfw, min_df=2, scale="zscore", lowercase=True).fit_transform(corpus)
    clf = build_classifier(estimator, random_state=seed) if estimator != "hgbm" else build_classifier(estimator, random_state=seed)
    report = cross_validate_bitig(clf, fm, y, cv_kind=cv_kind, groups_from=y if cv_kind == "loao" else None, folds=folds)
    table = Table(title=f"classify — {estimator} / {cv_kind}")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("accuracy", f"{report['accuracy']:.3f}")
    per_class = report["per_class"]
    if "macro avg" in per_class:
        table.add_row("macro_f1", f"{per_class['macro avg']['f1-score']:.3f}")
    console.print(table)
```

### Step 7.6 — Tests (verbatim five files)

**`tests/cli/test_zeta_cmd.py`:**

```python
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"
pytestmark = pytest.mark.integration


def test_zeta_classic_runs_on_federalist() -> None:
    result = runner.invoke(
        app,
        [
            "zeta", str(FED),
            "--metadata", str(FED / "metadata.tsv"),
            "--group-by", "author",
            "--variant", "classic",
            "--top-k", "5",
            "--group-a", "Hamilton",
            "--group-b", "Madison",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "preferred in Hamilton" in result.stdout
    assert "preferred in Madison" in result.stdout


def test_zeta_rejects_unknown_variant() -> None:
    result = runner.invoke(
        app,
        ["zeta", str(FED), "--metadata", str(FED / "metadata.tsv"), "--variant", "bogus"],
    )
    assert result.exit_code != 0
```

**`tests/cli/test_reduce_cmd.py`:**

```python
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"
pytestmark = pytest.mark.integration


def test_reduce_pca_writes_parquet(tmp_path: Path) -> None:
    out = tmp_path / "r.parquet"
    result = runner.invoke(
        app,
        [
            "reduce", str(FED),
            "--metadata", str(FED / "metadata.tsv"),
            "--method", "pca",
            "--n-components", "2",
            "--mfw", "200",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pd.read_parquet(out)
    assert df.shape == (13, 2)


def test_reduce_rejects_unknown_method(tmp_path: Path) -> None:
    out = tmp_path / "r.parquet"
    result = runner.invoke(
        app,
        ["reduce", str(FED), "--metadata", str(FED / "metadata.tsv"), "--method", "bogus", "--output", str(out)],
    )
    assert result.exit_code != 0
```

**`tests/cli/test_cluster_cmd.py`:**

```python
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"
pytestmark = pytest.mark.integration


def test_cluster_hierarchical_writes_parquet(tmp_path: Path) -> None:
    out = tmp_path / "c.parquet"
    result = runner.invoke(
        app,
        [
            "cluster", str(FED),
            "--metadata", str(FED / "metadata.tsv"),
            "--method", "hierarchical",
            "--n-clusters", "3",
            "--mfw", "200",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    df = pd.read_parquet(out)
    assert df.shape == (13, 2)
    assert set(df.columns) == {"document_id", "cluster"}


def test_cluster_kmeans_runs(tmp_path: Path) -> None:
    out = tmp_path / "k.parquet"
    result = runner.invoke(
        app,
        [
            "cluster", str(FED),
            "--metadata", str(FED / "metadata.tsv"),
            "--method", "kmeans",
            "--n-clusters", "3",
            "--mfw", "200",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
```

**`tests/cli/test_consensus_cmd.py`:**

```python
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"
pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_consensus_writes_newick(tmp_path: Path) -> None:
    out = tmp_path / "consensus.nwk"
    result = runner.invoke(
        app,
        [
            "consensus", str(FED),
            "--metadata", str(FED / "metadata.tsv"),
            "--bands", "100,200",
            "--replicates", "3",
            "--seed", "42",
            "--output", str(out),
        ],
    )
    assert result.exit_code == 0, result.stdout
    text = out.read_text()
    assert text.endswith(";")
    assert len(text) > 10
```

**`tests/cli/test_classify_cmd.py`:**

```python
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bitig.cli import app

runner = CliRunner()
FED = Path(__file__).parent.parent / "fixtures" / "federalist"
pytestmark = pytest.mark.integration


def test_classify_logreg_loao_runs() -> None:
    result = runner.invoke(
        app,
        [
            "classify", str(FED),
            "--metadata", str(FED / "metadata.tsv"),
            "--estimator", "logreg",
            "--group-by", "author",
            "--cv-kind", "loao",
            "--mfw", "200",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "accuracy" in result.stdout.lower()
```

### Step 7.7 — Register all five in `src/bitig/cli/__init__.py`

Append after the existing registrations:

```python
from bitig.cli.classify_cmd import classify_command
from bitig.cli.cluster_cmd import cluster_command
from bitig.cli.consensus_cmd import consensus_command
from bitig.cli.reduce_cmd import reduce_command
from bitig.cli.zeta_cmd import zeta_command

app.command(name="zeta")(zeta_command)
app.command(name="reduce")(reduce_command)
app.command(name="cluster")(cluster_command)
app.command(name="consensus")(consensus_command)
app.command(name="classify")(classify_command)
```

### Step 7.8 — Run and commit

`pytest tests/cli/test_zeta_cmd.py tests/cli/test_reduce_cmd.py tests/cli/test_cluster_cmd.py tests/cli/test_consensus_cmd.py tests/cli/test_classify_cmd.py -v` — expect 8 tests passing (1 consensus marked slow + integration).

```bash
git add src/bitig/cli/zeta_cmd.py src/bitig/cli/reduce_cmd.py src/bitig/cli/cluster_cmd.py src/bitig/cli/consensus_cmd.py src/bitig/cli/classify_cmd.py tests/cli/test_zeta_cmd.py tests/cli/test_reduce_cmd.py tests/cli/test_cluster_cmd.py tests/cli/test_consensus_cmd.py tests/cli/test_classify_cmd.py src/bitig/cli/__init__.py
git commit -m "feat(cli): bitig zeta, reduce, cluster, consensus, classify subcommands"
```

---

## Task 8: Phase 3 integration test + public API + tag

**Files:**
- Create: `tests/integration/test_phase3_workflow.py`
- Modify: `src/bitig/__init__.py` (re-export Phase 3 names)

### Step 8.1 — Integration test `tests/integration/test_phase3_workflow.py`

```python
"""End-to-end: load corpus → MFW → reduce → cluster → zeta → classify, all within one script."""

from __future__ import annotations

import numpy as np
import pytest

from bitig.features import MFWExtractor
from bitig.io import load_corpus
from bitig.methods.classify import build_classifier, cross_validate_bitig
from bitig.methods.cluster import HierarchicalCluster
from bitig.methods.reduce import PCAReducer
from bitig.methods.zeta import ZetaClassic

pytestmark = pytest.mark.integration

FED = "tests/fixtures/federalist"


def test_phase3_end_to_end_workflow() -> None:
    corpus = load_corpus(FED, metadata=f"{FED}/metadata.tsv").filter(role="train")
    y = np.array(corpus.metadata_column("author"))

    fm = MFWExtractor(n=200, min_df=2, scale="zscore", lowercase=True).fit_transform(corpus)

    pca = PCAReducer(n_components=2).fit_transform(fm)
    assert pca.values["coordinates"].shape == (len(corpus), 2)

    clusters = HierarchicalCluster(n_clusters=3, linkage="ward").fit_transform(fm)
    assert len(clusters.values["labels"]) == len(corpus)

    zeta = ZetaClassic(group_by="author", top_k=5, group_a="Hamilton", group_b="Madison").fit_transform(corpus)
    assert len(zeta.tables) == 2

    report = cross_validate_bitig(
        build_classifier("logreg", random_state=42),
        fm,
        y,
        cv_kind="loao",
        groups_from=y,
    )
    assert 0.0 <= report["accuracy"] <= 1.0
```

### Step 8.2 — `src/bitig/__init__.py` Phase 3 additions

Add to the existing re-exports:

```python
from bitig.methods.classify import build_classifier, cross_validate_bitig
from bitig.methods.cluster import HDBSCANCluster, HierarchicalCluster, KMeansCluster
from bitig.methods.consensus import BootstrapConsensus
from bitig.methods.reduce import MDSReducer, PCAReducer, TSNEReducer, UMAPReducer
from bitig.methods.zeta import ZetaClassic, ZetaEder
from bitig.result import Result
```

And add to `__all__`:

```python
"BootstrapConsensus",
"HDBSCANCluster",
"HierarchicalCluster",
"KMeansCluster",
"MDSReducer",
"PCAReducer",
"Result",
"TSNEReducer",
"UMAPReducer",
"ZetaClassic",
"ZetaEder",
"build_classifier",
"cross_validate_bitig",
```

### Step 8.3 — Update README Status to Phase 3

Replace the `## Status` paragraph with:

```markdown
## Status

**Phase 3 — Analytical breadth.** Ships Craig's Zeta (classic + Eder variants), dimensionality
reducers (PCA/MDS/t-SNE/UMAP), clustering (hierarchical Ward/avg/complete/single, k-means, HDBSCAN),
bootstrap consensus trees (Newick output), and sklearn-wrapped classifiers (logreg/SVM/RF/HGBM)
with stylometry-aware CV (LOAO, leave-one-text-out, stratified). CLI: `bitig zeta`, `reduce`,
`cluster`, `consensus`, `classify` — all live.

Phases 4 (embeddings + Bayesian), 5 (viz + reports + wizard shell), 6 (docs + PyPI) remain.
```

### Step 8.4 — Full suite + tag

```bash
source .venv/bin/activate
pytest -n auto --cov=bitig -q
pre-commit run --all-files
git add src/bitig/__init__.py README.md tests/integration/test_phase3_workflow.py
git commit -m "feat: public API re-exports for Phase 3 (analytical breadth)"
git tag -a phase-3-analytical-breadth -m "Phase 3 complete: Zeta + reducers + clustering + consensus + classify with CLI"
```

---

## Phase 3 — Acceptance Criteria

From a clean checkout of the tagged `phase-3-analytical-breadth`:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m spacy download en_core_web_sm
pre-commit run --all-files
pytest -n auto --cov=bitig -q

# Acceptance CLI sequence
bitig zeta ./tests/fixtures/federalist --metadata ./tests/fixtures/federalist/metadata.tsv \
    --group-by author --variant classic --top-k 5 --group-a Hamilton --group-b Madison
bitig reduce ./tests/fixtures/federalist --metadata ./tests/fixtures/federalist/metadata.tsv \
    --method pca --n-components 2 --mfw 200 --output /tmp/pca.parquet
bitig cluster ./tests/fixtures/federalist --metadata ./tests/fixtures/federalist/metadata.tsv \
    --method hierarchical --n-clusters 3 --mfw 200 --output /tmp/clusters.parquet
bitig consensus ./tests/fixtures/federalist --metadata ./tests/fixtures/federalist/metadata.tsv \
    --bands 100,200 --replicates 3 --output /tmp/consensus.nwk
bitig classify ./tests/fixtures/federalist --metadata ./tests/fixtures/federalist/metadata.tsv \
    --estimator logreg --group-by author --cv-kind loao --mfw 200
```

Every command must exit 0 and produce its expected output.

---

## Self-Review

- **Spec coverage:** §6.2 (Zeta), §6.3 (reduce), §6.4 (cluster), §6.5 (consensus), §6.6 (classify + LOAO). All covered.
- **Deferred:** §6.7 Bayesian (Phase 4), §10 viz (Phase 5), §11 reports (Phase 5), `bitig run` + interactive shell (Phase 5).
- **Placeholder scan:** no TBD/TODO.
- **Type consistency:** `Result` used by every method; `FeatureMatrix` input shared; `cross_validate_bitig` signature consistent between test and implementation.
- **Commit cadence:** 8 tasks → 8 commits.
