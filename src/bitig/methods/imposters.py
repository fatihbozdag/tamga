"""General Imposters -- Koppel & Winter (2014) authorship verification.

Bootstraps over feature subsets and asks: across many random projections of
the MFW space, how often is the target text closer to the *candidate* author
than to any *imposter* author? The aggregate score is in [0, 1]; values near
1 mean the candidate consistently wins, values near 0 mean an imposter does.

This is a *verification* method (one candidate per method config) -- it is
the natural complement to the regular delta runner branch, which is
*attribution* (pick the closest of all candidates). Targets are excluded
from the MFW vocabulary and z-score statistics so frequencies do not leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bitig.corpus import Corpus
from bitig.features.mfw import MFWExtractor, _tokenise
from bitig.methods.delta import (
    ArgamonLinearDelta,
    BurrowsDelta,
    CosineDelta,
    EderDelta,
    EderSimpleDelta,
    QuadraticDelta,
)
from bitig.result import Result

_BASE_DELTA: dict[str, type] = {
    "burrows": BurrowsDelta,
    "cosine": CosineDelta,
    "argamon_linear": ArgamonLinearDelta,
    "quadratic": QuadraticDelta,
    "eder": EderDelta,
    "eder_simple": EderSimpleDelta,
}


class GeneralImposters:
    """Koppel & Winter (2014) General Imposters verification.

    Parameters
    ----------
    target_ids : list[str]
        Document ids to verify. Each is asked whether it could plausibly be
        written by `candidate`.
    candidate : str
        The alleged author. Must appear in the `group_by` column of the
        training portion of the corpus (i.e., among non-target documents).
    group_by : str
        Metadata column naming the author of each training document.
    n_iter : int
        Number of bootstrap iterations; each samples a feature subset and
        votes on whether the candidate or some imposter is closer.
    feature_frac : float
        Fraction of MFW columns sampled per iteration. The classical GI
        value is 0.5; lower values produce noisier per-iteration votes but
        stabilise the aggregate score.
    base_delta : str
        Distance kernel; one of `burrows`, `cosine`, `argamon_linear`,
        `quadratic`, `eder`, `eder_simple`.
    mfw_n : int
        Top-N MFW vocabulary size (fit on training only).
    lowercase : bool
        Case-fold during tokenisation.
    threshold : float
        Decision cutoff; targets whose score >= threshold are reported as
        verified. Stored on Result.values so downstream code can re-decide.
    seed : int
        Seed for the per-iteration feature subsample.
    """

    def __init__(
        self,
        *,
        target_ids: list[str],
        candidate: str,
        group_by: str,
        n_iter: int = 100,
        feature_frac: float = 0.5,
        base_delta: str = "burrows",
        mfw_n: int = 200,
        lowercase: bool = True,
        threshold: float = 0.5,
        seed: int = 42,
    ) -> None:
        if base_delta not in _BASE_DELTA:
            raise ValueError(f"unknown base_delta {base_delta!r} (known: {sorted(_BASE_DELTA)})")
        if not target_ids:
            raise ValueError("general_imposters requires at least one target_id")
        if not (0.0 < feature_frac <= 1.0):
            raise ValueError("feature_frac must be in (0, 1]")
        if n_iter < 1:
            raise ValueError("n_iter must be >= 1")
        self.target_ids = list(target_ids)
        self.candidate = candidate
        self.group_by = group_by
        self.n_iter = int(n_iter)
        self.feature_frac = float(feature_frac)
        self.base_delta = base_delta
        self.mfw_n = int(mfw_n)
        self.lowercase = lowercase
        self.threshold = float(threshold)
        self.seed = int(seed)

    def fit_transform(self, corpus: Corpus) -> Result:
        target_set = set(self.target_ids)
        train_docs = [d for d in corpus.documents if d.id not in target_set]
        target_docs = [d for d in corpus.documents if d.id in target_set]
        if not train_docs:
            raise ValueError(
                "general_imposters needs at least one training document outside target_ids"
            )
        if not target_docs:
            raise ValueError(
                f"none of target_ids {self.target_ids!r} match documents in the corpus"
            )

        train_corpus = Corpus(documents=train_docs, language=corpus.language)
        mfw = MFWExtractor(n=self.mfw_n, scale="zscore", lowercase=self.lowercase)
        train_fm = mfw.fit_transform(train_corpus)
        x_train = train_fm.X
        y_train = np.array(train_corpus.metadata_column(self.group_by))
        if any(v is None for v in y_train):
            raise ValueError(f"some training documents lack metadata column {self.group_by!r}")
        authors = sorted({str(a) for a in y_train.tolist()})
        if self.candidate not in authors:
            raise ValueError(f"candidate {self.candidate!r} not in training authors {authors!r}")
        if len(authors) < 2:
            raise ValueError(
                "general_imposters needs at least 2 distinct authors in the training corpus"
            )

        # Project each target into the same MFW space (counts -> l1 -> z-score).
        # Direct internal-state access is intentional -- both classes ship in this package.
        vocab_index = {tok: i for i, tok in enumerate(mfw._vocabulary)}
        means = mfw._column_means
        stds = mfw._column_stds
        if means is None or stds is None:
            raise RuntimeError("MFW fit did not produce z-score statistics")
        target_vectors: list[np.ndarray] = []
        for doc in target_docs:
            counts = np.zeros(len(vocab_index), dtype=float)
            for tok in _tokenise(doc.text, self.lowercase):
                j = vocab_index.get(tok)
                if j is not None:
                    counts[j] += 1
            row_sum = counts.sum() or 1.0
            rel = counts / row_sum
            target_vectors.append((rel - means) / stds)

        delta_cls = _BASE_DELTA[self.base_delta]
        rng = np.random.default_rng(self.seed)
        n_features = x_train.shape[1]
        k = max(1, round(n_features * self.feature_frac))

        rows: list[dict[str, object]] = []
        for doc, tgt in zip(target_docs, target_vectors, strict=True):
            wins = 0
            for _ in range(self.n_iter):
                cols = rng.choice(n_features, size=k, replace=False)
                clf = delta_cls()
                clf.fit(x_train[:, cols], y_train)
                # decision_function returns -distance; argmax => nearest centroid.
                neg_dists = clf.decision_function(tgt[cols].reshape(1, -1))[0]
                if str(clf.classes_[int(np.argmax(neg_dists))]) == self.candidate:
                    wins += 1
            score = wins / self.n_iter
            rows.append(
                {
                    "target_id": doc.id,
                    "candidate": self.candidate,
                    "score": float(score),
                    "n_iter": self.n_iter,
                    "n_features_per_iter": int(k),
                    "verified": bool(score >= self.threshold),
                }
            )

        table = pd.DataFrame(rows)
        return Result(
            method_name=f"general_imposters_{self.base_delta}",
            params={
                "target_ids": self.target_ids,
                "candidate": self.candidate,
                "group_by": self.group_by,
                "n_iter": self.n_iter,
                "feature_frac": self.feature_frac,
                "base_delta": self.base_delta,
                "mfw_n": self.mfw_n,
                "lowercase": self.lowercase,
                "threshold": self.threshold,
                "seed": self.seed,
            },
            values={
                "candidate": self.candidate,
                "imposters": [a for a in authors if a != self.candidate],
                "threshold": self.threshold,
                "scores": {row["target_id"]: row["score"] for row in rows},
            },
            tables=[table],
        )
