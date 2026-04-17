"""Tests for Bayesian authorship attribution — requires tamga[bayesian]."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import is_classifier

from tamga.corpus import Corpus, Document
from tamga.features import MFWExtractor

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
    from tamga.methods.bayesian import BayesianAuthorshipAttributor

    assert is_classifier(BayesianAuthorshipAttributor())


def test_bayesian_attributor_separates_two_authors() -> None:
    from tamga.methods.bayesian import BayesianAuthorshipAttributor

    corpus = _corpus()
    y = np.array(corpus.metadata_column("author"))
    fm = MFWExtractor(n=5, scale="none", lowercase=True).fit_transform(corpus)
    clf = BayesianAuthorshipAttributor(prior_alpha=0.5).fit(fm, y)
    preds = clf.predict(fm)
    # On in-sample data with strong author-word associations, accuracy should be near 1.
    assert (preds == y).mean() >= 0.8


def test_bayesian_attributor_predict_proba_sums_to_one() -> None:
    from tamga.methods.bayesian import BayesianAuthorshipAttributor

    corpus = _corpus()
    y = np.array(corpus.metadata_column("author"))
    fm = MFWExtractor(n=5, scale="none", lowercase=True).fit_transform(corpus)
    clf = BayesianAuthorshipAttributor().fit(fm, y)
    probs = clf.predict_proba(fm)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)


def test_bayesian_raises_clear_error_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    from tamga.methods import bayesian

    monkeypatch.setattr(bayesian, "_pymc_available", False)
    with pytest.raises(ImportError, match=r"tamga\[bayesian\]"):
        bayesian.HierarchicalGroupComparison(group_by="author")
