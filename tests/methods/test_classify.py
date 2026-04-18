"""Tests for sklearn classifier wrappers and LOAO CV."""

from __future__ import annotations

import numpy as np
from sklearn.base import is_classifier

from tamga.corpus import Corpus, Document
from tamga.features import MFWExtractor
from tamga.methods.classify import build_classifier, cross_validate_tamga


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


def test_cross_validate_tamga_loao() -> None:
    corpus = _corpus()
    y = np.array(corpus.metadata_column("author"))
    mfw = MFWExtractor(n=5, scale="zscore", lowercase=True)
    X_fm = mfw.fit_transform(corpus)
    # Use four interleaved groups (0..3), each spanning both classes, so every
    # LOAO fold still has both classes in the training set (required by LogisticRegression).
    groups = np.array([i % 4 for i in range(20)])
    report = cross_validate_tamga(
        build_classifier("logreg", random_state=42),
        X_fm,
        y,
        cv_kind="loao",
        groups_from=groups,
    )
    assert "accuracy" in report
    assert "per_class" in report


def test_cross_validate_tamga_stratified() -> None:
    corpus = _corpus()
    y = np.array(corpus.metadata_column("author"))
    mfw = MFWExtractor(n=5, scale="zscore", lowercase=True)
    X_fm = mfw.fit_transform(corpus)
    report = cross_validate_tamga(
        build_classifier("rf", random_state=42),
        X_fm,
        y,
        cv_kind="stratified",
        folds=5,
    )
    assert "accuracy" in report


def test_cross_validate_seed_controls_stratified_folds() -> None:
    """Different seeds must produce different stratified fold assignments.

    Regression test for the audit finding that classify.py hardcoded random_state=42
    regardless of study.seed, making CV non-reproducible under user-supplied seeds.
    """
    corpus = _corpus()
    y = np.array(corpus.metadata_column("author"))
    mfw = MFWExtractor(n=5, scale="zscore", lowercase=True)
    X_fm = mfw.fit_transform(corpus)

    report_a = cross_validate_tamga(
        build_classifier("rf", random_state=0),
        X_fm,
        y,
        cv_kind="stratified",
        folds=5,
        seed=1,
    )
    report_b = cross_validate_tamga(
        build_classifier("rf", random_state=0),
        X_fm,
        y,
        cv_kind="stratified",
        folds=5,
        seed=999,
    )
    # Same seed must reproduce exactly.
    report_a2 = cross_validate_tamga(
        build_classifier("rf", random_state=0),
        X_fm,
        y,
        cv_kind="stratified",
        folds=5,
        seed=1,
    )
    assert np.array_equal(report_a["predictions"], report_a2["predictions"])
    # Different seeds must produce different fold assignments (and thus different
    # cross_val_predict output on this dataset).
    assert not np.array_equal(report_a["predictions"], report_b["predictions"])
