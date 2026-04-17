"""End-to-end: load corpus → MFW → reduce → cluster → zeta → classify, all within one script."""

from __future__ import annotations

import numpy as np
import pytest

from tamga.features import MFWExtractor
from tamga.io import load_corpus
from tamga.methods.classify import build_classifier, cross_validate_tamga
from tamga.methods.cluster import HierarchicalCluster
from tamga.methods.reduce import PCAReducer
from tamga.methods.zeta import ZetaClassic

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

    zeta = ZetaClassic(
        group_by="author", top_k=5, group_a="Hamilton", group_b="Madison"
    ).fit_transform(corpus)
    assert len(zeta.tables) == 2

    report = cross_validate_tamga(
        build_classifier("logreg", random_state=42),
        fm,
        y,
        cv_kind="loao",
        groups_from=y,
    )
    assert 0.0 <= report["accuracy"] <= 1.0
