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
