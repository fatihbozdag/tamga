"""Tests for Craig's Zeta."""

from __future__ import annotations

import pytest

from tamga.corpus import Corpus, Document
from tamga.methods.zeta import ZetaClassic, ZetaEder


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
