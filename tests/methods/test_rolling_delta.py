"""Unit tests for `tamga.methods.rolling_delta.RollingDelta`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tamga.corpus import Corpus
from tamga.corpus.document import Document
from tamga.methods.rolling_delta import RollingDelta


def _alice_text(n_paragraphs: int = 200) -> str:
    """Alice-flavoured prose: function-word distribution leans on 'the/of/and/in'."""
    rng = np.random.default_rng(0)
    function_words = ["the", "of", "and", "in", "a", "to"]
    content_words = ["alice", "garden", "rabbit", "story", "tea"]
    parts: list[str] = []
    for _ in range(n_paragraphs):
        n = int(rng.integers(15, 22))
        sent = " ".join(
            rng.choice(function_words, size=n // 2).tolist()
            + rng.choice(content_words, size=n - n // 2).tolist()
        )
        parts.append(sent.capitalize() + ".")
    return " ".join(parts)


def _bob_text(n_paragraphs: int = 200) -> str:
    """Bob-flavoured prose: function-word distribution skews to 'but/that/with/on'."""
    rng = np.random.default_rng(1)
    function_words = ["but", "that", "with", "on", "by", "from"]
    content_words = ["bob", "office", "report", "meeting", "memo"]
    parts: list[str] = []
    for _ in range(n_paragraphs):
        n = int(rng.integers(15, 22))
        sent = " ".join(
            rng.choice(function_words, size=n // 2).tolist()
            + rng.choice(content_words, size=n - n // 2).tolist()
        )
        parts.append(sent.capitalize() + ".")
    return " ".join(parts)


@pytest.fixture()
def synth_corpus() -> Corpus:
    """Two Alice training docs + two Bob training docs + one Alice-target doc."""
    return Corpus(
        documents=[
            Document(id="alice_train_1", text=_alice_text(), metadata={"author": "Alice"}),
            Document(id="alice_train_2", text=_alice_text(220), metadata={"author": "Alice"}),
            Document(id="bob_train_1", text=_bob_text(), metadata={"author": "Bob"}),
            Document(id="bob_train_2", text=_bob_text(220), metadata={"author": "Bob"}),
            Document(id="alice_target", text=_alice_text(300), metadata={"author": "Alice"}),
        ]
    )


def test_rolling_delta_returns_expected_table(synth_corpus: Corpus) -> None:
    rd = RollingDelta(
        target_ids=["alice_target"],
        group_by="author",
        window_size=400,
        step=100,
        base_delta="burrows",
        mfw_n=80,
    )
    result = rd.fit_transform(synth_corpus)
    assert result.method_name == "rolling_delta_burrows"
    assert len(result.tables) == 1
    table = result.tables[0]
    assert {"doc_id", "window_idx", "window_start_token", "nearest_author"} <= set(table.columns)
    assert (table["doc_id"] == "alice_target").all()
    # 2 candidate authors -> 2 distance columns.
    distance_cols = [c for c in table.columns if c.startswith("distance_")]
    assert sorted(distance_cols) == ["distance_Alice", "distance_Bob"]
    # Rolling correctness: most windows of an Alice-flavoured target should be classified Alice.
    alice_share = (table["nearest_author"] == "Alice").mean()
    assert alice_share > 0.6, f"alice_share={alice_share:.2f}; rolling delta picked Bob too often"


def test_rolling_delta_step_default(synth_corpus: Corpus) -> None:
    rd = RollingDelta(target_ids=["alice_target"], group_by="author", window_size=400, mfw_n=50)
    assert rd.step == 40  # window_size // 10


def test_rolling_delta_unknown_base_delta_raises() -> None:
    with pytest.raises(ValueError, match="unknown base_delta"):
        RollingDelta(target_ids=["x"], group_by="author", base_delta="not_a_real_kernel")


def test_rolling_delta_empty_targets_raises() -> None:
    with pytest.raises(ValueError, match="at least one target_id"):
        RollingDelta(target_ids=[], group_by="author")


def test_rolling_delta_target_not_in_corpus(synth_corpus: Corpus) -> None:
    rd = RollingDelta(target_ids=["does_not_exist"], group_by="author", window_size=400)
    with pytest.raises(ValueError, match="none of target_ids"):
        rd.fit_transform(synth_corpus)


def test_rolling_delta_window_larger_than_target(synth_corpus: Corpus) -> None:
    rd = RollingDelta(target_ids=["alice_target"], group_by="author", window_size=100_000, mfw_n=20)
    with pytest.raises(ValueError, match="less than window_size"):
        rd.fit_transform(synth_corpus)


def test_rolling_delta_table_round_trips(synth_corpus: Corpus, tmp_path) -> None:
    """Result.save() should write the windows table to parquet."""
    rd = RollingDelta(
        target_ids=["alice_target"],
        group_by="author",
        window_size=400,
        step=200,
        mfw_n=50,
    )
    result = rd.fit_transform(synth_corpus)
    result.save(tmp_path)
    parquet = tmp_path / "table_0.parquet"
    assert parquet.exists()
    reloaded = pd.read_parquet(parquet)
    assert len(reloaded) == len(result.tables[0])
