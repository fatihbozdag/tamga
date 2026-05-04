"""Tests for the FeatureMatrix dataclass."""

import numpy as np
import pandas as pd
import pytest

from bitig.features import FeatureMatrix


def _fm(X: np.ndarray, feature_names: list[str], doc_ids: list[str] | None = None) -> FeatureMatrix:  # noqa: N803
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
