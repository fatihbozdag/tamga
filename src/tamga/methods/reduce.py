"""Dimensionality reducers — PCA, MDS, t-SNE, UMAP — with a shared Result interface."""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

from tamga.features import FeatureMatrix
from tamga.result import Result


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

        return umap.UMAP  # type: ignore[no-any-return]
