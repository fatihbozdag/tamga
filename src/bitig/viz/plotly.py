"""Plotly (interactive) renderers -- parallel to `bitig.viz.mpl` for the most-used plot types.

These produce `plotly.graph_objects.Figure` instances; save them with `fig.write_html(path)`
or embed them in a notebook / NiceGUI page. The matplotlib path remains the default for
publication-quality static figures (PDF / PNG); use this module when interactive
hover / zoom / pan is more useful (large dendrograms, dense scatters, posterior heatmaps).

Requires the optional `bitig[interactive]` extra (just `plotly>=5.20`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from plotly.graph_objects import Figure
else:
    Figure = "Figure"

from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram


def _require_plotly() -> tuple:
    """Import plotly lazily so this module loads even without the extra."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "bitig.viz.plotly requires the optional `bitig[interactive]` extra "
            "(install with `pip install 'bitig[interactive]'` or `uv pip install plotly`)"
        ) from exc
    return (go,)


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    *,
    labels: list[str] | None = None,
    title: str = "Dendrogram",
    height: int = 500,
) -> Figure:
    """Interactive dendrogram from a precomputed linkage matrix."""
    (go,) = _require_plotly()
    ddata = scipy_dendrogram(linkage_matrix, labels=labels, no_plot=True)
    xs: list[float] = []
    ys: list[float] = []
    for x_quad, y_quad in zip(ddata["icoord"], ddata["dcoord"], strict=False):
        xs.extend([x_quad[0], x_quad[1], x_quad[2], x_quad[3], np.nan])
        ys.extend([y_quad[0], y_quad[1], y_quad[2], y_quad[3], np.nan])
    fig = go.Figure(
        go.Scatter(x=xs, y=ys, mode="lines", line={"color": "#444", "width": 1.2}, hoverinfo="skip")
    )
    leaf_xs = [5 + i * 10 for i in range(len(ddata["ivl"]))]
    fig.update_layout(
        title=title,
        height=height,
        xaxis={
            "tickmode": "array",
            "tickvals": leaf_xs,
            "ticktext": ddata["ivl"],
            "tickangle": -90,
            "showgrid": False,
        },
        yaxis={"title": "distance", "showgrid": True},
        margin={"l": 60, "r": 20, "t": 40, "b": 80},
    )
    return fig


def plot_scatter_2d(
    coordinates: np.ndarray,
    *,
    labels: list[str] | None = None,
    groups: list[str] | None = None,
    title: str = "2D projection",
    height: int = 500,
) -> Figure:
    """Hoverable 2-D scatter with optional group colouring + per-point labels."""
    (go,) = _require_plotly()
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(f"coordinates must be (n_docs, >=2); got {coordinates.shape}")
    fig = go.Figure()
    if groups is not None:
        unique = sorted(set(groups))
        for g in unique:
            mask = np.array([gg == g for gg in groups])
            text = (
                [labels[i] for i in range(len(labels)) if mask[i]]  # type: ignore[index]
                if labels
                else None
            )
            fig.add_trace(
                go.Scatter(
                    x=coordinates[mask, 0],
                    y=coordinates[mask, 1],
                    mode="markers",
                    name=g,
                    text=text,
                    hovertemplate="<b>%{text}</b><br>(%{x:.2f}, %{y:.2f})<extra></extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                mode="markers",
                text=labels,
                hovertemplate="<b>%{text}</b><br>(%{x:.2f}, %{y:.2f})<extra></extra>"
                if labels
                else None,
            )
        )
    fig.update_layout(
        title=title,
        height=height,
        xaxis={"title": "Component 1"},
        yaxis={"title": "Component 2"},
    )
    return fig


def plot_distance_heatmap(
    distances: np.ndarray,
    *,
    labels: list[str] | None = None,
    title: str = "Distance heatmap",
    colorscale: str = "Viridis",
    height: int = 500,
) -> Figure:
    """Symmetric distance heatmap with hoverable cell values."""
    (go,) = _require_plotly()
    fig = go.Figure(
        go.Heatmap(
            z=distances,
            x=labels,
            y=labels,
            colorscale=colorscale,
            colorbar={"title": "distance"},
            hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=height, yaxis={"autorange": "reversed"})
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Confusion matrix",
    height: int = 500,
) -> Figure:
    """Confusion matrix with cell-count annotations."""
    (go,) = _require_plotly()
    from sklearn.metrics import confusion_matrix as _cm

    cm = _cm(y_true, y_pred)
    labels = sorted(set(np.concatenate([y_true, y_pred])))
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            hovertemplate="observed=%{y}, predicted=%{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        xaxis={"title": "predicted"},
        yaxis={"title": "observed", "autorange": "reversed"},
    )
    return fig


def plot_pca_biplot(
    coordinates: np.ndarray,
    loadings: np.ndarray,
    feature_names: list[str],
    *,
    labels: list[str] | None = None,
    groups: list[str] | None = None,
    top_n: int = 15,
    explained_variance_ratio: np.ndarray | None = None,
    title: str = "PCA biplot",
    height: int = 600,
) -> Figure:
    """Interactive PCA biplot: hoverable docs + top-N loading arrows."""
    _require_plotly()
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(f"coordinates must be (n_docs, >=2); got {coordinates.shape}")
    if loadings.ndim != 2 or loadings.shape[0] < 2:
        raise ValueError(f"loadings must be (>=2, n_features); got {loadings.shape}")
    if loadings.shape[1] != len(feature_names):
        raise ValueError(
            f"loadings columns ({loadings.shape[1]}) != len(feature_names) ({len(feature_names)})"
        )

    fig = plot_scatter_2d(coordinates, labels=labels, groups=groups, title=title, height=height)

    plane = loadings[:2]
    norms = np.linalg.norm(plane, axis=0)
    top_idx = np.argsort(norms)[::-1][: min(top_n, plane.shape[1])]
    half_range = min(np.ptp(coordinates[:, 0]) / 2, np.ptp(coordinates[:, 1]) / 2)
    scale = (0.9 * half_range) / norms[top_idx[0]] if norms[top_idx[0]] > 0 else 1.0

    for i in top_idx:
        x = scale * float(plane[0, i])
        y = scale * float(plane[1, i])
        fig.add_annotation(
            x=x,
            y=y,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#a83232",
            arrowwidth=1.0,
            text=feature_names[i],
            font={"color": "#7a1f1f", "size": 10},
        )

    if explained_variance_ratio is not None and len(explained_variance_ratio) >= 2:
        fig.update_layout(
            xaxis_title=f"PC1 ({100 * float(explained_variance_ratio[0]):.1f}%)",
            yaxis_title=f"PC2 ({100 * float(explained_variance_ratio[1]):.1f}%)",
        )
    else:
        fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    return fig


def plot_posterior_heatmap(
    proba: np.ndarray,
    document_ids: list[str],
    classes: list[str],
    *,
    title: str = "Posterior probability",
    colorscale: str = "Magma",
    height: int = 500,
) -> Figure:
    """Interactive posterior heatmap (n_docs, n_classes) with hoverable P values."""
    (go,) = _require_plotly()
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2-D (n_docs, n_classes); got {proba.shape}")
    if proba.shape[0] != len(document_ids):
        raise ValueError(
            f"proba rows ({proba.shape[0]}) != len(document_ids) ({len(document_ids)})"
        )
    if proba.shape[1] != len(classes):
        raise ValueError(f"proba columns ({proba.shape[1]}) != len(classes) ({len(classes)})")
    fig = go.Figure(
        go.Heatmap(
            z=proba,
            x=classes,
            y=document_ids,
            colorscale=colorscale,
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": "P(author | doc)"},
            hovertemplate="doc=%{y}, author=%{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        xaxis={"title": "author"},
        yaxis={"title": "document", "autorange": "reversed"},
    )
    return fig
