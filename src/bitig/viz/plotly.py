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
import pandas as pd

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


def plot_rolling_delta(
    table: pd.DataFrame,
    *,
    title: str | None = None,
    x_axis: str = "window_start_token",
    height_per_doc: int = 320,
) -> Figure:
    """Interactive rolling-delta time-series, one subplot per `doc_id`.

    Mirrors `bitig.viz.mpl.plot_rolling_delta`: distance lines per author and
    a thin coloured strip showing which author wins each window.
    """
    (go,) = _require_plotly()
    from plotly.subplots import make_subplots

    if table.empty:
        raise ValueError("plot_rolling_delta needs at least one window")
    if x_axis not in table.columns:
        raise ValueError(f"x_axis {x_axis!r} not in table columns: {list(table.columns)}")
    distance_cols = [c for c in table.columns if c.startswith("distance_")]
    if not distance_cols:
        raise ValueError("table has no `distance_<author>` columns")
    authors = [c.removeprefix("distance_") for c in distance_cols]
    palette = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    color_for = {a: palette[i % len(palette)] for i, a in enumerate(authors)}

    doc_ids = list(dict.fromkeys(table["doc_id"]))
    n = len(doc_ids)
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[str(d) for d in doc_ids],
        vertical_spacing=0.08,
    )
    for row_idx, doc_id in enumerate(doc_ids, start=1):
        sub = table[table["doc_id"] == doc_id].sort_values(x_axis)
        x = sub[x_axis].to_numpy()
        for author, dcol in zip(authors, distance_cols, strict=True):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=sub[dcol].to_numpy(),
                    mode="lines",
                    name=author,
                    legendgroup=author,
                    showlegend=row_idx == 1,
                    line={"color": color_for[author], "width": 1.4},
                    hovertemplate=f"{author}<br>%{{x}}: %{{y:.3f}}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[0.02] * len(x),
                mode="markers",
                marker={
                    "color": [color_for.get(a, "#888") for a in sub["nearest_author"]],
                    "size": 6,
                    "symbol": "square",
                },
                hovertext=[f"winner: {a}" for a in sub["nearest_author"]],
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
                yaxis=f"y{row_idx}",
            ),
            row=row_idx,
            col=1,
        )
        fig.update_yaxes(title_text="delta distance", row=row_idx, col=1)

    fig.update_xaxes(title_text=x_axis.replace("_", " "), row=n, col=1)
    fig.update_layout(
        title=title or "Rolling delta",
        height=height_per_doc * n,
        legend={"orientation": "h", "y": 1.05},
    )
    return fig


def plot_imposters_scores(
    table: pd.DataFrame,
    *,
    threshold: float = 0.5,
    title: str | None = None,
    height: int = 420,
) -> Figure:
    """Interactive General Imposters bar chart with threshold cutoff."""
    (go,) = _require_plotly()
    if table.empty:
        raise ValueError("plot_imposters_scores needs at least one target")
    required = {"target_id", "score"}
    if not required.issubset(table.columns):
        raise ValueError(f"table missing required columns: {required - set(table.columns)}")

    sub = table.sort_values("score", ascending=False)
    colors = ["#1f7a4d" if s >= threshold else "#a83232" for s in sub["score"]]
    candidate = str(sub["candidate"].iloc[0]) if "candidate" in sub.columns else "candidate"
    fig = go.Figure(
        go.Bar(
            x=sub["target_id"].astype(str),
            y=sub["score"].to_numpy(),
            marker={"color": colors},
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=threshold,
        line={"dash": "dash", "color": "grey", "width": 1},
        annotation_text=f"threshold = {threshold:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=title or f"General Imposters: target vs '{candidate}'",
        height=height,
        xaxis={"title": "target"},
        yaxis={"title": "verification score", "range": [0, 1]},
    )
    return fig


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    classes: np.ndarray | None = None,
    n_bins: int = 10,
    title: str | None = None,
    height: int = 600,
) -> Figure:
    """Interactive reliability diagram (calibration bars + diagonal + confidence histogram)."""
    (go,) = _require_plotly()
    from plotly.subplots import make_subplots

    from bitig.metrics.calibration import (
        brier_score,
        calibration_curve,
        expected_calibration_error,
    )

    curve = calibration_curve(y_true, y_proba, classes=classes, n_bins=n_bins)
    ece = expected_calibration_error(y_true, y_proba, classes=classes, n_bins=n_bins)
    brier = brier_score(y_true, y_proba, classes=classes)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.05,
    )
    width = float(curve.bin_edges[1] - curve.bin_edges[0])
    has_data = curve.counts > 0
    fig.add_trace(
        go.Bar(
            x=curve.bin_centers[has_data],
            y=curve.accuracy[has_data],
            width=[width * 0.9] * int(has_data.sum()),
            marker={"color": "#4f81bd", "line": {"color": "#222", "width": 1}},
            name="observed accuracy",
            hovertemplate="confidence ~%{x:.2f}<br>accuracy %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": "grey", "width": 1},
            name="perfect",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=curve.bin_centers,
            y=curve.counts,
            width=[width * 0.9] * len(curve.bin_centers),
            marker={"color": "#888"},
            name="count",
            showlegend=False,
            hovertemplate="confidence ~%{x:.2f}<br>n=%{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="predicted confidence", range=[0, 1], row=2, col=1)
    fig.update_yaxes(title_text="accuracy", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="count", row=2, col=1)
    fig.update_layout(
        title=title or f"Reliability diagram (ECE={ece:.3f}, Brier={brier:.3f})",
        height=height,
        legend={"orientation": "h", "y": 1.08},
    )
    return fig
