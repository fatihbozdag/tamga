"""Matplotlib renderers for every major tamga plot type."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import confusion_matrix

from tamga.viz.style import figure_size


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    *,
    labels: list[str] | None = None,
    title: str = "Dendrogram",
    orientation: str = "top",
) -> Figure:
    fig, ax = plt.subplots(figsize=figure_size("one_and_half"))
    scipy_dendrogram(linkage_matrix, labels=labels, orientation=orientation, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_scatter_2d(
    coordinates: np.ndarray,
    *,
    labels: list[str] | None = None,
    groups: list[str] | None = None,
    title: str = "2D projection",
) -> Figure:
    fig, ax = plt.subplots(figsize=figure_size("single"))
    if groups is not None:
        unique = sorted(set(groups))
        palette = sns.color_palette(n_colors=len(unique))
        for i, g in enumerate(unique):
            mask = np.array([gg == g for gg in groups])
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], label=g, color=palette[i], s=40)
        ax.legend(fontsize=7)
    else:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], s=40)
    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(label, (coordinates[i, 0], coordinates[i, 1]), fontsize=6, alpha=0.7)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_distance_heatmap(
    distances: np.ndarray,
    *,
    labels: list[str] | None = None,
    title: str = "Distance heatmap",
    cmap: str = "viridis",
) -> Figure:
    fig, ax = plt.subplots(figsize=figure_size("one_and_half"))
    sns.heatmap(
        distances,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        square=True,
        ax=ax,
        cbar_kws={"label": "distance"},
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> Figure:
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
    labels = sorted(set(np.concatenate([y_true, y_pred])))
    fig, ax = plt.subplots(figsize=figure_size("single"))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("observed")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    names: list[str],
    importance: np.ndarray,
    *,
    top_n: int = 20,
    title: str = "Feature importance",
) -> Figure:
    order = np.argsort(importance)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=figure_size("single"))
    ax.barh([names[i] for i in order][::-1], importance[order][::-1])
    ax.set_xlabel("importance")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_bootstrap_consensus_tree(
    support: dict[str, float],
    leaves: list[str],
    *,
    title: str = "Bootstrap consensus tree",
    min_support_annotate: float = 0.5,
) -> Figure:
    """Render a bootstrap consensus tree from a clade-support map.

    `support` maps a comma-joined sorted leaf-id frozenset (the same key shape
    BootstrapConsensus emits) to its support fraction in [0, 1]. We construct
    a pairwise distance matrix where d(i,j) = 1 - max{support(c) | i,j ∈ c},
    run average-linkage clustering on that, and draw the dendrogram. Internal
    nodes whose support exceeds `min_support_annotate` are labelled with the
    rounded support percentage.

    The y-axis "1 - clade support" reads cleanly: short branches = strong
    consensus across the bootstrap MFW bands.
    """
    n = len(leaves)
    if n < 2:
        raise ValueError("plot_bootstrap_consensus_tree needs at least 2 leaves")
    idx = {leaf: i for i, leaf in enumerate(leaves)}

    # Pairwise consensus distance: 1 - max support of any clade containing both.
    D = np.ones((n, n), dtype=float)  # noqa: N806
    np.fill_diagonal(D, 0.0)
    for clade_str, freq in support.items():
        members = [m.strip() for m in clade_str.split(",")]
        ids = [idx[m] for m in members if m in idx]
        for i in ids:
            for j in ids:
                if i != j:
                    D[i, j] = min(D[i, j], 1.0 - float(freq))

    Z = scipy_linkage(squareform(D, checks=False), method="average")  # noqa: N806

    fig, ax = plt.subplots(figsize=figure_size("one_and_half"))
    ddata = scipy_dendrogram(Z, labels=leaves, ax=ax, leaf_rotation=90)

    # Annotate internal nodes with support % (≈ 1 - height) when meaningful.
    for x_coords, y_coords in zip(ddata["icoord"], ddata["dcoord"], strict=False):
        height = y_coords[1]  # top of the U-shape
        support_frac = max(0.0, 1.0 - height)
        if support_frac >= min_support_annotate:
            x_mid = 0.5 * (x_coords[1] + x_coords[2])
            ax.annotate(
                f"{round(support_frac * 100)}%",
                (x_mid, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#444",
            )
    ax.set_ylabel("1 - clade support")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_rolling_delta(
    table: pd.DataFrame,
    *,
    title: str | None = None,
    x_axis: str = "window_start_token",
) -> Figure:
    """Render the rolling-delta time-series for one or more target documents.

    Expects the table emitted by RollingDelta: one row per (doc_id, window),
    with a `nearest_author` column and `distance_<author>` columns. Produces
    one subplot per `doc_id`; each subplot shows one distance line per author
    and shades the strip where each author wins the window.
    """
    if table.empty:
        raise ValueError("plot_rolling_delta needs at least one window")
    if x_axis not in table.columns:
        raise ValueError(f"x_axis {x_axis!r} not in table columns: {list(table.columns)}")
    distance_cols = [c for c in table.columns if c.startswith("distance_")]
    if not distance_cols:
        raise ValueError("table has no `distance_<author>` columns")
    authors = [c.removeprefix("distance_") for c in distance_cols]
    palette = sns.color_palette(n_colors=len(authors))
    color_for = dict(zip(authors, palette, strict=True))

    doc_ids = list(dict.fromkeys(table["doc_id"]))
    n = len(doc_ids)
    fig, axes = plt.subplots(
        n, 1, figsize=figure_size("one_and_half" if n == 1 else "double"), sharex=False
    )
    axes_iter = axes if isinstance(axes, np.ndarray) else np.array([axes])

    for ax, doc_id in zip(axes_iter, doc_ids, strict=True):
        sub = table[table["doc_id"] == doc_id].sort_values(x_axis)
        x = sub[x_axis].to_numpy()
        for author, dcol in zip(authors, distance_cols, strict=True):
            ax.plot(x, sub[dcol].to_numpy(), label=author, color=color_for[author], lw=1.2)
        # Highlight winning author per window with a thin coloured strip on the bottom.
        ymin, ymax = ax.get_ylim()
        strip = (ymax - ymin) * 0.04
        for _, row in sub.iterrows():
            ax.axvspan(
                row[x_axis],
                row[x_axis] + 1,
                ymin=0,
                ymax=strip / (ymax - ymin) if ymax > ymin else 0.04,
                color=color_for.get(row["nearest_author"], "#888"),
                alpha=0.55,
                lw=0,
            )
        ax.set_title(str(doc_id), fontsize=9)
        ax.set_ylabel("delta distance")
        ax.legend(fontsize=7, loc="upper right")
    axes_iter[-1].set_xlabel(x_axis.replace("_", " "))
    fig.suptitle(title or "Rolling delta")
    fig.tight_layout()
    return fig


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    classes: np.ndarray | None = None,
    n_bins: int = 10,
    title: str | None = None,
) -> Figure:
    """Reliability diagram + confidence histogram for a probabilistic classifier.

    Top panel: per-bin observed accuracy vs mean predicted confidence with a
    diagonal "perfect calibration" reference. Bars below the diagonal mean
    over-confident predictions; above means under-confident. Bottom panel:
    histogram of per-prediction confidence to surface bin sparsity.
    """
    from tamga.metrics.calibration import (
        brier_score,
        calibration_curve,
        expected_calibration_error,
    )

    curve = calibration_curve(y_true, y_proba, classes=classes, n_bins=n_bins)
    ece = expected_calibration_error(y_true, y_proba, classes=classes, n_bins=n_bins)
    brier = brier_score(y_true, y_proba, classes=classes)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=figure_size("one_and_half"),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    width = curve.bin_edges[1] - curve.bin_edges[0]
    has_data = curve.counts > 0
    ax_top.bar(
        curve.bin_centers[has_data],
        curve.accuracy[has_data],
        width=width * 0.9,
        edgecolor="#222",
        color="#4f81bd",
        alpha=0.85,
        label="observed accuracy",
    )
    ax_top.plot([0, 1], [0, 1], "--", color="grey", lw=1, alpha=0.7, label="perfect")
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1.05)
    ax_top.set_ylabel("accuracy")
    ax_top.legend(loc="upper left", fontsize=7)
    ax_top.set_title(title or f"Reliability diagram (ECE={ece:.3f}, Brier={brier:.3f})")

    ax_bot.bar(
        curve.bin_centers,
        curve.counts,
        width=width * 0.9,
        color="#888",
        alpha=0.7,
    )
    ax_bot.set_xlabel("predicted confidence")
    ax_bot.set_ylabel("count")
    fig.tight_layout()
    return fig


def plot_imposters_scores(
    table: pd.DataFrame,
    *,
    threshold: float = 0.5,
    title: str | None = None,
) -> Figure:
    """Render General Imposters verification scores as a bar chart.

    Expects the table emitted by GeneralImposters: one row per target with
    `target_id`, `candidate`, and `score` columns. Bars are coloured by the
    decision (verified vs rejected at `threshold`); a horizontal cutoff line
    marks the threshold.
    """
    if table.empty:
        raise ValueError("plot_imposters_scores needs at least one target")
    required = {"target_id", "score"}
    if not required.issubset(table.columns):
        raise ValueError(f"table missing required columns: {required - set(table.columns)}")

    sub = table.sort_values("score", ascending=False)
    colors = ["#1f7a4d" if s >= threshold else "#a83232" for s in sub["score"]]
    fig, ax = plt.subplots(figsize=figure_size("one_and_half"))
    ax.bar(sub["target_id"].astype(str), sub["score"].to_numpy(), color=colors)
    ax.axhline(threshold, color="grey", lw=1, ls="--", alpha=0.7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("verification score")
    ax.set_xlabel("target")
    if len(sub) > 6:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    candidate = str(sub["candidate"].iloc[0]) if "candidate" in sub.columns else "candidate"
    ax.set_title(title or f"General Imposters: target vs {candidate!r}")
    fig.tight_layout()
    return fig


def plot_zeta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    label_a: str = "group A",
    label_b: str = "group B",
    title: str | None = None,
) -> Figure:
    df = pd.concat([df_a.assign(side=label_a), df_b.assign(side=label_b)])
    fig, ax = plt.subplots(figsize=figure_size("one_and_half"))
    for side, marker, color in [(label_a, "o", "#1f77b4"), (label_b, "s", "#d62728")]:
        sub = df[df["side"] == side]
        ax.scatter(sub["prop_a"], sub["prop_b"], marker=marker, color=color, label=side)
        for _, row in sub.iterrows():
            ax.annotate(row["word"], (row["prop_a"], row["prop_b"]), fontsize=6)
    ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.3)
    ax.set_xlabel(f"proportion in {label_a}")
    ax.set_ylabel(f"proportion in {label_b}")
    ax.set_title(title or f"Zeta preference: {label_a} vs {label_b}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
