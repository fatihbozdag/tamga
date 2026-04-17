"""Matplotlib renderers for every major tamga plot type."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
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
