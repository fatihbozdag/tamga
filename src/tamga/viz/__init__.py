"""tamga visualisation — matplotlib/seaborn static + plotly interactive."""

from tamga.viz.mpl import (
    plot_bootstrap_consensus_tree,
    plot_confusion_matrix,
    plot_dendrogram,
    plot_distance_heatmap,
    plot_feature_importance,
    plot_scatter_2d,
    plot_zeta,
)
from tamga.viz.style import apply_publication_style, figure_size

__all__ = [
    "apply_publication_style",
    "figure_size",
    "plot_bootstrap_consensus_tree",
    "plot_confusion_matrix",
    "plot_dendrogram",
    "plot_distance_heatmap",
    "plot_feature_importance",
    "plot_scatter_2d",
    "plot_zeta",
]
