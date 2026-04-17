"""Render the two showcase figures for the quickstart example.

Run after `tamga run examples/quickstart/study.yaml --name demo`. Produces:
  - results/demo/pca/pca.png        (the "who writes like whom" map)
  - results/demo/zeta/zeta.png      (the "tell-tale words" plot)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tamga.viz import apply_publication_style, plot_scatter_2d, plot_zeta

HERE = Path(__file__).resolve().parent
RUN_DIR = HERE / "results" / "demo"
META_PATH = HERE / "metadata.tsv"


def render() -> None:
    apply_publication_style(dpi=200)
    meta = pd.read_csv(META_PATH, sep="\t", index_col="filename")

    pca_dir = RUN_DIR / "pca"
    if pca_dir.is_dir():
        data = json.loads((pca_dir / "result.json").read_text())
        coords = np.array(data["values"]["coordinates"]["__ndarray__"]).reshape(
            data["values"]["coordinates"]["shape"]
        )
        ids = data["values"]["document_ids"]
        authors = [meta.at[f"{i}.txt", "author"] for i in ids]
        fig = plot_scatter_2d(
            coords,
            labels=ids,
            groups=authors,
            title="Quickstart PCA — writing style map (200 most frequent words)",
        )
        fig.savefig(pca_dir / "pca.png", dpi=200, bbox_inches="tight")
        print(f"wrote {pca_dir / 'pca.png'}")

    zeta_dir = RUN_DIR / "zeta"
    if zeta_dir.is_dir() and (zeta_dir / "table_0.parquet").is_file():
        df_a = pd.read_parquet(zeta_dir / "table_0.parquet")
        df_b = pd.read_parquet(zeta_dir / "table_1.parquet")
        fig = plot_zeta(df_a, df_b, label_a="Hamilton", label_b="Madison")
        fig.savefig(zeta_dir / "zeta.png", dpi=200, bbox_inches="tight")
        print(f"wrote {zeta_dir / 'zeta.png'}")


if __name__ == "__main__":
    render()
