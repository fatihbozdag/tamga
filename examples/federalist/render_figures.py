"""Render matplotlib figures for each method in a bitig run directory.

Run after `bitig run examples/federalist/study.yaml --name demo`, then `bitig report ...`
will pick up the PNGs automatically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bitig.viz import (
    apply_publication_style,
    plot_dendrogram,
    plot_scatter_2d,
    plot_zeta,
)


def render(run_dir: Path, metadata_path: Path) -> None:
    apply_publication_style(dpi=300)
    meta = pd.read_csv(metadata_path, sep="\t", index_col="filename")

    # PCA scatter
    pca_dir = run_dir / "pca"
    if pca_dir.is_dir():
        data = json.loads((pca_dir / "result.json").read_text())
        coords = np.array(data["values"]["coordinates"]["__ndarray__"]).reshape(
            data["values"]["coordinates"]["shape"]
        )
        ids = data["values"]["document_ids"]
        authors = [meta.at[f"{i}.txt", "author"] for i in ids]
        fig = plot_scatter_2d(coords, labels=ids, groups=authors, title="PCA — Federalist MFW=500")
        fig.savefig(pca_dir / "pca.png", dpi=300, bbox_inches="tight")
        print(f"wrote {pca_dir / 'pca.png'}")

    # Ward dendrogram
    ward_dir = run_dir / "ward"
    if ward_dir.is_dir():
        data = json.loads((ward_dir / "result.json").read_text())
        linkage = np.array(data["values"]["linkage"]["__ndarray__"]).reshape(
            data["values"]["linkage"]["shape"]
        )
        ids = data["values"]["document_ids"]
        fig = plot_dendrogram(linkage, labels=ids, title="Ward dendrogram — Federalist MFW=500")
        fig.set_size_inches(12, 6)
        fig.savefig(ward_dir / "ward.png", dpi=300, bbox_inches="tight")
        print(f"wrote {ward_dir / 'ward.png'}")

    # Zeta preference scatter
    zeta_dir = run_dir / "zeta_hamilton_madison"
    if zeta_dir.is_dir() and (zeta_dir / "table_0.parquet").is_file():
        df_a = pd.read_parquet(zeta_dir / "table_0.parquet")
        df_b = pd.read_parquet(zeta_dir / "table_1.parquet")
        fig = plot_zeta(df_a, df_b, label_a="Hamilton", label_b="Madison")
        fig.savefig(zeta_dir / "zeta.png", dpi=300, bbox_inches="tight")
        print(f"wrote {zeta_dir / 'zeta.png'}")


if __name__ == "__main__":
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/federalist/results/demo")
    meta = Path(sys.argv[2] if len(sys.argv) > 2 else "examples/federalist/metadata.tsv")
    render(run_dir, meta)
