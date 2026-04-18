"""`tamga classify <corpus>` — sklearn classifier + CV report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from tamga.features import MFWExtractor
from tamga.io import load_corpus
from tamga.methods.classify import build_classifier, cross_validate_tamga

console = Console()


def classify_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    metadata: Path = typer.Option(..., "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    estimator: str = typer.Option("logreg", "--estimator"),
    group_by: str = typer.Option("author", "--group-by"),
    cv_kind: str = typer.Option("loao", "--cv-kind"),
    folds: int = typer.Option(5, "--folds"),
    mfw: int = typer.Option(500, "--mfw"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Fit+cross-validate a classifier and print per-author metrics."""
    corpus = load_corpus(path, metadata=metadata)
    y = np.array(corpus.metadata_column(group_by))
    fm = MFWExtractor(n=mfw, min_df=2, scale="zscore", lowercase=True).fit_transform(corpus)
    clf = build_classifier(estimator, random_state=seed)
    report = cross_validate_tamga(
        clf,
        fm,
        y,
        cv_kind=cv_kind,
        groups_from=y if cv_kind == "loao" else None,
        folds=folds,
        seed=seed,
    )
    table = Table(title=f"classify — {estimator} / {cv_kind}")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("accuracy", f"{report['accuracy']:.3f}")
    per_class = report["per_class"]
    if "macro avg" in per_class:
        table.add_row("macro_f1", f"{per_class['macro avg']['f1-score']:.3f}")
    console.print(table)
