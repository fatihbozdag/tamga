"""`bitig delta <corpus>` — fit Burrows-family Delta and attribute held-out documents."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from bitig.features import MFWExtractor
from bitig.io import load_corpus
from bitig.methods.delta import (
    ArgamonLinearDelta,
    BurrowsDelta,
    CosineDelta,
    EderDelta,
    EderSimpleDelta,
    QuadraticDelta,
)

console = Console()

_METHODS = {
    "burrows": BurrowsDelta,
    "eder": EderDelta,
    "eder_simple": EderSimpleDelta,
    "argamon": ArgamonLinearDelta,
    "cosine": CosineDelta,
    "quadratic": QuadraticDelta,
}


def delta_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    method: str = typer.Option("burrows", "--method", help=f"One of {sorted(_METHODS)}"),
    mfw: int = typer.Option(1000, "--mfw", help="Top-N most frequent words"),
    mfw_min: int = typer.Option(2, "--mfw-min", help="Minimum document frequency for MFW"),
    metadata: Path = typer.Option(..., "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    group_by: str = typer.Option("author", "--group-by", help="Metadata column with author labels"),
    test_filter: str | None = typer.Option(
        None,
        "--test-filter",
        help="Key=value (e.g. 'role=test') selecting held-out documents; "
        "if not provided, fit+predict on the entire corpus.",
    ),
) -> None:
    """Fit a Delta classifier on a corpus and report per-document attributions."""
    if method not in _METHODS:
        console.print(f"[red]error:[/red] unknown method {method!r}. Known: {sorted(_METHODS)}")
        raise typer.Exit(code=1)

    corpus = load_corpus(path, metadata=metadata)

    if test_filter is not None:
        key, _, value = test_filter.partition("=")
        if not _:
            console.print(
                f"[red]error:[/red] --test-filter must be 'key=value'; got {test_filter!r}"
            )
            raise typer.Exit(code=1)
        test = corpus.filter(**{key: value})
        train_docs = [d for d in corpus.documents if d not in test.documents]
        from bitig.corpus import Corpus

        train = Corpus(documents=train_docs)
    else:
        train = corpus
        test = corpus

    extractor = MFWExtractor(n=mfw, min_df=mfw_min, scale="zscore", lowercase=True)
    train_fm = extractor.fit_transform(train)
    test_fm = extractor.transform(test)

    clf_cls = _METHODS[method]
    clf = clf_cls().fit(train_fm, np.asarray(train.metadata_column(group_by)))
    preds = clf.predict(test_fm)

    table = Table(title=f"Delta attribution — method={method}, mfw={mfw}")
    table.add_column("doc_id", style="cyan")
    table.add_column(f"{group_by} (observed)")
    table.add_column(f"{group_by} (predicted)")
    table.add_column("match")
    for doc, pred in zip(test.documents, preds, strict=True):
        observed = doc.metadata.get(group_by, "<unknown>")
        match = "yes" if observed == pred else "no"
        table.add_row(doc.id, str(observed), str(pred), match)
    console.print(table)
