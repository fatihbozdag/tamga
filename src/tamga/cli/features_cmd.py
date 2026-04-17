"""`tamga features <corpus>` — build a feature matrix and persist to parquet."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from tamga.features import (
    CharNgramExtractor,
    FunctionWordExtractor,
    MFWExtractor,
    PunctuationExtractor,
    WordNgramExtractor,
)
from tamga.io import load_corpus

console = Console()

_EXTRACTORS = {
    "mfw": MFWExtractor,
    "char_ngram": CharNgramExtractor,
    "word_ngram": WordNgramExtractor,
    "function_word": FunctionWordExtractor,
    "punctuation": PunctuationExtractor,
}


def features_command(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),  # noqa: B008
    type: str = typer.Option("mfw", "--type", help=f"Feature type: one of {sorted(_EXTRACTORS)}"),
    n: int = typer.Option(1000, "--n", help="Top-N for MFW, or n-gram order"),
    min_df: int = typer.Option(1, "--min-df"),
    scale: str = typer.Option("zscore", "--scale", help="none | zscore | l1 | l2"),
    lowercase: bool = typer.Option(False, "--lowercase"),
    metadata: Path | None = typer.Option(None, "--metadata", "-m", exists=True, dir_okay=False),  # noqa: B008
    output: Path = typer.Option(Path("features.parquet"), "--output", "-o"),  # noqa: B008
) -> None:
    """Build a feature matrix from a corpus and save to parquet."""
    if type not in _EXTRACTORS:
        console.print(
            f"[red]error:[/red] unknown feature type {type!r}. Known: {sorted(_EXTRACTORS)}"
        )
        raise typer.Exit(code=1)

    corpus = load_corpus(path, metadata=metadata)
    console.print(f"[green]loaded[/green] {len(corpus)} documents")

    extractor_cls = _EXTRACTORS[type]
    if type == "mfw":
        extractor = extractor_cls(n=n, min_df=min_df, scale=scale, lowercase=lowercase)
    elif type in ("char_ngram", "word_ngram"):
        extractor = extractor_cls(n=n, scale=scale)
    else:
        extractor = extractor_cls()

    fm = extractor.fit_transform(corpus)
    df = fm.as_dataframe()
    df.to_parquet(output)
    console.print(
        f"[green]wrote[/green] {output} ({fm.X.shape[0]} docs x {fm.n_features} features)"
    )
