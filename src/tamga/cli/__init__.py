"""Typer CLI entry point."""

from __future__ import annotations

import typer
from rich.console import Console

from tamga._version import __version__
from tamga.cli.bayesian_cmd import bayesian_command
from tamga.cli.cache_cmd import cache_app
from tamga.cli.classify_cmd import classify_command
from tamga.cli.cluster_cmd import cluster_command
from tamga.cli.consensus_cmd import consensus_command
from tamga.cli.delta_cmd import delta_command
from tamga.cli.embed_cmd import embed_command
from tamga.cli.features_cmd import features_command
from tamga.cli.info_cmd import info_command
from tamga.cli.ingest_cmd import ingest_command
from tamga.cli.init_cmd import init_command
from tamga.cli.plot_cmd import plot_command
from tamga.cli.reduce_cmd import reduce_command
from tamga.cli.report_cmd import report_command
from tamga.cli.run_cmd import run_command
from tamga.cli.shell_cmd import shell_command
from tamga.cli.zeta_cmd import zeta_command

console = Console()
app = typer.Typer(
    name="tamga",
    help="tamga — computational stylometry (next-generation Python replacement for R's Stylo).",
    no_args_is_help=True,
    add_completion=True,
)

app.command(name="init")(init_command)
app.command(name="ingest")(ingest_command)
app.command(name="info")(info_command)
app.command(name="features")(features_command)
app.command(name="delta")(delta_command)
app.command(name="zeta")(zeta_command)
app.command(name="reduce")(reduce_command)
app.command(name="cluster")(cluster_command)
app.command(name="consensus")(consensus_command)
app.command(name="classify")(classify_command)
app.command(name="embed")(embed_command)
app.command(name="bayesian")(bayesian_command)
app.command(name="run")(run_command)
app.command(name="report")(report_command)
app.command(name="plot")(plot_command)
app.command(name="shell")(shell_command)
app.add_typer(cache_app, name="cache")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"tamga {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    """tamga — computational stylometry."""
