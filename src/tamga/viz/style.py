"""Publication-grade styling defaults.

Sets matplotlib rcParams to journal-friendly values: 300 DPI, serif fonts, colorblind palette,
clean grid. Applied via `apply_publication_style()`.
"""

from __future__ import annotations

from typing import Literal

import matplotlib as mpl
import seaborn as sns

ColumnWidth = Literal["single", "one_and_half", "double"]

_WIDTHS: dict[str, tuple[float, float]] = {
    "single": (3.5, 2.5),
    "one_and_half": (5.0, 3.5),
    "double": (7.0, 5.0),
}


def apply_publication_style(
    *,
    dpi: int = 300,
    font_family: str = "serif",
    palette: str = "colorblind",
) -> None:
    """Set matplotlib + seaborn defaults for publication output."""
    mpl.rcParams["figure.dpi"] = dpi
    mpl.rcParams["savefig.dpi"] = dpi
    mpl.rcParams["font.family"] = font_family
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.linestyle"] = ":"
    mpl.rcParams["grid.alpha"] = 0.3
    sns.set_palette(palette)


def figure_size(width: ColumnWidth = "single") -> tuple[float, float]:
    """Return (width_in, height_in) for a standard journal column width."""
    return _WIDTHS[width]
