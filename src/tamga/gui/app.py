"""NiceGUI page definitions for the tamga desktop app.

Importing this module registers the ``@ui.page`` routes as a side effect; the
launcher imports it just before calling ``ui.run()``.
"""

from __future__ import annotations

from nicegui import ui

from tamga._version import __version__

# Palette matching the README badges.
BRAND_NAVY = "#0F1A2B"
BRAND_BRASS = "#C9A34A"

ui.colors(primary=BRAND_NAVY, secondary=BRAND_BRASS)


@ui.page("/")
def home() -> None:
    with ui.column().classes("w-full max-w-3xl mx-auto p-8 gap-4"):
        with ui.row().classes("items-center gap-3"):
            ui.label("tamga").classes("text-4xl font-bold").style(f"color: {BRAND_NAVY}")
            ui.label(f"v{__version__}").classes("text-sm").style(f"color: {BRAND_BRASS}")
        ui.label("Computational stylometry — desktop preview").classes("text-lg")
        ui.separator()
        ui.markdown(
            "This is the MVP window. The full workflow (Ingest / Study / Run / "
            "Results) ships in the next release."
        )
        ui.link(
            "Documentation",
            "https://fatihbozdag.github.io/tamga/",
            new_tab=True,
        ).classes("text-sm")
