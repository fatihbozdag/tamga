"""NiceGUI page definitions for the tamga desktop app.

Importing this module registers every ``@ui.page`` route as a side effect;
the launcher imports it just before calling ``ui.run()``.
"""

from __future__ import annotations

from nicegui import ui

from tamga.gui.layout import BRAND_BRASS, BRAND_NAVY
from tamga.gui.pages import ingest, results, run, study  # noqa: F401

ui.colors(primary=BRAND_NAVY, secondary=BRAND_BRASS)
