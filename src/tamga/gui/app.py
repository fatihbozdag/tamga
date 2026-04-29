"""NiceGUI page definitions for the tamga desktop app.

Importing this module registers every ``@ui.page`` route as a side effect;
the launcher imports it just before calling ``ui.run()``. No UI calls happen
at module scope — NiceGUI forbids that when using ``@ui.page``.
"""

from __future__ import annotations

from tamga.gui.pages import forensic, ingest, results, run, study  # noqa: F401
