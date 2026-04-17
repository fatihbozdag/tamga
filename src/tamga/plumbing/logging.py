"""Logging helpers.

Loggers are namespaced under `tamga.*`. Verbosity is controlled once at the root and inherited.
Integrates with Rich for readable terminal output when the Rich handler is installed by the CLI.
"""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_ROOT = logging.getLogger("tamga")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. Names not starting with `tamga.` are prefixed."""
    if not name.startswith("tamga"):
        name = f"tamga.{name}"
    _configure_once()
    return logging.getLogger(name)


def set_verbosity(level: str | int) -> None:
    """Set the root `tamga` logger verbosity. Accepts 'DEBUG', 'INFO', 'WARNING', 'ERROR' or int."""
    _configure_once()
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    _ROOT.setLevel(level)


def _configure_once() -> None:
    if _ROOT.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    _ROOT.addHandler(handler)
    _ROOT.setLevel(logging.INFO)
    _ROOT.propagate = False
