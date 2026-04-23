"""Desktop GUI for tamga (optional; install with ``tamga[gui]``).

Importing this sub-package only succeeds when NiceGUI and PyWebView are installed.
Production entry point is :func:`tamga.gui.launcher.launch`, normally invoked via
``tamga gui`` on the command line.
"""

from __future__ import annotations

from tamga.gui.launcher import launch

__all__ = ["launch"]
