"""Desktop GUI for bitig (optional; install with ``bitig[gui]``).

Importing this sub-package only succeeds when NiceGUI and PyWebView are installed.
Production entry point is :func:`bitig.gui.launcher.launch`, normally invoked via
``bitig gui`` on the command line.
"""

from __future__ import annotations

from bitig.gui.launcher import launch

__all__ = ["launch"]
