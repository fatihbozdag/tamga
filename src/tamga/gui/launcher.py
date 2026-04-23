"""Launch the tamga desktop GUI — NiceGUI rendered in a PyWebView native window."""

from __future__ import annotations


def launch(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    native: bool = True,
    window_width: int = 1200,
    window_height: int = 800,
    reload: bool = False,
) -> None:
    """Launch the desktop GUI.

    Parameters
    ----------
    host, port
        Bind address for the local NiceGUI server (only reachable from this machine).
    native
        When True (default), open a PyWebView-backed desktop window. When False,
        open the default browser against ``http://host:port``.
    window_width, window_height
        Native window dimensions in pixels; ignored when ``native=False``.
    reload
        Enable NiceGUI's hot-reload; only useful for GUI development.

    Raises
    ------
    ImportError
        If the ``gui`` extra (nicegui, pywebview) is not installed.
    """
    try:
        from nicegui import ui
    except ImportError as exc:
        raise ImportError(
            "tamga GUI requires the 'gui' extra. Install with: uv pip install 'tamga[gui]'"
        ) from exc

    if native:
        try:
            import webview  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Native desktop window requires pywebview (part of the 'gui' extra). "
                "Pass --no-native to open in a browser instead."
            ) from exc

    # Import registers @ui.page routes as a side-effect.
    from tamga.gui import app as _app  # noqa: F401

    ui.run(
        host=host,
        port=port,
        native=native,
        window_size=(window_width, window_height) if native else None,
        reload=reload,
        show=not native,
        title="tamga",
    )
