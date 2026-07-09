"""Helpers for future Colab terminal support."""
from __future__ import annotations

from urllib.parse import urlencode, urlparse

COLAB_TTY_PATH = "/colab/tty"


def build_colab_tty_ws_url(proxy_url: str, proxy_token: str) -> str:
    """Return the runtime-proxy WebSocket URL for Colab's raw tty endpoint.

    Dialeng does not expose a terminal UI yet. This helper codifies the
    google-colab-cli reference path so future terminal work does not tunnel
    shell traffic through the Jupyter kernel.
    """
    parsed = urlparse(proxy_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    params = urlencode({"colab-runtime-proxy-token": proxy_token})
    return f"{scheme}://{parsed.netloc}{COLAB_TTY_PATH}?{params}"
