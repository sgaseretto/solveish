"""
dialeng.dev - Development helpers for extension authors.

Usage from a notebook kernel:
    from dialeng.dev import reload_extensions
    reload_extensions()

This triggers re-extraction of #| export cells from AUTORUN notebooks,
reimports all extension modules, and auto-refreshes all connected browser tabs.
"""

import json
import urllib.request


def reload_extensions(host: str = "localhost", port: int = 8000) -> dict:
    """Reload all AUTORUN extensions and refresh connected notebooks.

    Re-extracts #| export cells from AUTORUN/*.ipynb, reimports the
    generated modules, and broadcasts a page refresh to all connected
    browser tabs via WebSocket.

    Args:
        host: Dialeng server host (default: localhost)
        port: Dialeng server port (default: 8000)

    Returns:
        dict with 'extracted', 'loaded', and 'errors' keys.
    """
    url = f"http://{host}:{port}/dialeng/reload-extensions"
    req = urllib.request.Request(url, method="POST", data=b"")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        print(f"Failed to reload extensions: {e}")
        return {"error": str(e)}

    extracted = result.get("extracted", [])
    loaded = result.get("loaded", [])
    errors = result.get("errors", [])

    if extracted:
        print(f"Extracted from: {', '.join(extracted)}")
    if loaded:
        print(f"Loaded {len(loaded)} extension(s): {', '.join(loaded)}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    if not loaded and not errors:
        print("No extensions found to reload.")

    return result
