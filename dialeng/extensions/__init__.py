"""Stable public facade for owner-aware Dialeng extensions.

Third-party extensions expose ``setup(api, ctx)`` through an ``extension.toml``
manifest. Imports stay lazy here because core registries capture contributions
without depending on the lifecycle manager during application bootstrap.
"""

__all__ = [
    "EXTENSION_API_VERSION",
    "ExtensionAPIPolicy",
    "ExtensionAPI",
    "ExtensionCellType",
    "ExtensionContext",
    "ExtensionKernel",
    "ExtensionManager",
    "ExtensionManifest",
    "ExtensionProvider",
    "ExtensionSetting",
    "ExtensionSource",
]


def __getattr__(name):
    """Load the public facade lazily so core registry imports stay acyclic."""
    if name in {"ExtensionAPI", "ExtensionContext"}:
        from .api import ExtensionAPI, ExtensionContext
        return {"ExtensionAPI": ExtensionAPI, "ExtensionContext": ExtensionContext}[name]
    if name == "ExtensionManager":
        from .manager import ExtensionManager
        return ExtensionManager
    if name in {
        "EXTENSION_API_VERSION", "ExtensionAPIPolicy", "ExtensionCellType",
        "ExtensionKernel", "ExtensionProvider", "ExtensionSetting",
    }:
        from .contracts import (
            EXTENSION_API_VERSION,
            ExtensionAPIPolicy,
            ExtensionCellType,
            ExtensionKernel,
            ExtensionProvider,
            ExtensionSetting,
        )
        return {
            "EXTENSION_API_VERSION": EXTENSION_API_VERSION,
            "ExtensionAPIPolicy": ExtensionAPIPolicy,
            "ExtensionCellType": ExtensionCellType,
            "ExtensionKernel": ExtensionKernel,
            "ExtensionProvider": ExtensionProvider,
            "ExtensionSetting": ExtensionSetting,
        }[name]
    if name in {"ExtensionManifest", "ExtensionSource"}:
        from .manifest import ExtensionManifest, ExtensionSource
        return {
            "ExtensionManifest": ExtensionManifest,
            "ExtensionSource": ExtensionSource,
        }[name]
    raise AttributeError(name)
