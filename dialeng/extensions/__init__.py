"""Stable public facade for owner-aware Dialeng extensions.

Third-party extensions expose ``setup(api, ctx)`` through an ``extension.toml``
manifest. Imports stay lazy here because core registries capture contributions
without depending on the lifecycle manager during application bootstrap.
"""

__all__ = [
    "EXTENSION_API_VERSION",
    "ExtensionAPI",
    "ExtensionContext",
    "ExtensionManager",
    "ExtensionManifest",
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
    if name in {"EXTENSION_API_VERSION", "ExtensionManifest", "ExtensionSource"}:
        from .manifest import EXTENSION_API_VERSION, ExtensionManifest, ExtensionSource
        return {
            "EXTENSION_API_VERSION": EXTENSION_API_VERSION,
            "ExtensionManifest": ExtensionManifest,
            "ExtensionSource": ExtensionSource,
        }[name]
    raise AttributeError(name)
