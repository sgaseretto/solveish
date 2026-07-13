"""FastHTML route registration modules."""

from .diagnostics import register_diagnostics_routes
from .e2e import register_e2e_routes
from .extensions import register_extension_routes
from .skills import register_skill_routes
from .websocket import register_websocket_routes

__all__ = [
    "register_diagnostics_routes",
    "register_e2e_routes",
    "register_extension_routes",
    "register_skill_routes",
    "register_websocket_routes",
]
