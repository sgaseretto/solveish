"""FastHTML route registration modules."""

from .diagnostics import register_diagnostics_routes
from .websocket import register_websocket_routes

__all__ = ["register_diagnostics_routes", "register_websocket_routes"]
