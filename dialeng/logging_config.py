"""Console logging configuration for Dialeng.

Provides a Uvicorn-compatible log config that:
- emits `dialeng.*` logger output to the terminal
- keeps Uvicorn's normal startup/error logs
- suppresses repetitive access-log noise from polling/static requests
"""
from __future__ import annotations

import copy
import logging
from typing import Any

import uvicorn.config


class DialengAccessFilter(logging.Filter):
    """Drop noisy access-log lines that drown out app-level runtime logs."""

    QUIET_EXACT_PATHS = {
        "/favicon.ico",
        "/render-markdown",
    }
    QUIET_PREFIXES = (
        "/static/",
    )
    QUIET_SUBSTRINGS = (
        "/kernel/snapshot",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        args = getattr(record, "args", ())
        if not isinstance(args, tuple) or len(args) < 5:
            return True

        path = str(args[2])
        try:
            status_code = int(args[4])
        except (TypeError, ValueError):
            status_code = 0

        # Keep errors and redirects visible even for quiet endpoints.
        if status_code >= 400 or (300 <= status_code < 400):
            return True
        if path in self.QUIET_EXACT_PATHS:
            return False
        if any(path.startswith(prefix) for prefix in self.QUIET_PREFIXES):
            return False
        if any(fragment in path for fragment in self.QUIET_SUBSTRINGS):
            return False
        return True


def build_log_config(level: str = "INFO") -> dict[str, Any]:
    """Build the Uvicorn log config used by Dialeng."""
    config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    config.setdefault("filters", {})
    config["filters"]["dialeng_access"] = {
        "()": "dialeng.logging_config.DialengAccessFilter",
    }
    config["formatters"]["dialeng"] = {
        "()": "uvicorn.logging.DefaultFormatter",
        "fmt": "%(levelprefix)s [%(name)s] %(message)s",
        "use_colors": None,
    }
    config["formatters"]["access"]["fmt"] = '%(levelprefix)s [http] %(client_addr)s - "%(request_line)s" %(status_code)s'
    config["handlers"]["dialeng"] = {
        "formatter": "dialeng",
        "class": "logging.StreamHandler",
        "stream": "ext://sys.stderr",
    }
    config["handlers"]["access"]["filters"] = ["dialeng_access"]
    config["loggers"]["dialeng"] = {
        "handlers": ["dialeng"],
        "level": level,
        "propagate": False,
    }
    return config
