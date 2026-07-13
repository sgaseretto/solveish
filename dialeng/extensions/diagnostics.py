"""Secret-redacted diagnostics for extension inventory and browser responses."""
from __future__ import annotations

import os
import re
from typing import Any


_ASSIGNMENT = re.compile(
    r"(?i)(authorization|cookie|set-cookie|access[_-]?token|refresh[_-]?token|api[_-]?key|secret)"
    r"\s*[:=]\s*([^\s,;]+)"
)


def redact_text(value: Any) -> str:
    text = str(value)
    text = _ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    for name, secret in os.environ.items():
        if not secret or len(secret) < 8:
            continue
        upper = name.upper()
        if any(marker in upper for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "COOKIE")):
            text = text.replace(secret, "[REDACTED]")
    return text


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(
                marker in str(key).lower()
                for marker in ("token", "secret", "password", "cookie", "authorization", "api_key")
            ) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item) for item in value]
    return redact_text(value) if isinstance(value, str) else value
