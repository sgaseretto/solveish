"""Small helpers for structured, correlation-friendly server logs."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any


def new_operation_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def log_operation(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    op_id: str | None = None,
    **fields: Any,
) -> str:
    """Log an event as stable key/value fields and return its operation id."""
    operation_id = op_id or new_operation_id(event.replace("_", "-"))
    all_fields = {"op_id": operation_id, **fields}
    rendered_fields = " ".join(f"{key}={_format_value(value)}" for key, value in all_fields.items())
    logger.log(level, "[operation] event=%s %s", event, rendered_fields)
    return operation_id


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _format_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.1f}"
    text = str(value).replace("\n", "\\n")
    if " " in text or "\t" in text:
        return repr(text)
    return text
