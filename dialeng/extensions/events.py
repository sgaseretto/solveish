"""Typed, ordered lifecycle events for public Dialeng extensions."""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


EventMode = Literal["observe", "transform", "guard"]


@dataclass(frozen=True)
class EventHandlerRegistration:
    name: str
    event: str
    handler: Callable
    mode: EventMode = "observe"
    order: int = 50
    timeout_seconds: float = 5.0


@dataclass
class EventOutcome:
    event: str
    payload: dict[str, Any]
    blocked: bool = False
    reason: str | None = None
    errors: list[dict[str, str]] = field(default_factory=list)


handlers: dict[str, EventHandlerRegistration] = {}
logger = logging.getLogger(__name__)


class ExtensionEventBus:
    def __init__(self) -> None:
        self.failure_counts: dict[str, int] = {}
        self.disabled_handlers: set[str] = set()
        self.last_diagnostics: list[dict[str, Any]] = []

    async def emit(self, event: str, payload: dict[str, Any] | None = None) -> EventOutcome:
        outcome = EventOutcome(event=event, payload=dict(payload or {}))
        operation_id = str(outcome.payload.get("operation_id") or uuid.uuid4().hex[:12])
        selected = sorted(
            (
                item for item in handlers.values()
                if item.event == event and item.name not in self.disabled_handlers
            ),
            key=lambda item: (item.order, item.name),
        )
        for registration in selected:
            started = time.perf_counter()
            try:
                result = registration.handler(outcome.payload)
                if inspect.isawaitable(result):
                    result = await asyncio.wait_for(result, registration.timeout_seconds)
                if registration.mode == "transform" and result is not None:
                    if not isinstance(result, dict):
                        raise TypeError("Transform handlers must return a dictionary or None")
                    outcome.payload = result
                elif registration.mode == "guard" and result:
                    if isinstance(result, str):
                        outcome.blocked = True
                        outcome.reason = result
                    elif isinstance(result, dict) and result.get("block"):
                        outcome.blocked = True
                        outcome.reason = str(result.get("reason") or "Blocked by extension")
                    if outcome.blocked:
                        break
                self.failure_counts.pop(registration.name, None)
            except Exception as exc:
                diagnostic = {
                    "handler": registration.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "event": event,
                    "operation_id": operation_id,
                    "duration_ms": f"{(time.perf_counter() - started) * 1000:.1f}",
                }
                outcome.errors.append(diagnostic)
                self.last_diagnostics.append(diagnostic)
                self.last_diagnostics = self.last_diagnostics[-100:]
                count = self.failure_counts.get(registration.name, 0) + 1
                self.failure_counts[registration.name] = count
                logger.warning(
                    "Extension event handler failed (handler=%s, event=%s, operation_id=%s, duration_ms=%s, failures=%s): %s",
                    registration.name, event, operation_id, diagnostic["duration_ms"], count, exc,
                )
                if count >= 3:
                    self.disabled_handlers.add(registration.name)
                    logger.error("Disabled repeatedly failing extension handler: %s", registration.name)
                if registration.mode == "guard":
                    outcome.blocked = True
                    outcome.reason = f"Extension guard failed: {registration.name}"
                    break
        return outcome

    def reset_handler(self, name: str) -> None:
        self.failure_counts.pop(name, None)
        self.disabled_handlers.discard(name)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "disabled_handlers": sorted(self.disabled_handlers),
            "failure_counts": dict(self.failure_counts),
            "recent_errors": list(self.last_diagnostics),
        }


event_bus = ExtensionEventBus()
