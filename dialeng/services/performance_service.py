"""Bounded, privacy-safe performance instrumentation for Dialeng."""

from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Mapping


_SECRET_PARTS = {
    "authorization",
    "cookie",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil((percentile / 100) * len(ordered)) - 1)
    return ordered[index]


def _safe_value(key: str, value: Any) -> Any:
    key_parts = set(key.lower().replace("-", "_").split("_"))
    if key_parts & _SECRET_PARTS or key.lower() in {"api_key", "account_email"}:
        return "[redacted]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _safe_value(str(k), v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(key, item) for item in list(value)[:20]]
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= 240 else f"{text[:237]}..."


def normalize_http_path(path: str) -> str:
    """Bound route cardinality without retaining notebook or cell identifiers."""
    static_dialeng_paths = (
        "/dialeng/diagnostics/",
        "/dialeng/file",
        "/dialeng/new",
        "/dialeng/reload-extensions",
        "/dialeng/workspace",
    )
    if path.startswith(static_dialeng_paths):
        return path
    normalized = re.sub(r"^/dialeng/[^/]+", "/dialeng/{notebook_id}", path)
    normalized = re.sub(r"/cell/[^/]+", "/cell/{cell_id}", normalized)
    normalized = re.sub(r"/ws/[^/]+", "/ws/{notebook_id}", normalized)
    return normalized


@dataclass
class _DurationSeries:
    max_samples: int

    def __post_init__(self) -> None:
        self.samples: deque[float] = deque(maxlen=self.max_samples)
        self.count = 0
        self.total_ms = 0.0
        self.min_ms: float | None = None
        self.max_ms = 0.0

    def observe(self, duration_ms: float) -> None:
        value = max(0.0, float(duration_ms))
        self.samples.append(value)
        self.count += 1
        self.total_ms += value
        self.min_ms = value if self.min_ms is None else min(self.min_ms, value)
        self.max_ms = max(self.max_ms, value)

    def snapshot(self) -> dict[str, Any]:
        values = list(self.samples)
        return {
            "count": self.count,
            "avg_ms": round(self.total_ms / self.count, 2) if self.count else 0.0,
            "min_ms": round(self.min_ms or 0.0, 2),
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(_percentile(values, 50), 2),
            "p95_ms": round(_percentile(values, 95), 2),
            "recent_samples": len(values),
        }


class PerformanceMonitor:
    """Collect bounded counters, gauges, durations, and recent events."""

    def __init__(
        self,
        *,
        max_events: int = 500,
        max_samples: int = 256,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.max_events = max(1, max_events)
        self.max_samples = max(1, max_samples)
        self._monotonic = monotonic
        self._started_at = monotonic()
        self._events: deque[dict[str, Any]] = deque(maxlen=self.max_events)
        self._counters: defaultdict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._durations: dict[tuple[str, tuple[tuple[str, str], ...]], _DurationSeries] = {}
        self._lock = threading.RLock()

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def record_event(self, name: str, **fields: Any) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "fields": {key: _safe_value(key, value) for key, value in fields.items()},
        }
        with self._lock:
            self._events.append(event)

    def observe_duration(
        self,
        name: str,
        duration_ms: float,
        *,
        labels: Mapping[str, Any] | None = None,
        record_event: bool = True,
        **fields: Any,
    ) -> None:
        safe_labels = {
            str(key): str(_safe_value(str(key), value))
            for key, value in (labels or {}).items()
        }
        series_key = (name, tuple(sorted(safe_labels.items())))
        with self._lock:
            series = self._durations.get(series_key)
            if series is None:
                series = _DurationSeries(self.max_samples)
                self._durations[series_key] = series
            series.observe(duration_ms)
        if record_event:
            self.record_event(
                name,
                duration_ms=round(float(duration_ms), 2),
                labels=safe_labels,
                **fields,
            )

    @contextmanager
    def measure(
        self,
        name: str,
        *,
        labels: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> Iterator[None]:
        start = self._monotonic()
        try:
            yield
        except BaseException as exc:
            self.observe_duration(
                name,
                (self._monotonic() - start) * 1000,
                labels=labels,
                outcome="error",
                error_type=type(exc).__name__,
                **fields,
            )
            raise
        else:
            self.observe_duration(
                name,
                (self._monotonic() - start) * 1000,
                labels=labels,
                outcome="success",
                **fields,
            )

    def snapshot(self, *, recent_limit: int = 50) -> dict[str, Any]:
        with self._lock:
            durations = []
            for (name, labels), series in sorted(self._durations.items()):
                durations.append({
                    "name": name,
                    "labels": dict(labels),
                    **series.snapshot(),
                })
            events = list(self._events)[-max(0, min(recent_limit, self.max_events)):]
            return {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": round(self._monotonic() - self._started_at, 2),
                "limits": {
                    "max_events": self.max_events,
                    "max_samples_per_series": self.max_samples,
                },
                "counters": dict(sorted(self._counters.items())),
                "gauges": {key: round(value, 3) for key, value in sorted(self._gauges.items())},
                "durations": durations,
                "recent_events": events,
            }

    def reset(self) -> None:
        with self._lock:
            self._started_at = self._monotonic()
            self._events.clear()
            self._counters.clear()
            self._gauges.clear()
            self._durations.clear()


class RequestMetricsMiddleware:
    """ASGI middleware that records bounded HTTP route metrics."""

    def __init__(self, app, monitor: PerformanceMonitor) -> None:
        self.app = app
        self.monitor = monitor

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 500
        response_bytes = 0

        async def send_with_metrics(message) -> None:
            nonlocal status_code, response_bytes
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
            elif message.get("type") == "http.response.body":
                response_bytes += len(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, receive, send_with_metrics)
        finally:
            route = normalize_http_path(scope.get("path", ""))
            method = scope.get("method", "GET")
            duration_ms = (time.perf_counter() - start) * 1000
            self.monitor.increment("http.requests.total")
            self.monitor.increment(f"http.responses.{status_code // 100}xx")
            self.monitor.observe_duration(
                "http.request",
                duration_ms,
                labels={"method": method, "route": route, "status": status_code},
                response_bytes=response_bytes,
            )


async def monitor_event_loop_lag(
    monitor: PerformanceMonitor,
    *,
    interval_seconds: float = 1.0,
    warning_ms: float = 100.0,
) -> None:
    """Continuously measure event-loop scheduling lag until cancelled."""
    loop = asyncio.get_running_loop()
    expected = loop.time() + interval_seconds
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            now = loop.time()
            lag_ms = max(0.0, (now - expected) * 1000)
            monitor.set_gauge("event_loop.lag_ms", lag_ms)
            monitor.observe_duration(
                "event_loop.lag",
                lag_ms,
                record_event=lag_ms >= warning_ms,
                outcome="warning" if lag_ms >= warning_ms else "ok",
            )
            if lag_ms >= warning_ms:
                monitor.increment("event_loop.lag_warnings")
            expected = now + interval_seconds
    except asyncio.CancelledError:
        raise


performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    return performance_monitor
