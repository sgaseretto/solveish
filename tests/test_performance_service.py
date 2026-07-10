"""Tests for bounded performance instrumentation."""

import asyncio
import json

from dialeng.services.performance_service import (
    PerformanceMonitor,
    RequestMetricsMiddleware,
    monitor_event_loop_lag,
    normalize_http_path,
)


def test_monitor_bounds_events_samples_and_calculates_percentiles():
    monitor = PerformanceMonitor(max_events=2, max_samples=3)
    for duration in (10, 20, 30, 40):
        monitor.observe_duration("notebook.save", duration, labels={"mode": "explicit"})

    snapshot = monitor.snapshot(recent_limit=10)
    series = snapshot["durations"][0]

    assert series["count"] == 4
    assert series["recent_samples"] == 3
    assert series["avg_ms"] == 25.0
    assert series["p50_ms"] == 30.0
    assert series["p95_ms"] == 40.0
    assert len(snapshot["recent_events"]) == 2


def test_monitor_redacts_sensitive_fields_and_truncates_values():
    monitor = PerformanceMonitor(max_events=5)
    monitor.record_event(
        "provider.probe",
        access_token="secret-token",
        account_email="person@example.com",
        detail="x" * 300,
    )

    fields = monitor.snapshot()["recent_events"][0]["fields"]
    assert fields["access_token"] == "[redacted]"
    assert fields["account_email"] == "[redacted]"
    assert fields["detail"].endswith("...")
    assert len(fields["detail"]) == 240


def test_measure_records_success_and_error_outcomes():
    monitor = PerformanceMonitor()
    with monitor.measure("successful"):
        pass

    try:
        with monitor.measure("failed"):
            raise ValueError("boom")
    except ValueError:
        pass

    events = monitor.snapshot()["recent_events"]
    assert [event["fields"]["outcome"] for event in events] == ["success", "error"]
    assert events[-1]["fields"]["error_type"] == "ValueError"


def test_normalize_http_path_removes_resource_ids():
    assert normalize_http_path("/dialeng/demo~project/cell/abc123/run") == (
        "/dialeng/{notebook_id}/cell/{cell_id}/run"
    )
    assert normalize_http_path("/ws/private-notebook") == "/ws/{notebook_id}"
    assert normalize_http_path("/dialeng/diagnostics/performance") == (
        "/dialeng/diagnostics/performance"
    )


def test_request_metrics_middleware_records_route_status_and_bytes():
    monitor = PerformanceMonitor()

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 201, "headers": []})
        await send({"type": "http.response.body", "body": b"created"})

    async def run():
        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        middleware = RequestMetricsMiddleware(app, monitor)
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/dialeng/demo/cell/c1/run",
            },
            receive,
            send,
        )
        return sent

    sent = asyncio.run(run())
    snapshot = monitor.snapshot()

    assert sent[-1]["body"] == b"created"
    assert snapshot["counters"]["http.requests.total"] == 1
    assert snapshot["counters"]["http.responses.2xx"] == 1
    assert snapshot["durations"][0]["labels"] == {
        "method": "POST",
        "route": "/dialeng/{notebook_id}/cell/{cell_id}/run",
        "status": "201",
    }
    assert snapshot["recent_events"][0]["fields"]["response_bytes"] == 7


def test_event_loop_lag_monitor_updates_gauge_and_stops_cleanly():
    monitor = PerformanceMonitor()

    async def run():
        task = asyncio.create_task(
            monitor_event_loop_lag(monitor, interval_seconds=0.001, warning_ms=10_000)
        )
        await asyncio.sleep(0.005)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())

    snapshot = monitor.snapshot()
    assert "event_loop.lag_ms" in snapshot["gauges"]
    assert snapshot["durations"][0]["name"] == "event_loop.lag"


def test_snapshot_is_json_serializable():
    monitor = PerformanceMonitor()
    monitor.increment("example")
    monitor.set_gauge("queue.depth", 2)

    json.dumps(monitor.snapshot())
