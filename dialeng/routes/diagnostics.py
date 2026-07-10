"""Content-free performance and runtime diagnostics routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def register_diagnostics_routes(
    rt,
    *,
    performance_monitor,
    runtime_registry,
    broadcast_manager,
    autosave_manager,
    build_kernel_snapshot: Callable[[str], dict[str, Any]],
    runtime_counts: Callable[[], dict[str, int]],
) -> None:
    """Register diagnostics routes against explicitly supplied services."""

    @rt("/dialeng/diagnostics/performance")
    def get(recent_limit: int = 50):
        counts = runtime_counts()
        for name, value in counts.items():
            performance_monitor.set_gauge(f"runtime.{name}", value)
        performance_monitor.set_gauge(
            "runtime.websocket_clients",
            sum(
                broadcast_manager.client_count(notebook_id)
                for notebook_id in broadcast_manager.notebook_ids()
            ),
        )
        snapshot = performance_monitor.snapshot(recent_limit=max(0, min(recent_limit, 200)))
        snapshot["runtime_registry"] = runtime_registry.diagnostics()
        return snapshot

    @rt("/dialeng/diagnostics/browser")
    def post(
        notebook_id: str = "",
        long_task_count: int = 0,
        long_task_total_ms: int = 0,
        long_task_max_ms: int = 0,
        editor_count: int = 0,
        visible_cell_count: int = 0,
        dom_cell_count: int = 0,
    ):
        performance_monitor.increment("browser.metrics_batches")
        performance_monitor.increment("browser.long_tasks", max(0, long_task_count))
        performance_monitor.set_gauge("browser.last_editor_count", max(0, editor_count))
        performance_monitor.set_gauge(
            "browser.last_visible_cell_count", max(0, visible_cell_count)
        )
        performance_monitor.set_gauge("browser.last_dom_cell_count", max(0, dom_cell_count))
        performance_monitor.set_gauge(
            "browser.last_long_task_max_ms", max(0, long_task_max_ms)
        )
        if long_task_count > 0:
            performance_monitor.observe_duration(
                "browser.long_task",
                long_task_total_ms / long_task_count,
                labels={"source": "browser_batch"},
                record_event=False,
            )
        return {"ok": True}

    @rt("/dialeng/{nb_id}/diagnostics")
    def get(nb_id: str):
        snapshot = build_kernel_snapshot(nb_id)
        runtime = runtime_registry.get(nb_id)
        return {
            "notebook_id": nb_id,
            "kernel": snapshot["kernel"],
            "queue": snapshot["queue"],
            "setup": snapshot["setup"],
            "auth": snapshot["auth"],
            "revisions": snapshot["revisions"],
            "runtime": {
                "lifecycle": runtime.lifecycle.value if runtime else "unknown",
                "kernel_generation": runtime.kernel_generation if runtime else 0,
                "mutation_locked": runtime.mutation_lock.locked() if runtime else False,
                "websocket_clients": broadcast_manager.client_count(nb_id),
                "autosave_pending": autosave_manager.has_pending(nb_id),
            },
        }
