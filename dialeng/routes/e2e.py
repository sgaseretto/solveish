"""Test-only fault injection routes for browser E2E scenarios."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


def register_e2e_routes(
    rt,
    *,
    kernel_service,
    broadcast_json: Callable[[str, dict[str, Any]], Awaitable[None]],
    broadcast_kernel_snapshot: Callable[[str], Awaitable[None]],
) -> None:
    """Register controls that are reachable only in an E2E server process."""

    @rt("/dialeng/{nb_id}/__e2e__/kernel/terminate")
    async def terminate_kernel(nb_id: str):
        kernel = kernel_service._kernels.get(nb_id)
        process = getattr(kernel, "process", None)
        if process is None or not process.is_alive():
            return {"ok": False, "reason": "no_live_local_kernel"}

        pid = process.pid
        process.terminate()
        await asyncio.to_thread(process.join, 2)
        if process.is_alive():
            process.kill()
            await asyncio.to_thread(process.join, 2)
        await broadcast_kernel_snapshot(nb_id)
        return {"ok": not process.is_alive(), "pid": pid}

    @rt("/dialeng/{nb_id}/__e2e__/websocket/disconnect")
    async def disconnect_websocket(nb_id: str):
        await broadcast_json(nb_id, {"type": "e2e_force_disconnect"})
        return {"ok": True}
