"""Framework-light control protocol for Dialeng browser WebSockets."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable


SendCallable = Callable[[str], Awaitable[None]]


def websocket_connection_id(scope: dict[str, Any]) -> str | None:
    """Return the stable per-socket key from a WebSocket handshake."""
    for name, value in scope.get("headers", []):
        if bytes(name).lower() == b"sec-websocket-key":
            return bytes(value).decode("ascii", errors="replace")
    return None


def deleted_notebook_payload(notebook_id: str) -> dict[str, Any]:
    return {
        "type": "notebook_deleted",
        "notebook_id": notebook_id,
        "workspace_url": "/dialeng/workspace?deleted=1",
        "message": "Notebook deleted.",
    }


def revision_mismatches(
    client_revisions: dict[str, Any],
    server_revisions: dict[str, int],
) -> list[str]:
    """Return durable streams that require authoritative recovery."""
    durable_streams = ("notebook", "output", "persistence")
    return [
        stream
        for stream in durable_streams
        if int(client_revisions.get(stream, 0) or 0) != server_revisions[stream]
    ]


async def handle_control_message(
    message: str | None,
    send: SendCallable,
    notebook_id: str,
    *,
    notebook_unavailable: Callable[[str], bool],
    revision_snapshot: Callable[[str], dict[str, int]],
    send_kernel_snapshot: Callable[[str, SendCallable], Awaitable[None]],
    cancel_cell: Callable[[str, str], Any],
    cancel_extension_command: Callable[[str], Any] | None = None,
) -> bool:
    """Handle one browser control message and report whether it was recognized."""
    if message is None or not isinstance(message, str) or not message:
        return False
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return False

    message_type = data.get("type")
    if message_type == "join":
        if notebook_unavailable(notebook_id):
            await send(json.dumps(deleted_notebook_payload(notebook_id)))
            return True
        server_revisions = revision_snapshot(notebook_id)
        mismatches = revision_mismatches(data.get("revisions") or {}, server_revisions)
        if mismatches:
            await send(json.dumps({
                "type": "resync_required",
                "notebook_id": notebook_id,
                "reason": "revision_mismatch",
                "streams": mismatches,
                "server_revisions": server_revisions,
            }))
        await send_kernel_snapshot(notebook_id, send)
        return True

    if message_type == "heartbeat":
        await send(json.dumps({
            "type": "heartbeat_ack",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
        return True

    if message_type == "cancel":
        cell_id = data.get("cell_id")
        if cell_id:
            result = cancel_cell(notebook_id, str(cell_id))
            if inspect.isawaitable(result):
                await result
        return True

    if message_type == "extension_command_cancel":
        operation_id = str(data.get("operation_id") or "")
        cancelled = False
        if operation_id and cancel_extension_command is not None:
            result = cancel_extension_command(operation_id)
            cancelled = bool(await result if inspect.isawaitable(result) else result)
        await send(json.dumps({
            "type": "extension_command_cancel_ack",
            "operation_id": operation_id,
            "cancelled": cancelled,
        }))
        return True

    return False
