"""WebSocket route registration and notebook connection lifecycle."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from dialeng.services.websocket_protocol import (
    deleted_notebook_payload,
    handle_control_message,
    websocket_connection_id,
)


def _notebook_id_from_scope(scope: dict[str, Any]) -> str:
    parts = scope.get("path", "").strip("/").split("/")
    return parts[1] if len(parts) > 1 else "default"


def register_websocket_routes(
    app,
    *,
    broadcast_manager,
    kernel_service,
    notebook_unavailable: Callable[[str], bool],
    revision_snapshot: Callable[[str], dict[str, int]],
    send_kernel_snapshot: Callable[[str, Callable[[str], Awaitable[None]]], Awaitable[None]],
    cancel_cell: Callable[[str, str], Awaitable[None]],
    logger,
    cancel_extension_command: Callable[[str], Awaitable[bool]] | None = None,
) -> None:
    """Register the notebook socket with explicit lifecycle dependencies."""

    async def on_connect(send, scope):
        notebook_id = _notebook_id_from_scope(scope)
        if notebook_unavailable(notebook_id):
            await send(json.dumps(deleted_notebook_payload(notebook_id)))
            return
        client_count = broadcast_manager.add_connection(
            notebook_id,
            send,
            connection_id=websocket_connection_id(scope),
        )
        kernel_service.set_client_count(notebook_id, client_count)
        logger.info(
            "WebSocket client connected (notebook=%s, connections=%s)",
            notebook_id,
            client_count,
        )

    async def on_disconnect(send, scope):
        notebook_id = _notebook_id_from_scope(scope)
        client_count = broadcast_manager.remove_connection(
            notebook_id,
            send,
            connection_id=websocket_connection_id(scope),
        )
        kernel_service.set_client_count(notebook_id, client_count)
        logger.info(
            "WebSocket client disconnected (notebook=%s, connections=%s)",
            notebook_id,
            client_count,
        )

    @app.ws("/ws/{nb_id}", conn=on_connect, disconn=on_disconnect)
    async def ws(msg: str, send, nb_id: str):
        await handle_control_message(
            msg,
            send,
            nb_id,
            notebook_unavailable=notebook_unavailable,
            revision_snapshot=revision_snapshot,
            send_kernel_snapshot=send_kernel_snapshot,
            cancel_cell=cancel_cell,
            cancel_extension_command=cancel_extension_command,
        )
