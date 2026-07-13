"""Unit tests for browser WebSocket control protocol behavior."""

import asyncio
import json

from dialeng.services.websocket_protocol import (
    handle_control_message,
    revision_mismatches,
    websocket_connection_id,
)


def test_connection_id_comes_from_handshake_key():
    scope = {"headers": [(b"host", b"localhost"), (b"Sec-WebSocket-Key", b"abc123")]}
    assert websocket_connection_id(scope) == "abc123"


def test_revision_mismatches_only_include_durable_recovery_streams():
    assert revision_mismatches(
        {"notebook": 1, "kernel": 0, "output": 2, "persistence": 0},
        {"notebook": 2, "kernel": 9, "output": 2, "persistence": 1},
    ) == ["notebook", "persistence"]


def test_cancel_dispatch_and_invalid_payload_are_isolated():
    async def run():
        sent = []
        cancelled = []

        async def send(message):
            sent.append(json.loads(message))

        async def send_snapshot(_notebook_id, _send):
            return None

        kwargs = {
            "notebook_unavailable": lambda _id: False,
            "revision_snapshot": lambda _id: {
                "notebook": 0,
                "kernel": 0,
                "outline": 0,
                "llm": 0,
                "output": 0,
                "persistence": 0,
            },
            "send_kernel_snapshot": send_snapshot,
            "cancel_cell": lambda notebook_id, cell_id: cancelled.append((notebook_id, cell_id)),
        }
        assert await handle_control_message("not-json", send, "demo", **kwargs) is False
        assert await handle_control_message(
            json.dumps({"type": "cancel", "cell_id": "cell-one"}),
            send,
            "demo",
            **kwargs,
        ) is True
        assert cancelled == [("demo", "cell-one")]
        assert sent == []

    asyncio.run(run())


def test_extension_command_cancel_uses_control_socket_and_acknowledges_result():
    async def run():
        sent = []
        cancelled = []

        async def send(message):
            sent.append(json.loads(message))

        async def cancel(operation_id):
            cancelled.append(operation_id)
            return operation_id == "ext-command-running"

        handled = await handle_control_message(
            json.dumps({
                "type": "extension_command_cancel",
                "operation_id": "ext-command-running",
            }),
            send,
            "demo",
            notebook_unavailable=lambda _id: False,
            revision_snapshot=lambda _id: {},
            send_kernel_snapshot=lambda _id, _send: None,
            cancel_cell=lambda _notebook_id, _cell_id: None,
            cancel_extension_command=cancel,
        )

        assert handled is True
        assert cancelled == ["ext-command-running"]
        assert sent == [{
            "type": "extension_command_cancel_ack",
            "operation_id": "ext-command-running",
            "cancelled": True,
        }]

    asyncio.run(run())
