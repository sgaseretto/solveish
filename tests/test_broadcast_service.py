"""Tests for bounded, independent WebSocket client delivery."""

import asyncio
import json
import time

from dialeng.services.broadcast_service import BroadcastManager
from dialeng.services.performance_service import PerformanceMonitor


def test_slow_client_does_not_delay_healthy_client():
    async def run():
        slow_release = asyncio.Event()
        healthy_received = asyncio.Event()
        healthy_messages = []

        async def slow_send(_message):
            await slow_release.wait()

        async def healthy_send(message):
            healthy_messages.append(json.loads(message))
            healthy_received.set()

        manager = BroadcastManager(send_timeout_seconds=1)
        manager.add_connection("demo", slow_send)
        manager.add_connection("demo", healthy_send)

        started_at = time.perf_counter()
        await manager.broadcast_json("demo", {"type": "cell_add", "cell_id": "one"})
        enqueue_elapsed = time.perf_counter() - started_at
        await asyncio.wait_for(healthy_received.wait(), timeout=0.2)

        assert enqueue_elapsed < 0.05
        assert healthy_messages[0]["cell_id"] == "one"
        slow_release.set()
        manager.drop_notebook("demo")

    asyncio.run(run())


def test_connection_id_removes_client_when_send_wrapper_changes():
    async def send_one(_message):
        return None

    async def send_two(_message):
        return None

    manager = BroadcastManager()
    manager.add_connection("demo", send_one, connection_id="socket-one")

    remaining = manager.remove_connection(
        "demo",
        send_two,
        connection_id="socket-one",
    )

    assert remaining == 0
    assert manager.client_count("demo") == 0


def test_drain_waits_for_terminal_event_before_connections_are_dropped():
    async def run():
        release = asyncio.Event()
        delivered = []

        async def send(message):
            await release.wait()
            delivered.append(json.loads(message))

        manager = BroadcastManager()
        manager.add_connection("demo", send)
        await manager.broadcast_json("demo", {"type": "notebook_deleted"})
        drain_task = asyncio.create_task(manager.drain_notebook("demo"))
        await asyncio.sleep(0)
        assert not drain_task.done()

        release.set()
        assert await drain_task is True
        manager.drop_notebook("demo")
        assert delivered == [{"type": "notebook_deleted"}]

    asyncio.run(run())


def test_removing_one_of_multiple_clients_updates_stored_collection():
    async def send_one(_message):
        return None

    async def send_two(_message):
        return None

    manager = BroadcastManager()
    manager.add_connection("demo", send_one, connection_id="socket-one")
    manager.add_connection("demo", send_two, connection_id="socket-two")

    remaining = manager.remove_connection(
        "demo",
        send_one,
        connection_id="socket-one",
    )

    assert remaining == 1
    assert manager.client_count("demo") == 1
    manager.drop_notebook("demo")


def test_snapshot_events_coalesce_to_latest_value():
    async def run():
        release = asyncio.Event()
        first_send = asyncio.Event()
        messages = []

        async def send(message):
            messages.append(json.loads(message))
            if len(messages) == 1:
                first_send.set()
                await release.wait()

        monitor = PerformanceMonitor()
        manager = BroadcastManager(monitor=monitor)
        manager.add_connection("demo", send)
        await manager.broadcast_json("demo", {"type": "cell_add", "cell_id": "block"})
        await asyncio.wait_for(first_send.wait(), timeout=0.2)
        for revision in (1, 2, 3):
            await manager.broadcast_json("demo", {
                "type": "kernel_snapshot",
                "revision": revision,
            })

        release.set()
        await asyncio.sleep(0.02)

        snapshots = [item for item in messages if item["type"] == "kernel_snapshot"]
        assert [item["revision"] for item in snapshots] == [3]
        assert monitor.snapshot()["counters"]["websocket.messages.coalesced"] == 2
        manager.drop_notebook("demo")

    asyncio.run(run())


def test_adjacent_stream_chunks_are_batched_without_reordering_terminal_event():
    async def run():
        release = asyncio.Event()
        first_send = asyncio.Event()
        messages = []

        async def send(message):
            messages.append(json.loads(message))
            if len(messages) == 1:
                first_send.set()
                await release.wait()

        manager = BroadcastManager()
        manager.add_connection("demo", send)
        await manager.broadcast_json("demo", {"type": "cell_add", "cell_id": "block"})
        await asyncio.wait_for(first_send.wait(), timeout=0.2)
        await manager.broadcast_json("demo", {
            "type": "code_stream_chunk", "cell_id": "c1", "stream": "stdout", "chunk": "a",
            "revision": 1, "event_id": "one",
        })
        await manager.broadcast_json("demo", {
            "type": "code_stream_chunk", "cell_id": "c1", "stream": "stdout", "chunk": "b",
            "revision": 2, "event_id": "two",
        })
        await manager.broadcast_json("demo", {
            "type": "code_stream_end", "cell_id": "c1", "has_error": False,
        })

        release.set()
        await asyncio.sleep(0.02)

        assert [item["type"] for item in messages] == [
            "cell_add", "code_stream_chunk", "code_stream_end",
        ]
        assert messages[1]["chunk"] == "ab"
        assert messages[1]["revision_start"] == 1
        assert messages[1]["revision"] == 2
        assert messages[1]["event_id"] == "two"
        manager.drop_notebook("demo")

    asyncio.run(run())


def test_structural_overflow_disconnects_client_instead_of_dropping_event():
    async def run():
        release = asyncio.Event()
        first_send = asyncio.Event()

        async def send(_message):
            first_send.set()
            await release.wait()

        monitor = PerformanceMonitor()
        manager = BroadcastManager(monitor=monitor, max_pending=2)
        manager.add_connection("demo", send)
        await manager.broadcast_json("demo", {"type": "cell_add", "cell_id": "running"})
        await asyncio.wait_for(first_send.wait(), timeout=0.2)
        await manager.broadcast_json("demo", {"type": "cell_add", "cell_id": "one"})
        await manager.broadcast_json("demo", {"type": "cell_add", "cell_id": "two"})
        await manager.broadcast_json("demo", {"type": "cell_add", "cell_id": "three"})

        assert manager.client_count("demo") == 0
        assert monitor.snapshot()["counters"]["websocket.clients.disconnected_overflow"] == 1
        release.set()

    asyncio.run(run())
