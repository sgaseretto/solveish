"""Bounded WebSocket delivery for notebook clients."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, MutableMapping

SendCallable = Callable[[str], Awaitable[None]]

_COALESCED_EVENTS = {"kernel_snapshot", "llm_status", "queue_update"}
_BATCHED_EVENTS = {"code_stream_chunk", "stream_chunk"}


@dataclass
class _OutboundMessage:
    text: str | None = None
    data: dict[str, Any] | None = None
    event_type: str = "html"
    key: tuple[Any, ...] = ()

    def render(self) -> str:
        return self.text if self.text is not None else json.dumps(self.data or {})


@dataclass
class _Client:
    send: SendCallable
    max_pending: int
    connection_id: str | None = None
    pending: deque[_OutboundMessage] = field(default_factory=deque)
    wakeup: asyncio.Event | None = None
    writer_task: asyncio.Task | None = None
    drained: asyncio.Event | None = None
    closed: bool = False


class BroadcastManager:
    """Deliver events independently so a slow tab cannot stall healthy tabs."""

    def __init__(
        self,
        connections: MutableMapping[str, list[Any]] | None = None,
        *,
        monitor: Any = None,
        max_pending: int = 256,
        send_timeout_seconds: float = 5.0,
        max_batch_bytes: int = 64 * 1024,
        on_client_count_changed: Callable[[str, int], None] | None = None,
    ):
        self._connections = connections if connections is not None else {}
        self._monitor = monitor
        self._max_pending = max(1, max_pending)
        self._send_timeout_seconds = max(0.01, send_timeout_seconds)
        self._max_batch_bytes = max(1024, max_batch_bytes)
        self._on_client_count_changed = on_client_count_changed

    def _notify_client_count(self, notebook_id: str) -> None:
        if self._on_client_count_changed is not None:
            self._on_client_count_changed(notebook_id, self.client_count(notebook_id))

    def _clients(self, notebook_id: str) -> list[_Client]:
        raw_clients = self._connections.get(notebook_id, [])
        if all(isinstance(item, _Client) for item in raw_clients):
            return raw_clients
        normalized: list[_Client] = []
        for item in raw_clients:
            normalized.append(item if isinstance(item, _Client) else _Client(item, self._max_pending))
        self._connections[notebook_id] = normalized
        return normalized

    def has_clients(self, notebook_id: str) -> bool:
        return bool(self._clients(notebook_id))

    def client_count(self, notebook_id: str) -> int:
        return len(self._clients(notebook_id))

    def notebook_ids(self) -> list[str]:
        return list(self._connections.keys())

    def add_connection(
        self,
        notebook_id: str,
        send: SendCallable,
        *,
        connection_id: str | None = None,
    ) -> int:
        client = _Client(
            send=send,
            max_pending=self._max_pending,
            connection_id=connection_id,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Synchronous tests can still register senders; delivery remains direct.
            loop = None
        if loop is not None:
            client.wakeup = asyncio.Event()
            client.drained = asyncio.Event()
            client.drained.set()
            client.writer_task = loop.create_task(
                self._writer_loop(notebook_id, client),
                name=f"dialeng-ws-writer-{notebook_id}",
            )
        self._connections.setdefault(notebook_id, []).append(client)
        self._update_gauges()
        self._notify_client_count(notebook_id)
        return self.client_count(notebook_id)

    def remove_connection(
        self,
        notebook_id: str,
        send: SendCallable,
        *,
        connection_id: str | None = None,
    ) -> int:
        clients = self._clients(notebook_id)
        for client in list(clients):
            same_connection = (
                connection_id is not None
                and client.connection_id == connection_id
            )
            if same_connection or (connection_id is None and client.send is send):
                self._close_client(client)
                clients.remove(client)
        if not clients:
            self._connections.pop(notebook_id, None)
        self._update_gauges()
        self._notify_client_count(notebook_id)
        return len(clients)

    def drop_notebook(self, notebook_id: str) -> None:
        for client in self._clients(notebook_id):
            self._close_client(client)
        self._connections.pop(notebook_id, None)
        self._update_gauges()
        self._notify_client_count(notebook_id)

    def _close_client(self, client: _Client) -> None:
        client.closed = True
        client.pending.clear()
        if client.drained is not None:
            client.drained.set()
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        if client.writer_task and client.writer_task is not current_task:
            client.writer_task.cancel()

    async def _writer_loop(self, notebook_id: str, client: _Client) -> None:
        assert client.wakeup is not None
        try:
            while not client.closed:
                await client.wakeup.wait()
                while client.pending and not client.closed:
                    outbound = client.pending.popleft()
                    self._update_gauges()
                    started_at = time.perf_counter()
                    await asyncio.wait_for(
                        client.send(outbound.render()),
                        timeout=self._send_timeout_seconds,
                    )
                    if self._monitor is not None:
                        self._monitor.observe_duration(
                            "websocket.client_send",
                            (time.perf_counter() - started_at) * 1000,
                            labels={"event_type": outbound.event_type},
                            record_event=False,
                        )
                if client.drained is not None:
                    client.drained.set()
                client.wakeup.clear()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._monitor is not None:
                self._monitor.increment("websocket.clients.disconnected_slow_or_failed")
                self._monitor.record_event(
                    "websocket.client_writer_failed",
                    notebook_id=notebook_id,
                    error_type=type(exc).__name__,
                )
            client.closed = True
            if client.drained is not None:
                client.drained.set()
            clients = self._clients(notebook_id)
            if client in clients:
                clients.remove(client)
            if not clients:
                self._connections.pop(notebook_id, None)
            self._update_gauges()
            self._notify_client_count(notebook_id)

    def _update_gauges(self) -> None:
        if self._monitor is None:
            return
        clients = [client for nb_id in self.notebook_ids() for client in self._clients(nb_id)]
        self._monitor.set_gauge("websocket.clients.total", len(clients))
        self._monitor.set_gauge(
            "websocket.outbound_queue_depth",
            sum(len(client.pending) for client in clients),
        )
        self._monitor.set_gauge(
            "websocket.outbound_queue_max",
            max((len(client.pending) for client in clients), default=0),
        )

    @staticmethod
    def _event_key(data: dict[str, Any]) -> tuple[Any, ...]:
        event_type = str(data.get("type") or "json")
        if event_type == "kernel_snapshot":
            return (event_type,)
        if event_type == "llm_status":
            return (event_type,)
        if event_type == "queue_update":
            return (event_type,)
        if event_type == "code_stream_chunk":
            return (
                event_type,
                data.get("cell_id"),
                data.get("output_stream") or data.get("stream"),
            )
        if event_type == "stream_chunk":
            return (event_type, data.get("cell_id"), bool(data.get("thinking")))
        return (event_type, data.get("cell_id"))

    def _enqueue(self, client: _Client, outbound: _OutboundMessage) -> str:
        if client.closed:
            return "closed"
        outbound = _OutboundMessage(
            text=outbound.text,
            data=dict(outbound.data) if outbound.data is not None else None,
            event_type=outbound.event_type,
            key=outbound.key,
        )

        if outbound.event_type in _COALESCED_EVENTS:
            for index in range(len(client.pending) - 1, -1, -1):
                if client.pending[index].key == outbound.key:
                    client.pending[index] = outbound
                    if self._monitor is not None:
                        self._monitor.increment("websocket.messages.coalesced")
                    return "coalesced"

        if outbound.event_type in _BATCHED_EVENTS and client.pending:
            previous = client.pending[-1]
            previous_chunk = str((previous.data or {}).get("chunk") or "")
            next_chunk = str((outbound.data or {}).get("chunk") or "")
            if (
                previous.key == outbound.key
                and len((previous_chunk + next_chunk).encode("utf-8", errors="replace"))
                <= self._max_batch_bytes
            ):
                previous_data = previous.data or {}
                outbound_data = outbound.data or {}
                previous_revision = previous_data.get("revision_start", previous_data.get("revision"))
                previous.data = {
                    **previous_data,
                    "chunk": previous_chunk + next_chunk,
                    "revision_start": previous_revision,
                    "revision": outbound_data.get("revision", previous_data.get("revision")),
                    "event_id": outbound_data.get("event_id", previous_data.get("event_id")),
                }
                if self._monitor is not None:
                    self._monitor.increment("websocket.messages.batched")
                return "batched"

        if len(client.pending) >= client.max_pending:
            if self._monitor is not None:
                self._monitor.increment("websocket.clients.disconnected_overflow")
            self._close_client(client)
            return "overflow"

        client.pending.append(outbound)
        if client.drained is not None:
            client.drained.clear()
        if client.wakeup is not None:
            client.wakeup.set()
        return "queued"

    async def drain_notebook(self, notebook_id: str, timeout: float = 5.0) -> bool:
        """Wait until queued messages have been sent to current notebook clients."""
        waits = [
            client.drained.wait()
            for client in list(self._clients(notebook_id))
            if client.drained is not None and not client.closed
        ]
        if not waits:
            return True
        try:
            await asyncio.wait_for(asyncio.gather(*waits), timeout=max(0.01, timeout))
            return True
        except asyncio.TimeoutError:
            return False

    async def _broadcast(
        self,
        notebook_id: str,
        outbound: _OutboundMessage,
        *,
        exclude_send: Any = None,
    ) -> int:
        clients = self._clients(notebook_id)
        if not clients:
            return 0

        started_at = time.perf_counter()
        direct: list[_Client] = []
        overflowed: list[_Client] = []
        attempted = 0
        for client in list(clients):
            if client.send is exclude_send:
                continue
            attempted += 1
            if client.writer_task is None:
                direct.append(client)
                continue
            if self._enqueue(client, outbound) == "overflow":
                overflowed.append(client)

        for client in overflowed:
            if client in clients:
                clients.remove(client)

        failed = 0
        if direct:
            results = await asyncio.gather(
                *(client.send(outbound.render()) for client in direct),
                return_exceptions=True,
            )
            for client, result in zip(direct, results):
                if isinstance(result, BaseException):
                    failed += 1
                    if client in clients:
                        clients.remove(client)

        if not clients:
            self._connections.pop(notebook_id, None)
        if overflowed or failed:
            self._notify_client_count(notebook_id)
        if self._monitor is not None:
            self._monitor.increment("websocket.broadcasts.total")
            self._monitor.increment("websocket.messages.attempted", attempted)
            self._monitor.increment("websocket.messages.failed", failed)
            self._monitor.observe_duration(
                "websocket.broadcast_enqueue",
                (time.perf_counter() - started_at) * 1000,
                labels={"event_type": outbound.event_type},
                payload_bytes=len(outbound.render().encode("utf-8", errors="replace")),
                failed_clients=failed,
                record_event=False,
            )
        self._update_gauges()
        return len(clients)

    async def broadcast_text(
        self,
        notebook_id: str,
        message: str,
        *,
        exclude_send: Any = None,
    ) -> int:
        """Queue pre-rendered HTML/text for every live notebook client."""
        return await self._broadcast(
            notebook_id,
            _OutboundMessage(text=message, event_type="html", key=("html",)),
            exclude_send=exclude_send,
        )

    async def broadcast_json(
        self,
        notebook_id: str,
        data: dict[str, Any],
        *,
        exclude_send: Any = None,
    ) -> int:
        event_type = str(data.get("type") or "json")
        return await self._broadcast(
            notebook_id,
            _OutboundMessage(
                data=dict(data),
                event_type=event_type,
                key=self._event_key(data),
            ),
            exclude_send=exclude_send,
        )

    async def broadcast_all_json(self, data: dict[str, Any]) -> None:
        await asyncio.gather(*(
            self.broadcast_json(notebook_id, data)
            for notebook_id in self.notebook_ids()
        ))
