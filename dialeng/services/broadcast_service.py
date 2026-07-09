"""WebSocket broadcast bookkeeping for notebook clients."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, MutableMapping

SendCallable = Callable[[str], Awaitable[None]]


class BroadcastManager:
    """Owns active websocket send callables and dead-connection cleanup."""

    def __init__(self, connections: MutableMapping[str, list[SendCallable]] | None = None):
        self._connections = connections if connections is not None else {}

    def has_clients(self, notebook_id: str) -> bool:
        return bool(self._connections.get(notebook_id))

    def client_count(self, notebook_id: str) -> int:
        return len(self._connections.get(notebook_id, []))

    def notebook_ids(self) -> list[str]:
        return list(self._connections.keys())

    def add_connection(self, notebook_id: str, send: SendCallable) -> int:
        self._connections.setdefault(notebook_id, []).append(send)
        return self.client_count(notebook_id)

    def remove_connection(self, notebook_id: str, send: SendCallable) -> int:
        connections = self._connections.get(notebook_id)
        if connections and send in connections:
            connections.remove(send)
        return self.client_count(notebook_id)

    def drop_notebook(self, notebook_id: str) -> None:
        self._connections.pop(notebook_id, None)

    async def broadcast_text(
        self,
        notebook_id: str,
        message: str,
        *,
        exclude_send: Any = None,
    ) -> int:
        """Send a pre-rendered message to every live client for a notebook."""
        connections = self._connections.get(notebook_id)
        if not connections:
            return 0

        alive: list[SendCallable] = []
        for send in list(connections):
            if send is exclude_send:
                alive.append(send)
                continue
            try:
                await send(message)
                alive.append(send)
            except Exception:
                # Browser refreshes and closed tabs are expected; remove dead senders.
                pass

        self._connections[notebook_id] = alive
        return len(alive)

    async def broadcast_json(
        self,
        notebook_id: str,
        data: dict[str, Any],
        *,
        exclude_send: Any = None,
    ) -> int:
        return await self.broadcast_text(
            notebook_id,
            json.dumps(data),
            exclude_send=exclude_send,
        )

    async def broadcast_all_json(self, data: dict[str, Any]) -> None:
        for notebook_id in self.notebook_ids():
            await self.broadcast_json(notebook_id, data)
