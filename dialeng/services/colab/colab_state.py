"""Persisted Colab runtime metadata for diagnostics and cleanup."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

STATE_DIR = Path.home() / ".dialeng"
STATE_FILE = STATE_DIR / "colab_sessions.json"


@dataclass
class ColabSessionMetadata:
    """Serializable metadata for a Dialeng-managed Colab runtime."""

    notebook_id: str
    endpoint: str
    proxy_url: str
    proxy_token_expires_at: float
    runtime_type: str
    accelerator: str = ""
    kernel_id: Optional[str] = None
    jupyter_session_id: Optional[str] = None
    connection_state: str = "disconnected"
    last_activity_at: float = 0.0
    last_activity_reason: str = ""
    updated_at: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "ColabSessionMetadata":
        return cls(
            notebook_id=str(data.get("notebook_id", "")),
            endpoint=str(data.get("endpoint", "")),
            proxy_url=str(data.get("proxy_url", "")),
            proxy_token_expires_at=float(data.get("proxy_token_expires_at", 0.0) or 0.0),
            runtime_type=str(data.get("runtime_type", "cpu")),
            accelerator=str(data.get("accelerator", "")),
            kernel_id=data.get("kernel_id"),
            jupyter_session_id=data.get("jupyter_session_id"),
            connection_state=str(data.get("connection_state", "disconnected")),
            last_activity_at=float(data.get("last_activity_at", 0.0) or 0.0),
            last_activity_reason=str(data.get("last_activity_reason", "")),
            updated_at=float(data.get("updated_at", 0.0) or 0.0),
        )


class ColabSessionStore:
    """Small JSON store keyed by notebook id.

    This is diagnostic state, not an ownership source of truth. Corrupted files
    are ignored so a broken metadata cache never blocks Dialeng startup.
    """

    def __init__(self, path: Path = STATE_FILE):
        self.path = path

    def load_all(self) -> dict[str, ColabSessionMetadata]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            return {
                key: ColabSessionMetadata.from_dict(value)
                for key, value in raw.items()
                if isinstance(value, dict)
            }
        except Exception as exc:
            logger.warning("Ignoring corrupted Colab session metadata at %s: %s", self.path, exc)
            return {}

    def get(self, notebook_id: str) -> Optional[ColabSessionMetadata]:
        return self.load_all().get(notebook_id)

    def save_all(self, sessions: dict[str, ColabSessionMetadata]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: asdict(value) for key, value in sessions.items()}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def upsert(self, metadata: ColabSessionMetadata) -> None:
        sessions = self.load_all()
        metadata.updated_at = time.time()
        sessions[metadata.notebook_id] = metadata
        self.save_all(sessions)

    def update(self, notebook_id: str, **fields) -> None:
        sessions = self.load_all()
        current = sessions.get(notebook_id)
        if not current:
            return
        for key, value in fields.items():
            if hasattr(current, key):
                setattr(current, key, value)
        current.updated_at = time.time()
        self.save_all(sessions)

    def remove(self, notebook_id: str) -> None:
        sessions = self.load_all()
        if notebook_id in sessions:
            del sessions[notebook_id]
            self.save_all(sessions)

    def prune_stale_endpoints(self, active_endpoints: set[str]) -> int:
        sessions = self.load_all()
        original = len(sessions)
        sessions = {
            notebook_id: metadata
            for notebook_id, metadata in sessions.items()
            if metadata.endpoint in active_endpoints
        }
        if len(sessions) != original:
            self.save_all(sessions)
        return original - len(sessions)
