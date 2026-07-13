"""Atomic trust records for in-process Dialeng extensions."""
from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from .manifest import ExtensionSource, permission_fingerprint


class ExtensionTrustStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or (Path.home() / ".dialeng" / "extension_trust.json"))
        self._lock = threading.RLock()
        self._once: set[str] = set()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "records": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "records": {}}
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), dict):
            return {"version": 1, "records": {}}
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _record(source: ExtensionSource, mode: str) -> dict[str, str]:
        return {
            "mode": mode,
            "extension_id": source.manifest.id,
            "version": source.manifest.version,
            "source_hash": source.source_hash,
            "root": str(source.root.resolve()),
            "origin": source.origin,
            "permissions": permission_fingerprint(source.manifest),
        }

    @staticmethod
    def _key(source: ExtensionSource) -> str:
        return f"{source.manifest.id}|{source.root.resolve()}"

    def trust(self, source: ExtensionSource, mode: str = "version") -> None:
        if mode not in {"once", "version", "project"}:
            raise ValueError(f"Unsupported trust mode: {mode}")
        record = self._record(source, mode)
        key = self._key(source)
        with self._lock:
            if mode == "once":
                self._once.add(source.identity)
                return
            payload = self._read()
            payload["records"][key] = record
            self._write(payload)

    def revoke(self, source: ExtensionSource) -> None:
        with self._lock:
            self._once.discard(source.identity)
            payload = self._read()
            if payload["records"].pop(self._key(source), None) is not None:
                self._write(payload)

    def is_trusted(self, source: ExtensionSource) -> bool:
        if source.origin == "builtin" or source.identity in self._once:
            return True
        with self._lock:
            record = self._read()["records"].get(self._key(source))
        if not record:
            return False
        expected = self._record(source, str(record.get("mode") or "version"))
        if record.get("mode") == "project":
            return all(
                record.get(key) == expected[key]
                for key in ("extension_id", "root", "origin", "permissions")
            )
        return record == expected
