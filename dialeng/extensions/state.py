"""Persist enabled extension IDs per canonical Dialeng project."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from collections.abc import MutableMapping, Iterator
from typing import Any


class ExtensionStateNamespace(MutableMapping[str, Any]):
    """Small JSON-backed mapping scoped to one project and extension."""

    def __init__(self, store: "ExtensionStateStore", project_root: Path, extension_id: str):
        self.store = store
        self.project_root = Path(project_root).resolve()
        self.extension_id = extension_id

    def _values(self) -> dict[str, Any]:
        return self.store.runtime_state(self.project_root, self.extension_id)

    def __getitem__(self, key: str) -> Any:
        return self._values()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        values = self._values()
        values[str(key)] = value
        self.store.set_runtime_state(self.project_root, self.extension_id, values)

    def __delitem__(self, key: str) -> None:
        values = self._values()
        del values[key]
        self.store.set_runtime_state(self.project_root, self.extension_id, values)

    def __iter__(self) -> Iterator[str]:
        return iter(self._values())

    def __len__(self) -> int:
        return len(self._values())


class ExtensionStateStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or (Path.home() / ".dialeng" / "extension_state.json"))
        self._lock = threading.RLock()

    @staticmethod
    def project_id(project_root: Path) -> str:
        canonical = str(Path(project_root).resolve())
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "projects": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "projects": {}}
        if not isinstance(payload, dict) or not isinstance(payload.get("projects"), dict):
            return {"version": 1, "projects": {}}
        return payload

    def _write(self, payload: dict) -> None:
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

    def enabled(self, project_root: Path) -> set[str]:
        with self._lock:
            project = self._read()["projects"].get(self.project_id(project_root), {})
        values = project.get("enabled", []) if isinstance(project, dict) else []
        return {str(value) for value in values}

    def set_enabled(self, project_root: Path, extension_id: str, enabled: bool) -> None:
        with self._lock:
            payload = self._read()
            key = self.project_id(project_root)
            project = payload["projects"].setdefault(key, {
                "root": str(Path(project_root).resolve()),
                "enabled": [],
            })
            values = {str(value) for value in project.get("enabled", [])}
            if enabled:
                values.add(extension_id)
            else:
                values.discard(extension_id)
            project["root"] = str(Path(project_root).resolve())
            project["enabled"] = sorted(values)
            self._write(payload)

    def runtime_state(self, project_root: Path, extension_id: str) -> dict[str, Any]:
        with self._lock:
            project = self._read()["projects"].get(self.project_id(project_root), {})
        states = project.get("runtime_state", {}) if isinstance(project, dict) else {}
        value = states.get(extension_id, {}) if isinstance(states, dict) else {}
        return dict(value) if isinstance(value, dict) else {}

    def set_runtime_state(
        self, project_root: Path, extension_id: str, values: dict[str, Any],
    ) -> None:
        # Fail early rather than corrupting the private state file with objects
        # that cannot be restored after restart.
        json.dumps(values)
        with self._lock:
            payload = self._read()
            key = self.project_id(project_root)
            project = payload["projects"].setdefault(key, {
                "root": str(Path(project_root).resolve()),
                "enabled": [],
            })
            project.setdefault("runtime_state", {})[extension_id] = values
            project["root"] = str(Path(project_root).resolve())
            self._write(payload)

    def namespace(self, project_root: Path, extension_id: str) -> ExtensionStateNamespace:
        return ExtensionStateNamespace(self, project_root, extension_id)

    def enabled_resources(self, project_root: Path, extension_id: str) -> set[str]:
        with self._lock:
            project = self._read()["projects"].get(self.project_id(project_root), {})
        resources = project.get("resources", {}) if isinstance(project, dict) else {}
        values = resources.get(extension_id, []) if isinstance(resources, dict) else []
        return {str(value) for value in values}

    def set_resource_enabled(
        self, project_root: Path, extension_id: str, kind: str, enabled: bool,
    ) -> None:
        with self._lock:
            payload = self._read()
            key = self.project_id(project_root)
            project = payload["projects"].setdefault(key, {
                "root": str(Path(project_root).resolve()),
                "enabled": [],
            })
            resources = project.setdefault("resources", {})
            values = {str(value) for value in resources.get(extension_id, [])}
            if enabled:
                values.add(kind)
            else:
                values.discard(kind)
            resources[extension_id] = sorted(values)
            self._write(payload)
