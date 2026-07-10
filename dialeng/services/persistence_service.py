"""Crash-safe notebook writes and optional server-side autosave."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ExternalFileChangedError(RuntimeError):
    """Raised when an explicit save would overwrite a newer external edit."""


def file_signature(path: Path) -> tuple[int, int, str] | None:
    if not path.exists():
        return None
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return stat.st_mtime_ns, stat.st_size, digest


def atomic_write_json(path: Path, data: dict[str, Any]) -> tuple[int, int, str]:
    """Flush a complete JSON document beside its target, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, existing_mode)
        os.replace(temp_path, path)
        temp_path = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not support directory fsync.
            pass
        signature = file_signature(path)
        assert signature is not None
        return signature
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def save_notebook_atomic(
    notebook,
    path: Path,
    *,
    check_external: bool = True,
    overwrite_external: bool = False,
) -> tuple[int, int, str]:
    current_signature = file_signature(path)
    expected_signature = getattr(notebook, "disk_signature", None)
    if (
        check_external
        and not overwrite_external
        and expected_signature is not None
        and current_signature != expected_signature
    ):
        raise ExternalFileChangedError(
            f"{path.name} changed on disk after Dialeng loaded it. Reload it or explicitly overwrite it."
        )
    signature = atomic_write_json(path, notebook.to_ipynb())
    notebook.path = path
    notebook.disk_signature = signature
    notebook.modified = False
    return signature


@dataclass(frozen=True)
class AutosaveSettings:
    enabled: bool = False
    debounce_ms: int = 1500
    idle_only: bool = True
    recovery_snapshots: bool = True
    max_recovery_files: int = 5


class AutosaveManager:
    """Coalesce opt-in notebook writes; no task is created while disabled."""

    def __init__(
        self,
        root_provider: Callable[[], Path],
        is_idle: Callable[[str], bool],
        *,
        monitor: Any = None,
        on_saved: Callable[[str, Any, Path], Any] | None = None,
    ) -> None:
        self.root_provider = root_provider
        self.is_idle = is_idle
        self.monitor = monitor
        self.on_saved = on_saved
        self._tasks: dict[str, asyncio.Task] = {}

    def pending_count(self) -> int:
        return sum(1 for task in self._tasks.values() if not task.done())

    def has_pending(self, notebook_id: str) -> bool:
        task = self._tasks.get(notebook_id)
        return bool(task and not task.done())

    def cancel_pending(self, notebook_id: str) -> None:
        task = self._tasks.pop(notebook_id, None)
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()

    def schedule(self, notebook_id: str, notebook, settings: AutosaveSettings) -> bool:
        if not settings.enabled:
            return False
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        self.cancel_pending(notebook_id)
        self._tasks[notebook_id] = asyncio.create_task(
            self._run(notebook_id, notebook, settings),
            name=f"dialeng-autosave-{notebook_id}",
        )
        if self.monitor is not None:
            self.monitor.increment("autosave.scheduled")
            self.monitor.set_gauge("autosave.pending", self.pending_count())
        return True

    async def _run(self, notebook_id: str, notebook, settings: AutosaveSettings) -> None:
        delay = max(0.05, settings.debounce_ms / 1000)
        max_idle_wait = max(15.0, delay * 10)
        waited = 0.0
        try:
            await asyncio.sleep(delay)
            while settings.idle_only and not self.is_idle(notebook_id) and waited < max_idle_wait:
                await asyncio.sleep(min(0.5, delay))
                waited += min(0.5, delay)
            path = Path(notebook.path) if notebook.path else (
                self.root_provider() / f"{notebook_id}.ipynb"
            )
            if settings.recovery_snapshots:
                self._write_recovery(notebook_id, notebook, settings.max_recovery_files)
            save_notebook_atomic(notebook, path)
            if self.on_saved is not None:
                callback_result = self.on_saved(notebook_id, notebook, path)
                if asyncio.iscoroutine(callback_result):
                    await callback_result
            if self.monitor is not None:
                self.monitor.increment("autosave.completed")
        except asyncio.CancelledError:
            raise
        except ExternalFileChangedError as exc:
            logger.warning("Autosave skipped because the notebook changed externally (%s): %s", notebook_id, exc)
            if self.monitor is not None:
                self.monitor.increment("autosave.external_conflicts")
        except Exception:
            logger.exception("Autosave failed (notebook=%s)", notebook_id)
            if self.monitor is not None:
                self.monitor.increment("autosave.failed")
        finally:
            if self._tasks.get(notebook_id) is asyncio.current_task():
                self._tasks.pop(notebook_id, None)
            if self.monitor is not None:
                self.monitor.set_gauge("autosave.pending", self.pending_count())

    def _write_recovery(self, notebook_id: str, notebook, max_files: int) -> Path:
        recovery_dir = self.root_provider() / ".dialeng" / "recovery"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_id = notebook_id.replace("/", "_").replace("~", "_")
        path = recovery_dir / f"{safe_id}.{timestamp}.ipynb"
        atomic_write_json(path, notebook.to_ipynb())
        candidates = sorted(recovery_dir.glob(f"{safe_id}.*.ipynb"), reverse=True)
        for stale in candidates[max(1, max_files):]:
            stale.unlink(missing_ok=True)
        return path

    async def flush_all(self, timeout_seconds: float = 3.0) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        if not tasks:
            return
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout_seconds)
        except asyncio.TimeoutError:
            for task in tasks:
                task.cancel()


def find_newer_recovery(
    root: Path,
    notebook_id: str,
    canonical_path: Path,
) -> Path | None:
    safe_id = notebook_id.replace("/", "_").replace("~", "_")
    recovery_dir = root / ".dialeng" / "recovery"
    candidates = sorted(recovery_dir.glob(f"{safe_id}.*.ipynb"), reverse=True)
    if not candidates:
        return None
    newest = candidates[0]
    canonical_mtime = canonical_path.stat().st_mtime_ns if canonical_path.exists() else 0
    return newest if newest.stat().st_mtime_ns > canonical_mtime else None


def discard_recovery(path: Path, root: Path) -> None:
    resolved = path.resolve()
    recovery_root = (root / ".dialeng" / "recovery").resolve()
    if not resolved.is_relative_to(recovery_root):
        raise ValueError("Recovery path escapes the project recovery directory")
    resolved.unlink(missing_ok=True)
